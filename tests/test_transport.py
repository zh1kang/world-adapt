"""Provider transport: request validation and retries, stream assembly, provider endpoints, and the spend ledger."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
import stat
import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace

import pytest

from tests.fakes import (PANEL, PANEL_TASK, PROTOCOL, ROOT, SPEND_VARIABLES, SolverClient, Stream, read, response,
                         run_cli, solutions)
from world_adapt.agent import AgentConfig, Budgets, PilotAgent, llm, streaming
from world_adapt.agent.spending import SpendLimitExceeded, SpendingClient, raise_ledger_limit, wrap_client
from world_adapt.environments import registry
from world_adapt.predictors.simulator import SimulatorPredictor
from world_adapt.runner import study
from world_adapt.runner.config import (PROVIDERS, agent_config, build_parser, check_args, make_client, parse_args,
                                       provider_base_url)


# Requests ----------------------------------------------------------------------------------------------------------


def invoke(callback, usage):
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=callback)))
    return llm.request(client, [], None, model='test', reasoning_effort=None,
                       max_completion_tokens=8, completion_window=None, max_retries=2,
                       retry_wait=0, log=lambda event: None, planning_usage=usage)


def test_http_400_is_not_retried():
    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        error = RuntimeError('invalid request')
        error.status_code = 400
        raise error
    with pytest.raises(llm.InfrastructureError):
        invoke(fail, {})
    assert len(calls) == 1


def test_timeout_is_bounded_to_two_requests():
    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        raise TimeoutError('timeout')
    usage = {}
    with pytest.raises(llm.InfrastructureError):
        invoke(fail, usage)
    assert len(calls) == usage['planning_requests'] == usage['planning_unknown_requests'] == 2


def test_transient_failure_retries_without_changing_messages():
    calls = []
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None))])
    def reply(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ConnectionError('lost reply')
        return response
    usage = {}
    assert invoke(reply, usage) is response
    assert calls[0] == calls[1]
    assert usage['planning_unknown_requests'] == 1


@pytest.mark.parametrize(("status", "header", "waited"), [
    (429, "7", 7.0), (503, "1.5", 1.5), (429, "3600", 60.0), (529, "5", 5.0),
    (429, "soon", 10.0), (429, None, 10.0), (500, "7", 10.0)])
def test_retry_after_is_honoured_up_to_a_bound_for_429_and_503(monkeypatch, status, header, waited):
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None))])
    calls = []

    def reply(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            error = RuntimeError("busy")
            error.status_code = status
            error.response = SimpleNamespace(headers={} if header is None else {"retry-after": header})
            raise error
        return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=reply)))
    assert llm.request(client, [], None, model="m", reasoning_effort=None, max_completion_tokens=8,
                       completion_window=None, max_retries=2, retry_wait=10, log=lambda event: None) is response
    assert sleeps == [waited]


def bare_response(*, usage=None, choices=None):
    return SimpleNamespace(
        choices=choices if choices is not None else [SimpleNamespace(message=SimpleNamespace(tool_calls=[]))],
        usage=usage,
    )


def usage_record(prompt=10, completion=5, total=15):
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


def test_required_usage_rejects_missing_or_inconsistent_accounting():
    with pytest.raises(llm.InfrastructureError, match="did not contain usage"):
        llm.validate_response(bare_response(), require_usage=True)
    with pytest.raises(llm.InfrastructureError, match="smaller"):
        llm.validate_response(bare_response(usage=usage_record(total=10)), require_usage=True)
    assert llm.validate_response(bare_response(usage=usage_record()), require_usage=True).choices


def test_response_without_choices_is_rejected_before_dispatch():
    with pytest.raises(llm.InfrastructureError, match="did not contain a choice"):
        llm.validate_response(bare_response(choices=[]))


def test_request_raises_infrastructure_error_for_missing_required_usage():
    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**_kwargs):
                    return bare_response()

    events = []
    with pytest.raises(llm.InfrastructureError, match="did not contain usage"):
        llm.request(Client(), [], None, model="test", reasoning_effort=None, max_completion_tokens=None,
                    completion_window=None, max_retries=1, log=events.append, require_usage=True)
    assert events == [{"type": "response_validation_error", "error": "model response did not contain usage for required accounting",
                       "usage_required": True}]


def test_request_retries_transient_failure_without_unbound_response():
    class Client:
        def __init__(self):
            self.calls = 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("temporary")
            return bare_response(usage=usage_record())

    events = []
    client = Client()
    result = llm.request(client, [], None, model="test", reasoning_effort=None, max_completion_tokens=None,
                         completion_window=None, max_retries=2, retry_wait=0, log=events.append)
    assert result.choices and client.calls == 2
    assert [event["type"] for event in events] == ["api_error"]


def test_request_exhaustion_reports_infrastructure_error_after_all_retries():
    class Client:
        def __init__(self):
            self.calls = 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **_kwargs):
            self.calls += 1
            raise ConnectionError("offline")

    events = []
    with pytest.raises(llm.InfrastructureError, match="failed 2 times"):
        llm.request(Client(), [], None, model="test", reasoning_effort=None, max_completion_tokens=None,
                    completion_window=None, max_retries=2, retry_wait=0, log=events.append)
    assert len(events) == 2 and all(event["type"] == "api_error" for event in events)


def test_streaming_assembles_fragments_by_tool_index_and_requires_terminal_usage():
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "think", "content": "go", "tool_calls": [
            {"index": 1, "id": "call-", "function": {"name": "exe", "arguments": '{"actions":["'}},
            {"index": 0, "id": "call-0", "function": {"name": "con", "arguments": '{"reason":"'}}
        ]}}]},
        {"choices": [{"delta": {"content": " now", "tool_calls": [
            {"index": 1, "id": "1", "function": {"name": "cute", "arguments": 'right"]}'}},
            {"index": 0, "id": "", "function": {"name": "cede", "arguments": 'x"}'}}
        ]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
         "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}},
    ]
    result = streaming.assemble(chunks)
    choice = result.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content == "go now" and choice.message.reasoning_content == "think"
    assert [call.id for call in choice.message.tool_calls] == ["call-0", "call-1"]
    assert [call.function.name for call in choice.message.tool_calls] == ["concede", "execute"]
    assert choice.message.tool_calls[1].function.arguments == '{"actions":["right"]}'
    assert choice.message.model_dump(exclude_none=True)["tool_calls"][0]["function"]["name"] == "concede"
    with pytest.raises(llm.InfrastructureError, match="without a terminal"):
        streaming.assemble(chunks[:1])
    with pytest.raises(llm.InfrastructureError, match="without final usage"):
        streaming.assemble(chunks[:2] + [{"choices": [{"delta": {}, "finish_reason": "stop"}]}])
    with pytest.raises(llm.InfrastructureError, match="after an early usage"):
        streaming.assemble(chunks[:1] + [{"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                                                    "total_tokens": 2}}, chunks[1],
                            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}])
    with pytest.raises(llm.InfrastructureError, match="after the terminal"):
        streaming.assemble(chunks + [{"choices": [{"delta": {"content": "late"}}]}])
    with pytest.raises(llm.InfrastructureError, match="multiple choices"):
        streaming.assemble([{"choices": [{"delta": {"content": "a"}}, {"delta": {"content": "b"}}]}])
    with pytest.raises(llm.InfrastructureError, match="nonempty ids and names"):
        streaming.assemble([{"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "execute", "arguments": "{}"}}]},
            "finish_reason": "tool_calls"}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}])


def test_streaming_create_enables_final_usage_and_fails_closed_on_eof():
    class Stream:
        def __init__(self, chunks):
            self.chunks = chunks
            self.closed = False

        def __iter__(self):
            return iter(self.chunks)

        def close(self):
            self.closed = True

    class Client:
        def __init__(self):
            self.kwargs = None
            self.stream = Stream([{"choices": [{"delta": {}, "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}])
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            self.kwargs = kwargs
            return self.stream

    client = Client()
    result = streaming.create(client, model="test", messages=[])
    assert result.usage.total_tokens == 2
    assert client.kwargs["stream"] is True and client.kwargs["stream_options"] == {"include_usage": True}
    assert client.stream.closed

    class BrokenStream(Stream):
        def __iter__(self):
            yield {"choices": [{"delta": {"content": "partial"}}]}
            raise RuntimeError("connection reset")

    client.stream = BrokenStream([])
    with pytest.raises(llm.InfrastructureError, match="stream failed"):
        streaming.create(client, model="test", messages=[])
    assert client.stream.closed


def test_planning_retry_wait_is_bounded_by_remaining_task_time(monkeypatch):
    now, requests, waits, usage = [0.0], [], [], {}
    monkeypatch.setattr(llm.time, "monotonic", lambda: now[0])

    def wait(seconds):
        waits.append(seconds)
        now[0] += seconds

    def create(**kwargs):
        requests.append(kwargs)
        raise TimeoutError("uncertain")

    monkeypatch.setattr(llm.time, "sleep", wait)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(llm.InfrastructureError) as caught:
        llm.request(client, [], [], model="fake", reasoning_effort=None, max_completion_tokens=20,
                    completion_window=None, max_retries=5, retry_wait=10, log=lambda e: None,
                    deadline=2.0, planning_usage=usage)
    assert not isinstance(caught.value, llm.PlanningDeadlineExceeded)
    assert len(requests) == 1 and requests[0]["timeout"] == 2.0
    assert waits == [2.0] and now[0] == 2.0
    assert usage["planning_requests"] == usage["planning_unknown_requests"] == 1


@pytest.mark.parametrize("deadline", [None, 7200.0])
def test_remaining_task_time_never_increases_the_transport_timeout(monkeypatch, deadline):
    requests = []
    monkeypatch.setattr(llm.time, "monotonic", lambda: 0.0)

    def create(**kwargs):
        requests.append(kwargs)
        return response("ready")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm.request(client, [], [], model="fake", reasoning_effort=None, max_completion_tokens=20,
                completion_window=None, max_retries=1, log=lambda e: None, deadline=deadline, planning_usage={})
    assert requests[0]["timeout"] == llm.REQUEST_TIMEOUT_SECONDS


def test_streamed_replies_drive_the_agent_and_every_stream_is_closed():
    class RawClient:
        def __init__(self):
            self.streams = []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            if not kwargs.get("tools"):
                return response("streamed notes")
            stream = Stream([{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {
                                 "name": "execute", "arguments": '{"actions":["press 1"]}'}}]}}]},
                             {"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                              "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}}])
            self.streams.append(stream)
            return stream

    raw = RawClient()
    agent = PilotAgent(streaming.StreamingChatAdapter(raw, require_usage=True), AgentConfig(model="m", require_usage=True))
    result, notebook = agent.run_task(PANEL, PANEL_TASK, SimulatorPredictor(PANEL), Budgets(2, 2), "", goal="g")
    assert result.success and notebook == "streamed notes"
    assert len(raw.streams) == 1 and raw.streams[0].closed


def test_a_slow_stream_stops_being_consumed_at_the_task_deadline(monkeypatch):
    import world_adapt.episode as episode
    now = [0.0]
    monkeypatch.setattr(episode.time, "monotonic", lambda: now[0])

    class Slow(Stream):
        reads = 0

        def __iter__(self):
            for i in range(4):
                self.reads += 1
                if i:
                    now[0] = 2.0
                yield {"choices": [{"delta": {"content": "slow"}}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

    slow = Slow([])
    raw = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: slow if kwargs.get("tools") else response("notes"))))
    with pytest.raises(llm.InfrastructureError, match="deadline"):
        PilotAgent(streaming.StreamingChatAdapter(raw), AgentConfig(model="m")).run_task(
            PANEL, PANEL_TASK, SimulatorPredictor(PANEL), Budgets(6, 6, max_task_seconds=1), "", goal="g")
    assert slow.reads == 2 and slow.closed


class StreamingSolver(SolverClient):
    """Streams reference solutions; can omit the final usage record of play or notebook replies."""

    def __init__(self, solved, missing_usage=None):
        super().__init__(solved, stream=True)
        self.missing_usage = missing_usage

    def create(self, **kwargs):
        reply = super().create(**kwargs)
        if kwargs.get("tools"):
            assert kwargs["stream"] and kwargs["stream_options"]["include_usage"]
            if self.missing_usage == "play":
                reply.chunks = reply.chunks[:-1]
        elif self.missing_usage == "notebook":
            reply.usage = None
        return reply


def test_paid_usage_settings_reach_the_agent_from_runner_arguments():
    parser = build_parser()
    args = parse_args(parser, ["--env", "switchboard", "--require-usage", "--stream"])
    check_args(parser, args)
    assert args.stream and agent_config(args, registry.get(args.env)).require_usage


@pytest.mark.parametrize("missing_usage", [None, "play", "notebook"])
def test_streamed_runs_require_usage_before_a_task_counts(tmp_path, monkeypatch, missing_usage):
    argv = PROTOCOL + ["--stream", "--require-usage", "--infra-retries", "0"]
    client = StreamingSolver(solutions(argv), missing_usage)
    if missing_usage == "play":
        with pytest.raises(SystemExit) as stop:
            run_cli(argv, tmp_path, client, monkeypatch)
        assert stop.value.code == 2
        run_dir = next((tmp_path / "runs").iterdir())
    else:
        run_dir = run_cli(argv, tmp_path, client, monkeypatch)
    summary = read(run_dir, "summary.json")
    events = read(run_dir, "events.jsonl")
    assert all(stream.closed for stream in client.streams)
    if missing_usage == "play":
        assert summary["status"] == "incomplete" and summary["tasks_completed"] == 0
        assert len(summary["infrastructure_failures"]) == 1 and not any(e["type"] == "execute" for e in events)
    else:
        assert summary["status"] == "complete" and summary["success_rate"] == 1.0
    if missing_usage == "notebook":
        assert all(row["usage"]["writer_unknown_requests"] == 1 for row in summary["results"])


# Providers ---------------------------------------------------------------------------------------------------------


def test_modal_provider_uses_configured_endpoint(monkeypatch):
    monkeypatch.setattr('world_adapt.runner.config.load_env', lambda: None)
    monkeypatch.setenv('MODAL_BASE_URL', 'https://example.modal.direct/v1/')
    args = parse_args(build_parser(), ['--provider', 'modal', '--model', 'qualification-model'])
    check_args(build_parser(), args)
    assert provider_base_url(args.provider) == 'https://example.modal.direct/v1'
    assert args.model == 'qualification-model'


def test_modal_missing_endpoint_is_an_error(monkeypatch):
    monkeypatch.setattr('world_adapt.runner.config.load_env', lambda: None)
    monkeypatch.delenv('MODAL_BASE_URL', raising=False)
    with pytest.raises(SystemExit, match='MODAL_BASE_URL'):
        provider_base_url('modal')


def test_modal_requires_explicit_model():
    parser = build_parser()
    args = parse_args(parser, ['--provider', 'modal'])
    with pytest.raises(SystemExit):
        check_args(parser, args)


@pytest.mark.parametrize('url', [
    'http://example.modal.direct/v1',
    'https://unrelated.example/v1',
    'https://example.modal.direct.evil.example/v1',
    'https://user:secret@example.modal.direct/v1',
    'https://example.modal.direct/v1?key=secret',
    'https://example.modal.direct/v1#fragment',
])
def test_modal_rejects_unsafe_credential_destinations(monkeypatch, url):
    monkeypatch.setattr('world_adapt.runner.config.load_env', lambda: None)
    monkeypatch.setenv('MODAL_BASE_URL', url)
    with pytest.raises(SystemExit, match='HTTPS Modal endpoint'):
        provider_base_url('modal')


# Spend ledger ------------------------------------------------------------------------------------------------------


def usage_reply(prompt=10, completion=5):
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion))


def raw_client(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def wrapped(path, create, limit=0.0002):
    return SpendingClient(raw_client(create), path, limit=limit, input_price=1, output_price=1,
                          input_limit=100, agent="test")


def test_usage_releases_unused_reservation_and_persists_across_clients(tmp_path):
    path = tmp_path / "spend.json"
    wrapped(path, lambda **kw: usage_reply()).create(max_completion_tokens=50)
    wrapped(path, lambda **kw: usage_reply()).create(max_completion_tokens=50)
    data = json.loads(path.read_text())
    assert data["charged_usd"] == pytest.approx(0.00003)
    assert data["reservations"] == {}
    assert data["completed_requests"] == 2


def test_ledger_directory_is_synced_before_provider_submission(tmp_path, monkeypatch):
    synced = []
    fsync = os.fsync

    def record_sync(fd):
        synced.append(stat.S_ISDIR(os.fstat(fd).st_mode))
        fsync(fd)

    monkeypatch.setattr(os, 'fsync', record_sync)

    def provider(**kwargs):
        assert True in synced, 'the reservation rename must be durable before submission'
        return usage_reply()

    wrapped(tmp_path / 'spend.json', provider).create(max_completion_tokens=50)
    assert synced.count(True) == 2  # Reservation and final charge.


def test_parallel_reservations_cannot_exceed_limit(tmp_path):
    path = tmp_path / "spend.json"
    def lost(**kw):
        raise ConnectionError("lost response")
    def attempt(_):
        try:
            wrapped(path, lost).create(max_completion_tokens=50)
        except (ConnectionError, SpendLimitExceeded) as error:
            return type(error).__name__
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(attempt, range(4)))
    assert outcomes.count("ConnectionError") == 1
    assert outcomes.count("SpendLimitExceeded") == 3
    data = json.loads(path.read_text())
    assert len(data["reservations"]) == 1
    assert next(iter(data["reservations"].values()))["status"] == "unknown"


def test_missing_usage_is_an_infrastructure_error_that_keeps_its_reservation(tmp_path):
    path = tmp_path / "spend.json"
    with pytest.raises(llm.InfrastructureError, match="usage"):
        wrapped(path, lambda **kw: SimpleNamespace()).create(max_completion_tokens=50)
    assert [r["status"] for r in json.loads(path.read_text())["reservations"].values()] == ["unknown"]
    with pytest.raises(SpendLimitExceeded, match="limit"):
        wrapped(path, lambda **kw: usage_reply()).create(max_completion_tokens=50)


def http_error(status):
    error = RuntimeError(f"HTTP {status}")
    error.status_code = status
    return error


def test_rejected_http_requests_are_released_and_do_not_exhaust_the_cap(tmp_path):
    path = tmp_path / "spend.json"
    codes = iter([400, 503, 429, 400, 503, 400, 503, 429])

    def rejected(**kw):
        raise http_error(next(codes))

    for _ in range(8):  # Each reservation alone would be 0.00015 of the 0.0002 cap.
        with pytest.raises(RuntimeError, match="HTTP"):
            wrapped(path, rejected).create(max_completion_tokens=50)
    data = json.loads(path.read_text())
    assert data["reservations"] == {} and data["charged_usd"] == 0
    assert data["rejected_requests"] == 8 and data["completed_requests"] == 0
    wrapped(path, lambda **kw: usage_reply()).create(max_completion_tokens=50)


@pytest.mark.parametrize("error", [TimeoutError("timed out"), ConnectionError("reset"), http_error(200),
                                   http_error(408), http_error(500), http_error(502), http_error(504)])
def test_requests_that_may_have_been_billed_stay_unknown(tmp_path, error):
    """A timeout, a connection error, a mid-stream cut-off (SDKs report the already-sent 200) and an ambiguous
    gateway or server status may all arrive after the model ran, so the request may have been paid."""
    path = tmp_path / "spend.json"

    def lost(**kw):
        raise error

    with pytest.raises(type(error)):
        wrapped(path, lost).create(max_completion_tokens=50)
    data = json.loads(path.read_text())
    assert [r["status"] for r in data["reservations"].values()] == ["unknown"]
    assert data.get("rejected_requests", 0) == 0


def test_an_unknown_request_no_longer_closes_submissions_by_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_ADAPT_BLOCK_UNKNOWN_REQUESTS", "1")
    path = tmp_path / "spend.json"
    with pytest.raises(ConnectionError):
        wrapped(path, lambda **kw: (_ for _ in ()).throw(ConnectionError("lost")), limit=1).create(max_completion_tokens=50)
    wrapped(path, lambda **kw: usage_reply(), limit=1).create(max_completion_tokens=50)


def test_default_wrapper_preserves_client(monkeypatch):
    monkeypatch.delenv("WORLD_ADAPT_SPEND_LEDGER", raising=False)
    obj = raw_client(lambda **kw: usage_reply())
    assert wrap_client(obj) is obj


def test_spend_settings_cannot_disable_guard_with_an_empty_path(monkeypatch):
    monkeypatch.setenv('WORLD_ADAPT_SPEND_LEDGER', '')
    monkeypatch.setenv('WORLD_ADAPT_SPEND_LIMIT_USD', '475')
    with pytest.raises(ValueError, match='nonempty'):
        wrap_client(raw_client(lambda **kw: usage_reply()))


def test_rate_changes_cannot_reuse_ledger(tmp_path):
    path = tmp_path / "spend.json"
    wrapped(path, lambda **kw: usage_reply()).create(max_completion_tokens=50)
    changed = SpendingClient(raw_client(lambda **kw: usage_reply()), path, limit=.0002,
                             input_price=2, output_price=1, input_limit=100, agent="test")
    with pytest.raises(ValueError, match="rate"):
        changed.create(max_completion_tokens=50)


def test_reported_cached_tokens_use_the_frozen_cached_price(tmp_path):
    reply = usage_reply(10, 5)
    reply.usage.prompt_tokens_details = SimpleNamespace(cached_tokens=8)
    path = tmp_path / 'spend.json'
    obj = SpendingClient(raw_client(lambda **kw: reply), path, limit=.01, input_price=1,
                         cached_input_price=.1, output_price=1, input_limit=100, agent='test')
    obj.create(max_completion_tokens=50)
    data = json.loads(path.read_text())
    assert data['charged_usd'] == pytest.approx(.0000078)
    assert data['agents']['test']['cached_tokens'] == 8


def test_invalid_ledger_amounts_stop_before_a_provider_request(tmp_path):
    path = tmp_path / 'spend.json'
    obj = wrapped(path, lambda **kw: usage_reply())
    obj.create(max_completion_tokens=50)
    data = json.loads(path.read_text())
    data['charged_usd'] = -1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='invalid spend ledger'):
        obj.create(max_completion_tokens=50)


def test_streaming_transport_preserves_cache_usage_for_spend_accounting(tmp_path):
    from world_adapt.agent.streaming import assemble

    reply = assemble([
        {'choices': [{'index': 0, 'delta': {'content': 'done'}, 'finish_reason': 'stop'}]},
        {'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15,
                                 'prompt_tokens_details': {'cached_tokens': 8}}},
    ])
    path = tmp_path / 'spend.json'
    obj = SpendingClient(raw_client(lambda **kw: reply), path, limit=.01, input_price=1,
                         cached_input_price=.1, output_price=1, input_limit=100, agent='test')
    obj.create(max_completion_tokens=50)
    assert json.loads(path.read_text())['charged_usd'] == pytest.approx(.0000078)


def test_a_spend_stop_exits_75_and_keeps_completed_tasks_scored(tmp_path):
    script = tmp_path / "stop.py"
    script.write_text(textwrap.dedent('''
        import runpy
        import sys
        from unittest.mock import patch
        from tests.fakes import PROTOCOL, SolverClient, plan_observations, solutions
        from world_adapt.agent.spending import SpendLimitExceeded

        second = plan_observations(PROTOCOL)[1]

        class StopClient(SolverClient):
            def create(self, **kwargs):
                if self.task(kwargs["messages"]) == second:
                    raise SpendLimitExceeded("test spend limit reached")
                return super().create(**kwargs)

        sys.argv = ["world_adapt.runner.run", *PROTOCOL, "--label", "spend-stop"]
        with patch("world_adapt.runner.config.make_client", return_value=StopClient(solutions(PROTOCOL))):
            runpy.run_module("world_adapt.runner.run", run_name="__main__")
    '''))
    environment = {k: v for k, v in os.environ.items() if k not in SPEND_VARIABLES}
    environment["PYTHONPATH"] = str(ROOT)
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=environment,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 75, result.stderr
    run = next((tmp_path / "runs").iterdir())
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    complete = [e for e in events if e["type"] == "task_complete"]
    attempts = [e for e in events if e["type"] == "attempt_start"]
    assert len(complete) == 1 and complete[0]["result"]["success"] is True
    assert len(attempts) == 2 and attempts[-1]["task_id"] not in {e["task_id"] for e in complete}


def test_a_spend_stop_pauses_only_jobs_that_share_the_ledger(tmp_path):
    job = study.Job("active", "a", "patrol", "faulty-predictor-saved-notebook", 0, "adjacent_capture", "preserve")
    shared = study.Job("shared", "b", "patrol", "faulty-predictor-reset-notebook", 0, "adjacent_capture", "reset")
    other = study.Job("other", "c", "patrol", "faulty-predictor-reset-notebook", 1, "adjacent_capture", "reset")
    ledger = lambda name: {"WORLD_ADAPT_SPEND_LEDGER": f"outputs/{name}/spend-ledger.json"}
    agents = {"a": study.Agent("a", tmp_path / "a.json", "sail", ledger("one")),
              "b": study.Agent("b", tmp_path / "b.json", "sail", ledger("one")),
              "c": study.Agent("c", tmp_path / "c.json", "sail", ledger("two"))}
    scheduler = study.Scheduler([shared, other], agents, runs=tmp_path / "runs", logs=tmp_path / "logs",
                                workers=1, max_attempts=3)
    scheduler.running[job.label] = (SimpleNamespace(poll=lambda: 75), job, 0)
    scheduler.attempts[job.label] = 1
    (tmp_path / "logs").mkdir()
    scheduler.reap()
    assert scheduler.attempts[job.label] == 0  # A pause does not use up an attempt.
    assert scheduler.queue == [other] and scheduler.running == {}
    assert scheduler.results["active"].startswith("paused:") and scheduler.results["shared"].startswith("paused:")
    assert "other" not in scheduler.results


# Modal proxy auth ---------------------------------------------------------------------------------------------------


def modal_client(monkeypatch, **environment):
    monkeypatch.setattr("world_adapt.runner.config.load_env", lambda: None)
    for name in ("MODAL_PROXY_KEY", "MODAL_PROXY_SECRET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODAL_API_KEY", "bearer-token")
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    return make_client("modal", "https://example.modal.direct/v1", "MODAL_API_KEY")


def test_modal_proxy_credentials_are_sent_as_default_headers_when_set(monkeypatch):
    headers = modal_client(monkeypatch, MODAL_PROXY_KEY="wk-id", MODAL_PROXY_SECRET="ws-secret").default_headers
    assert headers["Modal-Key"] == "wk-id" and headers["Modal-Secret"] == "ws-secret"
    assert modal_client(monkeypatch).auth_headers == {"Authorization": "Bearer bearer-token"}
    assert "Modal-Key" not in modal_client(monkeypatch).default_headers


def test_modal_proxy_credentials_must_be_set_together(monkeypatch):
    with pytest.raises(SystemExit, match="together"):
        modal_client(monkeypatch, MODAL_PROXY_KEY="wk-id")


def test_other_providers_never_receive_modal_proxy_headers(monkeypatch):
    monkeypatch.setattr("world_adapt.runner.config.load_env", lambda: None)
    monkeypatch.setenv("SAIL_API_KEY", "key")
    monkeypatch.setenv("MODAL_PROXY_KEY", "wk-id")
    monkeypatch.setenv("MODAL_PROXY_SECRET", "ws-secret")
    assert "Modal-Key" not in make_client("sail", "https://api.sailresearch.com/v1", "SAIL_API_KEY").default_headers


# Anthropic ---------------------------------------------------------------------------------------------------------


class Block(SimpleNamespace):
    def model_dump(self, exclude_none=False):
        return dict(vars(self))


def thinking(text="plan", signature="sig-1"):
    return Block(type="thinking", thinking=text, signature=signature)


def tool_use(call_id, name, **arguments):
    return Block(type="tool_use", id=call_id, name=name, input=arguments)


def native_reply(*content, stop="tool_use", usage=(10, 5, 0), model="claude-opus-5-5"):
    fresh, output, cached = usage
    return SimpleNamespace(
        id="msg_1", model=model, stop_reason=stop, content=list(content),
        usage=SimpleNamespace(input_tokens=fresh, output_tokens=output, cache_read_input_tokens=cached,
                              cache_creation_input_tokens=0))


class NativeStream:
    def __init__(self, reply, events=("event",), failure=None):
        self.reply, self.events, self.failure = reply, events, failure

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        yield from self.events
        if self.failure is not None:
            raise self.failure

    def get_final_message(self):
        return self.reply


class NativeSdk:
    """Replays scripted Messages API streams, or raises a scripted error when the request is opened."""

    def __init__(self, *script):
        self.script, self.requests = list(script), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self.stream))

    def stream(self, **options):
        self.requests.append(options)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item if isinstance(item, NativeStream) else NativeStream(item)


def status_error_for(status, **headers):
    import anthropic
    import httpx2 as httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIStatusError("provider error", response=httpx.Response(status, headers=headers, request=request),
                                    body=None)


TOOLS = [{"type": "function", "function": {"name": "execute", "description": "Apply actions.",
                                            "parameters": {"type": "object", "properties": {}}}}]
OPENING = [{"role": "system", "content": "Be brief."}, {"role": "system", "content": "Play."},
           {"role": "user", "content": "Observe."}]


def ask(adapter, messages=OPENING, **kwargs):
    kwargs = {"model": "claude-opus-5-5", "reasoning_effort": "high", "max_completion_tokens": 100, **kwargs}
    return adapter.chat.completions.create(messages=messages, tools=TOOLS, **kwargs)


def test_chat_messages_and_tools_become_a_messages_request_with_adaptive_thinking():
    from world_adapt.agent.anthropic import THINKING_BINDING_BETA, AnthropicChatAdapter

    sdk = NativeSdk(native_reply(thinking(), tool_use("toolu_1", "execute", actions=["a"])),
                    native_reply(Block(type="text", text="done"), stop="end_turn"))
    adapter = AnthropicChatAdapter(sdk)
    first = ask(adapter)
    message = llm.assistant_message(first.choices[0].message)
    history = [*OPENING, message, {"role": "tool", "tool_call_id": "toolu_1", "name": "execute", "content": "{}"},
               {"role": "user", "content": "Respond with a tool call."}]
    ask(adapter, history, timeout=30)
    opening, second = sdk.requests
    assert opening["system"] == "Be brief.\n\nPlay." and opening["max_tokens"] == 100
    assert opening["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Observe."}]}]
    assert opening["tools"] == [{"name": "execute", "description": "Apply actions.",
                                 "input_schema": {"type": "object", "properties": {}}}]
    assert opening["thinking"] == {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}}
    assert opening["output_config"] == {"effort": "high"} and opening["betas"] == [THINKING_BINDING_BETA]
    assert second["timeout"] == 30
    # The tool result and the following prompt share one user turn, results first.
    assert second["messages"][-1] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "{}"},
        {"type": "text", "text": "Respond with a tool call."}]}


def test_a_tool_call_reply_is_parsed_into_a_chat_response_without_exposing_thinking():
    from world_adapt.agent.anthropic import AnthropicChatAdapter

    sdk = NativeSdk(native_reply(thinking("secret plan"), Block(type="text", text="Pressing."),
                                 tool_use("toolu_1", "execute", actions=["press 1"]), usage=(7, 5, 3)))
    reply = ask(AnthropicChatAdapter(sdk))
    choice = reply.choices[0]
    [call] = choice.message.tool_calls
    assert choice.finish_reason == "tool_calls" and choice.message.content == "Pressing."
    assert (call.id, call.function.name, json.loads(call.function.arguments)) == ("toolu_1", "execute",
                                                                                  {"actions": ["press 1"]})
    assert "secret plan" not in json.dumps(llm.assistant_message(choice.message))
    assert (reply.usage.prompt_tokens, reply.usage.completion_tokens, reply.usage.total_tokens) == (10, 5, 15)
    assert reply.usage.prompt_tokens_details.cached_tokens == 3
    llm.validate_response(reply, require_usage=True)


def test_thinking_blocks_round_trip_into_the_next_request_and_a_new_conversation_forgets_them():
    from world_adapt.agent.anthropic import AnthropicChatAdapter

    blocks = [thinking(signature="sig-9"), tool_use("toolu_1", "execute", actions=["a"])]
    sdk = NativeSdk(native_reply(*blocks), native_reply(Block(type="text", text="ok"), stop="end_turn"),
                    native_reply(Block(type="text", text="fresh"), stop="end_turn"))
    adapter = AnthropicChatAdapter(sdk)
    message = llm.assistant_message(ask(adapter).choices[0].message)
    tool = {"role": "tool", "tool_call_id": "toolu_1", "name": "execute", "content": "{}"}
    ask(adapter, [*OPENING, message, tool])
    assert sdk.requests[1]["messages"][1] == {"role": "assistant", "content": [b.model_dump() for b in blocks]}
    ask(adapter)  # No assistant message: a new conversation.
    assert list(adapter.history.values()) == [[{"type": "text", "text": "fresh"}]]
    with pytest.raises(ValueError, match="native blocks"):
        adapter.translate([*OPENING, message, tool])


def test_a_text_reply_is_restored_from_its_own_blocks_and_unknown_text_falls_back_to_text():
    from world_adapt.agent.anthropic import AnthropicChatAdapter

    adapter = AnthropicChatAdapter(NativeSdk())
    _, native = adapter.translate([*OPENING, {"role": "assistant", "content": "[reply cut off]"},
                                   {"role": "user", "content": "Shorter."}])
    assert native[1] == {"role": "assistant", "content": [{"type": "text", "text": "[reply cut off]"}]}


@pytest.mark.parametrize(("effort", "thinking", "output_config"), [
    ("none", {"type": "disabled"}, None), (None, None, None), ("max", "adaptive", {"effort": "max"})])
def test_reasoning_effort_maps_to_thinking_settings(effort, thinking, output_config):
    from world_adapt.agent.anthropic import AnthropicChatAdapter

    sdk = NativeSdk(native_reply(Block(type="text", text="ok"), stop="end_turn"))
    ask(AnthropicChatAdapter(sdk), reasoning_effort=effort)
    options = sdk.requests[0]
    assert (options.get("thinking") or {}).get("type") == (thinking["type"] if isinstance(thinking, dict) else thinking)
    assert options.get("output_config") == output_config and ("betas" in options) == (effort == "max")


@pytest.mark.parametrize("options", [{"reasoning_effort": "extreme"}, {"seed": 1}, {"tool_choice": "required"},
                                     {"metadata": {"completion_window": "asap"}}, {"max_completion_tokens": None}])
def test_unsupported_request_options_are_rejected_before_any_request(options):
    from world_adapt.agent.anthropic import AnthropicChatAdapter

    sdk = NativeSdk()
    with pytest.raises(ValueError):
        ask(AnthropicChatAdapter(sdk), **options)
    assert sdk.requests == []


@pytest.mark.parametrize(("reply", "match"), [
    (native_reply(Block(type="text", text="x"), stop="refusal"), "stop reason"),
    (native_reply(Block(type="server_tool_use"), stop="end_turn"), "block"),
    (native_reply(Block(type="text", text="x"), stop="end_turn", model="claude-other"), "served model"),
])
def test_unsupported_replies_are_infrastructure_errors(reply, match):
    from world_adapt.agent.anthropic import AnthropicChatAdapter

    with pytest.raises(llm.InfrastructureError, match=match):
        ask(AnthropicChatAdapter(NativeSdk(reply)))


def test_cache_writes_are_refused_because_they_are_not_priced():
    from world_adapt.agent.anthropic import AnthropicChatAdapter

    reply = native_reply(Block(type="text", text="x"), stop="end_turn")
    reply.usage.cache_creation_input_tokens = 4
    with pytest.raises(llm.InfrastructureError, match="cache"):
        ask(AnthropicChatAdapter(NativeSdk(reply)))


def test_a_stream_that_never_finishes_fails_at_the_request_deadline(monkeypatch):
    from world_adapt.agent import anthropic as module

    clock = iter([0.0, 5.0, 5.0])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    with pytest.raises(llm.InfrastructureError, match="deadline"):
        ask(module.AnthropicChatAdapter(NativeSdk(NativeStream(None))), timeout=1)


def anthropic_ledger(tmp_path, sdk):
    from world_adapt.agent.anthropic import AnthropicChatAdapter

    client = SpendingClient(AnthropicChatAdapter(sdk), tmp_path / "spend.json", limit=1, input_price=5,
                            cached_input_price=.5, output_price=25, input_limit=2000, agent="claude")
    return client, tmp_path / "spend.json"


def test_native_usage_is_charged_to_the_spend_ledger_including_cache_reads(tmp_path):
    client, path = anthropic_ledger(tmp_path, NativeSdk(native_reply(Block(type="text", text="x"), stop="end_turn",
                                                                       usage=(100, 40, 900))))
    ask(client)
    data = json.loads(path.read_text())
    assert data["charged_usd"] == pytest.approx((100 * 5 + 900 * .5 + 40 * 25) / 1e6)
    assert data["agents"]["claude"]["cached_tokens"] == 900 and data["agents"]["claude"]["prompt_tokens"] == 1000
    assert data["reservations"] == {}


def test_a_429_and_a_529_are_retried_honouring_retry_after_and_are_never_billed(tmp_path, monkeypatch):
    from world_adapt.agent.anthropic import AnthropicChatAdapter

    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    ok = native_reply(Block(type="text", text="x"), stop="end_turn")
    sdk = NativeSdk(status_error_for(429, **{"retry-after": "4"}), status_error_for(529), ok)
    path = tmp_path / "spend.json"
    client = SpendingClient(AnthropicChatAdapter(sdk), path, limit=1, input_price=5, output_price=25,
                            input_limit=2000, agent="claude")
    reply = llm.request(client, OPENING, TOOLS, model="claude-opus-5-5", reasoning_effort="high",
                        max_completion_tokens=100, completion_window=None, max_retries=3, retry_wait=1,
                        log=lambda event: None, require_usage=True)
    assert reply.choices[0].finish_reason == "stop" and len(sdk.requests) == 3
    assert sleeps == [4.0, 2.0]  # The server's Retry-After, then the doubled default backoff.
    data = json.loads(path.read_text())
    assert data["rejected_requests"] == 2 and data["completed_requests"] == 1 and data["reservations"] == {}


def test_a_stream_cut_off_after_it_started_stays_unknown_in_the_ledger(tmp_path):
    cut = NativeStream(None, failure=status_error_for(200))  # SDKs raise stream error events with the 200 status.
    client, path = anthropic_ledger(tmp_path, NativeSdk(cut))
    with pytest.raises(llm.InfrastructureError, match="cut|failed"):
        ask(client)
    assert [r["status"] for r in json.loads(path.read_text())["reservations"].values()] == ["unknown"]


def test_claude_plays_a_whole_task_with_thinking_carried_through_the_tool_loop(tmp_path):
    from world_adapt.agent.anthropic import AnthropicChatAdapter

    sdk = NativeSdk(
        native_reply(thinking(signature="sig-a"), tool_use("toolu_1", "execute", actions=["press 1"])),
        native_reply(Block(type="text", text="Button 1 lights lamp 1."), stop="end_turn"))
    events = []
    agent = PilotAgent(AnthropicChatAdapter(sdk), AgentConfig(model="claude-opus-5-5", max_completion_tokens=500,
                                                              require_usage=True), events.append)
    result, notebook = agent.run_task(PANEL, PANEL_TASK, SimulatorPredictor(PANEL), Budgets(6, 6), "", goal="g")
    assert result.success and notebook == "Button 1 lights lamp 1."
    assert result.usage["total_tokens"] == 30 and "usage_unknown_requests" not in result.usage
    planning, writer_request = sdk.requests
    assert "tools" in planning and "tools" not in writer_request
    assert "sig-a" not in json.dumps(events)


def test_anthropic_credentials_clients_and_flag_checks(monkeypatch):
    monkeypatch.setattr("world_adapt.runner.config.load_env", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    base_url, key_env, model = PROVIDERS["anthropic"]
    assert (model, key_env, provider_base_url("anthropic")) == ("claude-opus-5-5", "ANTHROPIC_API_KEY",
                                                                "https://api.anthropic.com/v1")
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        make_client("anthropic", base_url, key_env)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    client = make_client("anthropic", base_url, key_env)
    assert str(client.base_url).rstrip("/") == "https://api.anthropic.com" and client.max_retries == 0
    base = ["--provider", "anthropic", "--stream", "--max-completion-tokens", "1000"]
    parser = build_parser()
    check_args(parser, parse_args(parser, base))
    for bad in (["--provider", "anthropic", "--max-completion-tokens", "1000"],
                ["--provider", "anthropic", "--stream"], base + ["--action-tool-choice", "required"],
                base + ["--completion-window", "asap"]):
        with pytest.raises(SystemExit):
            parser = build_parser()
            check_args(parser, parse_args(parser, bad))


def window_client(path, create, *, limit, input_limit=1_000_000):
    return SpendingClient(raw_client(create), path, limit=limit, input_price=1, output_price=1,
                          input_limit=input_limit, agent="test")


def small_request(**extra):
    return dict(messages=[{"role": "user", "content": "hello world " * 50}], tools=[], max_completion_tokens=100, **extra)


def test_reservation_bounds_this_prompt_rather_than_the_served_window(tmp_path):
    path, seen = tmp_path / "spend.json", []

    def provider(**kw):
        seen.append(json.loads(path.read_text())["reservations"])
        return usage_reply(600, 5)

    window_client(path, provider, limit=1).create(**small_request())
    [reservation] = seen[0].values()
    assert reservation["reserved_prompt_tokens"] < 5_000
    assert reservation["reserved_usd"] == pytest.approx((reservation["reserved_prompt_tokens"] + 100) / 1_000_000)


def test_reservation_is_capped_at_the_served_window(tmp_path):
    path, seen = tmp_path / "spend.json", []

    def provider(**kw):
        seen.append(json.loads(path.read_text())["reservations"])
        return usage_reply(10, 5)

    huge = dict(messages=[{"role": "user", "content": "word " * 5_000}], max_completion_tokens=100)
    window_client(path, provider, limit=1, input_limit=2_000).create(**huge)
    [reservation] = seen[0].values()
    assert reservation["reserved_prompt_tokens"] == 2_000


def test_concurrent_small_requests_do_not_exhaust_a_small_cap(tmp_path):
    path, started = tmp_path / "spend.json", threading.Barrier(6)

    def provider(**kw):
        started.wait(timeout=10)  # All six requests hold their reservation at once.
        return usage_reply(100, 5)

    errors = []

    def work():
        try:
            window_client(path, provider, limit=0.05).create(**small_request())
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=work) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == [] and json.loads(path.read_text())["completed_requests"] == 6


def test_a_reply_whose_real_prompt_exceeds_the_estimate_is_charged_its_real_usage(tmp_path):
    path = tmp_path / "spend.json"
    window_client(path, lambda **kw: usage_reply(50_000, 5), limit=1).create(**small_request())
    data = json.loads(path.read_text())
    assert data["charged_usd"] == pytest.approx(50_005 / 1_000_000) and data["reservations"] == {}


def test_a_raised_ledger_limit_is_honoured_but_a_lowered_one_is_not(tmp_path):
    path = tmp_path / "spend.json"
    window_client(path, lambda **kw: usage_reply(), limit=0.001, input_limit=100).create(max_completion_tokens=50)
    raise_ledger_limit(path, 0.002)
    window_client(path, lambda **kw: usage_reply(), limit=0.001, input_limit=100).create(max_completion_tokens=50)
    assert json.loads(path.read_text())["limit_usd"] == 0.002
    with pytest.raises(ValueError, match="limit"):
        window_client(path, lambda **kw: usage_reply(), limit=0.003, input_limit=100).create(max_completion_tokens=50)
