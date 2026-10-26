"""The agent loop, its tools and prompts, context trimming, and the notebook writer, with scripted clients."""

import json
from types import SimpleNamespace

import pytest

from tests.fakes import PANEL, PANEL_TASK, Call, ScriptedClient, call, response
from world_adapt.agent import AgentConfig, Budgets, PilotAgent, writer
from world_adapt.agent.context import conversation_tokens
from world_adapt.agent.llm import InfrastructureError
from world_adapt.environments import registry
from world_adapt.environments.base import Task
from world_adapt.perturbations import registry as perturbations
from world_adapt.predictors.simulator import SimulatorPredictor

BUDGETS = Budgets(real_actions=6, predicted_transitions=6)


def play(script, *, notes=(), budgets=BUDGETS, notebook="old notes", game=PANEL, task=PANEL_TASK, predictor=None,
         **config):
    client = ScriptedClient(script, notes)
    events, outcomes = [], []
    agent = PilotAgent(client, AgentConfig(model="offline-model", **config), events.append)
    result, written = agent.run_task(game, task, predictor or SimulatorPredictor(game), budgets, notebook,
                                     goal="Turn on every lamp.", rules="Buttons toggle lamps.",
                                     commit_outcome=outcomes.append)
    return result, written, client, events, outcomes


def tool_payloads(messages):
    return [json.loads(m["content"]) for m in messages if m["role"] == "tool"]


def public_records(events):
    return next(e["experience"]["public_records"] for e in events if e["type"] == "writer_experience")


# Playing a task --------------------------------------------------------------------------------------------------


def test_predict_then_execute_solves_the_task_and_rewrites_the_notebook():
    result, notebook, client, events, outcomes = play([(None, [call("predict", ["press 1"], 1)]),
                                                       (None, [call("execute", ["press 1"], 2)])],
                                                      notes=["Button 1 lights lamp 1."])
    assert result.success and result.end_reason == "success"
    assert (result.real_actions, result.predicted_transitions, result.predict_calls, result.turns) == (1, 1, 1, 2)
    assert notebook == "Button 1 lights lamp 1." and outcomes == [result]
    first = client.requests[0]
    assert [t["function"]["name"] for t in first["tools"]] == ["predict", "execute", "concede"]
    assert "old notes" in first["messages"][1]["content"]
    system = first["messages"][0]["content"]
    assert "Environment mechanics: Buttons toggle lamps." in system and "Predicted actions: 6 per task" in system
    # The full assistant message, including provider reasoning, is passed back to the planner only.
    assert client.requests[1]["messages"][2]["reasoning_content"] == "hidden reasoning"
    writer_request = client.requests[-1]
    assert "tools" not in writer_request and "hidden reasoning" not in json.dumps(writer_request)
    assert [e["type"] for e in events].count("notebook") == 1


def test_identity_and_faulty_predictors_send_identical_prompts_tools_and_limits():
    spec = registry.get("switchboard")
    game = spec.make_game(spec.default_options)
    level = spec.make_levels(game, 1, 3000, spec.default_options)[0]
    first = {}
    for profile in ("identity", "breaker_retarget_enabled_only"):
        predictor = SimulatorPredictor(game, perturbations.profiles("switchboard")[profile])
        _, _, client, _, _ = play([(None, [call("concede", {}, 1)])], game=game, task=Task("t", level.state),
                                  predictor=predictor)
        first[profile] = client.requests[0]
    assert first["identity"] == first["breaker_retarget_enabled_only"]
    assert "breaker_retarget" not in json.dumps(first["identity"])


def test_budget_errors_and_real_budget_cuts_are_reported_to_the_agent():
    result, _, client, events, _ = play([(None, [call("predict", ["press 2"] * 4, 1)]),
                                    ("thinking out loud", None),
                                    (None, [call("execute", ["press 2"] * 5, 2)])],
                                   budgets=Budgets(real_actions=4, predicted_transitions=3))
    assert result.malformed == 2 and result.real_actions == 4
    assert not result.success and result.end_reason == "real_budget"
    errors = tool_payloads(client.requests[1]["messages"])
    assert "prediction budget" in errors[0]["error"] and errors[0]["budget_remaining"]["predicted_actions"] == 3
    cut = json.loads(public_records(events)[-1]["content"])
    assert "only the first 4 actions" in cut["note"]


def test_concede_ends_the_task_as_a_failure():
    result, *_ = play([(None, [call("concede", None, 1)])])
    assert not result.success and result.end_reason == "conceded" and result.turns == 1 and result.malformed == 0


def test_a_provably_unsolvable_real_state_ends_the_task():
    result, _, _, events, _ = play([(None, [call("execute", ["press 3"], 1)])])
    assert result.end_reason == "dead_state" and result.real_actions == 1
    assert "no longer be solved" in json.loads(public_records(events)[0]["content"])["task_over"]


def test_malformed_arguments_are_sanitized_in_history_and_counted():
    broken = call("predict", '{"actions": ["press 1",', 1)
    as_list = call("execute", '["press 1"]', 2)
    result, _, client, events, _ = play([(None, [broken]), (None, [as_list]), (None, [call("execute", ["press 1"], 3)])])
    assert result.success and result.malformed == 2 and result.predict_calls == 0
    history = client.requests[2]["messages"]
    echoed = [tc for m in history if m["role"] == "assistant" for tc in m.get("tool_calls", [])]
    assert all(isinstance(json.loads(tc["function"]["arguments"]), dict) for tc in echoed)
    first = json.loads(echoed[0]["function"]["arguments"])
    assert first["invalid_arguments"] == '{"actions": ["press 1",' and "not valid JSON" in first["problem"]
    replies = tool_payloads(history)
    assert "not valid JSON" in replies[0]["error"] and "a JSON list, not an object" in replies[1]["error"]
    logged = [e for e in events if e["type"] == "tool_result"]
    assert logged[0]["invalid_arguments"] and logged[2]["invalid_arguments"] is None


@pytest.mark.parametrize("tool", ["predict", "execute"])
@pytest.mark.parametrize("bad", [None, 7, {}, ["press 1"]])
def test_a_bad_action_element_is_one_malformed_call_without_partial_execution(tool, bad):
    result, _, _, events, _ = play([(None, [call(tool, ["press 2", bad], 1), call("execute", ["press 1"], 2)])],
                                   budgets=Budgets(real_actions=6, predicted_transitions=6, max_malformed=1))
    assert result.end_reason == "malformed_limit" and result.malformed == 1
    assert result.real_actions == result.predicted_transitions == result.execute_calls == result.predict_calls == 0
    records = public_records(events)
    assert json.loads(records[0]["arguments"]) == {"actions": ["press 2", bad]}
    assert "not executed" in json.loads(records[1]["content"])["error"]
    assert not any(e["type"] == "execute" for e in events)


@pytest.mark.parametrize("raw", [7, True, {}, ["press 1"]])
def test_non_text_tool_arguments_are_malformed_and_kept_as_public_json(raw):
    bad = Call(id="bad", function=SimpleNamespace(name="execute", arguments=raw))
    result, _, _, events, _ = play([(None, [bad])], budgets=Budgets(6, 6, max_malformed=1))
    assert result.end_reason == "malformed_limit" and result.real_actions == 0
    assert public_records(events)[0]["arguments"] == raw


def test_invalid_tool_identity_is_rejected_before_any_call_in_the_batch_runs():
    invalid = Call(id=None, function=SimpleNamespace(name="execute", arguments='{"actions":["press 1"]}'))
    events = []
    with pytest.raises(InfrastructureError, match="tool call"):
        PilotAgent(ScriptedClient([(None, [call("execute", ["press 1"], 1), invalid])]),
                   AgentConfig(model="m"), events.append).run_task(PANEL, PANEL_TASK, SimulatorPredictor(PANEL),
                                                                    BUDGETS, "", goal="g")
    assert not any(e["type"] == "execute" for e in events)


def test_reused_provider_call_ids_keep_every_dispatched_record():
    duplicate = call("concede", {}, 1)
    result, _, _, events, _ = play([(None, [call("predict", ["press 2"], 1), call("execute", ["press 2"], 1),
                                            duplicate])])
    records = public_records(events)
    assert [r["name"] for r in records] == ["predict", "execute", "concede"]
    assert len({r["record_id"] for r in records}) == 3
    assert json.loads(records[-1]["content"]) == {"kind": "task over", "note": "You conceded; the task ended as a failure."}
    assert result.end_reason == "conceded"


# Reply cut-offs and the malformed-call limit -------------------------------------------------------------------


def test_cut_off_reply_is_dropped_from_history_and_prompts_a_brief_retry():
    result, _, client, events, _ = play([("reasoning that overran the reply budget", None, "length"),
                                         (None, [call("execute", ["press 1"], 1)])], max_completion_tokens=256)
    assert result.success and result.malformed == 1
    history = client.requests[1]["messages"]
    assert not any(m.get("content") == "reasoning that overran the reply budget" for m in history)
    assert history[2] == {"role": "assistant", "content": "[reply cut off at the token limit before any tool call]"}
    assert "256-token reply limit" in history[3]["content"]
    assert [e["type"] for e in events].count("reply_cutoff") == 1


def test_a_cut_off_batch_executes_none_of_its_tool_calls():
    partial = call("concede", '{"reason":', 2)
    result, _, _, events, _ = play([(None, [call("execute", ["press 1"], 1), partial], "length"),
                                    (None, [call("execute", ["press 1"], 3)])], max_completion_tokens=256)
    assert result.success and result.malformed == 1 and result.real_actions == 1
    assert not any(e["type"] == "tool_result" and e["turn"] == 1 for e in events)


def test_reaching_the_malformed_limit_stops_the_rest_of_the_batch():
    invalid = call("predict", '["press 1"]', 1)
    result, _, client, events, _ = play([(None, [invalid, call("execute", ["press 1"], 2)])],
                                        budgets=Budgets(6, 6, max_malformed=1))
    assert result.end_reason == "malformed_limit" and result.real_actions == 0
    assert not any(e["type"] == "execute" for e in events)
    assert "not executed" in public_records(events)[1]["content"]


@pytest.mark.parametrize("script,reason", [
    ([(None, [call("execute", ["press 1"], 1)], "length")], "reply_cutoff_limit"),
    ([(None, [call("predict", '["x"]', 1)]), (None, [call("execute", ["press 1"], 2)], "length")], "response_limit"),
    ([(None, [call("execute", ["press 1"], 1)], "length"), (None, [call("predict", '["x"]', 2)])], "response_limit"),
])
def test_cut_off_and_malformed_limits_have_distinct_end_reasons(script, reason):
    limit = len(script)
    result, *_ = play(script, budgets=Budgets(6, 6, max_malformed=limit), max_completion_tokens=256)
    assert result.end_reason == reason and result.malformed == limit and not result.success


# Token, time and context limits ----------------------------------------------------------------------------------


@pytest.mark.parametrize("budgets,reason,turns", [
    (Budgets(6, 6, max_task_tokens=40), "token_limit", 3),
    (Budgets(6, 6, max_task_seconds=1e-9), "time_limit", 0),
])
def test_token_and_time_caps_end_the_task_and_the_writer_still_runs(budgets, reason, turns):
    result, notebook, client, _, _ = play([("let me think", None)] * 3, budgets=budgets, notes=["kept going"])
    assert result.end_reason == reason and result.turns == turns
    assert notebook == "kept going" and "tools" not in client.requests[-1]


def test_a_late_planning_reply_cannot_execute_but_is_accounted(monkeypatch):
    import world_adapt.episode as episode
    now = [0.0]
    monkeypatch.setattr(episode.time, "monotonic", lambda: now[0])

    class Late(ScriptedClient):
        def create(self, **kwargs):
            reply = super().create(**kwargs)
            if kwargs.get("tools"):
                now[0] = 10.0
            return reply

    client, events, outcomes = Late([(None, [call("execute", ["press 1"], 1)])], ["Retained."]), [], []
    result, note = PilotAgent(client, AgentConfig(model="m"), events.append).run_task(
        PANEL, PANEL_TASK, SimulatorPredictor(PANEL), Budgets(6, 6, max_task_seconds=1), "", goal="g",
        commit_outcome=outcomes.append)
    assert result.end_reason == "time_limit" and result.real_actions == 0
    assert result.usage["total_tokens"] == 30 and note == "Retained." and len(outcomes) == 1
    assert client.requests[0]["timeout"] <= 1
    assert public_records(events) == []


def test_the_deadline_is_rechecked_between_tools_in_one_batch(monkeypatch):
    import world_adapt.episode as episode
    now = [0.0]
    monkeypatch.setattr(episode.time, "monotonic", lambda: now[0])

    class Slow(SimulatorPredictor):
        def predict(self, *args):
            prediction = super().predict(*args)
            now[0] = 2.0
            return prediction

    result, _, _, events, _ = play([(None, [call("predict", ["press 2"], 1), call("execute", ["press 1"], 2)])],
                                   budgets=Budgets(6, 6, max_task_seconds=1), predictor=Slow(PANEL))
    assert result.end_reason == "time_limit" and result.real_actions == 0 and result.predict_calls == 1
    assert "time limit" in json.loads(public_records(events)[1]["content"])["error"]


def test_context_cap_drops_the_oldest_exchanges_and_keeps_the_present_state():
    script = [(None, [call("predict", ["press 2"], i)]) for i in range(1, 13)] + [(None, [call("execute", ["press 1"], 20)])]
    budgets = Budgets(real_actions=6, predicted_transitions=20)
    result, _, client, events, _ = play(script, budgets=budgets, max_context_tokens=1500, keep_recent_turns=2)
    assert result.success
    trims = [e for e in events if e["type"] == "context_trim"]
    assert trims and all(e["tokens_after"] < e["tokens_before"] for e in trims)
    last = client.requests[-2]["messages"]
    assert last[0]["role"] == "system" and "old notes" in last[1]["content"]
    assert last[2]["role"] == "user" and last[2]["content"].startswith("[Context window:") and "Lamps off" in last[2]["content"]
    for i, m in enumerate(last):
        if m["role"] == "assistant" and m.get("tool_calls"):
            assert last[i + 1]["tool_call_id"] == m["tool_calls"][0]["id"]
    assert conversation_tokens(last, client.requests[-2]["tools"]) <= 1500
    # The writer still receives every public record, including the trimmed ones.
    access = next(e for e in events if e["type"] == "notebook_input")["available_tool_call_ids"]
    assert access == [f"c{i}" for i in range(1, 13)] + ["c20"]


def test_a_planning_prompt_that_cannot_fit_is_rejected_before_any_request():
    client, events = ScriptedClient([]), []
    with pytest.raises(InfrastructureError, match="context cap"):
        PilotAgent(client, AgentConfig(model="m", max_context_tokens=1000), events.append).run_task(
            PANEL, PANEL_TASK, SimulatorPredictor(PANEL), BUDGETS, "large prior note " * 2000, goal="g")
    assert not client.requests and any(e["type"] == "planning_context_not_admitted" for e in events)


# The notebook writer ---------------------------------------------------------------------------------------------


def test_writer_input_is_public_experience_with_the_previous_notebook_as_a_separate_field():
    _, note, client, events, _ = play([(None, [call("predict", ["press 2"], 1)]), (None, [call("execute", ["press 1"], 2)])],
                                      notes=["Public notes."], notebook="prior notebook")
    experience = next(e["experience"] for e in events if e["type"] == "writer_experience")
    assert experience["input_version"] == writer.INPUT_VERSION
    assert "prior notebook" not in json.dumps(experience) and "hidden reasoning" not in json.dumps(experience)
    assert [r["call_id"] for r in experience["public_records"]] == ["c1", "c2"]
    payload = json.loads(client.requests[-1]["messages"][1]["content"])
    assert payload["previous_notebook"] == "prior notebook"
    assert payload["evidence_access"]["selected_record_ids"] == [r["record_id"] for r in experience["public_records"]]
    logged = next(e for e in events if e["type"] == "notebook")
    assert logged["writer"] == "freeform-v1" and logged["status"] == "written" and note == "Public notes."


def test_record_selection_ignores_outcomes_and_lists_every_omission():
    records = [writer.record("t/attempt-1", i, call_id="same", name="predict", arguments="not json", content="x " * 100)
               for i in range(1, 7)]
    experience = dict(public_records=records, messages=[{"role": "user", "content": "public instructions"}],
                      current_observation="public state", information_limits={"context_tokens": 2000, "evidence_tokens": 600},
                      outcome={"success": True, "end_reason": "PRIVATE_REASON"})
    messages, ids, dropped, _ = writer.prepare_input(experience, "previous", 100)
    assert messages == writer.prepare_input({**experience, "outcome": {"success": False}}, "previous", 100)[0]
    access = writer.access(messages)
    assert dropped > 0 and len(access["selected_record_ids"]) + len(access["omitted_records"]) == 6
    assert [r["record_id"] for r in access["omitted_records"]] == [r["record_id"] for r in records[:dropped]]
    assert ids == [] and access["ambiguous_tool_call_ids"] == ["same"]
    assert writer.evidence_tokens(messages) <= 600 and conversation_tokens(messages) <= 2000
    assert "PRIVATE" not in json.dumps(messages)


def test_an_oversized_latest_record_never_falls_back_to_older_evidence():
    records = [writer.record("t", 1, call_id="old", name="execute", arguments="{}", content="small"),
               writer.record("t", 2, call_id="large", name="predict", arguments="{}", content="large " * 5000)]
    experience = dict(public_records=records, messages=[], current_observation="state",
                      information_limits={"context_tokens": 2000, "evidence_tokens": 400})
    messages, ids, dropped, _ = writer.prepare_input(experience, "", 100)
    assert ids == [] and dropped == 2
    assert [x["call_id"] for x in writer.access(messages)["omitted_records"]] == ["old", "large"]


def test_the_archive_never_crosses_task_boundaries():
    client = ScriptedClient([(None, [call("execute", ["press 1"], 1)]), (None, [call("execute", ["press 1"], 2)])],
                            ["first note", "second note"])
    events = []
    agent = PilotAgent(client, AgentConfig(model="m"), events.append)
    for previous in ("", "first note"):
        agent.run_task(PANEL, PANEL_TASK, SimulatorPredictor(PANEL), BUDGETS, previous, goal="g")
    archives = [e["experience"] for e in events if e["type"] == "writer_experience"]
    assert [[r["call_id"] for r in a["public_records"]] for a in archives] == [["c1"], ["c2"]]
    assert "first note" not in json.dumps(archives[1]) and "first note" in json.dumps(client.requests[-1]["messages"])


def test_an_over_budget_note_gets_one_repair_and_then_keeps_the_previous_notebook():
    long_note = "lesson " * 3000
    result, note, client, events, _ = play([(None, [call("execute", ["press 1"], 1)])], notes=[long_note, long_note],
                                           notebook="prior")
    assert note == "prior" and len([r for r in client.requests if "tools" not in r]) == 2
    assert "Notebook update rejected: over_budget" in client.requests[-1]["messages"][-1]["content"]
    logged = next(e for e in events if e["type"] == "notebook")
    assert logged["status"] == "retained_previous" and logged["rejections"] == ["over_budget", "over_budget"]
    assert result.usage["writer_failed_updates"] == 1 and result.usage["writer_requests"] == 2


def test_a_writer_allowance_blocks_requests_before_submission():
    result, note, client, events, _ = play([(None, [call("execute", ["press 1"], 1)])], notes=["unused"],
                                           notebook="prior", notebook_max_tokens=10)
    assert result.success and note == "prior" and len(client.requests) == 1
    assert result.usage["writer_requests"] == 0 and result.usage["writer_failed_updates"] == 1
    assert not any(e["type"] in ("writer_request_start", "notebook_input") for e in events)


def test_an_evidence_envelope_that_cannot_fit_retains_the_previous_note_and_logs_omissions():
    result, note, client, events, _ = play([(None, [call("execute", ["press 1"], 1)])], notebook="prior",
                                           writer_evidence_tokens=32)
    assert result.success and note == "prior" and len(client.requests) == 1
    assert len(next(e for e in events if e["type"] == "writer_evidence_selection")["omitted_records"]) == 1


class FlakyWriter(ScriptedClient):
    """Fails the first notebook requests with `errors`, then answers them from `notes`."""

    def __init__(self, script, errors, notes=()):
        super().__init__(script, notes)
        self.errors = list(errors)

    def create(self, **kwargs):
        if not kwargs.get("tools") and self.errors:
            self.requests.append(kwargs)
            raise self.errors.pop(0)
        return super().create(**kwargs)


def status_error(status):
    error = RuntimeError(f"HTTP {status}")
    error.status_code = status
    return error


def write_with(client):
    events = []
    result, note = PilotAgent(client, AgentConfig(model="m", max_retries=5, retry_wait=0), events.append).run_task(
        PANEL, PANEL_TASK, SimulatorPredictor(PANEL), BUDGETS, "prior", goal="g")
    return result, note, events


@pytest.mark.parametrize("error", [status_error(503), status_error(429), ConnectionError("reset")])
def test_a_transient_writer_failure_is_retried_once_within_the_writer_budget(error):
    client = FlakyWriter([(None, [call("execute", ["press 1"], 1)])], [error], ["fresh notebook"])
    result, note, events = write_with(client)
    assert note == "fresh notebook" and len([r for r in client.requests if "tools" not in r]) == 2
    logged = next(e for e in events if e["type"] == "notebook")
    assert logged["status"] == "written" and logged["requests"] == 2 and logged["successful_responses"] == 1
    assert result.usage["writer_requests"] == 2 and result.usage["writer_failed_updates"] == 0


def test_a_writer_retry_stops_after_the_second_request():
    client = FlakyWriter([(None, [call("execute", ["press 1"], 1)])], [status_error(503), status_error(503)])
    result, note, _ = write_with(client)
    assert note == "prior" and len([r for r in client.requests if "tools" not in r]) == 2
    assert result.usage["writer_failed_updates"] == 1


@pytest.mark.parametrize("error", [status_error(400), TimeoutError("timed out")])
def test_a_writer_request_that_may_have_been_billed_or_is_invalid_is_not_retried(error):
    client = FlakyWriter([(None, [call("execute", ["press 1"], 1)])], [error], ["unused"])
    result, note, _ = write_with(client)
    assert note == "prior" and len([r for r in client.requests if "tools" not in r]) == 1


def test_a_writer_timeout_keeps_its_reservation_unknown():
    client = FlakyWriter([(None, [call("execute", ["press 1"], 1)])], [TimeoutError("lost reply")])
    result, note, _ = write_with(client)
    assert result.success and note == "prior" and len(client.requests) == 2
    assert result.usage["writer_unknown_requests"] == 1 and result.usage["writer_unknown_reserved_tokens"] > 32768
    assert client.requests[-1]["timeout"] > 0 and client.requests[-1]["max_completion_tokens"] == 32768


def test_a_late_writer_reply_is_charged_but_rejected(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(writer.time, "monotonic", lambda: clock[0])

    class LateWriter(ScriptedClient):
        def create(self, **kwargs):
            reply = super().create(**kwargs)
            if not kwargs.get("tools"):
                clock[0] += 10
            return reply

    client = LateWriter([(None, [call("execute", ["press 1"], 1)])], ["too late"])
    result, note = PilotAgent(client, AgentConfig(model="m", notebook_max_seconds=1)).run_task(
        PANEL, PANEL_TASK, SimulatorPredictor(PANEL), BUDGETS, "prior", goal="g")
    assert result.success and note == "prior"
    assert result.usage["total_tokens"] == 30 and result.usage["writer_failed_updates"] == 1


def test_missing_planning_usage_stays_unknown_even_when_the_writer_reports_usage():
    client = ScriptedClient([response(None, [call("execute", ["press 1"], 1)], usage=None)])
    result, _ = PilotAgent(client, AgentConfig(model="m")).run_task(PANEL, PANEL_TASK, SimulatorPredictor(PANEL),
                                                                    BUDGETS, "", goal="g")
    assert result.success and result.usage["usage_unknown_requests"] == 1


@pytest.mark.parametrize("options", [{"writer_evidence_tokens": 0}, {"max_context_tokens": 999},
                                     {"writer_context_tokens": 999}, {"notebook_tokens": 0},
                                     {"writer_evidence_tokens": 70_000}, {"notebook_max_seconds": float("inf")}])
def test_unbounded_or_inconsistent_writer_limits_are_rejected(options):
    with pytest.raises(ValueError):
        AgentConfig(model="m", **options)
