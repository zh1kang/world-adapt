"""Scripted model clients, a tiny hand-wired task, and helpers that run the real CLI offline.

Nothing here calls a provider. Scripted clients know evaluator witnesses, so they are correctness fixtures, never
evidence of model performance.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import world_adapt.runner.run as run_module
from world_adapt.environments import registry
from world_adapt.environments.base import Task
from world_adapt.environments.switchboard.game import Switchboard, SwitchState, Wiring
from world_adapt.runner import config, study
from world_adapt.runner.protocol import seed_bundle

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs" / "study.json"
SPEND_VARIABLES = ("WORLD_ADAPT_SPEND_LEDGER", "WORLD_ADAPT_SPEND_LIMIT_USD", "WORLD_ADAPT_SPEND_AGENT")

PROTOCOL = ["--env", "switchboard", "--seed", "5", "--adaptation-tasks", "4", "--heldout-per-tier", "1",
            "--max-retries", "1", "--model", "offline-model"]
"""A short switchboard history: 4 adaptation tasks and 3 held-out tasks."""


# Responses -----------------------------------------------------------------------------------------------------


class Message(SimpleNamespace):
    def model_dump(self, exclude_none=False):
        data = {"role": "assistant", "content": self.content, "reasoning_content": "hidden reasoning"}
        if self.tool_calls:
            data["tool_calls"] = [c.model_dump() for c in self.tool_calls]
        return data


class Call(SimpleNamespace):
    def model_dump(self):
        return {"id": self.id, "type": "function",
                "function": {"name": self.function.name, "arguments": self.function.arguments}}


def call(name: str, arguments, call_id) -> Call:
    """A tool call; `arguments` that are not a string are JSON-encoded (actions lists become {"actions": ...})."""
    if isinstance(arguments, list):
        arguments = json.dumps({"actions": arguments})
    elif not isinstance(arguments, str) and arguments is not None:
        arguments = json.dumps(arguments)
    return Call(id=f"c{call_id}" if isinstance(call_id, int) else call_id,
                function=SimpleNamespace(name=name, arguments=arguments))


def response(content=None, calls=None, finish_reason="stop", usage=(10, 5, 15)):
    usage = None if usage is None else SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1],
                                                       total_tokens=usage[2])
    return SimpleNamespace(choices=[SimpleNamespace(message=Message(content=content, tool_calls=calls),
                                                    finish_reason=finish_reason)], usage=usage)


def stream_chunks(content=None, calls=(), finish_reason="tool_calls", usage=(10, 5, 15)) -> list[dict]:
    """The chunks of one streamed chat completion; `calls` are (id, name, arguments)."""
    delta = {} if content is None else {"content": content}
    if calls:
        delta["tool_calls"] = [{"index": i, "id": i_d, "function": {"name": name, "arguments": arguments}}
                               for i, (i_d, name, arguments) in enumerate(calls)]
    chunks = [{"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}]
    if usage is not None:
        chunks.append({"choices": [], "usage": dict(zip(("prompt_tokens", "completion_tokens", "total_tokens"), usage))})
    return chunks


class Stream:
    def __init__(self, chunks):
        self.chunks, self.closed = chunks, False

    def __iter__(self):
        return iter(self.chunks)

    def close(self):
        self.closed = True


class ScriptedClient:
    """Replays scripted replies to planning requests; answers notebook requests with `notes` (then "notes")."""

    def __init__(self, script, notes=()):
        self.script, self.notes, self.requests = list(script), list(notes), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not kwargs.get("tools"):
            return response(self.notes.pop(0) if self.notes else "notes")
        item = self.script.pop(0)
        return item if not isinstance(item, tuple) else response(*item)


def task_observation(messages: list[dict]) -> str:
    """The task's first public prompt, for planning and notebook requests alike."""
    first = messages[1]["content"]
    if first.startswith("{"):  # A notebook request carries the public instructions as JSON evidence.
        first = json.loads(first)["public_instructions"][1]["content"]
    return first


class SolverClient:
    """Plays each task's evaluator witness, one `execute` chunk per turn, recognising the task from its prompt.

    Notebooks name the task they follow. `crash_after=(observation, executed_calls)` raises KeyboardInterrupt.
    `stream=True` answers with chunk streams, as the study configuration requests.
    """

    def __init__(self, solutions: dict[str, tuple], *, chunk: int = 1, stream: bool = False, predict: bool = False,
                 crash_after: tuple[str, int] | None = None):
        self.solutions, self.chunk, self.stream, self.predict = solutions, chunk, stream, predict
        self.crash_after, self.requests, self.streams = crash_after, [], []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def task(self, messages) -> str:
        first = task_observation(messages)
        (obs,) = [o for o in self.solutions if o in first]
        return obs

    def notebook(self, obs: str) -> str:
        return f"note after task {sorted(self.solutions).index(obs)}"

    def create(self, **kwargs):
        self.requests.append(kwargs)
        messages = kwargs["messages"]
        obs = self.task(messages)
        if not kwargs.get("tools"):
            return response(self.notebook(obs))
        executed = [m for m in messages if m.get("role") == "tool" and m.get("name") == "execute"]
        if self.crash_after == (obs, len(executed)):
            raise KeyboardInterrupt("simulated crash")
        done = sum(len(json.loads(m["content"]).get("steps", [])) for m in executed)
        actions = list(self.solutions[obs][done:done + self.chunk])
        names = ("predict", "execute") if self.predict else ("execute",)
        calls = [(f"{name}-{len(self.requests)}", name, json.dumps({"actions": actions})) for name in names]
        if self.stream:
            stream = Stream(stream_chunks(calls=calls))
            self.streams.append(stream)
            return stream
        return response(None, [Call(id=i, function=SimpleNamespace(name=n, arguments=a)) for i, n, a in calls],
                        "tool_calls")


class FailingClient:
    def __init__(self, error=ConnectionError):
        self.error = error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        raise self.error("provider down")


# A hand-wired task -----------------------------------------------------------------------------------------------

PANEL = Switchboard(3)
"""Three lamps; button k toggles lamp k; button 3 is a breaker that disables button 1."""
PANEL_TASK = Task("panel", SwitchState(Wiring(1, 3, (0b001, 0b010, 0b100), (-1, -1, 0)), 0b110))
"""`press 1` solves it; `press 3` makes it unsolvable (lamp 1 can no longer be switched on)."""


# The real CLI ------------------------------------------------------------------------------------------------------


@contextmanager
def cwd(path: Path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def args_for(argv: list[str]):
    parser = config.build_parser()
    args = config.parse_args(parser, argv)
    config.check_args(parser, args)
    return args


def planned_levels(argv: list[str]):
    """The game, level pool and plan the runner will build for `argv`."""
    args = args_for(argv)
    spec = registry.get(args.env)
    options = config.env_options(args, spec)
    game = spec.make_game(options)
    levels, _, plan = run_module.build_tasks(args, spec, options, game, seed_bundle(args.seed))
    return game, levels, plan


def solutions(argv: list[str]) -> dict[str, tuple]:
    """Observation -> reference solution for every level the run will generate."""
    game, levels, _ = planned_levels(argv)
    out = {game.observe(level.state): level.solution for level in levels}
    if getattr(game, "close", None):
        game.close()
    return out


def plan_observations(argv: list[str]) -> list[str]:
    """Observations of the planned tasks, in play order."""
    game, levels, plan = planned_levels(argv)
    return [game.observe(levels[p.level_index].state) for p in plan]


def run_cli(argv: list[str], tmp: Path, client, monkeypatch) -> Path | None:
    """Run `world_adapt.runner.run` in `tmp` with `client`; return the newest run folder."""
    monkeypatch.setattr(run_module, "make_client", lambda *a, **k: client)
    for name in SPEND_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    with cwd(tmp):
        try:
            run_module.main(argv)
        finally:
            dirs = sorted((tmp / "runs").iterdir()) if (tmp / "runs").exists() else []
    return dirs[-1] if dirs else None


def read(run_dir: Path, name: str):
    path = run_dir / name
    return [json.loads(line) for line in path.open()] if name.endswith(".jsonl") else json.loads(path.read_text())


def completes(run_dir: Path) -> dict[str, dict]:
    return {e["task_id"]: e for e in read(run_dir, "events.jsonl") if e["type"] == "task_complete"}


def study_job_argv(env: str, condition: str, runs: Path, **overrides) -> list[str]:
    """Runner arguments for one job of the study template, as the scheduler would launch it."""
    design = study.load_study(TEMPLATE)
    agent = study.agents(design)[0]
    [job] = study.jobs(design, [agent], envs=[env], conditions=[condition], seeds=[3000])
    job = replace(job, args={**job.args, "max_retries": 1, **overrides})
    return study.command(job, str(agent.config), runs)[3:]
