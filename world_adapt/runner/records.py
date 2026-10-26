"""What a run writes to disk, in one fixed format.

runs/<timestamp>-<label>/
  manifest.json    what the run is: code, specification, tasks, predictor, budgets, agent, seeds, machine
  events.jsonl     what the agent saw and did, plus resource accounting; every line has run_id and task_id
  evaluator.jsonl  evaluator-only data: reference solutions, perturbation activations, real-state events, solver checks
  notebooks.json   the notebook each task started from and the one the agent wrote after it
  summary.json     status, held-out and adaptation success, per-tier success, resource totals, per-task results

A task counts as complete only once its `task_complete` event is written (after the notebook), so a run resumes
from task boundaries only. Each attempt at a task plays in a fresh environment from the task's initial state;
an attempt without `task_complete` is reported as an infrastructure failure, never continued under the default
policy. An opted-in continuation policy can record the task as missing, with its incoming notebook preserved.
The scored outcome is fsynced as `task_outcome` before the notebook request starts. An interrupted writer is then
finalized with that immutable outcome and the prior notebook, without repeating real actions.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from world_adapt.runner.protocol import TIERS


class RunRecord:
    def __init__(self, run_dir: Path, mode: str):
        import fcntl

        self.run_dir = run_dir
        self.run_id = run_dir.name
        with ExitStack() as resources:
            writer = resources.enter_context((run_dir / ".writer.lock").open("a+b"))
            try:
                fcntl.flock(writer.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit(f"run already has an active writer: {run_dir}") from None
            old = read_events(run_dir) if mode == "a" else []
            self.completed: dict[str, dict] = completed_tasks(run_dir) if mode == "a" else {}
            self.missing: dict[str, dict] = missing_tasks(run_dir) if mode == "a" else {}
            if set(self.completed) & set(self.missing):
                raise ValueError("task cannot be both completed and missing")
            if mode == "a":
                for name in ("events.jsonl", "evaluator.jsonl"):
                    _recover_tail(run_dir / name)
            self._events = resources.enter_context((run_dir / "events.jsonl").open(mode))
            self._evaluator = resources.enter_context((run_dir / "evaluator.jsonl").open(mode))
            outcomes = [e for e in old if e["type"] == "task_outcome"]
            if len({e["task_id"] for e in outcomes}) != len(outcomes):
                raise ValueError("duplicate committed task outcomes")
            self.pending = {e["task_id"]: e for e in outcomes if e["task_id"] not in self.completed}
            self.attempts: Counter = Counter(e["task_id"] for e in old if e["type"] == "attempt_start")
            self.failures: list[dict] = [{k: e[k] for k in ("task_id", "attempt", "error")}
                                         for e in old if e["type"] == "infrastructure_failure"]
            self._resources = resources.pop_all()

    @classmethod
    def create(cls, label: str, manifest: dict[str, Any], evaluator_manifest: dict[str, Any],
               root: Path = Path("runs")) -> "RunRecord":
        run_dir = root / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}"
        run_dir.mkdir(parents=True)
        manifest = {**manifest, "run_id": run_dir.name,
                    "artifacts": {name: str(run_dir / name) for name in
                                  ("manifest.json", "events.jsonl", "evaluator.jsonl", "notebooks.json", "summary.json")}}
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        record = cls(run_dir, "w")
        record.log_evaluator({"type": "evaluator_manifest", **evaluator_manifest})
        return record

    @classmethod
    def resume(cls, run_dir: Path, manifest: dict[str, Any]) -> "RunRecord":
        """Reopen an interrupted run. Refuses if the configuration, tasks or plan differ from the original."""
        saved = json.loads((run_dir / "manifest.json").read_text())
        for key in ("fingerprint",):
            if saved.get(key) != manifest.get(key):
                raise SystemExit(f"resume: {key} differs from the original run (configuration, tasks or plan changed)")
        return cls(run_dir, "a")

    def log(self, event: dict) -> None:
        self._events.write(json.dumps({"t": time.time(), "run_id": self.run_id, **event}, default=str) + "\n")
        self._events.flush()
        if event["type"] in ("task_complete", "task_outcome", "task_missing", "writer_request_start", "writer_request_end"):
            os.fsync(self._events.fileno())

    def log_evaluator(self, event: dict) -> None:
        self._evaluator.write(json.dumps({"t": time.time(), "run_id": self.run_id, **event}, default=str) + "\n")
        self._evaluator.flush()
        if event["type"] == "task_diagnostics":
            os.fsync(self._evaluator.fileno())

    def start_attempt(self, task_id: str) -> int:
        self.attempts[task_id] += 1
        self.log({"type": "attempt_start", "task_id": task_id, "attempt": self.attempts[task_id]})
        return self.attempts[task_id]

    def fail_attempt(self, task_id: str, attempt: int, error: str) -> None:
        """Record an infrastructure failure once per attempt; the attempt is never scored or continued."""
        if any(f["task_id"] == task_id and f["attempt"] == attempt for f in self.failures):
            return
        failure = {"task_id": task_id, "attempt": attempt, "error": error}
        self.failures.append(failure)
        self.log({"type": "infrastructure_failure", **failure})

    def mark_missing(self, planned, incoming_notebook: str, reason: str, attempt: int | None = None) -> str:
        """Persist an unscored task omission and return the notebook supplied to that task.

        Missing tasks are terminal for continuation purposes, but are never included in scored results.  The saved
        incoming notebook is the only state that may flow to the next acquisition task.
        """
        existing = self.missing.get(planned.task_id)
        if existing is not None:
            if existing.get("incoming_notebook") != incoming_notebook:
                raise ValueError("conflicting task omission notebook")
            return existing["incoming_notebook"]
        if planned.task_id in self.completed or planned.task_id in self.pending:
            raise ValueError("cannot mark a completed or pending task missing")
        attempt = self.attempts[planned.task_id] if attempt is None else attempt
        event = {"type": "task_missing", "task_id": planned.task_id, "phase": planned.phase,
                 "tier": planned.tier, "level_index": planned.level_index, "attempt": attempt,
                 "reason": reason, "incoming_notebook": incoming_notebook, "start_notebook": incoming_notebook}
        self.log(event)
        self.missing[planned.task_id] = event
        return incoming_notebook

    def commit_outcome(self, planned, attempt: int, result: dict, start_notebook: str) -> None:
        """Persist the task's score before the notebook request can start."""
        if planned.task_id in self.pending or planned.task_id in self.completed:
            raise ValueError("task outcome already committed")
        self.log_evaluator({"type": "task_diagnostics", "task_id": planned.task_id, "attempt": attempt,
                            "diagnostics": result.get("diagnostics", {})})
        event = {"type": "task_outcome", "task_id": planned.task_id, "attempt": attempt,
                 "result": {k: v for k, v in result.items() if k != "diagnostics"},
                 "start_notebook": start_notebook}
        self.log(event)
        self.pending[planned.task_id] = event

    def recover_outcome(self, planned) -> str | None:
        """Finalize an interrupted writer once; never resubmit a possibly paid request."""
        outcome = self.pending.get(planned.task_id)
        if outcome is None:
            return None
        result = json.loads(json.dumps(outcome["result"]))
        result["diagnostics"] = next((e["diagnostics"] for e in reversed(read_events(self.run_dir, "evaluator.jsonl"))
                                      if e["type"] == "task_diagnostics" and e["task_id"] == planned.task_id
                                      and e["attempt"] == outcome["attempt"]), {})
        usage = result["usage"]
        active = False
        starts, ends = {}, {}
        for event in read_events(self.run_dir):
            if event["type"] == "task_outcome" and event["task_id"] == planned.task_id:
                active = True
            elif active and event.get("task_id") == planned.task_id:
                if event["type"] == "writer_request_start":
                    if event["request"] in starts:
                        raise ValueError("duplicate writer request start")
                    starts[event["request"]] = event
                elif event["type"] == "writer_request_end":
                    if event["request"] in ends or event["request"] not in starts:
                        raise ValueError("invalid writer request end")
                    ends[event["request"]] = event
        usage["writer_requests"] = len(starts)
        usage["writer_failed_updates"] = usage["writer_interrupted_updates"] = 1
        for request, start in starts.items():
            known = ends.get(request, {}).get("usage")
            if known is None:
                usage["writer_unknown_requests"] = usage.get("writer_unknown_requests", 0) + 1
                usage["writer_unknown_reserved_tokens"] = usage.get("writer_unknown_reserved_tokens", 0) + start["reserved_tokens"]
            else:
                for key, value in known.items():
                    usage[key] = usage.get(key, 0) + value
        note = outcome["start_notebook"]
        self.log({"type": "notebook", "task_id": planned.task_id, "status": "interrupted_retained_previous",
                  "notebook": note, "requests": len(starts)})
        self.complete(planned, outcome["attempt"], result, note, note)
        return note

    def complete(self, planned, attempt: int, result: dict, start_notebook: str, notebook: str) -> None:
        outcome = self.pending.get(planned.task_id)
        if outcome is not None:
            if outcome["attempt"] != attempt or outcome["start_notebook"] != start_notebook:
                raise ValueError("committed task outcome identity changed")
            for key, value in outcome["result"].items():
                if key not in ("usage", "seconds") and result.get(key) != value:
                    raise ValueError(f"committed task outcome changed: {key}")
        public = {k: v for k, v in result.items() if k != "diagnostics"}
        event = {"type": "task_complete", "task_id": planned.task_id, "phase": planned.phase, "tier": planned.tier,
                 "level_index": planned.level_index, "attempt": attempt, "result": public,
                 "start_notebook": start_notebook, "notebook": notebook}
        if planned.task_id in self.completed:
            done = self.completed[planned.task_id]
            previous = {**done, "result": {k: v for k, v in done["result"].items() if k != "diagnostics"}}
            if all(previous.get(k) == v for k, v in event.items()):
                return
            raise ValueError("conflicting task completion")
        self.log_evaluator({"type": "task_diagnostics", "task_id": planned.task_id, "attempt": attempt,
                            "diagnostics": result.get("diagnostics", {})})
        self.log(event)
        self.completed[planned.task_id] = {**event, "result": result}
        self.pending.pop(planned.task_id, None)

    def finish(self, label: str, plan: list, status: str) -> dict:
        if self.missing:
            status = "incomplete"
        summary = summarize(label, self.run_id, plan, list(self.completed.values()), self.failures, status,
                            list(self.missing.values()))
        write_json(self.run_dir / "summary.json", summary)
        write_json(self.run_dir / "notebooks.json", notebooks(plan, self.completed, self.missing))
        self._resources.close()
        return summary


def read_events(run_dir: Path, name: str = "events.jsonl") -> list[dict]:
    data = (run_dir / name).read_bytes()
    committed = data[:data.rfind(b"\n") + 1]
    return [json.loads(line) for line in committed.splitlines()]


def _recover_tail(path: Path) -> None:
    """Preserve an interrupted final write before reopening the committed journal for append."""
    data = path.read_bytes()
    boundary = data.rfind(b"\n") + 1
    tail = data[boundary:]
    if not tail:
        return
    digest = hashlib.sha256(tail).hexdigest()
    backup = path.with_name(f"{path.name}.interrupted-{digest}.bin")
    if backup.exists():
        if backup.read_bytes() != tail:
            raise ValueError(f"interrupted journal backup differs: {backup}")
    else:
        with backup.open("xb") as stream:
            stream.write(tail)
            stream.flush()
            os.fsync(stream.fileno())
    with path.open("r+b") as stream:
        stream.truncate(boundary)
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path: Path, data: Any) -> None:
    """Replace a snapshot atomically after its complete contents have reached the filesystem."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def completed_tasks(run_dir: Path) -> dict[str, dict]:
    """`task_complete` events by task, with their evaluator-only diagnostics merged back into the result."""
    diagnostics = {(e["task_id"], e["attempt"]): e["diagnostics"]
                   for e in read_events(run_dir, "evaluator.jsonl") if e["type"] == "task_diagnostics"}
    out = {}
    for e in read_events(run_dir):
        if e["type"] == "task_complete":
            out[e["task_id"]] = {**e, "result": {**e["result"],
                                                 "diagnostics": diagnostics.get((e["task_id"], e["attempt"]), {})}}
    return out


def missing_tasks(run_dir: Path) -> dict[str, dict]:
    """`task_missing` events by task, preserving the incoming notebook for continuation."""
    out = {}
    for event in read_events(run_dir):
        if event["type"] != "task_missing":
            continue
        previous = out.get(event["task_id"])
        if previous is not None and previous != event:
            raise ValueError("duplicate conflicting task omissions")
        out[event["task_id"]] = event
    return out


RESOURCES = ("real_actions", "predicted_transitions", "predict_calls", "execute_calls", "turns", "malformed")
USAGE = ("prompt_tokens", "completion_tokens", "total_tokens")


def _rate(rows: list[dict]) -> float | None:
    return sum(r["success"] for r in rows) / len(rows) if rows else None


def summarize(label: str, run_id: str, plan: list, completed: list[dict], infra: list[dict], status: str,
              missing: list[dict] | None = None) -> dict:
    """Scores and resource totals from `task_complete` events only (also used to rescore a run from its logs)."""
    order = {p.task_id: k for k, p in enumerate(plan)}
    done = sorted(completed, key=lambda e: order[e["task_id"]])
    rows = []
    for e in done:
        rows.append({"task_id": e["task_id"], "phase": e["phase"], "tier": e["tier"], "attempt": e["attempt"],
                     **e["result"]})
    by_phase = {phase: [r for r in rows if r["phase"] == phase] for phase in ("adaptation", "heldout")}
    totals: dict[str, float] = {k: sum(r[k] for r in rows) for k in RESOURCES}
    totals.update({k: sum(r["usage"].get(k, 0) for r in rows) for k in USAGE})
    totals["seconds"] = round(sum(r.get("seconds", 0.0) for r in rows), 3)
    for key in ("writer_requests", "writer_unknown_requests", "writer_unknown_reserved_tokens",
                "writer_failed_updates", "writer_interrupted_updates", "planning_requests", "planning_unknown_requests",
                "usage_unknown_requests"):
        if any(key in r["usage"] for r in rows):
            totals[key] = sum(r["usage"].get(key, 0) for r in rows)
    missing_rows = []
    for event in sorted(missing or [], key=lambda e: order.get(e["task_id"], len(order))):
        missing_rows.append({k: event[k] for k in ("task_id", "phase", "tier", "level_index", "attempt", "reason")})
    if missing_rows:
        status = "incomplete"
    return {
        "label": label,
        "run_id": run_id,
        "status": status,
        "tasks_planned": len(plan),
        "tasks_completed": len(rows),
        "tasks_missing": len(missing_rows),
        "missing_count": len(missing_rows),
        "missing": missing_rows,
        "success_rate": _rate(by_phase["heldout"]),
        "primary_metric": "heldout success",
        "heldout_success_rate": _rate(by_phase["heldout"]),
        "heldout_by_tier": {t: _rate([r for r in by_phase["heldout"] if r["tier"] == t]) for t in TIERS}
        if by_phase["heldout"] else {},
        "adaptation_success_rate": _rate(by_phase["adaptation"]),
        "totals": totals,
        "infrastructure_failures": infra,
        "results": rows,
    }


def notebooks(plan: list, completed: dict[str, dict], missing: dict[str, dict] | None = None) -> dict:
    out: dict[str, Any] = {}
    missing = missing or {}
    for p in plan:
        e = completed.get(p.task_id)
        if e is not None:
            entry = {"task_id": p.task_id, "tier": p.tier, "start": e["start_notebook"], "written": e["notebook"]}
        else:
            e = missing.get(p.task_id)
            if e is None:
                continue
            entry = {"task_id": p.task_id, "tier": p.tier, "status": "missing",
                     "start": e["incoming_notebook"], "written": e["incoming_notebook"],
                     "attempt": e["attempt"], "reason": e["reason"]}
        out.setdefault(p.phase, []).append(entry)
    if ("adaptation" in out and len(out["adaptation"]) == sum(p.phase == "adaptation" for p in plan)
            and all(p.task_id in completed for p in plan if p.phase == "adaptation")):
        out["saved_after_adaptation"] = out["adaptation"][-1]["written"]
    return out
