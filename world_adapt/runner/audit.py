"""Checks on a study's run folders: which runs exist against the study plan, and whether each summary agrees
with its own event logs.

The log check needs no environment or model: every scored episode's `summary.json` row is compared with its
`task_complete` and `task_end` events, its counters and success are recomputed from the attempt's `execute` and
`predict` events, its end reason is checked against the logged events and the task's budgets, and the run-level
rates and totals are recomputed with `records.summarize`. `world_adapt.runner.rescore` goes further (it replays the
logged actions in the real environment) and is available through `analyze --replay`.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from world_adapt.runner.protocol import PlannedTask
from world_adapt.runner.records import summarize
from world_adapt.runner.manifest import ROOT
from world_adapt.runner.study import CONDITIONS, Agent, Job, jobs, seed_blocks

COUNTERS = ("success", "end_reason", "real_actions", "predicted_transitions", "predict_calls", "execute_calls",
            "turns", "malformed", "usage")
SUMMARY_KEYS = ("status", "tasks_planned", "tasks_completed", "tasks_missing", "missing_count", "missing",
                "success_rate", "heldout_success_rate", "heldout_by_tier", "adaptation_success_rate", "totals")
WANTED = ("attempt_start", "execute", "predict", "concede", "task_complete", "task_end", "api_error", "context_trim",
          "reply_cutoff", "task_missing")


@dataclass(frozen=True)
class RunFolder:
    path: Path
    env: str
    cond: str
    seed: int


def run_folders(runs: Path, prefix: str, envs: list[str] | None = None) -> list[RunFolder]:
    """Folders named `<timestamp>-<prefix>-<env>-<condition>-s<seed>`, oldest first; `envs` restricts the match."""
    env = "|".join(map(re.escape, envs)) if envs else "[a-z0-9-]+"
    cond = "|".join(re.escape(slug) for slug in CONDITIONS)
    pattern = re.compile(rf"^\d{{8}}-\d{{6}}-{re.escape(prefix)}-(?P<env>{env})-(?P<cond>{cond})-s(?P<seed>\d+)$")
    out = []
    for d in sorted(runs.iterdir()) if runs.exists() else []:
        m = pattern.match(d.name)
        if m and d.is_dir():
            out.append(RunFolder(d, m["env"], m["cond"], int(m["seed"])))
    return out


def latest(folders: list[RunFolder]) -> dict[tuple[str, str, int], tuple[RunFolder, list[str]]]:
    """The newest folder per (env, condition, seed), as the launcher resumes it, with the older folders it supersedes."""
    out: dict[tuple[str, str, int], tuple[RunFolder, list[str]]] = {}
    for f in folders:
        key = (f.env, f.cond, f.seed)
        older = [out[key][0].path.name, *out[key][1]] if key in out else []
        out[key] = (f, older)
    return out


def scan_events(run_dir: Path) -> dict[str, Any]:
    """One pass over events.jsonl (and evaluator.jsonl for discrepancy exposure).

    Returns API-error, context-trim and reply-cutoff counts (scored attempts, and all attempts), the
    execute/predict/concede events of each (task, attempt), the last `task_complete` per task, `task_end` results by
    episode id, and per-episode exposure. Lines are filtered by event type before parsing; transcripts are never
    parsed.
    """
    out: dict[str, Any] = {"api_errors": 0, "context_trims": 0, "reply_cutoffs": 0, "reply_cutoffs_all": 0,
                           "completes": {}, "missing": {}, "missing_problems": [], "task_ends": {}, "exposure": {},
                           "attempts": defaultdict(lambda: {"execute": [], "predict": [], "concede": [],
                                                             "reply_cutoff": []})}
    current: dict[str, int] = {}
    path = run_dir / "events.jsonl"
    for line in path.open() if path.exists() else []:
        if not any(f'"type": "{t}"' in line for t in WANTED):
            continue
        e = json.loads(line)
        kind = e.get("type")
        if kind == "api_error":
            out["api_errors"] += 1
        elif kind == "context_trim":
            out["context_trims"] += 1
        elif kind == "attempt_start":
            current[e["task_id"]] = e["attempt"]
            out["attempts"][(e["task_id"], e["attempt"])]
        elif kind in ("execute", "predict", "concede", "reply_cutoff"):
            out["attempts"][(e["task_id"], current.get(e["task_id"]))][kind].append(e)
        elif kind == "task_complete":
            out["completes"][e["task_id"]] = e
        elif kind == "task_missing":
            if e["task_id"] in out["missing"]:
                out["missing_problems"].append(f"{e['task_id']}: duplicate task_missing events")
            out["missing"][e["task_id"]] = e
        elif kind == "task_end":
            out["task_ends"][e["episode_id"]] = e["result"]
    scored = {(task_id, event["attempt"]) for task_id, event in out["completes"].items()}
    for key, events in out["attempts"].items():
        count = len(events["reply_cutoff"])
        out["reply_cutoffs_all"] += count
        if key in scored:
            out["reply_cutoffs"] += count
    evaluator = run_dir / "evaluator.jsonl"
    for line in evaluator.open() if evaluator.exists() else []:
        if '"type": "task_end_private"' not in line:
            continue
        e = json.loads(line)
        traces = exposure_traces(e)
        out["exposure"][e["episode_id"]] = {
            "predictions": len(traces), "exposed_predictions": sum(any(step for step in t) for t in traces),
            "changed_steps": sum(1 for t in traces for step in t if step),
            "state_changed_steps": sum(1 for t in e["prediction_changed"] for step in t if step),
            "feedback": e["prediction_feedback"]}
    return out


def exposure_traces(event: dict[str, Any]) -> list:
    """Per prediction call, which predicted steps the faulty profile visibly changed (validated)."""
    state = event.get("prediction_changed")
    version = event.get("visible_change_recording_version")
    visible = event.get("prediction_visible_changed")
    requested = event.get("prediction_requested_transitions")
    if (type(version) is not int or version != 1 or not isinstance(visible, list)
            or not isinstance(state, list) or len(visible) != len(state)
            or not isinstance(requested, list) or len(requested) != len(visible)
            or any(not isinstance(call, list) or not isinstance(latent, list)
                   or type(limit) is not int or limit < 1 or not len(latent) <= len(call) <= limit
                   or any(type(flag) is not bool for flag in call)
                   for call, latent, limit in zip(visible, state, requested))):
        raise ValueError(f"invalid visible exposure record for {event.get('episode_id')}")
    return visible


def _end_reason_problem(row: dict[str, Any], events: dict[str, list], budget: dict[str, Any] | None) -> str | None:
    reason, executes = row["end_reason"], events["execute"]
    last = executes[-1] if executes else None
    succeeded = bool(last and last["success"])
    if succeeded != (reason == "success"):
        return f"end_reason {reason!r} but the last logged execute has success={succeeded}"
    if reason == "success":
        return None
    if budget is None:
        return "no budget for the task in the run manifest"
    tokens = row.get("usage", {}).get("total_tokens")
    plausible = {
        "terminal_failure": last is not None and last["terminal"],
        "dead_state": last is not None,
        "conceded": bool(events["concede"]),
        "real_budget": row["real_actions"] >= budget["real_actions"],
        "turn_limit": row["turns"] >= budget["max_turns"],
        "malformed_limit": row["malformed"] >= budget["max_malformed"] and
        not events["reply_cutoff"],
        "reply_cutoff_limit": bool(events["reply_cutoff"]) and
        len(events["reply_cutoff"]) == row["malformed"] and
        row["malformed"] >= budget["max_malformed"],
        "response_limit": bool(events["reply_cutoff"]) and
        len(events["reply_cutoff"]) < row["malformed"] and
        row["malformed"] >= budget["max_malformed"],
        "token_limit": budget.get("max_task_tokens") is not None and tokens is not None
        and tokens >= budget["max_task_tokens"],
        "time_limit": budget.get("max_task_seconds") is not None and row.get("seconds", 0) >= budget["max_task_seconds"],
    }
    return None if plausible.get(reason, False) else f"end_reason {reason!r} is not supported by the logs and budgets"


def check_logs(summary: dict[str, Any], manifest: dict[str, Any], scan: dict[str, Any]) -> list[str]:
    """Disagreements between summary.json and the run's own event log (empty when consistent)."""
    problems: list[str] = []
    rows = {r["task_id"]: r for r in summary["results"]}
    completes = scan["completes"]
    budgets = {b["task_id"]: b for b in manifest.get("budgets", {}).get("per_task", [])}
    planned = {p['task_id']: p for p in manifest.get('tasks', {}).get('plan', [])}
    missing = scan.get('missing', {})
    problems.extend(scan.get('missing_problems', []))
    for task, event in missing.items():
        if task not in planned:
            problems.append(f'{task}: task_missing event outside the manifest plan')
        if task in rows or task in completes:
            problems.append(f'{task}: both missing and scored')
        for key in ('phase', 'tier', 'level_index'):
            if event.get(key) != planned.get(task, {}).get(key):
                problems.append(f'{task}: missing {key} differs from the manifest plan')
    for task in sorted(set(rows) ^ set(completes)):
        problems.append(f"{task}: in {'summary.json' if task in rows else 'task_complete events'} only")
    for task, row in rows.items():
        done = completes.get(task)
        if done is None:
            continue
        if row["attempt"] != done["attempt"]:
            problems.append(f"{task}: summary attempt {row['attempt']}, task_complete attempt {done['attempt']}")
        for key in COUNTERS:
            if row.get(key) != done["result"].get(key):
                problems.append(f"{task}: {key} summary {row.get(key)!r}, task_complete {done['result'].get(key)!r}")
        ended = scan["task_ends"].get(f"{task}/attempt-{done['attempt']}")
        if ended is None:
            problems.append(f"{task}: no task_end event for attempt {done['attempt']}")
        else:
            # task_end is logged before the notebook rewrite, whose tokens are added to usage afterwards.
            problems += [f"{task}: {key} summary {row.get(key)!r}, task_end {ended.get(key)!r}"
                         for key in COUNTERS if key != "usage" and row.get(key) != ended.get(key)]
        events = scan["attempts"][(task, done["attempt"])]
        executes, predicts = events["execute"], events["predict"]
        recomputed = {"success": bool(executes and executes[-1]["success"]),
                      "real_actions": sum(e["executed"] for e in executes), "execute_calls": len(executes),
                      "predict_calls": len(predicts), "predicted_transitions": sum(e["charged"] for e in predicts)}
        problems += [f"{task}: {key} summary {row.get(key)!r}, recomputed from events {value!r}"
                     for key, value in recomputed.items() if row.get(key) != value]
        reason = _end_reason_problem(row, events, budgets.get(task))
        if reason:
            problems.append(f"{task}: {reason}")
    tasks = [PlannedTask(**p) for p in manifest.get("tasks", {}).get("plan", [])]
    if {p.task_id for p in tasks} >= set(completes):
        status = 'incomplete' if set(completes) != set(planned) else summary['status']
        again = summarize(summary.get("label", ""), summary.get("run_id", ""), tasks, list(completes.values()),
                          summary["infrastructure_failures"], status, list(missing.values()))
        problems += [f"summary {key}: recorded {summary.get(key)!r}, recomputed {again[key]!r}"
                     for key in SUMMARY_KEYS if summary.get(key) != again[key]]
    else:
        problems.append("task_complete events name tasks outside the manifest plan")
    return problems


def exclusions(study: dict[str, Any]) -> dict[str, str | None]:
    """Per study environment, why it is excluded from the pooled core average (None when pooled)."""
    out: dict[str, str | None] = {}
    for env, spec in study["environments"].items():
        pooled = spec.get("pooled", True)
        if not isinstance(pooled, bool):
            raise ValueError(f"'pooled' of {env} must be true or false, not {pooled!r}")
        out[env] = ("pooled: false in the study manifest" if not pooled
                    else "single planned seed block" if len(seed_blocks(study, env)) < 2 else None)
    return out


def plan(study: dict[str, Any], agent: str) -> dict[tuple[str, str, int], Job]:
    """Every planned run of one agent, by (env, condition, seed), exactly as the launcher enumerates them."""
    return {(j.env, j.condition, j.seed): j for j in jobs(study, [Agent(agent, ROOT, "")])}


def verify_plan(study: dict[str, Any], agent: str, folders: list[RunFolder], runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Runs and episodes found against the plan: missing, unfinished, incomplete, extra and mismatched runs.

    `folders` are all folders with the agent's prefix (any environment); `runs` are the loaded latest runs.
    """
    planned = plan(study, agent)
    newest = latest(folders)
    loaded = {(r["env"], r["cond"], r["seed"]): r for r in runs}
    out: dict[str, Any] = {"agent": agent, "planned_runs": len(planned), "planned_episodes": sum(j.episodes for j in planned.values()),
                           "complete_runs": 0, "episodes": 0, "missing": [], "unfinished": [], "incomplete": [],
                           "extra": [], "superseded": [], "mismatched": [], "cells": {}}
    for key, (folder, older) in newest.items():
        out["superseded"] += older
        if key not in planned:
            reason = "environment not in the study manifest" if folder.env not in study["environments"] else "seed block not planned"
            out["extra"].append({"run": folder.path.name, "reason": reason})
    for key, job in planned.items():
        env, slug, seed = key
        cell = out["cells"].setdefault(f"{env}/{slug}",
                                       {"planned_runs": 0, "complete_runs": 0, "planned_episodes": 0, "episodes": 0})
        cell["planned_runs"] += 1
        cell["planned_episodes"] += job.episodes
        r = loaded.get(key)
        if r is None:
            if key in newest:
                out["unfinished"].append(newest[key][0].path.name)
            else:
                out["missing"].append(job.label)
            continue
        s = r["summary"]
        for field, want in (("profile", job.profile), ("memory", job.memory)):
            if r["settings"].get(field) != want:
                out["mismatched"].append({"run": r["dir"], "field": field, "planned": want, "found": r["settings"].get(field)})
        if s["status"] != "complete" or s["tasks_completed"] != job.episodes:
            out["incomplete"].append({"run": r["dir"], "status": s["status"], "episodes": s["tasks_completed"],
                                      "planned": job.episodes})
            continue
        cell["complete_runs"] += 1
        cell["episodes"] += s["tasks_completed"]
        out["complete_runs"] += 1
        out["episodes"] += s["tasks_completed"]
    return out
