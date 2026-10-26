"""Recompute a run's scores and resource totals from its logs, with no agent or predictor call.

    uv run python -m world_adapt.runner.rescore outputs/my-study/runs/<run>

Regenerates the task pool from the manifest (checking the initial observations against the recorded hashes),
replays each scored attempt's logged real actions in the real environment, and recomputes success and resource
totals from the logged events. Exits 1 if anything disagrees with summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from world_adapt.environments import registry
from world_adapt.environments.base import Task
from world_adapt.environments.real import RealEnvironment
from world_adapt.runner.config import env_options
from world_adapt.runner.protocol import SEED_STREAMS
from world_adapt.runner.records import completed_tasks, missing_tasks, read_events, summarize
from world_adapt.runner.run import build_tasks


def rescore(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    args = argparse.Namespace(**manifest["settings"])
    spec = registry.get(args.env)
    options = env_options(args, spec)
    game = spec.make_game(options)
    seeds = {k: manifest["seeds"][k] for k in SEED_STREAMS}
    levels, hashes, plan = build_tasks(args, spec, options, game, seeds)
    mismatches: list[str] = []
    if hashes != manifest["tasks"]["observation_sha256"]:
        mismatches.append("regenerated task pool differs from the recorded observation hashes")

    events = read_events(run_dir)
    completed = completed_tasks(run_dir)
    missing = missing_tasks(run_dir)
    by_level = {p.task_id: p.level_index for p in plan}
    rescored = []
    for task_id, done in completed.items():
        episode = f"{task_id}/attempt-{done['attempt']}/"
        executes = [e for e in events if e["type"] == "execute" and e["execution_id"].startswith(episode)]
        predicts = [e for e in events if e["type"] == "predict" and e["query_id"].startswith(episode)]
        level = levels[by_level[task_id]]
        real = RealEnvironment(game, Task(task_id, level.state))
        for e in executes:
            chunk = e["actions"][:e["executed"]]
            if chunk and real.execute(chunk).transitions_executed != len(chunk):
                mismatches.append(f"{task_id}: replayed chunk {e['execution_id']} executed a different length")
        result = dict(done["result"])
        recomputed = {
            "success": game.is_success(real.snapshot),
            "real_actions": sum(e["executed"] for e in executes),
            "execute_calls": len(executes),
            "predicted_transitions": sum(e["charged"] for e in predicts),
            "predict_calls": len(predicts),
        }
        for key, value in recomputed.items():
            if result[key] != value:
                mismatches.append(f"{task_id}: {key} logged {result[key]!r}, recomputed {value!r}")
        rescored.append({**done, "result": {**result, **recomputed}})

    infra = [{k: e[k] for k in ("task_id", "attempt", "error")} for e in events if e["type"] == "infrastructure_failure"]
    old = json.loads((run_dir / "summary.json").read_text())
    status = 'incomplete' if set(completed) != set(by_level) else old['status']
    summary = summarize(old["label"], old["run_id"], plan, rescored, infra, status, list(missing.values()))
    for key in ("status", "tasks_planned", "tasks_completed", "tasks_missing", "missing_count", "missing",
                "success_rate", "heldout_success_rate", "heldout_by_tier", "adaptation_success_rate", "totals"):
        if summary[key] != old.get(key):
            mismatches.append(f"summary {key}: recorded {old.get(key)!r}, recomputed {summary[key]!r}")
    return {"summary": summary, "mismatches": mismatches}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    out = rescore(parser.parse_args().run_dir)
    s = out["summary"]
    print(f"{s['primary_metric']}: {s['success_rate']}; held-out by tier: {s['heldout_by_tier']}; totals: {s['totals']}")
    for m in out["mismatches"]:
        print("MISMATCH", m)
    print("ok: logs reproduce summary.json" if not out["mismatches"] else f"{len(out['mismatches'])} mismatches")
    sys.exit(1 if out["mismatches"] else 0)


if __name__ == "__main__":
    main()
