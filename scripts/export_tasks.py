"""Export the benchmark's task data: every task of the study, exactly as a run builds it.

For each environment the script builds the run command a study job would use (configs/study.json), parses it with
the runner's own parser, and generates the task pool and plan with `build_tasks`, the function every run calls. It
writes one line per task with the initial observation the agent sees, its SHA-256 (the same hash each run records
in manifest.json), the reference solution, and the step budgets, plus one record per environment with the goal and
rules the agent is given and the faulty predictor's profile (evaluator-only: agents never see it).

Run from the repository root:

    uv run --locked python scripts/export_tasks.py

Output: data/tasks/seed-<seed>.jsonl and data/environments.json. The output is deterministic and makes no model calls.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from world_adapt.environments import registry
from world_adapt.perturbations import registry as perturbations
from world_adapt.runner.config import budgets_for, build_parser, check_args, env_options, parse_args
from world_adapt.runner.protocol import seed_bundle
from world_adapt.runner.run import build_tasks
from world_adapt.runner.study import agents, command, jobs, load_study

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"


def main() -> None:
    study = load_study(ROOT / "configs/study.json")
    agent = agents(study)[0]
    environments, tasks_by_seed = {}, {}
    # Task generation does not depend on the condition, so one job per environment and seed covers all four.
    for job in jobs(study, [agent], conditions=["accurate-predictor-saved-notebook"]):
        parser = build_parser("export")
        args = parse_args(parser, command(job, str(agent.config), Path("/nonexistent"))[3:])
        check_args(parser, args)
        spec = registry.get(job.env)
        options = env_options(args, spec)
        game = spec.make_game(options)
        levels, hashes, plan = build_tasks(args, spec, options, game, seed_bundle(job.seed))
        profile = study["environments"][job.env]["primary_profile"]
        environments[job.env] = {"goal": spec.goal, "rules": spec.rules, "actions": list(game.actions()),
                                 "faulty_predictor_profile": profile,
                                 "faulty_predictor": perturbations.profiles(job.env)[profile].describe()}
        for p in plan:
            level = levels[p.level_index]
            tasks_by_seed.setdefault(job.seed, []).append({
                "environment": job.env, "seed": job.seed, "task_id": p.task_id, "phase": p.phase, "tier": p.tier,
                "level_index": p.level_index, "observation": game.observe(level.state),
                "observation_sha256": hashes[p.level_index], "reference_solution": [str(a) for a in level.solution],
                "budgets": asdict(budgets_for(level, args))})
        close = getattr(game, "close", None)
        if close is not None:
            close()
    (OUT / "tasks").mkdir(parents=True, exist_ok=True)
    for seed, rows in tasks_by_seed.items():
        path = OUT / "tasks" / f"seed-{seed}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        print(f"wrote {path.relative_to(ROOT)} ({len(rows)} tasks)")
    (OUT / "environments.json").write_text(json.dumps(environments, indent=2, sort_keys=True) + "\n")
    print(f"wrote data/environments.json ({len(environments)} environments)")


if __name__ == "__main__":
    main()
