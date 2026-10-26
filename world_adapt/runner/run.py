"""Runs: one agent configuration plays the protocol in one environment.

    uv run --extra openai python -m world_adapt.runner.run --agent-config configs/agents/example.json \
        --env switchboard --profile identity --memory preserve --seed 3000

A run plays 8 adaptation tasks in a fixed tier-balanced order, then 2 held-out tasks per tier, each branching from
a copy of the notebook saved after adaptation (empty under --memory reset). Studies launch runs through
`world_adapt.runner.local`, which passes every protocol flag from the study manifest.

Logs go to the selected run directory; see world_adapt/runner/records.py.
An infrastructure failure (the model API failing after every retry, or an interrupted attempt found on resume)
is retried --infra-retries times in a fresh environment with the same notebook; after that the default stops the
history, while --infrastructure-policy skip records an unscored missing task and continues. The process exits with
status 2 when the final summary contains missing work.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

from world_adapt.agent import PilotAgent, carry_over, writer
from world_adapt.agent.llm import InfrastructureError
from world_adapt.agent.memory import TOKENIZER
from world_adapt.environments import registry
from world_adapt.environments.base import Task
from world_adapt.perturbations import registry as perturbation_registry
from world_adapt.predictors.simulator import SimulatorPredictor
from world_adapt.runner import manifest as manifests
from world_adapt.runner.config import (BUDGET_RULE, PROVIDERS, agent_config, budgets_for, build_parser, check_args,
                                   default_label, env_options, make_client, parse_args, provider_base_url,
                                   settings)
from world_adapt.runner.protocol import plan_protocol, pool_size, seed_bundle, task_record
from world_adapt.runner.records import RunRecord

__all__ = ["PROVIDERS", "build_tasks", "main", "make_client"]


def build_tasks(args, spec, options, game, seeds):
    """The level pool and the plan. Deterministic in the configuration and seeds, so it is identical across the
    paired conditions of a seed block and can be regenerated to rescore a run."""
    n = pool_size(args.adaptation_tasks, args.heldout_per_tier)
    levels = spec.make_levels(game, n, seeds["task_generation"], options)
    if len(levels) < n:
        raise SystemExit(f"generated only {len(levels)} of {n} levels; change --seed or level settings")
    hashes = [hashlib.sha256(game.observe(lvl.state).encode()).hexdigest() for lvl in levels]
    if len(set(hashes)) != len(hashes):
        raise SystemExit("the task pool contains duplicate initial observations")
    try:
        plan = plan_protocol(levels, args.adaptation_tasks, args.heldout_per_tier, seeds["task_order"])
    except ValueError as error:
        raise SystemExit(f"cannot build the task split: {error}")
    return levels, hashes, plan


def main(argv: list[str] | None = None) -> None:
    parser = build_parser(__doc__)
    args = parse_args(parser, argv)
    check_args(parser, args)
    base_url, key_env = provider_base_url(args.provider), PROVIDERS[args.provider][1]
    client = make_client(args.provider, base_url, key_env)
    if args.provider == "anthropic":
        from world_adapt.agent.anthropic import AnthropicChatAdapter

        client = AnthropicChatAdapter(client)
    elif args.stream:
        from world_adapt.agent.streaming import StreamingChatAdapter

        client = StreamingChatAdapter(client, require_usage=args.require_usage)
    # The spending wrapper is opt-in through its own environment-backed ledger. Keeping this import and call
    # here leaves the runner's default transport and all existing local tests unchanged.
    from world_adapt.agent.spending import wrap_client

    client = wrap_client(client)

    spec = registry.get(args.env)
    options = env_options(args, spec)
    game = spec.make_game(options)
    available = perturbation_registry.profiles(args.env)
    if args.profile not in available:
        raise SystemExit(f"unknown profile {args.profile!r} for {args.env}; choose from {', '.join(sorted(available))}")
    profile = available[args.profile]
    seeds = seed_bundle(args.seed)
    levels, hashes, plan = build_tasks(args, spec, options, game, seeds)
    budgets = {p.task_id: budgets_for(levels[p.level_index], args) for p in plan}

    config = agent_config(args, spec)
    label = default_label(args)
    run_settings = {**settings(args), "budget_rule": BUDGET_RULE,
                    "protocol": f"{args.adaptation_tasks} adaptation + {args.heldout_per_tier} held-out per tier"}
    recorded_agent = asdict(config)
    if config.action_tool_choice is None:
        recorded_agent.pop("action_tool_choice")
    recorded_agent.update(writer=writer.WRITER, writer_input_version=writer.INPUT_VERSION,
                          writer_provider_attempts=writer.WRITER_PROVIDER_ATTEMPTS, writer_infrastructure_retries=0)
    manifest = manifests.build(
        env=args.env, options=options, game=game, profile=profile, settings=run_settings,
        budgets=[{"task_id": p.task_id, **asdict(budgets[p.task_id])} for p in plan], agent=recorded_agent,
        plan=[task_record(p) for p in plan], task_hashes=hashes, seeds=seeds, tokenizer=TOKENIZER,
        provider={"name": args.provider, "base_url": base_url, "api": "chat", "model": args.model})
    if args.resume_dir:
        record = RunRecord.resume(Path(args.resume_dir), manifest)
        record.log({"type": "resume", "completed": sorted(record.completed), "code": manifest["code"],
                    "started": manifest["started"]})
    else:
        record = RunRecord.create(label, manifest, {
            # Evaluator-only: never supplied to the agent.
            "profile": profile.describe(), "rules": spec.rules, "levels": [lvl.features for lvl in levels],
            "task_observation_hashes": hashes, "reference_solutions": [list(lvl.solution) for lvl in levels],
            "plan": [task_record(p) for p in plan]}, root=args.runs_dir)
    agent = PilotAgent(client, config, record.log, record.log_evaluator)

    def play(planned, start_notebook: str) -> str | None:
        """Play one task, retrying infrastructure failures; optionally record an unscored omission."""
        done = record.completed.get(planned.task_id)
        if done is not None:
            return done["notebook"]
        recovered = record.recover_outcome(planned)
        if recovered is not None:
            return recovered
        missing = record.missing.get(planned.task_id)
        if missing is not None:
            return missing["incoming_notebook"]
        # Attempts started before an interruption were never confirmed: record them as failed, but give every resume
        # a full retry budget. An interruption is not a task failure, and counting it would let one outage leave the
        # run unresumable; the launcher's max-attempts bounds how often a run is resumed.
        for prior in range(1, record.attempts[planned.task_id] + 1):
            record.fail_attempt(planned.task_id, prior, "attempt interrupted before task_complete; not continued")
        last_error = None
        for _ in range(args.infra_retries + 1):
            attempt = record.start_attempt(planned.task_id)
            level = levels[planned.level_index]
            task = Task(planned.task_id, level.state, {"reference_length": level.features["solution_length"]})
            # A fresh predictor per attempt: no cache, history or rollout state crosses a task boundary.
            predictor = SimulatorPredictor(game, profile, max_horizon=args.max_chunk)
            try:
                result, notebook = agent.run_task(game, task, predictor, budgets[planned.task_id], start_notebook,
                                                  goal=spec.goal, prune=spec.prune,
                                                  episode_id=f"{planned.task_id}/attempt-{attempt}",
                                                  rules=spec.rules,
                                                  commit_outcome=lambda outcome: record.commit_outcome(
                                                      planned, attempt, asdict(outcome), start_notebook))
            except InfrastructureError as error:
                last_error = str(error)
                record.fail_attempt(planned.task_id, attempt, str(error))
                continue
            except BaseException:
                # Preserve the journal/snapshots and release the writer lock before pausing or crashing.
                # A v1 outcome is durable and will not replay when this run resumes.
                record.recover_outcome(planned)
                record.finish(label, plan, "incomplete")
                raise
            record.complete(planned, attempt, asdict(result), start_notebook, notebook)
            b = budgets[planned.task_id]
            status = "solved" if result.success else f"failed ({result.end_reason})"
            print(f"{planned.task_id} [{planned.tier or '-'}]: {status} | real {result.real_actions}/{b.real_actions} "
                  f"(ref {level.features['solution_length']}) | predict calls {result.predict_calls}, "
                  f"predicted {result.predicted_transitions} | turns {result.turns} | "
                  f"tokens {result.usage.get('total_tokens', '?')}", flush=True)
            return notebook
        if args.infrastructure_policy == "skip":
            return record.mark_missing(
                planned, start_notebook,
                f"infrastructure retries exhausted: {last_error or 'unknown infrastructure failure'}",
                attempt=record.attempts[planned.task_id],
            )
        return None

    complete = True
    notebook = ""
    for planned in [p for p in plan if p.phase == "adaptation"]:
        written = play(planned, carry_over(args.memory, notebook))
        if written is None:
            complete = False
            break
        notebook = written
    if complete:
        saved = notebook
        for planned in [p for p in plan if p.phase == "heldout"]:
            # Every held-out branch starts from the same saved notebook; what it writes goes nowhere.
            if play(planned, carry_over(args.memory, saved)) is None:
                complete = False
                break

    summary = record.finish(label, plan, "complete" if complete else "incomplete")
    rate = summary["success_rate"]
    print(f"{summary['primary_metric']} {'n/a' if rate is None else f'{rate:.0%}'} ({summary['status']}); "
          f"logs in {record.run_dir}")
    if not complete or summary["tasks_missing"]:
        raise SystemExit(2)


if __name__ == "__main__":
    from world_adapt.agent.spending import SpendLimitExceeded

    try:
        main()
    except SpendLimitExceeded as error:
        print(f"Inference paused: {error}", flush=True)
        raise SystemExit(75) from None
