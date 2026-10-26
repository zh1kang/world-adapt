"""Study analysis: run discovery, log audits, paired contrasts, pooling, the plan check, labels and exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tests.fakes import PROTOCOL, ScriptedClient, SolverClient, plan_observations, run_cli, solutions
from world_adapt.runner import analyze, audit
from world_adapt.runner.protocol import PlannedTask
from world_adapt.runner.records import summarize
from world_adapt.runner.report import DIFFS, stratified_ci
from world_adapt.runner.study import CONDITIONS

PREFIX = "t-study"
TIERS = ("easy", "medium", "hard")
FS, FR, AS, AR = ("faulty-predictor-saved-notebook", "faulty-predictor-reset-notebook",
                  "accurate-predictor-saved-notebook", "accurate-predictor-reset-notebook")


def _run(runs: Path, env: str, cond: str, seed: int, heldout: list[bool], adaptation: list[bool],
         predict_calls: list[int], budget: int = 20, status: str = "complete", prefix: str = PREFIX,
         stamp: str | None = None, usage: dict | None = None) -> Path:
    """A run folder whose summary.json, events.jsonl and evaluator.jsonl agree, as the runner writes them."""
    label = f"{prefix}-{env}-{cond}-s{seed}"
    d = runs / f"{stamp or f'20261010-0{seed:05d}'}-{label}"
    d.mkdir(parents=True)
    usage = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110} if usage is None else usage
    faulty = CONDITIONS[cond].predictor == "perturbed"
    plan, completed, per_task, events, private = [], [], [], [], []
    for i, (success, calls) in enumerate(zip(adaptation + heldout, predict_calls)):
        tid = f"t{i}"
        phase = "adaptation" if i < len(adaptation) else "heldout"
        plan.append(PlannedTask(tid, phase, TIERS[i % 3], i))
        result = {"task_id": tid, "success": success, "real_actions": 5, "predicted_transitions": calls * 2,
                  "predict_calls": calls, "execute_calls": 2, "turns": 3, "malformed": 0,
                  "end_reason": "success" if success else "real_budget", "usage": usage, "seconds": 1.0}
        events.append({"type": "attempt_start", "task_id": tid, "attempt": 1})
        events += [{"type": "predict", "task_id": tid, "query_id": f"{tid}/attempt-1/predict-{k + 1}", "charged": 2}
                   for k in range(calls)]
        events += [{"type": "execute", "task_id": tid, "execution_id": f"{tid}/attempt-1/execute-{k + 1}",
                    "executed": n, "success": success and k == 1, "terminal": False} for k, n in enumerate((2, 3))]
        events.append({"type": "task_end", "episode_id": f"{tid}/attempt-1",
                       "result": {k: v for k, v in result.items() if k != "seconds"}})
        done = {"type": "task_complete", "task_id": tid, "phase": phase, "tier": TIERS[i % 3], "level_index": i,
                "attempt": 1, "result": result}
        events.append(done)
        completed.append(done)
        visible = [[faulty and k == 0, False] for k in range(calls)]
        private.append({"type": "task_end_private", "task_id": tid, "episode_id": f"{tid}/attempt-1",
                        "prediction_changed": [[[]] for _ in range(calls)], "prediction_visible_changed": visible,
                        "prediction_requested_transitions": [2] * calls, "visible_change_recording_version": 1,
                        "prediction_feedback": {"matched_prediction_queries": calls, "matched_prediction_steps": calls,
                                                "observed_mismatch_queries": int(faulty and calls > 0),
                                                "observed_mismatch_steps": int(faulty and calls > 0),
                                                "first_mismatch_real_action": None}})
        per_task.append({"task_id": tid, "real_actions": 5, "predicted_transitions": budget, "max_turns": 40,
                         "max_malformed": 5, "max_task_tokens": None, "max_task_seconds": None})
    events += [{"type": "api_error"}, {"type": "context_trim"}, {"type": "context_trim"}]
    (d / "summary.json").write_text(json.dumps(summarize(label, d.name, plan, completed, [], status)))
    (d / "manifest.json").write_text(json.dumps({
        "fingerprint": "f", "budgets": {"per_task": per_task},
        "tasks": {"plan": [vars(p) for p in plan], "observation_sha256": [f"h{i}" for i in range(len(plan))]},
        "settings": {"profile": "skew" if faulty else "identity", "memory": CONDITIONS[cond].memory}}))
    (d / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (d / "evaluator.jsonl").write_text("".join(json.dumps(e) + "\n" for e in private))
    return d


def _block(runs: Path, env: str, seed: int, rates: dict[str, float], **kw) -> None:
    """One seed block; each run plays 2 adaptation and 3 held-out tasks with the given held-out success rate."""
    for cond, rate in rates.items():
        wins = round(rate * 3)
        _run(runs, env, cond, seed, heldout=[k < wins for k in range(3)], adaptation=[True, False],
             predict_calls=[1] * 5, **kw)


FULL = {FS: 1.0, FR: 0.0, AS: 1.0, AR: 1.0}


def _manifest(tmp: Path) -> Path:
    design = {"name": "t-study", "launch_approved": True, "sources": {}, "budgets": {},
              "protocol": {"seed_blocks": [0, 1], "runner_args": {"adaptation_tasks": 2, "heldout_per_tier": 1}},
              "conditions": {slug: {"predictor": c.predictor, "memory": c.memory} for slug, c in CONDITIONS.items()},
              "environments": {"patrol": {"primary_profile": "skew"},
                               "iceslide": {"primary_profile": "skew", "seed_blocks": [0]},
                               "depot": {"primary_profile": "skew", "pooled": False}},
              "agents": {"m1": {"config": "c1.json", "backend": "sail"}}}
    path = tmp / "study.json"
    path.write_text(json.dumps(design))
    return path


@pytest.fixture
def two_blocks(tmp_path) -> Path:
    for seed in (0, 1):
        _run(tmp_path, "patrol", FS, seed, heldout=[True, False], adaptation=[True, True], predict_calls=[0, 0, 4, 4])
        _run(tmp_path, "patrol", FR, seed, heldout=[False, False], adaptation=[True, False], predict_calls=[2, 2, 2, 2])
        _run(tmp_path, "patrol", AS, seed, heldout=[True, True], adaptation=[True, True], predict_calls=[5, 5, 5, 5])
        _run(tmp_path, "patrol", AR, seed, heldout=[True, False], adaptation=[True, True], predict_calls=[10] * 4)
    (tmp_path / f"20261010-000000-other-patrol-{FS}-s0").mkdir()
    return tmp_path


# Aggregation ------------------------------------------------------------------------------------------------------


def test_runs_are_found_by_prefix_with_their_budgets_and_event_counts(two_blocks):
    runs = analyze.load_runs(two_blocks, PREFIX)
    assert len(runs) == 8 and {r["cond"] for r in runs} == set(CONDITIONS)
    assert runs[0]["budgets"]["t0"]["predicted_transitions"] == 20
    assert runs[0]["api_errors"] == 1 and runs[0]["context_trims"] == 2
    assert all(r["log_mismatches"] == [] for r in runs)
    assert analyze.load_runs(two_blocks, "nothing") == []


def test_aggregate_reports_success_paired_contrasts_and_predictor_use(two_blocks):
    results = analyze.aggregate(analyze.load_runs(two_blocks, PREFIX), boot=200)
    patrol = results["environments"]["patrol"]
    assert patrol["blocks"] == 2 and patrol["pooled"]
    assert patrol["conditions"][FS]["heldout_success"] == 0.5 and patrol["conditions"][AS]["heldout_success"] == 1.0
    differences = {k: v["mean"] for k, v in patrol["differences"].items()}
    assert differences == {"faulty-predictor-memory-benefit": 0.5, "accurate-predictor-memory-benefit": 0.5,
                           "memory-benefit-difference": 0.0, "predictor-error-cost-saved-notebook": 0.5,
                           "predictor-error-cost-reset-notebook": 0.5}
    usage = patrol["conditions"][FS]["predictor"]
    assert usage == {"mean_predict_calls": 2.0, "zero_predict_share": 0.5, "prediction_budget_share": 0.2, "episodes": 8}
    assert patrol["conditions"][FS]["predictor_heldout"]["zero_predict_share"] == 0.0
    assert patrol["conditions"][FS]["exposure"]["exposed_episode_share"] == 0.5
    assert patrol["conditions"][FS]["exposure"]["exposed_prediction_share"] == 0.25
    assert patrol["conditions"][AS]["exposure"]["exposed_episode_share"] == 0.0
    assert patrol["conditions"][FS]["feedback_heldout"]["mismatch_episode_share"] == 1.0
    assert patrol["conditions"][FS]["per_episode"]["prompt_tokens"] == 100
    assert results["pooled"]["conditions"][FS]["heldout_success"] == 0.5
    assert results["counts"]["episodes"] == 32 and results["tokens"]["prompt_tokens"] == 3200


def test_incomplete_runs_never_enter_protocol_metrics(two_blocks):
    _run(two_blocks, "patrol", FS, 2, heldout=[True, True], adaptation=[True, True], predict_calls=[1] * 4,
         status="incomplete")
    results = analyze.aggregate(analyze.load_runs(two_blocks, PREFIX), boot=200)
    assert results["environments"]["patrol"]["blocks"] == 2
    assert results["counts"]["incomplete_runs"] == [f"20261010-000002-{PREFIX}-patrol-{FS}-s2"]


def test_outputs_use_full_condition_names_and_slugs(two_blocks, tmp_path):
    runs = analyze.load_runs(two_blocks, PREFIX)
    md = analyze.write_outputs(tmp_path / "out", runs, analyze.aggregate(runs, boot=200))
    assert "Faulty predictor + saved notebook: success" in md and "| patrol | 2 | 100.0%" in md
    assert "Faulty predictor: saved minus reset notebook" in md
    assert "| **Pooled core (equally weighted)** | patrol 2 |" in md
    for short in ("P-M", "U-R", "| PM", "G_P", "condition_code"):
        assert short not in md
    with (tmp_path / "out" / "runs.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert {r["condition"] for r in rows} == {c.name for c in CONDITIONS.values()}
    assert {r["condition_slug"] for r in rows} == set(CONDITIONS)
    episodes = (tmp_path / "out" / "episodes.csv").read_text().splitlines()
    assert episodes[0].split(",")[:4] == ["env", "condition", "condition_slug", "seed"] and len(episodes) == 33
    results = json.loads((tmp_path / "out" / "results.json").read_text())
    assert set(results["environments"]["patrol"]["conditions"]) == set(CONDITIONS)
    assert set(results["difference_labels"]) == set(DIFFS)


def test_stratified_bootstrap_known_answers():
    two = {"A": {0: {"x": 0.0}, 1: {"x": 1.0}}, "B": {0: {"x": 1.0}, 1: {"x": 1.0}}}
    assert stratified_ci(two, ("x",), "k", boot=2000) == {"x": (0.5, 1.0)}
    # Blocks never move between environments; environments are weighted equally whatever their block counts.
    strata = {"A": {s: {"x": 0.0} for s in range(2)}, "B": {s: {"x": 1.0} for s in range(10)}}
    assert stratified_ci(strata, ("x",), "k", boot=500) == {"x": (0.5, 0.5)}
    assert stratified_ci({"A": {0: {"x": 1.0}}}, ("x",), "k") == {"x": None}
    assert stratified_ci(two, ("x",), "seed") == stratified_ci(two, ("x",), "seed")


def test_pooling_weights_environments_equally_and_reports_single_blocks_separately(tmp_path):
    for seed in (0, 1):
        _block(tmp_path, "patrol", seed, {FS: 1 / 3, FR: 0.0, AS: 1.0, AR: 2 / 3})
    for seed in range(4):
        _block(tmp_path, "iceslide", seed, {FS: 1.0, FR: 1.0, AS: 1.0, AR: 1.0})
    _block(tmp_path, "depot", 0, {FS: 0.0, FR: 0.0, AS: 0.0, AR: 0.0})
    results = analyze.aggregate(analyze.load_runs(tmp_path, PREFIX), boot=500)
    p = results["pooled"]
    assert p["environments"] == ["iceslide", "patrol"] and p["blocks"] == {"iceslide": 4, "patrol": 2}
    assert p["excluded"] == {"depot": "single seed block"}
    assert p["conditions"][FS]["heldout_success"] == pytest.approx(2 / 3)
    assert p["differences"]["faulty-predictor-memory-benefit"]["mean"] == pytest.approx(1 / 6)
    depot = results["environments"]["depot"]
    assert not depot["pooled"] and depot["conditions"][FS]["ci95"] is None
    md = analyze.render(results)
    assert "## Separately reported environments" in md and "Excluded from the pooled core average: depot" in md


def test_each_contrast_uses_its_own_complete_and_comparable_pairs(tmp_path):
    for seed in (0, 1):
        _block(tmp_path, "patrol", seed, {FS: 1.0, FR: 0.0})
    _block(tmp_path, "patrol", 2, FULL)
    results = analyze.aggregate(analyze.load_runs(tmp_path, PREFIX), boot=100)
    differences = results["environments"]["patrol"]["differences"]
    assert differences["faulty-predictor-memory-benefit"]["blocks"] == 3
    assert differences["accurate-predictor-memory-benefit"]["blocks"] == 1
    assert differences["memory-benefit-difference"]["blocks"] == 1
    coverage = results["pooled"]["quantity_coverage"]
    assert coverage["faulty-predictor-memory-benefit"]["blocks"] == {"patrol": 3}
    assert coverage[FS]["excluded"] == {"patrol": "single complete paired seed block"}


def test_a_task_identity_mismatch_excludes_only_the_affected_pairs(tmp_path):
    _block(tmp_path, "patrol", 0, FULL)
    folder = next(tmp_path.glob(f"*-{AR}-s0"))
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["tasks"]["observation_sha256"][0] = "different"
    (folder / "manifest.json").write_text(json.dumps(manifest))
    results = analyze.aggregate(analyze.load_runs(tmp_path, PREFIX), boot=50)
    differences = results["environments"]["patrol"]["differences"]
    assert differences["faulty-predictor-memory-benefit"]["blocks"] == 1
    assert differences["accurate-predictor-memory-benefit"]["blocks"] == 0
    assert any(i["reason"] == "task identity or budget mismatch" for i in results["pairing_issues"]["patrol"])


def test_duplicate_run_identities_are_rejected(tmp_path):
    runs = analyze.load_runs(_run(tmp_path, "patrol", FS, 0, [True], [True], [1, 1]).parent, PREFIX)
    with pytest.raises(ValueError, match="duplicate run identity"):
        analyze.aggregate(runs + runs, boot=10)


def test_unknown_or_partial_usage_is_never_summed_as_zero():
    for usage in ({"prompt_tokens": 10, "completion_tokens": 5}, {"prompt_tokens": 10, "completion_tokens": 5,
                                                                   "total_tokens": 0},
                  {"prompt_tokens": True, "completion_tokens": 5, "total_tokens": 6},
                  {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "writer_unknown_reserved_tokens": 9},
                  {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "planning_unknown_requests": 1}):
        run = {"summary": {"results": [{"task_id": "a", "attempt": 1, "usage": usage}], "infrastructure_failures": []},
               "budgets": {}, "exposure": {}}
        totals = analyze.token_totals([run])
        assert totals["unknown_usage_episodes"] == 1 and totals["prompt_tokens"] is None


# The study plan ---------------------------------------------------------------------------------------------------


def test_the_plan_check_flags_missing_extra_superseded_and_incomplete_runs(tmp_path):
    from world_adapt.runner.study import load_study
    design = load_study(_manifest(tmp_path))
    runs, prefix = tmp_path / "runs", "t-study-m1"
    _block(runs, "patrol", 0, FULL, prefix=prefix)
    _block(runs, "patrol", 1, {FR: 0.0, AS: 1.0}, prefix=prefix)
    _run(runs, "patrol", AR, 1, heldout=[True], adaptation=[True, True], predict_calls=[1] * 3, prefix=prefix,
         status="incomplete")
    _block(runs, "patrol", 7, {FS: 1.0}, prefix=prefix)
    _block(runs, "iceslide", 0, {FS: 0.0, FR: 0.0, AS: 1.0, AR: 1.0}, prefix=prefix)
    _block(runs, "iceslide", 0, {FS: 1.0}, prefix=prefix, stamp="20261011-000000")
    (runs / f"20261010-000000-{prefix}-depot-{FS}-s0").mkdir()
    assert audit.exclusions(design) == {"patrol": None, "iceslide": "single planned seed block",
                                        "depot": "pooled: false in the study manifest"}
    _, results = analyze.analyze_agent(runs, design, "m1", boot=100)
    plan = results["plan"]
    assert plan["planned_runs"] == 8 + 4 + 8 and plan["planned_episodes"] == 20 * 5
    assert f"{prefix}-patrol-{FS}-s1" in plan["missing"]
    assert plan["unfinished"] == [f"20261010-000000-{prefix}-depot-{FS}-s0"]
    assert plan["incomplete"] == [{"run": f"20261010-000001-{prefix}-patrol-{AR}-s1", "status": "incomplete",
                                   "episodes": 3, "planned": 5}]
    assert {x["reason"] for x in plan["extra"]} == {"seed block not planned"}
    assert plan["superseded"] == [f"20261010-000000-{prefix}-iceslide-{FS}-s0"]
    assert plan["cells"][f"patrol/{FS}"] == {"planned_runs": 2, "complete_runs": 1, "planned_episodes": 10, "episodes": 5}
    assert results["environments"]["iceslide"]["conditions"][FS]["heldout_success"] == 1.0
    md = analyze.render(results)
    assert "Plan (m1): 10/20 runs complete" in md and "patrol / Faulty predictor + saved notebook 5/10" in md


def test_the_cli_reads_a_prepared_study_directory(tmp_path, capsys):
    path = _manifest(tmp_path)
    _block(tmp_path / "runs", "patrol", 0, FULL, prefix="t-study-m1")
    assert analyze.main(["--study", str(path)]) == 0
    capsys.readouterr()
    results = json.loads((tmp_path / "analysis" / "results.json").read_text())
    assert results["plan"]["complete_runs"] == 4
    assert analyze.main(["--study", str(path), "--runs", str(tmp_path / "empty")]) == 1
    with pytest.raises(SystemExit):
        analyze.main(["--study", str(path), "--agent", "unknown"])


# Log audits ---------------------------------------------------------------------------------------------------------


def test_the_log_check_detects_a_summary_that_disagrees_with_its_events(tmp_path):
    d = _run(tmp_path, "patrol", FS, 0, heldout=[True, False], adaptation=[True, False], predict_calls=[1] * 4)
    summary = json.loads((d / "summary.json").read_text())
    summary["results"][3].update(success=True, end_reason="success")
    summary["results"][1]["end_reason"] = "turn_limit"
    summary["heldout_success_rate"] = 1.0
    (d / "summary.json").write_text(json.dumps(summary))
    mismatches = analyze.load_runs(tmp_path, PREFIX)[0]["log_mismatches"]
    assert "t3: success summary True, recomputed from events False" in mismatches
    assert "t3: end_reason 'success' but the last logged execute has success=False" in mismatches
    assert "t1: end_reason 'turn_limit' is not supported by the logs and budgets" in mismatches
    assert "summary heldout_success_rate: recorded 1.0, recomputed 0.5" in mismatches
    results = analyze.aggregate(analyze.load_runs(tmp_path, PREFIX), boot=100)
    assert list(results["counts"]["log_mismatches"]) == [d.name]
    assert results["environments"]["patrol"]["conditions"][FS]["heldout_success"] is None


def test_reply_cutoffs_are_counted_for_scored_and_all_attempts(tmp_path):
    d = _run(tmp_path, "patrol", FS, 0, heldout=[True], adaptation=[True], predict_calls=[1, 1])
    with (d / "events.jsonl").open("a") as stream:
        for event in ({"type": "attempt_start", "task_id": "t0", "attempt": 2}, {"type": "reply_cutoff", "task_id": "t0"}):
            stream.write(json.dumps(event) + "\n")
    lines = (d / "events.jsonl").read_text().splitlines()
    lines.insert(1, json.dumps({"type": "reply_cutoff", "task_id": "t0", "turn": 1}))
    (d / "events.jsonl").write_text("\n".join(lines) + "\n")
    scan = audit.scan_events(d)
    assert scan["reply_cutoffs"] == 1 and scan["reply_cutoffs_all"] == 2


def test_the_audit_requires_end_reasons_to_match_the_cutoff_mix():
    budget = {"max_malformed": 2, "real_actions": 5, "max_turns": 4, "max_task_tokens": None, "max_task_seconds": None}

    def problem(reason, cutoffs):
        row = {"end_reason": reason, "malformed": 2, "real_actions": 0, "turns": 2, "usage": {}}
        events = {"execute": [], "predict": [], "concede": [], "reply_cutoff": [{"turn": t} for t in range(cutoffs)]}
        return audit._end_reason_problem(row, events, budget)

    assert problem("response_limit", 1) is None and problem("reply_cutoff_limit", 2) is None
    assert problem("malformed_limit", 0) is None
    for reason, cutoffs in (("malformed_limit", 1), ("reply_cutoff_limit", 1), ("malformed_limit", 2),
                            ("response_limit", 2)):
        assert problem(reason, cutoffs) is not None


def test_runner_output_passes_the_log_check_and_replays_after_a_resume(tmp_path, monkeypatch):
    argv = [*PROTOCOL, "--profile", "breaker_retarget_enabled_only", "--label", f"{PREFIX}-switchboard-{FS}-s5"]
    solved = solutions(argv)
    with pytest.raises(KeyboardInterrupt):
        run_cli(argv, tmp_path, SolverClient(solved, predict=True, crash_after=(plan_observations(argv)[1], 2)),
                monkeypatch)
    (run_dir,) = list((tmp_path / "runs").iterdir())
    run_cli([*argv, "--resume-dir", str(run_dir)], tmp_path, SolverClient(solved, predict=True), monkeypatch)
    silent = [*PROTOCOL, "--memory", "reset", "--label", f"{PREFIX}-switchboard-{AR}-s5", "--adaptation-tasks", "1"]
    run_cli(silent, tmp_path, ScriptedClient([("thinking", None)] * 30), monkeypatch)
    runs = analyze.load_runs(tmp_path / "runs", PREFIX, replay=True)
    assert [r["log_mismatches"] for r in runs] == [[], []]
    accurate, faulty = sorted(runs, key=lambda r: r["cond"])
    assert faulty["summary"]["infrastructure_failures"] and faulty["exposure"]
    assert {row["end_reason"] for row in accurate["summary"]["results"]} == {"malformed_limit"}
    results = analyze.aggregate(runs, boot=50)
    assert results["environments"]["switchboard"]["conditions"][FS]["exposure"]["episodes"] == 7
    assert results["tokens"]["unrecorded_attempts"] == 1


def test_missing_tasks_are_reported_without_being_scored(tmp_path, monkeypatch):
    argv = PROTOCOL + ["--infra-retries", "0", "--infrastructure-policy", "skip", "--label",
                       f"missing-switchboard-{FS}-s5"]

    class FailOnSecond(SolverClient):
        def create(self, **kwargs):
            if self.task(kwargs["messages"]) == plan_observations(argv)[1]:
                raise ConnectionError("provider unavailable")
            return super().create(**kwargs)

    with pytest.raises(SystemExit):
        run_cli(argv, tmp_path, FailOnSecond(solutions(argv)), monkeypatch)
    runs = analyze.load_runs(tmp_path / "runs", "missing")
    results = analyze.aggregate(runs, boot=10)
    assert results["counts"]["missing_task_count"] == 1
    assert results["counts"]["missing_tasks"][0]["task_id"] == "adapt-1"
    assert "Explicitly missing tasks: 1" in analyze.render(results)
    assert runs[0]["log_mismatches"] == []
