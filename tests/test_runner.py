"""The run protocol, run records, resume and infrastructure policies, rescoring, and the study scheduler."""

import json
import os
import subprocess
import sys

import pytest

import world_adapt.runner.run as run_module
from tests.fakes import (PROTOCOL, ROOT, SPEND_VARIABLES, TEMPLATE, FailingClient, SolverClient, completes,
                         plan_observations, read, response, run_cli, solutions)
from world_adapt.agent.llm import InfrastructureError
from world_adapt.agent.spending import SpendLimitExceeded
from world_adapt.environments import registry
from world_adapt.perturbations.registry import profiles
from world_adapt.predictors.simulator import SimulatorPredictor
from world_adapt.runner import audit, study
from world_adapt.runner.config import build_parser, check_args, parse_args, settings
from world_adapt.runner.protocol import plan_protocol, pool_size, seed_bundle
from world_adapt.runner.rescore import rescore

FAULTY = "breaker_retarget_enabled_only"


@pytest.fixture(scope="module")
def solved():
    return solutions(PROTOCOL)


# The protocol ------------------------------------------------------------------------------------------------------


def test_task_split_is_tier_balanced_disjoint_and_identical_across_conditions():
    spec = registry.get("switchboard")
    game = spec.make_game(spec.default_options)
    levels = spec.make_levels(game, pool_size(8, 2), 3000, spec.default_options)
    plan = plan_protocol(levels, 8, 2, seed_bundle(3000)["task_order"])
    assert plan == plan_protocol(levels, 8, 2, seed_bundle(3000)["task_order"])
    adaptation = [p for p in plan if p.phase == "adaptation"]
    heldout = [p for p in plan if p.phase == "heldout"]
    assert len(adaptation) == 8 and len(heldout) == 6
    assert sorted(sum(p.tier == t for p in adaptation) for t in ("easy", "medium", "hard")) == [2, 3, 3]
    assert all(sum(p.tier == t for p in heldout) == 2 for t in ("easy", "medium", "hard"))
    assert not {p.level_index for p in adaptation} & {p.level_index for p in heldout}
    assert len({p.tier for p in adaptation[:3]}) == 3


def test_seed_bundle_keeps_the_task_seed_and_derives_independent_streams():
    bundle = seed_bundle(3000)
    assert bundle["task_generation"] == 3000 and bundle == seed_bundle(3000)
    assert len(set(bundle.values())) == len(bundle)


def test_scripted_agent_completes_the_protocol_and_the_logs_rescore(tmp_path, monkeypatch, solved):
    run_dir = run_cli(PROTOCOL, tmp_path, SolverClient(solved), monkeypatch)
    summary = read(run_dir, "summary.json")
    assert summary["status"] == "complete" and summary["tasks_completed"] == summary["tasks_planned"] == 7
    assert summary["success_rate"] == summary["heldout_success_rate"] == 1.0
    assert summary["heldout_by_tier"] == {"easy": 1.0, "medium": 1.0, "hard": 1.0}
    out = rescore(run_dir)
    assert out["mismatches"] == []
    assert out["summary"]["totals"]["real_actions"] == summary["totals"]["real_actions"] > 0
    summary["totals"]["real_actions"] += 1
    (run_dir / "summary.json").write_text(json.dumps(summary))
    assert any("totals" in m for m in rescore(run_dir)["mismatches"])


def test_hidden_fields_stay_out_of_agent_requests_and_the_public_log(tmp_path, monkeypatch, solved):
    client = SolverClient(solved, predict=True)
    run_dir = run_cli(PROTOCOL + ["--profile", FAULTY, "--label", "hidden-check"], tmp_path, client, monkeypatch)
    sent = json.dumps([r["messages"] for r in client.requests])
    assert FAULTY not in sent and "reference" not in sent and "hidden reasoning" not in json.dumps(
        [r["messages"] for r in client.requests if not r.get("tools")])
    public = (run_dir / "events.jsonl").read_text()
    for hidden in ("reference_solutions", "prediction_changed", "dead_check", "diagnostics", FAULTY):
        assert hidden not in public
    evaluator = read(run_dir, "evaluator.jsonl")
    assert evaluator[0]["type"] == "evaluator_manifest" and evaluator[0]["profile"]["name"] == FAULTY
    assert "atoms" not in json.dumps(read(run_dir, "manifest.json")["predictor"])


def test_the_manifest_identifies_the_run(tmp_path, monkeypatch, solved):
    run_dir = run_cli(PROTOCOL, tmp_path, SolverClient(solved), monkeypatch)
    m = read(run_dir, "manifest.json")
    for key in ("specification", "code", "environment", "tasks", "predictor", "observation_action_adapter", "budgets",
                "memory", "agent", "seeds", "dependencies", "hardware", "started", "artifacts", "fingerprint"):
        assert key in m, key
    assert m["memory"]["tokenizer"] == "tiktoken/o200k_base" and m["seeds"]["task_order"] == seed_bundle(5)["task_order"]
    assert len(m["budgets"]["per_task"]) == 7
    assert m["agent"]["settings"]["writer"] == "freeform-v1"
    assert m["agent"]["harness_source_sha256"] and m["agent"]["prompt_source_sha256"]


def test_saved_notebooks_chain_adaptation_and_branch_every_heldout_task(tmp_path, monkeypatch, solved):
    client = SolverClient(solved)
    run_dir = run_cli(PROTOCOL, tmp_path, client, monkeypatch)
    done = completes(run_dir)
    plan = [e["task_id"] for e in read(run_dir, "manifest.json")["tasks"]["plan"]]
    adaptation = [t for t in plan if t.startswith("adapt")]
    heldout = [t for t in plan if t.startswith("heldout")]
    assert done[adaptation[0]]["start_notebook"] == ""
    for before, after in zip(adaptation, adaptation[1:]):
        assert done[after]["start_notebook"] == done[before]["notebook"] != ""
    saved = done[adaptation[-1]]["notebook"]
    assert all(done[t]["start_notebook"] == saved for t in heldout)
    heldout_notes = {done[t]["notebook"] for t in heldout} - {saved}
    planning = [r["messages"][1]["content"] for r in client.requests if r.get("tools")]
    assert heldout_notes and not any(note in text for note in heldout_notes for text in planning)
    notebooks = read(run_dir, "notebooks.json")
    assert notebooks["saved_after_adaptation"] == saved and len(notebooks["heldout"]) == 3


def test_reset_notebooks_start_every_task_empty_but_still_record_notes(tmp_path, monkeypatch, solved):
    client = SolverClient(solved)
    run_dir = run_cli(PROTOCOL + ["--memory", "reset"], tmp_path, client, monkeypatch)
    for e in completes(run_dir).values():
        assert e["start_notebook"] == "" and e["notebook"].startswith("note after task")
    assert all("(empty)" in r["messages"][1]["content"] for r in client.requests if r.get("tools"))


def test_every_task_attempt_gets_a_fresh_predictor(tmp_path, monkeypatch, solved):
    made = []

    class Recording(SimulatorPredictor):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    monkeypatch.setattr(run_module, "SimulatorPredictor", Recording)
    run_cli(PROTOCOL, tmp_path, SolverClient(solved), monkeypatch)
    assert len(made) == 7 and len({id(p) for p in made}) == 7


# Resume and infrastructure failures ---------------------------------------------------------------------------------


def test_an_interrupted_task_resumes_in_a_fresh_attempt_without_repeating_actions(tmp_path, monkeypatch, solved):
    crash = plan_observations(PROTOCOL)[1]
    with pytest.raises(KeyboardInterrupt):
        run_cli(PROTOCOL, tmp_path, SolverClient(solved, crash_after=(crash, 2)), monkeypatch)
    (run_dir,) = list((tmp_path / "runs").iterdir())
    before = read(run_dir, "events.jsonl")
    assert [e["task_id"] for e in before if e["type"] == "task_complete"] == ["adapt-0"]
    run_cli(PROTOCOL + ["--resume-dir", str(run_dir)], tmp_path, SolverClient(solved), monkeypatch)
    events = read(run_dir, "events.jsonl")
    attempts = [(e["task_id"], e["attempt"]) for e in events if e["type"] == "attempt_start"]
    assert attempts.count(("adapt-0", 1)) == 1 and ("adapt-1", 2) in attempts
    summary = read(run_dir, "summary.json")
    assert summary["infrastructure_failures"] == [{"task_id": "adapt-1", "attempt": 1,
                                                   "error": "attempt interrupted before task_complete; not continued"}]
    assert completes(run_dir)["adapt-1"]["result"]["real_actions"] == len(solved[crash])
    assert summary["status"] == "complete" and rescore(run_dir)["mismatches"] == []


def test_repeated_interruptions_do_not_exhaust_the_retry_budget(tmp_path, monkeypatch, solved):
    argv = PROTOCOL + ["--infra-retries", "1"]
    first = plan_observations(argv)[0]
    for resume in range(5):
        with pytest.raises(KeyboardInterrupt):
            extra = ["--resume-dir", str(next((tmp_path / "runs").iterdir()))] if resume else []
            run_cli(argv + extra, tmp_path, SolverClient(solved, crash_after=(first, 1)), monkeypatch)
    run_dir = next((tmp_path / "runs").iterdir())
    run_cli(argv + ["--resume-dir", str(run_dir)], tmp_path, SolverClient(solved), monkeypatch)
    assert read(run_dir, "summary.json")["status"] == "complete" and rescore(run_dir)["mismatches"] == []


def test_a_persistent_api_failure_is_retried_then_stops_the_history(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as stop:
        run_cli(PROTOCOL + ["--infra-retries", "1"], tmp_path, FailingClient(), monkeypatch)
    assert stop.value.code == 2
    summary = read(next((tmp_path / "runs").iterdir()), "summary.json")
    assert summary["status"] == "incomplete" and summary["results"] == []
    assert [(f["task_id"], f["attempt"]) for f in summary["infrastructure_failures"]] == [("adapt-0", 1), ("adapt-0", 2)]


def test_resume_refuses_a_changed_configuration(tmp_path, monkeypatch, solved):
    run_dir = run_cli(PROTOCOL, tmp_path, SolverClient(solved), monkeypatch)
    with pytest.raises(SystemExit, match="fingerprint"):
        run_cli(PROTOCOL + ["--memory", "reset", "--resume-dir", str(run_dir)], tmp_path, SolverClient(solved),
                monkeypatch)


def test_completed_tasks_are_never_replayed_on_resume(tmp_path, monkeypatch, solved):
    run_dir = run_cli(PROTOCOL, tmp_path, SolverClient(solved), monkeypatch)
    resumed = SolverClient(solved)
    run_cli(PROTOCOL + ["--resume-dir", str(run_dir)], tmp_path, resumed, monkeypatch)
    assert not resumed.requests and read(run_dir, "summary.json")["tasks_missing"] == 0


class FailOn(SolverClient):
    """Keeps one task unavailable while the rest of the scripted run proceeds."""

    def __init__(self, solved, observation):
        super().__init__(solved)
        self.observation = observation

    def create(self, **kwargs):
        if self.task(kwargs["messages"]) == self.observation:
            raise ConnectionError("provider unavailable")
        return super().create(**kwargs)


SKIP = PROTOCOL + ["--infra-retries", "0", "--infrastructure-policy", "skip"]


def test_continuation_policies_are_recorded_in_the_run_settings():
    recorded = settings(parse_args(build_parser(), SKIP))
    assert recorded["infrastructure_policy"] == "skip" and "interrupted_policy" not in recorded
    assert settings(parse_args(build_parser(), PROTOCOL))["infrastructure_policy"] == "stop"


def test_a_skipped_task_is_unscored_and_the_next_task_keeps_the_saved_notebook(tmp_path, monkeypatch, solved):
    with pytest.raises(SystemExit) as stop:
        run_cli(SKIP, tmp_path, FailOn(solved, plan_observations(SKIP)[1]), monkeypatch)
    assert stop.value.code == 2
    run_dir = next((tmp_path / "runs").iterdir())
    summary = read(run_dir, "summary.json")
    assert summary["status"] == "incomplete" and summary["tasks_missing"] == summary["missing_count"] == 1
    assert summary["tasks_completed"] == summary["tasks_planned"] - 1
    assert next(e for e in read(run_dir, "events.jsonl") if e["type"] == "task_missing")["task_id"] == "adapt-1"
    done = completes(run_dir)
    assert done["adapt-2"]["start_notebook"] == done["adapt-0"]["notebook"] != ""
    assert rescore(run_dir)["mismatches"] == []


@pytest.mark.parametrize("field,value", [("status", "complete"), ("tasks_missing", 0), ("missing", [])])
def test_tampered_missingness_is_detected(tmp_path, monkeypatch, solved, field, value):
    with pytest.raises(SystemExit):
        run_cli(SKIP, tmp_path, FailOn(solved, plan_observations(SKIP)[1]), monkeypatch)
    run_dir = next((tmp_path / "runs").iterdir())
    summary = read(run_dir, "summary.json")
    summary[field] = value
    (run_dir / "summary.json").write_text(json.dumps(summary))
    assert any(field in issue for issue in rescore(run_dir)["mismatches"])
    assert any(field in issue for issue in audit.check_logs(summary, read(run_dir, "manifest.json"),
                                                            audit.scan_events(run_dir)))


def test_an_interrupted_attempt_is_retried_on_resume_and_never_skipped(tmp_path, monkeypatch, solved):
    crash = plan_observations(PROTOCOL)[1]
    with pytest.raises(KeyboardInterrupt):
        run_cli(PROTOCOL, tmp_path, SolverClient(solved, crash_after=(crash, 2)), monkeypatch)
    run_dir = next((tmp_path / "runs").iterdir())
    run_cli(PROTOCOL + ["--resume-dir", str(run_dir)], tmp_path, SolverClient(solved), monkeypatch)
    events = read(run_dir, "events.jsonl")
    assert not any(e["type"] == "task_missing" for e in events)
    done = completes(run_dir)
    assert "adapt-1" in done and read(run_dir, "summary.json")["status"] == "complete"
    assert all(done[t]["start_notebook"] == done["adapt-3"]["notebook"]
               for t in ("heldout-easy-0", "heldout-medium-0", "heldout-hard-0"))
    with pytest.raises(SystemExit):
        parse_args(build_parser(), PROTOCOL + ["--interrupted-policy", "skip"])


def test_a_spend_stop_is_never_recorded_as_a_missing_task(tmp_path, monkeypatch, solved):
    with pytest.raises(SpendLimitExceeded):
        run_cli(SKIP, tmp_path, FailingClient(SpendLimitExceeded), monkeypatch)
    run_dir = next((tmp_path / "runs").iterdir())
    assert read(run_dir, "summary.json")["tasks_missing"] == 0
    assert not any(e["type"] == "task_missing" for e in read(run_dir, "events.jsonl"))


# Notebook writer failures ------------------------------------------------------------------------------------------


class WriterClient(SolverClient):
    def __init__(self, solved, failure=None):
        super().__init__(solved)
        self.failure = failure

    def create(self, **kwargs):
        if not kwargs.get("tools") and self.failure is not None:
            self.requests.append(kwargs)
            raise self.failure("writer unavailable")
        if not kwargs.get("tools"):
            self.requests.append(kwargs)
            return response("public note")
        return super().create(**kwargs)


@pytest.mark.parametrize(("failure", "writer_requests"), [(TimeoutError, 1), (ConnectionError, 2)])
def test_a_writer_network_failure_keeps_scored_outcomes_without_task_retries(tmp_path, monkeypatch, solved, failure,
                                                                             writer_requests):
    """A timeout may have been billed and is not repeated; a connection error is asked once more."""
    monkeypatch.setattr("world_adapt.agent.writer.time.sleep", lambda seconds: None)
    run_dir = run_cli(PROTOCOL, tmp_path, WriterClient(solved, failure), monkeypatch)
    summary = read(run_dir, "summary.json")
    assert summary["status"] == "complete" and summary["success_rate"] == 1 and not summary["infrastructure_failures"]
    events = read(run_dir, "events.jsonl")
    assert sum(e["type"] == "attempt_start" for e in events) == sum(e["type"] == "task_outcome" for e in events) == 7
    assert all(row["usage"]["writer_unknown_requests"] == writer_requests for row in summary["results"])
    assert all(not e["notebook"] for e in completes(run_dir).values())
    assert audit.check_logs(summary, read(run_dir, "manifest.json"), audit.scan_events(run_dir)) == []


def test_resume_after_a_writer_interrupt_keeps_the_outcome_and_the_previous_note(tmp_path, monkeypatch, solved):
    with pytest.raises(KeyboardInterrupt):
        run_cli(PROTOCOL, tmp_path, WriterClient(solved, KeyboardInterrupt), monkeypatch)
    [run_dir] = list((tmp_path / "runs").iterdir())
    [outcome] = [e for e in read(run_dir, "events.jsonl") if e["type"] == "task_outcome"]
    run_cli(PROTOCOL + ["--resume-dir", str(run_dir)], tmp_path, WriterClient(solved), monkeypatch)
    events = read(run_dir, "events.jsonl")
    assert sum(e["type"] == "attempt_start" and e["task_id"] == outcome["task_id"] for e in events) == 1
    done = completes(run_dir)[outcome["task_id"]]
    assert done["result"]["success"] and done["notebook"] == done["start_notebook"] == ""
    assert done["result"]["usage"]["writer_interrupted_updates"] == done["result"]["usage"]["writer_unknown_requests"] == 1
    assert read(run_dir, "summary.json")["status"] == "complete"


def test_an_abrupt_exit_during_the_writer_recovers_without_replaying_actions(tmp_path, monkeypatch, solved):
    script = '''
import os
from unittest.mock import patch
from tests.fakes import PROTOCOL, solutions
from tests.test_runner import WriterClient
from world_adapt.runner.run import main
class HardExit(WriterClient):
    def create(self, **kwargs):
        if not kwargs.get("tools"):
            os._exit(17)
        return super().create(**kwargs)
with patch("world_adapt.runner.run.make_client", return_value=HardExit(solutions(PROTOCOL))):
    main(PROTOCOL)
'''
    env = {k: v for k, v in os.environ.items() if k not in SPEND_VARIABLES}
    env["PYTHONPATH"] = str(ROOT)
    stopped = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env, capture_output=True, timeout=60)
    assert stopped.returncode == 17, stopped.stderr
    [run_dir] = list((tmp_path / "runs").iterdir())
    events = read(run_dir, "events.jsonl")
    assert sum(e["type"] == "writer_request_start" for e in events) == 1
    assert not any(e["type"] == "writer_request_end" for e in events)
    run_cli(PROTOCOL + ["--resume-dir", str(run_dir)], tmp_path, WriterClient(solved), monkeypatch)
    assert [e["attempt"] for e in read(run_dir, "events.jsonl")
            if e["type"] == "attempt_start" and e["task_id"] == "adapt-0"] == [1]
    done = completes(run_dir)["adapt-0"]
    assert done["result"]["success"] and done["notebook"] == "" and done["result"]["usage"]["writer_unknown_requests"] == 1


def test_a_writer_spend_stop_keeps_the_score_and_stops_before_the_next_task(tmp_path, monkeypatch, solved):
    with pytest.raises(SpendLimitExceeded):
        run_cli(PROTOCOL, tmp_path, WriterClient(solved, SpendLimitExceeded), monkeypatch)
    [run_dir] = list((tmp_path / "runs").iterdir())
    summary = read(run_dir, "summary.json")
    assert summary["status"] == "incomplete" and summary["tasks_completed"] == 1
    assert summary["results"][0]["success"] and summary["totals"]["writer_unknown_requests"] == 1


def test_a_writer_infrastructure_error_keeps_the_previous_notebook_and_never_fails_the_task(tmp_path, monkeypatch,
                                                                                            solved):
    argv = PROTOCOL + ["--infra-retries", "0"]

    class FailWriter(SolverClient):
        def create(self, **kwargs):
            if not kwargs.get("tools"):
                raise InfrastructureError("writer unavailable")
            return super().create(**kwargs)

    run_dir = run_cli(argv, tmp_path, FailWriter(solved), monkeypatch)
    assert read(run_dir, "summary.json")["status"] == "complete"
    assert rescore(run_dir)["mismatches"] == []


# Configuration -----------------------------------------------------------------------------------------------------


def test_agent_config_files_are_overridden_by_flags_and_unknown_keys_are_rejected(tmp_path):
    agent = tmp_path / "agent.json"
    agent.write_text(json.dumps({"model": "m-1", "reasoning_effort": "low", "notebook_tokens": 500}))
    args = parse_args(build_parser(), ["--agent-config", str(agent), "--reasoning-effort", "medium"])
    assert (args.model, args.reasoning_effort, args.notebook_tokens) == ("m-1", "medium", 500)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"prompt_variant": "v2"}))
    with pytest.raises(SystemExit):
        parse_args(build_parser(), ["--agent-config", str(bad)])


def test_the_example_agent_config_is_valid_for_every_environment():
    for env in registry.names():
        parser = build_parser()
        args = parse_args(parser, ["--agent-config", str(ROOT / "configs/agents/example.json"), "--env", env])
        check_args(parser, args)
        assert args.stream and args.require_usage


# The study template and scheduler -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def template():
    return study.load_study(TEMPLATE)


def test_the_template_enrolls_six_environments_four_conditions_and_fourteen_tasks(template):
    assert set(template["environments"]) == set(registry.names())
    for env, spec in template["environments"].items():
        assert spec["primary_profile"] in profiles(env) and spec["primary_profile"] != "identity"
    jobs = study.jobs(template, study.agents(template))
    assert len(jobs) == len({j.label for j in jobs}) == 24 and all(j.episodes == 14 for j in jobs)
    assert {j.condition for j in jobs} == set(study.CONDITIONS)
    assert template["sources"] == {} and template["launch_approved"] is False


def test_conditions_set_the_predictor_profile_and_notebook_policy(template):
    jobs = {j.condition: j for j in study.jobs(template, study.agents(template), envs=["patrol"])}
    assert (jobs["accurate-predictor-saved-notebook"].profile, jobs["accurate-predictor-saved-notebook"].memory) == \
        ("identity", "preserve")
    assert (jobs["faulty-predictor-reset-notebook"].profile, jobs["faulty-predictor-reset-notebook"].memory) == \
        ("adjacent_capture", "reset")
    assert jobs["faulty-predictor-saved-notebook"].label == \
        "six-environment-adaptation-example-patrol-faulty-predictor-saved-notebook-s3000"
    assert study.condition_name("faulty-predictor-saved-notebook") == "Faulty predictor + saved notebook"


def test_every_job_command_is_accepted_by_the_runner(template, tmp_path):
    agent = study.agents(template)[0]
    for job in study.jobs(template, [agent]):
        cmd = study.command(job, str(agent.config), tmp_path, "py")
        assert cmd[:3] == ["py", "-m", "world_adapt.runner.run"]
        parser = build_parser()
        args = parse_args(parser, cmd[3:])
        check_args(parser, args)
        assert (args.env, args.profile, args.memory, args.seed, args.label) == (
            job.env, job.profile, job.memory, job.seed, job.label)


@pytest.mark.parametrize("change,match", [
    (lambda s: s["conditions"].update({"no-predictor-saved-notebook": {"predictor": "identity", "memory": "preserve"}}), "unknown condition"),
    (lambda s: s["conditions"]["faulty-predictor-saved-notebook"].update(memory="reset"), "faulty-predictor-saved"),
    (lambda s: s["environments"]["patrol"].update(seed_blocks=[99]), "patrol"),
    (lambda s: s.pop("sources"), "sources"),
])
def test_inconsistent_manifests_are_rejected(tmp_path, change, match):
    value = json.loads(TEMPLATE.read_text())
    change(value)
    path = tmp_path / "study.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=match):
        study.load_study(path)


def test_run_folder_state_decides_new_resume_complete_or_broken(tmp_path):
    runs = tmp_path / "runs"
    for name, files in (("20261010-000000-p-a", {"manifest.json": "{}", "summary.json": '{"status": "complete"}'}),
                        ("20261010-000001-p-b", {"manifest.json": "{}"}), ("20261010-000002-p-c", {})):
        (runs / name).mkdir(parents=True)
        for file, text in files.items():
            (runs / name / file).write_text(text)
    assert [study.state(runs, f"p-{x}") for x in "abcd"] == ["complete", "resume", "broken", "new"]
    job = study.Job("p-b", "a", "patrol", "faulty-predictor-reset-notebook", 0, "adjacent_capture", "reset",
                    {"max_turns": 40})
    assert study.command(job, "agent.json", runs, "py")[-2:] == ["--resume-dir", str(runs / "20261010-000001-p-b")]


def test_source_drift_names_changed_pins():
    pins = study.pin_sources(("world_adapt/runner", "pyproject.toml"))
    assert study.source_drift({"sources": pins}) == {}
    tampered = {"sources": {**pins, "world_adapt/runner": "0" * 64}}
    assert set(study.source_drift(tampered)) == {"world_adapt/runner"}


def test_the_scheduler_runs_jobs_and_requeues_failures(tmp_path, monkeypatch):
    runs, logs = tmp_path / "runs", tmp_path / "logs"
    runs.mkdir()
    script = tmp_path / "fake_runner.py"
    script.write_text(
        "import json, sys, time, pathlib\n"
        "label = sys.argv[sys.argv.index('--label') + 1]\n"
        "runs = pathlib.Path(sys.argv[sys.argv.index('--runs') + 1])\n"
        "marker = runs / f'.{label}.tried'\n"
        "if 'flaky' in label and not marker.exists():\n"
        "    marker.write_text('1'); sys.exit(1)\n"
        "d = runs / f'{time.strftime(\"%Y%m%d-%H%M%S\")}-{label}'\n"
        "d.mkdir()\n"
        "(d / 'manifest.json').write_text('{}')\n"
        "(d / 'summary.json').write_text(json.dumps({'status': 'complete'}))\n")
    monkeypatch.setattr(study, "command", lambda job, agent, runs_, python: [
        python, str(script), "--label", job.label, "--runs", str(runs_)])
    jobs = [study.Job("t-steady", "a", "patrol", "faulty-predictor-saved-notebook", 0, "adjacent_capture", "preserve"),
            study.Job("t-flaky", "a", "patrol", "accurate-predictor-saved-notebook", 0, "identity", "preserve")]
    scheduler = study.Scheduler(jobs, {"a": study.Agent("a", tmp_path / "agent.json", "sail")}, runs=runs, logs=logs,
                                workers=2, max_attempts=3, poll_seconds=0.1, stagger_seconds=0)
    assert scheduler.run() == {"t-steady": "complete", "t-flaky": "complete"}
    assert scheduler.attempts == {"t-steady": 1, "t-flaky": 2}
    assert json.loads((logs / "status.json").read_text())["queued"] == []
    assert (logs / "t-flaky.log").read_text().count("=== attempt") == 2


def test_the_scheduler_persists_attempts_and_never_relaunches_exhausted_jobs(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "attempts.json").write_text(json.dumps({"bounded": 4}))
    agent = {"a": study.Agent("a", tmp_path / "agent.json", "sail")}
    exhausted = study.Job("bounded", "a", "patrol", "faulty-predictor-saved-notebook", 0, "adjacent_capture", "preserve")
    scheduler = study.Scheduler([exhausted], agent, runs=tmp_path / "runs", logs=logs, workers=1, max_attempts=4)
    monkeypatch.setattr(study.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("exhausted job was relaunched"))
    assert scheduler.run()["bounded"] != "complete"
    fresh = study.Job("fresh", "a", "patrol", "faulty-predictor-saved-notebook", 0, "adjacent_capture", "preserve")

    def spawn(*_a, **_kw):
        assert json.loads((logs / "attempts.json").read_text())["fresh"] == 1
        return object()

    monkeypatch.setattr(study.subprocess, "Popen", spawn)
    study.Scheduler([fresh], agent, runs=tmp_path / "runs", logs=logs, workers=1, max_attempts=4).launch(fresh)


def test_the_template_cannot_launch_but_can_be_planned(tmp_path, capsys):
    common = [str(TEMPLATE), "--runs-dir", str(tmp_path / "runs"), "--logs-dir", str(tmp_path / "logs")]
    assert study.main(common) == 2
    assert study.main(common + ["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "24 runs, 336 episodes" in out and out.count("world_adapt.runner.run") == 24
    assert "-faulty-predictor-saved-notebook-s3000" in out
