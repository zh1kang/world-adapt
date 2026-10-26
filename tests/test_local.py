"""Preparing, validating and launching a local study, offline preflight, and the launcher lock."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.fakes import ROOT
from world_adapt.agent import spending
from world_adapt.runner import local, manifest, preflight, study

ARM = "faulty-predictor-saved-notebook"


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    """A miniature repository root holding the real template and example agent config."""
    monkeypatch.setattr(local, "load_env", lambda: None)
    for relative in study.SOURCE_PINS + (".env.example", "world_adapt/runner/local.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "world_adapt":
            path.mkdir(exist_ok=True)
            (path / "module.py").write_text("# pinned\n")
        else:
            path.write_text("# pinned\n")
    for relative in ("configs/study.json", "configs/agents/example.json"):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, tmp_path / relative)
    monkeypatch.setattr(local, "ROOT", tmp_path)
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(manifest, "ROOT", tmp_path)
    monkeypatch.setattr(local, "TEMPLATE", tmp_path / "configs/study.json")
    monkeypatch.setattr(local, "AGENT_CONFIG", tmp_path / "configs/agents/example.json")
    return tmp_path


def prepare(repo: Path, *extra: str, provider: str = "sail", out: str = "outputs/example") -> Path:
    assert local.main(["prepare", "--provider", provider, "--model", "vendor/model", "--budget-usd", "2",
                       "--input-price", "1", "--output-price", "4", "--input-limit", "131072",
                       "--out", str(repo / out), *extra]) == 0
    return repo / out / "study.json"


def no_launch(monkeypatch):
    monkeypatch.setattr(local.study, "main", lambda _argv: pytest.fail("must reject before launch"))


# Preparing a study ---------------------------------------------------------------------------------------------------


def test_prepare_writes_a_source_pinned_single_agent_study(repo):
    path = prepare(repo, "--env", "patrol", "--seed", "3000")
    prepared = json.loads(path.read_text())
    config = json.loads((path.parent / "agent.json").read_text())
    assert prepared["launch_approved"] is True and list(prepared["environments"]) == ["patrol"]
    assert prepared["conditions"][ARM] == {"predictor": "perturbed", "memory": "preserve"}
    assert list(prepared["agents"]) == ["vendor-model"] and prepared["name"] == "local-example"
    assert (config["provider"], config["model"], config["reasoning_effort"]) == ("sail", "vendor/model", "high")
    assert prepared["budget_provenance"]["budget_usd"] == 2.0
    assert set(prepared["sources"]) == {*study.SOURCE_PINS, "configs/study.json", "outputs/example/agent.json"}
    assert prepared["source_snapshot"]["archive_sha256"]
    with pytest.raises(SystemExit):  # Never overwrite a prepared study.
        prepare(repo)


@pytest.mark.parametrize(("option", "value"), [("--budget-usd", "0"), ("--input-price", "nan"),
                                                  ("--output-price", "inf"), ("--input-limit", "0")])
def test_prepare_rejects_invalid_spending_inputs(repo, option, value):
    argv = ["prepare", "--provider", "sail", "--model", "x/y", "--budget-usd", "1", "--input-price", "1",
            "--output-price", "1", "--input-limit", "100000", "--out", str(repo / "out")]
    argv[argv.index(option) + 1] = value
    with pytest.raises(SystemExit):
        local.main(argv)


def test_prepare_rejects_an_output_directory_outside_the_repository(repo, tmp_path_factory):
    outside = tmp_path_factory.mktemp("elsewhere") / "study"
    with pytest.raises(SystemExit):
        local.main(["prepare", "--provider", "sail", "--model", "x/y", "--budget-usd", "1", "--input-price", "1",
                    "--output-price", "1", "--input-limit", "100000", "--out", str(outside)])


def test_prepare_rejects_an_input_limit_below_the_context_budgets(repo):
    with pytest.raises(SystemExit):
        local.main(["prepare", "--provider", "sail", "--model", "x/y", "--budget-usd", "1", "--input-price", "1",
                    "--output-price", "1", "--input-limit", "1000", "--out", str(repo / "out")])


@pytest.mark.parametrize(("field", "value"), [
    ("provider", "modal"), ("model", "another/model"), ("runs_dir", "elsewhere"), ("resume_dir", "elsewhere"),
    ("seed", 999), ("label", "unplanned"), ("max_context_tokens", 200000),
])
def test_prepare_rejects_template_overrides_that_escape_the_declared_run(repo, field, value):
    template = repo / "configs/study.json"
    data = json.loads(template.read_text())
    data["environments"]["patrol"]["runner_args"][field] = value
    template.write_text(json.dumps(data))
    with pytest.raises(SystemExit):
        prepare(repo, "--env", "patrol")


def test_the_source_snapshot_restores_the_prepared_inputs(repo):
    path = prepare(repo)
    restored = repo / "restored"
    with zipfile.ZipFile(path.parent / "source-snapshot.zip") as archive:
        archive.extractall(restored)
    receipt = json.loads((path.parent / "source-receipt.json").read_text())
    assert (restored / "world_adapt/module.py").is_file() and (restored / ".env.example").is_file()
    assert (restored / "configs/study.json").is_file() and "outputs/example/agent.json" in receipt["files"]
    assert receipt["study_sha256"]


def test_prepare_quick_check_flags_write_a_smaller_valid_study(repo):
    path = prepare(repo, "--env", "switchboard", "--condition", ARM, "--condition", "faulty-predictor-reset-notebook",
                   "--adaptation-tasks", "1", "--heldout-per-tier", "1")
    prepared = json.loads(path.read_text())
    assert list(prepared["conditions"]) == [ARM, "faulty-predictor-reset-notebook"]
    assert prepared["protocol"]["runner_args"] == {"adaptation_tasks": 1, "heldout_per_tier": 1, "infra_retries": 2}
    jobs = study.jobs(prepared, study.agents(prepared))
    assert len(jobs) == 2 and {job.episodes for job in jobs} == {4}
    full = json.loads(prepare(repo, out="outputs/full").read_text())
    assert list(full["conditions"]) == list(study.CONDITIONS)
    assert full["protocol"]["runner_args"] == {"adaptation_tasks": 8, "heldout_per_tier": 2, "infra_retries": 2}


@pytest.mark.parametrize("flags", [["--condition", "faulty-predictor"], ["--adaptation-tasks", "-1"],
                                   ["--adaptation-tasks", "many"], ["--heldout-per-tier", "0"],
                                   ["--heldout-per-tier", "1.5"]])
def test_prepare_rejects_invalid_quick_check_values(repo, flags):
    with pytest.raises(SystemExit):
        prepare(repo, *flags)
    assert not (repo / "outputs/example/study.json").exists()


def test_an_anthropic_study_is_prepared_and_needs_its_own_credential(repo, monkeypatch, capsys):
    path = prepare(repo, "--env", "patrol", provider="anthropic")
    prepared = json.loads(path.read_text())
    assert prepared["agents"]["vendor-model"]["backend"] == "anthropic"
    assert json.loads((path.parent / "agent.json").read_text())["provider"] == "anthropic"
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    no_launch(monkeypatch)
    with pytest.raises(SystemExit):
        local.main(["run", "--study", str(path)])
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


# Running a prepared study -------------------------------------------------------------------------------------------


def test_a_dry_run_delegates_to_the_scheduler_without_credentials(repo, monkeypatch):
    path = prepare(repo, provider="modal")
    seen = []
    monkeypatch.setattr(local.study, "main", lambda argv: seen.extend(argv) or 0)
    assert local.main(["run", "--study", str(path), "--dry-run"]) == 0
    assert "--dry-run" in seen and seen[seen.index("--max-attempts") + 1] == "3"
    assert seen[seen.index("--runs-dir") + 1] == str(path.parent / "runs")
    assert seen[seen.index("--logs-dir") + 1] == str(path.parent / "logs")


def test_max_attempts_is_exposed_on_run_and_must_be_positive(repo, monkeypatch):
    path = prepare(repo)
    seen = []
    monkeypatch.setattr(local.study, "main", lambda argv: seen.extend(argv) or 0)
    assert local.main(["run", "--study", str(path), "--dry-run", "--max-attempts", "5"]) == 0
    assert seen[seen.index("--max-attempts") + 1] == "5"
    with pytest.raises(SystemExit):
        local.main(["run", "--study", str(path), "--dry-run", "--max-attempts", "0"])


def paused_study(repo, monkeypatch) -> tuple[Path, list[str]]:
    """A study whose fake runner stops on a spend limit while STOP_AFTER_PLANNING_REQUESTS is set."""
    path = prepare(repo, "--env", "switchboard", "--condition", ARM, "--adaptation-tasks", "2",
                   "--heldout-per-tier", "1")
    script = repo / "fake_runner.py"
    script.write_text(textwrap.dedent('''
        import os, runpy, sys
        from unittest.mock import patch
        from tests.fakes import SolverClient, solutions
        from world_adapt.agent.spending import SpendLimitExceeded

        argv, stop_after = sys.argv[1:], int(os.environ.get("STOP_AFTER_PLANNING_REQUESTS", "0"))

        class Client(SolverClient):
            def create(self, **kwargs):
                planning = sum(1 for request in self.requests if request.get("tools"))
                if kwargs.get("tools") and stop_after and planning >= stop_after:
                    raise SpendLimitExceeded("test spend limit reached")
                return super().create(**kwargs)

        sys.argv = ["world_adapt.runner.run", *argv]
        with patch("world_adapt.runner.config.make_client", return_value=Client(solutions(argv), stream=True)):
            runpy.run_module("world_adapt.runner.run", run_name="__main__")
    '''))
    real_main, real_command, real_sleep = study.main, study.command, time.sleep

    def launch(argv):  # Replace only the child process, after the launcher's own validation has run.
        with monkeypatch.context() as patch:
            patch.setattr(study, "command",
                          lambda *a, **k: [sys.executable, str(script), *real_command(*a, **k)[3:]])
            return real_main(argv)

    monkeypatch.setattr(local.study, "main", launch)
    monkeypatch.setattr(study.time, "sleep", lambda seconds: real_sleep(min(seconds, 0.05)))
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.setenv("SAIL_API_KEY", "unused")
    run = ["run", "--study", str(path), "--max-attempts", "1"]
    return path, run


def test_rerunning_a_paused_study_resumes_unfinished_histories_without_repeating_completed_tasks(
        repo, monkeypatch):
    """A spend stop exits 75. It must not use up the attempt (here the only one), and the next `run` resumes."""
    path, run = paused_study(repo, monkeypatch)
    monkeypatch.setenv("STOP_AFTER_PLANNING_REQUESTS", "6")
    assert local.main(run) == 1
    [label] = json.loads((path.parent / "logs/attempts.json").read_text())
    assert json.loads((path.parent / "logs/attempts.json").read_text())[label] == 0
    [run_dir] = (path.parent / "runs").iterdir()
    paused = [e["task_id"] for e in read_events(run_dir) if e["type"] == "task_complete"]
    assert paused and len(paused) < 5

    monkeypatch.delenv("STOP_AFTER_PLANNING_REQUESTS")
    assert local.main(run) == 0
    assert local.main(run) == 0  # Idempotent: a complete study launches nothing.
    events = read_events(run_dir)
    assert sorted(e["task_id"] for e in events if e["type"] == "task_complete") == sorted(
        ["adapt-0", "adapt-1", "heldout-easy-0", "heldout-medium-0", "heldout-hard-0"])
    started = [e["task_id"] for e in events if e["type"] == "attempt_start"]
    assert all(started.count(task) == 1 for task in paused)
    assert json.loads((run_dir / "summary.json").read_text())["status"] == "complete"
    assert len(list((path.parent / "runs").iterdir())) == 1


def test_raising_the_budget_resumes_a_study_paused_by_its_cap(repo, monkeypatch):
    path, run = paused_study(repo, monkeypatch)
    ledger = path.parent / "spend-ledger.json"
    spending.raise_ledger_limit(ledger, 2.0)  # What the first child process would have created.
    monkeypatch.setenv("STOP_AFTER_PLANNING_REQUESTS", "6")
    assert local.main(run) == 1
    [run_dir] = (path.parent / "runs").iterdir()
    paused = [e["task_id"] for e in read_events(run_dir) if e["type"] == "task_complete"]
    assert paused and len(paused) < 5

    monkeypatch.delenv("STOP_AFTER_PLANNING_REQUESTS")
    with pytest.raises(SystemExit):  # A lower cap is refused.
        local.main([*run, "--budget-usd", "1"])
    assert local.main([*run, "--budget-usd", "5"]) == 0
    assert json.loads(ledger.read_text())["limit_usd"] == 5.0
    [entry] = json.loads((path.parent / "budget.json").read_text())["history"]
    assert (entry["from"], entry["to"]) == (2.0, 5.0) and entry["at"].endswith("Z")
    events = read_events(run_dir)
    assert sorted(e["task_id"] for e in events if e["type"] == "task_complete") == sorted(
        ["adapt-0", "adapt-1", "heldout-easy-0", "heldout-medium-0", "heldout-hard-0"])
    started = [e["task_id"] for e in events if e["type"] == "attempt_start"]
    assert all(started.count(task) == 1 for task in paused)


def test_budget_top_up_is_idempotent_never_lowers_and_keeps_the_pinned_study_intact(repo, monkeypatch):
    path = prepare(repo)
    pinned = path.read_bytes()
    monkeypatch.setattr(local.study, "main", lambda _argv: 0)
    monkeypatch.setenv("SAIL_API_KEY", "unused")
    run = ["run", "--study", str(path)]
    with pytest.raises(SystemExit):  # A dry run never changes state.
        local.main([*run, "--dry-run", "--budget-usd", "9"])
    assert local.main([*run, "--budget-usd", "2"]) == 0  # Equal to the current budget: no-op.
    assert not (path.parent / "budget.json").exists()
    assert local.main([*run, "--budget-usd", "4"]) == 0
    assert local.main([*run, "--budget-usd", "4"]) == 0
    assert local.main([*run, "--budget-usd", "6"]) == 0
    with pytest.raises(SystemExit):
        local.main([*run, "--budget-usd", "5"])
    with pytest.raises(SystemExit):
        local.main([*run, "--budget-usd", "nan"])
    history = json.loads((path.parent / "budget.json").read_text())["history"]
    assert [(h["from"], h["to"]) for h in history] == [(2.0, 4.0), (4.0, 6.0)]
    assert json.loads((path.parent / "spend-ledger.json").read_text())["limit_usd"] == 6.0
    assert path.read_bytes() == pinned  # Hashed study content is untouched; the top-up lives beside it.
    assert local.main(run) == 0  # Integrity checks pass with the raised cap.


def test_a_ledger_that_disagrees_with_the_recorded_budget_is_rejected(repo, monkeypatch):
    path = prepare(repo)
    spending.raise_ledger_limit(path.parent / "spend-ledger.json", 9.0)
    no_launch(monkeypatch)
    with pytest.raises(SystemExit):
        local.main(["run", "--study", str(path), "--dry-run"])


# Modal cold start ----------------------------------------------------------------------------------------------------


@pytest.fixture
def modal(repo, monkeypatch):
    """A prepared Modal study, a scripted /models endpoint, and a fake clock that sleeping advances."""
    monkeypatch.setenv("MODAL_BASE_URL", "https://example--vllm.modal.run/v1")
    monkeypatch.setenv("MODAL_API_KEY", "secret")
    monkeypatch.delenv("MODAL_PROXY_KEY", raising=False)
    monkeypatch.delenv("MODAL_PROXY_SECRET", raising=False)
    state = SimpleNamespace(now=0.0, polls=[], statuses=[], launched=[])
    monkeypatch.setattr(local, "sleep", lambda seconds: setattr(state, "now", state.now + seconds))
    monkeypatch.setattr(local, "monotonic", lambda: state.now)

    def fetch(url, headers):
        state.polls.append((url, headers))
        return state.statuses.pop(0) if len(state.statuses) > 1 else state.statuses[0]

    monkeypatch.setattr(local, "_fetch_status", fetch)
    monkeypatch.setattr(local.study, "main", lambda argv: state.launched.append(argv) or 0)
    state.path = prepare(repo, provider="modal")
    return state


def test_modal_run_waits_for_the_endpoint_before_launching(modal, monkeypatch, capsys):
    monkeypatch.setenv("MODAL_PROXY_KEY", "k")
    monkeypatch.setenv("MODAL_PROXY_SECRET", "s")
    modal.statuses = [503, 503, 200]
    assert local.main(["run", "--study", str(modal.path)]) == 0
    assert len(modal.polls) == 3 and modal.now == 20 and len(modal.launched) == 1
    url, headers = modal.polls[0]
    assert url == "https://example--vllm.modal.run/v1/models"
    assert headers == {"Authorization": "Bearer secret", "Modal-Key": "k", "Modal-Secret": "s"}
    assert "503" in capsys.readouterr().out


def test_modal_run_times_out_without_launching_anything(modal):
    modal.statuses = [503]
    with pytest.raises(SystemExit):
        local.main(["run", "--study", str(modal.path), "--endpoint-timeout", "35"])
    assert modal.launched == [] and 3 <= len(modal.polls) <= 5


def test_modal_wait_is_skipped_for_dry_runs_and_other_providers(modal, repo):
    modal.statuses = [503]
    assert local.main(["run", "--study", str(modal.path), "--dry-run"]) == 0
    os.environ["SAIL_API_KEY"] = "unused"
    try:
        sail = prepare(repo, out="outputs/sail")
        assert local.main(["run", "--study", str(sail)]) == 0
    finally:
        del os.environ["SAIL_API_KEY"]
    assert modal.polls == [] and len(modal.launched) == 2


def read_events(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]


def test_a_dry_run_lists_every_job_and_resumes_only_inside_the_prepared_runs(repo, capsys):
    path = prepare(repo)
    prepared = json.loads(path.read_text())
    job = study.jobs(prepared, study.agents(prepared))[0]
    resume_dir = path.parent / "runs" / f"20261023-000000-{job.label}"
    resume_dir.mkdir(parents=True)
    (resume_dir / "manifest.json").write_text("{}")
    assert local.main(["run", "--study", str(path), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "24 runs, 336 episodes" in output and output.count("world_adapt.runner.run") == 24
    assert f"--resume-dir {resume_dir}" in output


def test_billable_runs_require_provider_credentials(repo, monkeypatch):
    path = prepare(repo)
    monkeypatch.delenv("SAIL_API_KEY", raising=False)
    no_launch(monkeypatch)
    with pytest.raises(SystemExit):
        local.main(["run", "--study", str(path)])


def test_a_modal_run_rejects_an_unknown_endpoint(repo, monkeypatch):
    path = prepare(repo, provider="modal")
    monkeypatch.setenv("MODAL_API_KEY", "secret")
    monkeypatch.setenv("MODAL_BASE_URL", "https://example.invalid/v1")
    no_launch(monkeypatch)
    with pytest.raises(SystemExit):
        local.main(["run", "--study", str(path)])


@pytest.mark.parametrize("tamper", ["spend_limit", "agent_stream", "study_budget", "source"])
def test_a_tampered_study_is_rejected_before_launch(repo, monkeypatch, tamper):
    path = prepare(repo)
    prepared = json.loads(path.read_text())
    if tamper == "spend_limit":
        prepared["agents"]["vendor-model"]["env"]["WORLD_ADAPT_SPEND_LIMIT_USD"] = "999"
        path.write_text(json.dumps(prepared))
    elif tamper == "agent_stream":
        config = json.loads((path.parent / "agent.json").read_text())
        (path.parent / "agent.json").write_text(json.dumps({**config, "stream": False}))
    elif tamper == "study_budget":
        prepared["budgets"]["real_factor"] = 99
        path.write_text(json.dumps(prepared))
    else:
        (repo / "world_adapt/module.py").write_text("changed\n")
    no_launch(monkeypatch)
    with pytest.raises(SystemExit):
        local.main(["run", "--study", str(path), "--dry-run"])


# Offline preflight ----------------------------------------------------------------------------------------------------


def job(env: str, profile: str, seed: int = 0, **args) -> study.Job:
    protocol = {"adaptation_tasks": 2, "heldout_per_tier": 1, "max_turns": 100, "max_chunk": 8, **args}
    return study.Job("offline", "test", env, ARM if profile != "identity" else "accurate-predictor-saved-notebook",
                     seed, profile, "preserve", protocol)


AGENT = study.Agent("test", ROOT / "configs/agents/example.json", "sail")


def test_preflight_verifies_witnesses_and_identity_predictions_without_model_calls(monkeypatch):
    import world_adapt.runner.config as config
    monkeypatch.setattr(config, "make_client", lambda *a, **k: pytest.fail("preflight must not create a client"))
    report = preflight.check_job(job("switchboard", "identity"), AGENT)
    assert report["model_calls"] == 0 and report["writer"] == "freeform-v1"
    assert all(len(t["state_sha256"]) == 64 and t["witness_final_action_success"] for t in report["tasks"])
    assert not any(t["visible_changed_predictions"] for t in report["tasks"])


def test_preflight_shows_the_faulty_profile_changes_valid_predictions():
    report = preflight.check_job(job("depot", "cargo_and_stacks", 3000), AGENT)
    assert len(report["tasks"]) == 5 and all(t["reference_verified"] for t in report["tasks"])
    assert sum(t["changed_predicted_steps"] for t in report["tasks"]) > 0
    assert all(t["rejected_steps"] == 0 for t in report["tasks"])
    fingerprints = [t["task_fingerprint"] for t in report["tasks"]]
    assert len(set(fingerprints)) == 5


def test_preflight_native_start_identity_ignores_derived_coverage():
    state = {"agent": (1.0, 2.0), "block": (3.0, 4.0, 0.0), "goal": (5.0, 6.0, 0.0), "coverage": 0.1,
             "status": "active"}
    game = SimpleNamespace(name="pusht", physical=lambda _: state)
    first = preflight.state_hash(game, None)
    state["coverage"] += 1e-16
    assert preflight.state_hash(game, None) == first
    state["agent"] = (1.0, 2.00000000000001)
    assert preflight.state_hash(game, None) != first


def test_preflight_rejects_structural_overlap_but_allows_paired_arms():
    task = {"task_id": "adapt-0", "phase": "adaptation", "task_fingerprint": "a" * 64}
    first = {"environment": "depot", "seed": 3000, "tasks": [task]}
    preflight.check_structural_tasks([first, {**first, "condition": "other-arm"}])
    with pytest.raises(ValueError, match="structurally equivalent"):
        preflight.check_structural_tasks([first, {**first, "seed": 3001}])
    with pytest.raises(ValueError, match="structurally equivalent"):
        preflight.check_structural_tasks([{**first, "tasks": [task, {**task, "task_id": "heldout", "phase": "heldout"}]}])
    with pytest.raises(ValueError, match="missing.*fingerprint"):
        preflight.check_structural_tasks([{**first, "tasks": [{"task_id": "x", "phase": "adaptation"}]}])
    with pytest.raises(ValueError, match="paired.*inventory"):
        preflight.check_structural_tasks([first, {**first, "tasks": [{**task, "task_fingerprint": "b" * 64}]}])


# The launcher lock ----------------------------------------------------------------------------------------------------


def test_scheduler_owner_lock_rejects_second_launcher_and_cleans_up(tmp_path, monkeypatch):
    runs, logs = tmp_path / "runs", tmp_path / "logs"
    runs.mkdir()
    script = tmp_path / "slow_runner.py"
    marker = tmp_path / "started"
    script.write_text(
        "import json, pathlib, sys, time\n"
        f"marker = pathlib.Path({str(marker)!r})\n"
        "label = sys.argv[sys.argv.index('--label') + 1]\n"
        "runs = pathlib.Path(sys.argv[sys.argv.index('--runs') + 1])\n"
        "marker.write_text('started')\n"
        "time.sleep(0.5)\n"
        "stamp = time.strftime('%Y%m%d-%H%M%S')\n"
        "d = runs / f'{stamp}-{label}'\n"
        "d.mkdir()\n"
        "(d / 'manifest.json').write_text('{}')\n"
        "(d / 'summary.json').write_text(json.dumps({'status': 'complete'}))\n")
    monkeypatch.setattr(local.study, "command", lambda job, _agent, runs_, python: [
        python, str(script), "--label", job.label, "--runs", str(runs_)])
    job = local.study.Job("slow", "a", "patrol", ARM, 0, "identity", "preserve")
    agent = local.study.Agent("a", tmp_path / "agent.json", "sail")
    lock_path = tmp_path / ".local-run.lock"
    first = local.study.Scheduler([job], {"a": agent}, runs=runs, logs=logs, workers=1,
                                  max_attempts=1, poll_seconds=0.05,
                                  stagger_seconds=0, owner_lock=lock_path)
    error: list[BaseException] = []

    def run_first():
        try:
            first.run()
        except BaseException as exc:  # pragma: no cover - failure is asserted below
            error.append(exc)

    thread = threading.Thread(target=run_first)
    thread.start()
    deadline = time.time() + 3
    while not marker.exists() and time.time() < deadline:
        time.sleep(0.02)
    assert marker.exists()
    second = local.study.Scheduler([job], {"a": agent}, runs=runs, logs=tmp_path / "logs-2", workers=1,
                                   max_attempts=1, poll_seconds=0.05,
                                   stagger_seconds=0, owner_lock=lock_path)
    with pytest.raises(ValueError, match="owns"):
        second.run()
    thread.join(timeout=3)
    assert not thread.is_alive() and not error


def test_scheduler_exception_terminates_children_before_releasing_lock(tmp_path, monkeypatch):
    runs, logs = tmp_path / "runs", tmp_path / "logs"
    runs.mkdir()
    script = tmp_path / "failing_parent_runner.py"
    pid_path = tmp_path / "child.pid"
    script.write_text(
        "import os, pathlib, sys, time\n"
        f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(10)\n")
    monkeypatch.setattr(local.study, "command", lambda job, _agent, runs_, python: [
        python, str(script), "--label", job.label, "--runs", str(runs_)])
    scheduler = local.study.Scheduler(
        [local.study.Job("failing", "a", "patrol", ARM, 0, "identity", "preserve")],
        {"a": local.study.Agent("a", tmp_path / "agent.json", "sail")}, runs=runs, logs=logs,
        workers=1, max_attempts=1, poll_seconds=0.01, stagger_seconds=0,
        owner_lock=tmp_path / ".local-run.lock")
    def fail_after_spawn():
        deadline = time.time() + 2
        while not pid_path.exists() and time.time() < deadline:
            time.sleep(0.01)
        raise RuntimeError("test failure")

    monkeypatch.setattr(scheduler, "reap", fail_after_spawn)
    with pytest.raises(RuntimeError, match="test failure"):
        scheduler.run()
    assert pid_path.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_path.read_text()), 0)
    # The lock is available again after the child has been reaped.
    followup = local.study.Scheduler([], {}, runs=runs, logs=tmp_path / "logs-2", workers=1,
                                     max_attempts=1, owner_lock=tmp_path / ".local-run.lock")
    assert followup.run() == {}


def test_scheduler_lock_survives_killed_launcher_until_worker_exits(tmp_path):
    lock_path = tmp_path / ".local-run.lock"
    pid_path = tmp_path / "child.pid"
    child = tmp_path / "child.py"
    child.write_text(
        "import pathlib, os, time\n"
        f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(30)\n")
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "import pathlib, sys\n"
        "from world_adapt.runner import study\n"
        f"child = {str(child)!r}\n"
        "study.command = lambda job, agent, runs, python: [python, child]\n"
        f"job = study.Job('orphan', 'a', 'patrol', {ARM!r}, 0, 'identity', 'preserve')\n"
        "agent = study.Agent('a', pathlib.Path('agent.json'), 'sail')\n"
        f"study.Scheduler([job], {{'a': agent}}, runs=pathlib.Path({str(tmp_path / 'runs')!r}), "
        f"logs=pathlib.Path({str(tmp_path / 'logs')!r}), workers=1, max_attempts=1, "
        f"poll_seconds=0.05, stagger_seconds=0, owner_lock=pathlib.Path({str(lock_path)!r})).run()\n")
    parent = subprocess.Popen([sys.executable, str(launcher)], cwd=local.ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.time() + 5
        while not pid_path.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert pid_path.exists()
        parent.kill()
        parent.wait(timeout=5)
        probe = tmp_path / "probe.py"
        probe.write_text(
            "import pathlib\n"
            "from world_adapt.runner import study\n"
            f"study.Scheduler([], {{}}, runs=pathlib.Path({str(tmp_path / 'runs')!r}), "
            f"logs=pathlib.Path({str(tmp_path / 'probe-logs')!r}), workers=1, max_attempts=1, "
            f"owner_lock=pathlib.Path({str(lock_path)!r})).run()\n")
        blocked = subprocess.run([sys.executable, str(probe)], cwd=local.ROOT,
                                 text=True, capture_output=True)
        assert blocked.returncode != 0
        assert "owns" in blocked.stderr
    finally:
        if pid_path.exists():
            os.kill(int(pid_path.read_text()), signal.SIGTERM)
        try:
            parent.wait(timeout=2)
        except subprocess.TimeoutExpired:
            parent.kill()
    released = subprocess.run([sys.executable, str(probe)], cwd=local.ROOT,
                              text=True, capture_output=True)
    assert released.returncode == 0


def test_scheduler_reloads_attempts_after_owner_lock_is_acquired(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    lock_path = tmp_path / ".local-run.lock"
    (logs / "attempts.json").write_text(json.dumps({"bounded": 1}))
    owner = local.study.Scheduler([], {}, runs=tmp_path / "runs", logs=logs, workers=1,
                                  max_attempts=4, owner_lock=lock_path)
    owner._acquire_owner()
    try:
        (logs / "attempts.json").write_text(json.dumps({"bounded": 4}))
        contender = local.study.Scheduler(
            [local.study.Job("bounded", "a", "patrol", ARM, 0, "identity", "preserve")],
            {"a": local.study.Agent("a", tmp_path / "agent.json", "sail")}, runs=tmp_path / "runs",
            logs=logs, workers=1, max_attempts=4, owner_lock=lock_path)
        assert contender.attempts == {}
    finally:
        owner._release_owner()
    assert contender.run() == {"bounded": "failed after 4 cumulative attempts"}
    assert contender.attempts == {"bounded": 4}
