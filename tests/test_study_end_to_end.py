"""Every environment through the real runner with the study template, offline, and the protocol it fixes.

A scripted client streams each task's evaluator witness, so these runs check the harness, not model ability.
`tests/golden/study.json` pins the task pools, plans, budgets, system prompts and tool schemas of seed block 3000;
a difference means runs are no longer comparable with runs made before the change.
"""

import hashlib
import json
from pathlib import Path

import pytest

from tests.fakes import SolverClient, read, run_cli, solutions, study_job_argv
from world_adapt.runner.rescore import rescore

GOLDEN = json.loads((Path(__file__).parent / "golden" / "study.json").read_text())
ARMS = ("faulty-predictor-saved-notebook", "accurate-predictor-reset-notebook")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("condition", ARMS)
@pytest.mark.parametrize("env", sorted(GOLDEN))
def test_the_study_protocol_runs_end_to_end_and_matches_the_recorded_protocol(env, condition, tmp_path, monkeypatch):
    argv = study_job_argv(env, condition, tmp_path / "runs")
    client = SolverClient(solutions(argv), chunk=8, stream=True, predict=True)
    monkeypatch.setenv("MODAL_BASE_URL", "https://offline-check.modal.run")
    run_dir = run_cli(argv, tmp_path, client, monkeypatch)
    summary = read(run_dir, "summary.json")
    assert summary["status"] == "complete" and summary["tasks_completed"] == 14
    assert all(row["success"] for row in summary["results"]) and not summary["infrastructure_failures"]
    assert rescore(run_dir)["mismatches"] == []
    assert all(stream.closed for stream in client.streams)

    events = read(run_dir, "events.jsonl")
    done = [e for e in events if e["type"] == "task_complete"]
    adaptation = [e for e in done if e["phase"] == "adaptation"]
    heldout = [e for e in done if e["phase"] == "heldout"]
    assert len(adaptation) == 8 and len(heldout) == 6
    if condition.endswith("saved-notebook"):
        assert all(after["start_notebook"] == before["notebook"] for before, after in zip(adaptation, adaptation[1:]))
        assert all(e["start_notebook"] == adaptation[-1]["notebook"] != "" for e in heldout)
    else:
        assert all(e["start_notebook"] == "" for e in done)
    assert sum(e["type"] == "task_outcome" for e in events) == sum(e["type"] == "notebook_input" for e in events) == 14
    assert "hidden reasoning" not in (run_dir / "events.jsonl").read_text()

    manifest = read(run_dir, "manifest.json")
    starts = [e for e in events if e["type"] == "task_start"]
    recorded = {"task_observations_sha256": digest(manifest["tasks"]["observation_sha256"]),
                "task_plan_sha256": digest(manifest["tasks"]["plan"]),
                "budgets_sha256": digest(manifest["budgets"]["per_task"]),
                "system_prompts_sha256": digest([e["messages"][0]["content"] for e in starts]),
                "tools_sha256": digest(starts[0]["tools"])}
    assert recorded == GOLDEN[env]
