"""Launch or resume a prepared study: every (agent, environment, condition, seed block) run, in parallel.

Users start studies with `python -m world_adapt.runner.local run`, which validates the prepared study and calls
`main` here. A study manifest pins the environments, their faulty predictor profiles, the conditions, the seed
blocks, the protocol parameters, the per-run budgets, the agents (config file, backend, environment overrides) and
the source hashes of the code the results depend on. Each job is one `python -m world_adapt.runner.run` process
whose run folder is `runs/<timestamp>-<label>` with `<label> = <study>-<agent>-<env>-<condition>-s<seed>`.
Re-running the same command is safe: complete runs are skipped, interrupted runs are resumed with `--resume-dir`,
and a job that exits non-zero is re-queued up to `--max-attempts` times. A job that pauses on the spend limit
(exit 75) does not use up an attempt. Use one launcher per output directory; its
worker count can be changed while a launcher runs by writing a number to `<logs>/workers`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from world_adapt.runner.manifest import ROOT, source_hash
from world_adapt.runner.records import write_json

@dataclass(frozen=True)
class Condition:
    name: str
    """Human-readable label used in tables and reports."""
    predictor: str
    """'identity' (accurate predictor) or 'perturbed' (the environment's faulty profile)."""
    memory: str
    """'preserve' (saved notebook) or 'reset' (every task starts with an empty notebook)."""


CONDITIONS = {
    "accurate-predictor-saved-notebook": Condition("Accurate predictor + saved notebook", "identity", "preserve"),
    "accurate-predictor-reset-notebook": Condition("Accurate predictor + reset notebook", "identity", "reset"),
    "faulty-predictor-saved-notebook": Condition("Faulty predictor + saved notebook", "perturbed", "preserve"),
    "faulty-predictor-reset-notebook": Condition("Faulty predictor + reset notebook", "perturbed", "reset"),
}
"""Condition slug (used in manifests, run folder names and exports) -> condition."""


def condition_name(slug: str) -> str:
    """The human-readable label of a condition slug."""
    return CONDITIONS[slug].name


SOURCE_PINS = ("world_adapt", "world_adapt/environments/depot/domain.pddl", "README.md", "pyproject.toml", "uv.lock")
"""Sources every prepared study pins in addition to its template and agent config."""


@dataclass(frozen=True)
class Agent:
    name: str
    config: Path
    backend: str
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables for the runner process; `${NAME}` values are read from the launcher's environment."""

    def resolved_env(self, environ: dict[str, str] | None = None) -> dict[str, str]:
        environ = os.environ if environ is None else environ
        out = {}
        for key, value in self.env.items():
            if value.startswith("${") and value.endswith("}"):
                variable = value[2:-1]
                value = environ.get(variable, "")
                if not value:
                    raise ValueError(f"{self.name}: required environment variable {variable} is not set")
            out[key] = value
        return out


@dataclass(frozen=True)
class Job:
    label: str
    agent: str
    env: str
    condition: str
    seed: int
    profile: str
    memory: str
    args: dict[str, Any] = field(default_factory=dict)
    """Runner flags (dest names) beyond env/profile/memory/seed."""

    @property
    def episodes(self) -> int:
        return int(self.args["adaptation_tasks"]) + 3 * int(self.args["heldout_per_tier"])


def load_study(path: Path) -> dict[str, Any]:
    study = json.loads(Path(path).read_text())
    for key in ("name", "protocol", "conditions", "environments", "budgets", "agents", "sources"):
        if key not in study:
            raise ValueError(f"study manifest {path} lacks '{key}'")
    if "launch_approved" in study and type(study["launch_approved"]) is not bool:
        raise ValueError("launch_approved must be a boolean")
    for slug, cond in study["conditions"].items():
        if slug not in CONDITIONS:
            raise ValueError(f"unknown condition {slug!r}; use one of {', '.join(CONDITIONS)}")
        c = CONDITIONS[slug]
        if (cond.get("predictor"), cond.get("memory")) != (c.predictor, c.memory):
            raise ValueError(f"condition {slug} must be predictor={c.predictor}, memory={c.memory}")
    for env, spec in study["environments"].items():
        if "primary_profile" not in spec:
            raise ValueError(f"environment {env} lacks 'primary_profile'")
        extra = set(spec.get("seed_blocks", [])) - set(study["protocol"]["seed_blocks"])
        if extra:
            raise ValueError(f"environment {env} has seed blocks {sorted(extra)} outside the study's seed_blocks")
    for name, spec in study["agents"].items():
        if "config" not in spec or "backend" not in spec:
            raise ValueError(f"agent {name} needs 'config' and 'backend'")
    return study


def agents(study: dict[str, Any], names: list[str] | None = None) -> list[Agent]:
    out = []
    for name, spec in study["agents"].items():
        if names is not None and name not in names:
            continue
        out.append(Agent(name, ROOT / spec["config"], spec["backend"], dict(spec.get("env", {}))))
    return out


def runner_args(study: dict[str, Any], env: str) -> dict[str, Any]:
    """Protocol parameters and budgets for one environment: study-wide defaults, then the environment's overrides."""
    spec = study["environments"][env]
    return {**study["protocol"].get("runner_args", {}), **study["budgets"], **spec.get("runner_args", {})}


def label(study: dict[str, Any], agent: str, env: str, condition: str, seed: int) -> str:
    return f"{study['name']}-{agent}-{env}-{condition}-s{seed}"


def seed_blocks(study: dict[str, Any], env: str) -> list[int]:
    """The environment's seed blocks: its own `seed_blocks`, else the study's."""
    return list(study["environments"][env].get("seed_blocks", study["protocol"]["seed_blocks"]))


def jobs(study: dict[str, Any], selected: list[Agent], envs: list[str] | None = None,
         conditions: list[str] | None = None, seeds: list[int] | None = None) -> list[Job]:
    """All runs for the selected agents, interleaved across agents."""
    unknown = sorted(set(conditions or ()) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"unknown condition(s) {', '.join(unknown)}; use one of {', '.join(CONDITIONS)}")
    out = []
    for seed in study["protocol"]["seed_blocks"]:
        if seeds is not None and seed not in seeds:
            continue
        for env, spec in study["environments"].items():
            if envs is not None and env not in envs:
                continue
            if seed not in seed_blocks(study, env):
                continue
            for slug in study["conditions"]:
                if conditions is not None and slug not in conditions:
                    continue
                c = CONDITIONS[slug]
                profile = spec["primary_profile"] if c.predictor == "perturbed" else "identity"
                for agent in selected:
                    out.append(Job(label(study, agent.name, env, slug, seed), agent.name, env, slug, seed,
                                   profile, c.memory, runner_args(study, env)))
    return out


def planned_episodes(study: dict[str, Any]) -> dict[str, int]:
    """Episodes per environment and in total for one agent."""
    one = [Agent("x", ROOT, "x")]
    counts = {env: sum(j.episodes for j in jobs(study, one) if j.env == env) for env in study["environments"]}
    return {**counts, "total": sum(counts.values())}


def plan_table(selected_jobs: list[Job]) -> list[str]:
    """Dry-run plan: runs and episodes per agent and environment."""
    cells: dict[tuple[str, str], list[Job]] = {}
    for job in selected_jobs:
        cells.setdefault((job.agent, job.env), []).append(job)
    lines = [f"{'agent':<24} {'environment':<12} {'conditions':>10} {'seeds':>5} {'runs':>5} {'episodes':>8}"]
    for (agent, env), group in cells.items():
        lines.append(f"{agent:<24} {env:<12} {len({j.condition for j in group}):>10} {len({j.seed for j in group}):>5} "
                     f"{len(group):>5} {sum(j.episodes for j in group):>8}")
    lines.append(f"{'total':<24} {'':<12} {'':>10} {'':>5} {len(selected_jobs):>5} "
                 f"{sum(j.episodes for j in selected_jobs):>8}")
    return lines


def source_drift(study: dict[str, Any]) -> dict[str, dict[str, str | None]]:
    """Pinned source hashes that no longer match the working tree."""
    drift = {}
    for rel, pinned in study["sources"].items():
        current = source_hash(rel) if (ROOT / rel).exists() else None
        if current != pinned:
            drift[rel] = {"pinned": pinned, "current": current}
    return drift


def pin_sources(paths: tuple[str, ...]) -> dict[str, str]:
    return {rel: source_hash(rel) for rel in paths}


def run_dir(runs: Path, label: str) -> Path | None:
    found = sorted(runs.glob(f"*-{label}")) if runs.exists() else []
    return found[-1] if found else None


def state(runs: Path, label: str) -> str:
    """new | complete | resume (interrupted with a manifest) | broken (folder without a manifest)."""
    d = run_dir(runs, label)
    if d is None:
        return "new"
    summary = d / "summary.json"
    if summary.exists():
        try:
            if json.loads(summary.read_text()).get("status") == "complete":
                return "complete"
        except json.JSONDecodeError:
            pass
    return "resume" if (d / "manifest.json").exists() else "broken"


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def command(job: Job, agent_config: str, runs: Path, python: str = sys.executable) -> list[str]:
    cmd = [python, "-m", "world_adapt.runner.run", "--agent-config", agent_config, "--env", job.env,
           "--profile", job.profile, "--memory", job.memory, "--seed", str(job.seed), "--label", job.label,
           "--runs-dir", str(runs.resolve())]
    for key, value in sorted(job.args.items()):
        if isinstance(value, bool):
            if value:
                cmd.append(_flag(key))
        elif isinstance(value, list):
            for item in value:
                cmd += [_flag(key), str(item)]
        else:
            cmd += [_flag(key), str(value)]
    if state(runs, job.label) == "resume":
        cmd += ["--resume-dir", str(run_dir(runs, job.label).resolve())]
    return cmd


class Scheduler:
    def __init__(self, queue: list[Job], agents: dict[str, Agent], *, runs: Path, logs: Path, workers: int,
                 max_attempts: int, python: str = sys.executable, poll_seconds: float = 15.0,
                 stagger_seconds: float = 2.0, notes: dict[str, Any] | None = None,
                 owner_lock: Path | None = None):
        self.queue, self.agents, self.runs, self.logs = list(queue), agents, runs, logs
        self.default_workers, self.max_attempts, self.python = workers, max_attempts, python
        self.poll_seconds, self.notes = poll_seconds, notes or {}
        self.stagger_seconds = stagger_seconds
        self.owner_lock = owner_lock
        self._owner_stream = None
        self._log_handles: dict[str, Any] = {}
        self.attempts: dict[str, int] = {}
        if self.owner_lock is None:
            self._load_attempts()
        self.running: dict[str, tuple[subprocess.Popen, Job, float]] = {}
        self.results: dict[str, str] = {}

    def workers(self) -> int:
        try:
            return int((self.logs / "workers").read_text().strip())
        except (OSError, ValueError):
            return self.default_workers

    def write_status(self) -> None:
        write_json(self.logs / "status.json", {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "queued": [j.label for j in self.queue],
            "running": sorted(self.running), "results": self.results, "attempts": self.attempts, **self.notes})

    def launch(self, job: Job) -> None:
        agent = self.agents[job.agent]
        self.attempts[job.label] = self.attempts.get(job.label, 0) + 1
        self.logs.mkdir(parents=True, exist_ok=True)
        write_json(self.logs / "attempts.json", self.attempts)
        cmd = command(job, str(agent.config), self.runs, self.python)
        log = (self.logs / f"{job.label}.log").open("a")
        log.write(f"\n=== attempt {self.attempts[job.label]} {time.strftime('%H:%M:%S')}: {' '.join(cmd)}\n")
        log.flush()
        try:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, **agent.resolved_env()},
                                    pass_fds=((self._owner_stream.fileno(),) if self._owner_stream else ()))
        except BaseException:
            log.close()
            raise
        self._log_handles[job.label] = log
        self.running[job.label] = (proc, job, time.time())
        print(f"[{time.strftime('%H:%M:%S')}] start {job.label} (attempt {self.attempts[job.label]})", flush=True)

    def ledger(self, job: Job) -> str | None:
        """The spend ledger the job's agent charges, if any; jobs on different ledgers cannot exhaust each other."""
        return self.agents[job.agent].env.get("WORLD_ADAPT_SPEND_LEDGER")

    def reap(self) -> None:
        for label in list(self.running):
            proc, job, started = self.running[label]
            code = proc.poll()
            if code is None:
                continue
            del self.running[label]
            log = self._log_handles.pop(label, None)
            if log is not None:
                log.close()
            minutes = (time.time() - started) / 60
            if code == 0 and state(self.runs, label) == "complete":
                self.results[label] = "complete"
                print(f"[{time.strftime('%H:%M:%S')}] done  {label} in {minutes:.0f} min", flush=True)
            elif code == 75:
                # A pause is not a failure of the job: the next launch resumes it with its full attempt allowance.
                self.attempts[label] -= 1
                write_json(self.logs / "attempts.json", self.attempts)
                self.results[label] = "paused: inference spend limit or unknown usage"
                ledger = self.ledger(job)
                paused = [j for j in self.queue if ledger is not None and self.ledger(j) == ledger]
                for queued in paused:
                    self.results[queued.label] = "paused: shared spend ledger limit or unknown usage"
                self.queue = [j for j in self.queue if j not in paused]
                print(f"[{time.strftime('%H:%M:%S')}] pause {job.label}: spend guard; {len(paused)} queued jobs "
                      f"sharing its ledger preserved", flush=True)
            elif self.attempts[label] < self.max_attempts:
                print(f"[{time.strftime('%H:%M:%S')}] FAIL  {label} exit {code} after {minutes:.0f} min; re-queued", flush=True)
                self.queue.append(job)
            else:
                self.results[label] = f"failed after {self.attempts[label]} attempts (exit {code})"
                print(f"[{time.strftime('%H:%M:%S')}] GIVE UP {label} exit {code}", flush=True)

    def _acquire_owner(self) -> None:
        if self.owner_lock is None:
            return
        import fcntl

        self.owner_lock.parent.mkdir(parents=True, exist_ok=True)
        stream = self.owner_lock.open("a+")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            stream.close()
            raise ValueError(f"another study launcher owns {self.owner_lock}") from error
        self._owner_stream = stream
        try:
            self._load_attempts()
        except BaseException:
            stream.close()
            self._owner_stream = None
            raise

    def _load_attempts(self) -> None:
        ledger = self.logs / "attempts.json"
        if not ledger.exists():
            return
        saved = json.loads(ledger.read_text())
        if not isinstance(saved, dict) or any(not isinstance(key, str) or isinstance(value, bool)
                or not isinstance(value, int) or value < 0 for key, value in saved.items()):
            raise ValueError(f"invalid scheduler attempt ledger: {ledger}")
        self.attempts = saved

    def _release_owner(self) -> None:
        if self._owner_stream is None:
            return
        self._owner_stream.close()
        self._owner_stream = None

    def _cleanup_children(self) -> None:
        """Stop children on launcher failure before releasing the inherited owner lock."""
        live = [proc for proc, _job, _started in self.running.values() if proc.poll() is None]
        for proc in live:
            proc.terminate()
        deadline = time.time() + 5.0
        for proc in live:
            remaining = max(0.0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for log in self._log_handles.values():
            log.close()
        self._log_handles.clear()
        self.running.clear()

    def run(self) -> dict[str, str]:
        self.logs.mkdir(parents=True, exist_ok=True)
        self._acquire_owner()
        try:
            while self.queue or self.running:
                i = 0
                while i < len(self.queue) and len(self.running) < self.workers():
                    job = self.queue[i]
                    current = state(self.runs, job.label)
                    if current == "complete":
                        self.queue.pop(i)
                        self.results[job.label] = "complete"
                        continue
                    if self.attempts.get(job.label, 0) >= self.max_attempts:
                        self.queue.pop(i)
                        self.results[job.label] = f"failed after {self.attempts[job.label]} cumulative attempts"
                        continue
                    if current == "broken":
                        self.queue.pop(i)
                        self.results[job.label] = "broken run folder (no manifest); move it aside to rerun"
                        continue
                    self.queue.pop(i)
                    self.launch(job)
                    time.sleep(self.stagger_seconds)  # stagger run-folder timestamps and API bursts
                self.reap()
                self.write_status()
                if self.queue or self.running:
                    time.sleep(self.poll_seconds)
            bad = {k: v for k, v in self.results.items() if v != "complete"}
            print(f"finished: {len(self.results) - len(bad)} complete, {len(bad)} not complete: {bad}", flush=True)
            return self.results
        finally:
            self._cleanup_children()
            self._release_owner()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("study", type=Path, help="prepared study manifest (study.json)")
    p.add_argument("--agent", action="append", dest="agents", help="agent name from the manifest (repeatable; default all)")
    p.add_argument("--runs-dir", type=Path, required=True)
    p.add_argument("--logs-dir", type=Path, required=True, help="scheduler logs, attempt ledger and status")
    p.add_argument("--workers", type=int, default=1, help="concurrent runner processes for this launcher")
    p.add_argument("--max-attempts", type=int, default=1)
    p.add_argument("--dry-run", action="store_true", help="print the plan and the commands without launching")
    p.add_argument("--owner-lock", type=Path, help="non-blocking launcher lock inherited by worker processes")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    a = parser.parse_args(argv)
    study = load_study(a.study)
    if study.get("launch_approved") is not True and not a.dry_run:
        print("refusing to launch: this manifest is not an approved prepared study", flush=True)
        return 2
    if not study["sources"] and not a.dry_run:
        print("refusing to launch: study has no source pins", flush=True)
        return 2
    drift = source_drift(study)
    if drift and not a.dry_run:
        print("refusing to launch: pinned sources changed: " + ", ".join(sorted(drift)), flush=True)
        return 2
    unknown = sorted(set(a.agents or []) - set(study["agents"]))
    if unknown:
        parser.error(f"unknown agents {', '.join(unknown)}; the manifest knows {', '.join(sorted(study['agents']))}")
    selected = agents(study, a.agents)
    if not selected:
        parser.error("no agents selected")
    all_jobs = jobs(study, selected)
    states = {j.label: state(a.runs_dir, j.label) for j in all_jobs}
    queue = [j for j in all_jobs if states[j.label] != "complete"]
    print(f"study {study['name']}: agents {', '.join(ag.name for ag in selected)}; {len(all_jobs)} runs, "
          f"{sum(j.episodes for j in all_jobs)} episodes; {len(queue)} to run "
          f"({sum(s == 'resume' for s in states.values())} resumed), {len(all_jobs) - len(queue)} complete", flush=True)
    if a.dry_run:
        print("\n".join(plan_table(all_jobs)), flush=True)
        by_name = {ag.name: ag for ag in selected}
        for job in queue:
            print(" ".join(command(job, str(by_name[job.agent].config), a.runs_dir)))
        return 0
    for ag in selected:
        if not ag.config.exists():
            parser.error(f"agent config {ag.config} does not exist")
    scheduler = Scheduler(queue, {ag.name: ag for ag in selected}, runs=a.runs_dir, logs=a.logs_dir, workers=a.workers,
                          max_attempts=a.max_attempts, owner_lock=a.owner_lock,
                          notes={"study": study["name"], "agents": [ag.name for ag in selected]})
    for job in all_jobs:
        if states[job.label] == "complete":
            scheduler.results[job.label] = "complete"
    results = scheduler.run()
    return 0 if all(v == "complete" for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
