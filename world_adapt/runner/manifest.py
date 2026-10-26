"""The run manifest: enough to identify, reproduce and audit a run."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SPEC_VERSION = "README.md, research specification of October 5, 2026"


def sha256_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def source_hash(*relative: str) -> str:
    """One hash over the named source files or directories (every .py file inside, in sorted order)."""
    h = hashlib.sha256()
    for rel in relative:
        path = ROOT / rel
        files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for f in files:
            h.update(str(f.relative_to(ROOT)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True,
                              env={**os.environ, "GIT_EDITOR": "true"}).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def code_version() -> dict[str, Any]:
    status = _git("status", "--porcelain", "--untracked-files=no")
    diff = _git("diff", "HEAD")
    return {"commit": _git("rev-parse", "HEAD"), "dirty": bool(status),
            "diff_sha256": sha256_text(diff) if diff else None}


def dependencies() -> dict[str, Any]:
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    return {"python": platform.python_version(), "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
            "packages": dict(sorted(versions.items(), key=lambda kv: kv[0].lower()))}


def hardware() -> dict[str, Any]:
    return {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "hostname_sha256": sha256_text(platform.node())[:16]}


def environment_module(env: str) -> str:
    from world_adapt.environments.registry import _ENTRIES

    return _ENTRIES[env].split(":")[0].rsplit(".", 1)[0]


def build(*, env: str, options: dict, game, profile, settings: dict, budgets: list[dict], agent: dict,
          plan: list[dict], task_hashes: list[str], seeds: dict[str, int], tokenizer: str,
          provider: dict) -> dict[str, Any]:
    """Everything except evaluator-only task data. `fingerprint` is what a resume must match."""
    env_package = environment_module(env).replace(".", "/")
    described = profile.describe()
    runtime = dependencies()
    identity = {
        "environment": {"name": env, "version": getattr(game, "version", None), "options": options,
                        "adapter": game.spec(), "source_sha256": source_hash(env_package)},
        "tasks": {"plan": plan, "observation_sha256": task_hashes, "task_generation_seed": seeds["task_generation"],
                  "task_order_seed": seeds["task_order"]},
        "predictor": {"type": "SimulatorPredictor", "checkpoint_sha256": None, "decoder_sha256": None,
                      "inference": {"max_horizon": settings["max_chunk"], "deterministic": True},
                      "source_sha256": source_hash("world_adapt/predictors", "world_adapt/perturbations"),
                      # Full profile parameters are evaluator-only (evaluator.jsonl); this identifies them.
                      "profile": {"name": profile.name, "version": profile.version,
                                  "parameters_sha256": sha256_text(repr(described))}},
        "observation_action_adapter": {"observation": "text",
                                       "source_sha256": source_hash("world_adapt/environments/base.py",
                                                                    "world_adapt/environments/real.py",
                                                                    "world_adapt/session.py", "world_adapt/episode.py",
                                                                    "world_adapt/feedback.py")},
        "budgets": {"per_task": budgets, "rule": settings["budget_rule"]},
        "memory": {"policy": settings["memory"], "notebook_tokens": agent["notebook_tokens"], "tokenizer": tokenizer,
                   "protocol": settings["protocol"]},
        "agent": {"provider": provider, "settings": agent,
                  "harness_source_sha256": source_hash("world_adapt/agent"),
                  "runner_source_sha256": source_hash("world_adapt/runner"),
                  "runtime": runtime,
                  "prompt_source_sha256": source_hash("world_adapt/agent/prompts.py", "world_adapt/agent/tools.py",
                                                      "world_adapt/agent/writer.py")},
        "seeds": {**seeds, "notes": {
            "environment": "unused: environments are deterministic",
            "predictor": "unused: the simulator predictor is deterministic",
            "perturbation": "unused: profile parameters are fixed",
            "agent": "not sent: provider seed support is not assumed"}},
        "settings": settings,
    }
    return {
        "specification": {"version": SPEC_VERSION, "readme_sha256": sha256_file(ROOT / "README.md")},
        "code": code_version(),
        **identity,
        "fingerprint": sha256_text(repr(sorted((k, repr(v)) for k, v in identity.items()))),
        "dependencies": runtime,
        "hardware": hardware(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
