r"""Prepare and run a reproducible local World-Adapt study.

`prepare` writes a source-pinned study directory (study.json, agent.json, source snapshot and receipt) from the
six-environment study template. `run` validates it and delegates execution to the study scheduler, which calls an
already configured SAIL or Modal OpenAI-compatible endpoint or the Anthropic API. Nothing is deployed.

    python -m world_adapt.runner.local prepare \
        --provider sail --model example/model \
        --budget-usd 20 --input-price 1 --output-price 4 --input-limit 131072 \
        --out outputs/my-study
    python -m world_adapt.runner.local run --study outputs/my-study/study.json --dry-run

The price values in this example are illustrative accounting inputs. Re-running `run` on the same study resumes
unfinished histories and skips complete ones. For a quick check, `prepare` accepts `--env`, `--condition`,
`--adaptation-tasks` and `--heldout-per-tier`; the result is a smaller study, not comparable to the full protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic, sleep
from typing import Any

from world_adapt.agent import spending
from world_adapt.runner import config as run_config
from world_adapt.runner import study
from world_adapt.secrets import load_env


ROOT = study.ROOT
BUDGET_FILE = "budget.json"
"""Budget top-ups beside the hashed study.json: [{from, to, at}]. The ledger cap follows the latest."""
ENDPOINT_POLL_SECONDS = 10
TEMPLATE = ROOT / "configs" / "study.json"
AGENT_CONFIG = ROOT / "configs" / "agents" / "example.json"


def _finite_positive(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive finite number") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _positive_int(value: Any, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if isinstance(value, bool) or number < 1 or str(number) != str(value).strip():
        raise ValueError(f"{name} must be a positive integer")
    return number


def _count(value: Any, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _safe_slug(value: str, *, fallback: str = "study") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or fallback


def _inside_root(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must be inside the repository: {ROOT}") from error
    return resolved


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _copy_agent_config(source: Path | None, destination: Path, *, provider: str, model: str,
                       effort: str) -> dict[str, Any]:
    source = _inside_root(AGENT_CONFIG if source is None else source, "--agent-config")
    try:
        config = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"agent config is not valid JSON: {source}") from error
    if not isinstance(config, dict):
        raise ValueError("agent config must contain a JSON object")
    config = dict(config)
    config["provider"] = provider
    config["model"] = model
    config["reasoning_effort"] = effort
    config.setdefault("max_retries", 3)
    config.setdefault("stream", True)
    config.setdefault("require_usage", True)
    if config.get("stream") is not True:
        raise ValueError("local spending requires streamed chat responses")
    if config.get("require_usage") is not True:
        raise ValueError("local spending requires provider usage accounting")
    destination.write_text(json.dumps(config, indent=2) + "\n")
    return config


def _pinned_paths(agent_config: str, template_path: Path) -> list[str]:
    return list(dict.fromkeys([*study.SOURCE_PINS, _relative(template_path), agent_config]))


def _snapshot_files(agent_config: str, template_path: Path, study_path: Path | None = None) -> list[Path]:
    paths = [*_pinned_paths(agent_config, template_path), ".env.example"]
    if study_path is not None:
        paths.append(_relative(study_path))
    files: list[Path] = []
    for relative in dict.fromkeys(paths):
        path = ROOT / relative
        if path.is_dir():
            files.extend(candidate for candidate in sorted(path.rglob("*"))
                         if candidate.is_file() and "__pycache__" not in candidate.parts)
        elif path.is_file():
            files.append(path)
    return sorted(set(files), key=lambda value: value.relative_to(ROOT).as_posix())


def _write_source_snapshot(out: Path, agent_config: str, template_path: Path,
                           study_path: Path | None = None) -> dict[str, Any]:
    archive = out / "source-snapshot.zip"
    files = _snapshot_files(agent_config, template_path, study_path)
    entries: dict[str, str] = {}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            payload = path.read_bytes()
            entries[relative] = hashlib.sha256(payload).hexdigest()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            stream.writestr(info, payload)
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    receipt = {
        "format": "world_adapt-local-source-v1",
        "archive": archive.name,
        "archive_sha256": archive_sha,
        "files": entries,
    }
    receipt_path = out / "source-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    receipt["receipt"] = receipt_path.name
    return receipt


def _validate_selections(template: dict[str, Any], envs: list[str] | None,
                         seeds: list[int] | None) -> tuple[list[str], list[int]]:
    selected_envs = list(template["environments"] if envs is None else envs)
    unknown_envs = sorted(set(selected_envs) - set(template["environments"]))
    if unknown_envs:
        raise ValueError(f"unknown environment(s): {', '.join(unknown_envs)}")
    if not selected_envs:
        raise ValueError("at least one environment is required")
    template_seeds = list(template["protocol"]["seed_blocks"])
    selected_seeds = template_seeds if seeds is None else list(dict.fromkeys(seeds))
    if not selected_seeds:
        raise ValueError("at least one seed is required")
    unknown_seeds = sorted(set(selected_seeds) - set(template_seeds))
    if unknown_seeds:
        raise ValueError(f"seed(s) are outside the template inventory: {unknown_seeds}")
    return selected_envs, selected_seeds


def _relative_repo_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{name} must be a repository-relative path")
    path = _inside_root(ROOT / value, name)
    if path == ROOT:
        raise ValueError(f"{name} must name a file inside the repository")
    return path


def _validate_prepared_manifest(manifest: dict[str, Any], study_path: Path,
                                config: dict[str, Any] | None = None) -> None:
    agents = manifest.get("agents")
    if not isinstance(agents, dict) or len(agents) != 1:
        raise ValueError("a local study must contain exactly one agent")
    agent_name, agent_spec = next(iter(agents.items()))
    if not isinstance(agent_spec, dict) or agent_spec.get("backend") not in run_config.PROVIDERS:
        raise ValueError(f"local study agent backend must be one of {', '.join(sorted(run_config.PROVIDERS))}")
    provider = agent_spec["backend"]
    local_execution = manifest.get("local_execution")
    provenance = manifest.get("budget_provenance")
    if not isinstance(local_execution, dict) or local_execution.get("endpoint_provider") != provider:
        raise ValueError("local execution provider does not match the agent")
    if not isinstance(provenance, dict):
        raise ValueError("study is missing budget provenance")
    source_snapshot = manifest.get("source_snapshot")
    if not isinstance(source_snapshot, dict):
        raise ValueError("study is missing source snapshot metadata")
    archive_name = source_snapshot.get("archive")
    receipt_name = source_snapshot.get("receipt")
    if not isinstance(archive_name, str) or Path(archive_name).name != archive_name:
        raise ValueError("source snapshot archive must be a local filename")
    if not isinstance(receipt_name, str) or Path(receipt_name).name != receipt_name:
        raise ValueError("source snapshot receipt must be a local filename")
    archive_path = study_path.parent / archive_name
    receipt_path = study_path.parent / receipt_name
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"source receipt is not valid JSON: {receipt_path}") from error
    if not isinstance(receipt, dict) or receipt.get("format") != "world_adapt-local-source-v1":
        raise ValueError("source receipt has an unsupported format")
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if archive_sha != source_snapshot.get("archive_sha256") or archive_sha != receipt.get("archive_sha256"):
        raise ValueError("source snapshot archive hash does not match its receipt")
    final_study_sha = hashlib.sha256(study_path.read_bytes()).hexdigest()
    if receipt.get("study_sha256") != final_study_sha:
        raise ValueError("prepared study hash does not match its source receipt")
    entries = receipt.get("files")
    if not isinstance(entries, dict):
        raise ValueError("source receipt is missing file hashes")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = set(archive.namelist())
            for relative, expected in entries.items():
                if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise ValueError("source receipt contains an unsafe path")
                if not isinstance(expected, str) or len(expected) != 64:
                    raise ValueError(f"source receipt has an invalid hash for {relative}")
                if relative not in members:
                    raise ValueError(f"source snapshot is missing {relative}")
                if hashlib.sha256(archive.read(relative)).hexdigest() != expected:
                    raise ValueError(f"source snapshot hash mismatch for {relative}")
                if relative != _relative(study_path):
                    current = ROOT / relative
                    if not current.is_file() or hashlib.sha256(current.read_bytes()).hexdigest() != expected:
                        raise ValueError(f"source receipt file mismatch for {relative}")
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"cannot validate source snapshot: {archive_path}") from error
    budget = _finite_positive(provenance.get("budget_usd"), "budget provenance")
    input_price = _finite_positive(provenance.get("input_price_usd_per_million_tokens"), "input price")
    cached_price = _finite_positive(provenance.get("cached_input_price_usd_per_million_tokens"), "cached input price")
    output_price = _finite_positive(provenance.get("output_price_usd_per_million_tokens"), "output price")
    input_limit = _positive_int(provenance.get("input_limit_tokens"), "input limit")
    if cached_price > input_price:
        raise ValueError("cached input price cannot exceed input price")
    ledger_path = _relative_repo_path(provenance.get("ledger"), "budget ledger")
    config_path = _relative_repo_path(agent_spec.get("config"), "agent config")
    try:
        loaded_config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"agent config is not valid JSON: {config_path}") from error
    if not isinstance(loaded_config, dict):
        raise ValueError("agent config must contain a JSON object")
    if config is not None and loaded_config != config:
        raise ValueError("generated agent config changed before validation")
    if loaded_config.get("provider") != provider or loaded_config.get("model") != local_execution.get("endpoint_model"):
        raise ValueError("agent config provider or model does not match local execution")
    if loaded_config.get("stream") is not True:
        raise ValueError("local spending requires streamed chat requests")
    if loaded_config.get("require_usage") is not True:
        raise ValueError("local spending requires usage accounting")
    completion_limit = loaded_config.get("max_completion_tokens")
    if type(completion_limit) is not int or completion_limit < 1:
        raise ValueError("agent config must set a positive max_completion_tokens")
    context_limits = [manifest.get("budgets", {}).get("max_context_tokens"),
                      manifest.get("budgets", {}).get("writer_context_tokens")]
    context_limits = [int(limit) for limit in context_limits if isinstance(limit, (int, float)) and limit is not None]
    if context_limits and input_limit < max(context_limits):
        raise ValueError("input limit is smaller than the planning or writer context budget")
    if manifest.get("budgets", {}).get("reasoning_effort") != loaded_config.get("reasoning_effort"):
        raise ValueError("manifest reasoning effort does not match the agent config")
    expected_env = {
        "WORLD_ADAPT_SPEND_LEDGER": provenance["ledger"],
        "WORLD_ADAPT_SPEND_LIMIT_USD": str(budget),
        "WORLD_ADAPT_INPUT_PRICE": str(input_price),
        "WORLD_ADAPT_CACHED_INPUT_PRICE": str(cached_price),
        "WORLD_ADAPT_OUTPUT_PRICE": str(output_price),
        "WORLD_ADAPT_INPUT_LIMIT": str(input_limit),
        "WORLD_ADAPT_SPEND_AGENT": agent_name,
        "WORLD_ADAPT_SPEND_STUDY": manifest.get("name"),
    }
    if agent_spec.get("env") != expected_env:
        raise ValueError("agent spend settings do not match the frozen budget provenance")
    if ledger_path != _inside_root(study_path.parent / "spend-ledger.json", "budget ledger"):
        raise ValueError("budget ledger must remain inside the prepared study directory")
    selected = study.agents(manifest, [agent_name])
    for job in study.jobs(manifest, selected):
        command = study.command(job, str(config_path), study_path.parent / "runs", sys.executable)
        parser = run_config.build_parser()
        parsed = run_config.parse_args(parser, command[3:])
        run_config.check_args(parser, parsed)
        if parsed.provider != provider or parsed.model != local_execution.get("endpoint_model"):
            raise ValueError(f"resolved job {job.label} changes the prepared provider or model")
        if not parsed.agent_config or Path(parsed.agent_config).resolve() != config_path.resolve():
            raise ValueError(f"resolved job {job.label} changes the prepared agent config")
        if (parsed.env, parsed.profile, parsed.memory, parsed.seed, parsed.label) != (
                job.env, job.profile, job.memory, job.seed, job.label):
            raise ValueError(f"resolved job {job.label} changes its task identity")
        if not parsed.runs_dir or Path(parsed.runs_dir).resolve() != (study_path.parent / "runs").resolve():
            raise ValueError(f"resolved job {job.label} changes the prepared run directory")
        if parsed.resume_dir is not None:
            resume = Path(parsed.resume_dir).resolve()
            try:
                resume.relative_to((study_path.parent / "runs").resolve())
            except ValueError as error:
                raise ValueError(f"resolved job {job.label} resumes outside the prepared run directory") from error
        if parsed.stream is not True or parsed.require_usage is not True:
            raise ValueError(f"resolved job {job.label} does not use streamed usage-accounted chat")
        for name in ("max_context_tokens", "writer_context_tokens"):
            value = getattr(parsed, name)
            if value is not None and value > input_limit:
                raise ValueError(f"resolved job {job.label} exceeds the declared input limit with {name}")
        if parsed.max_completion_tokens != completion_limit:
            raise ValueError(f"resolved job {job.label} changes max_completion_tokens")


def prepare(args: argparse.Namespace) -> Path:
    provider = args.provider
    model = (args.model or "").strip()
    if not model:
        raise ValueError("--model is required")
    budget = _finite_positive(args.budget_usd, "--budget-usd")
    input_price = _finite_positive(args.input_price, "--input-price")
    output_price = _finite_positive(args.output_price, "--output-price")
    input_limit = _positive_int(args.input_limit, "--input-limit")
    effort = args.reasoning_effort
    if effort not in {"none", "low", "medium", "high", "max", "off"}:
        raise ValueError("--reasoning-effort must be one of none, low, medium, high, max, or off")
    out = _inside_root(Path(args.out), "--out")
    if out.exists():
        raise ValueError(f"refusing to overwrite existing prepared study: {out}")
    template_path = _inside_root(Path(args.template), "--template")
    if not template_path.is_file():
        raise ValueError(f"study template does not exist: {template_path}")
    template = json.loads(template_path.read_text())
    envs, seeds = _validate_selections(template, args.env, args.seed)
    conditions = list(dict.fromkeys(args.condition or template["conditions"]))
    unknown = sorted(set(conditions) - set(study.CONDITIONS))
    if unknown:
        raise ValueError(f"unknown condition(s): {', '.join(unknown)}; use one of {', '.join(study.CONDITIONS)}")
    protocol = {}
    if args.adaptation_tasks is not None:
        protocol["adaptation_tasks"] = _count(args.adaptation_tasks, "--adaptation-tasks", 0)
    if args.heldout_per_tier is not None:
        protocol["heldout_per_tier"] = _count(args.heldout_per_tier, "--heldout-per-tier", 1)

    out.mkdir(parents=True)
    config_path = out / "agent.json"
    config = _copy_agent_config(args.agent_config, config_path, provider=provider, model=model, effort=effort)
    config_rel = _relative(config_path)
    ledger_rel = _relative(out / "spend-ledger.json")
    study_name = f"local-{_safe_slug(out.name)}"
    generated = json.loads(json.dumps(template))
    generated["name"] = study_name
    generated["launch_approved"] = True
    generated["protocol"]["seed_blocks"] = seeds
    generated["protocol"]["runner_args"].update(protocol)
    generated["conditions"] = {slug: generated["conditions"][slug] for slug in conditions}
    generated["budgets"]["reasoning_effort"] = effort
    generated["environments"] = {name: generated["environments"][name] for name in envs}
    for spec in generated["environments"].values():
        spec["seed_blocks"] = list(seeds)
    generated["agents"] = {
        _safe_slug(model, fallback="model"): {
            "config": config_rel,
            "backend": provider,
            "env": {
                "WORLD_ADAPT_SPEND_LEDGER": ledger_rel,
                "WORLD_ADAPT_SPEND_LIMIT_USD": str(budget),
                "WORLD_ADAPT_INPUT_PRICE": str(input_price),
                "WORLD_ADAPT_CACHED_INPUT_PRICE": str(input_price),
                "WORLD_ADAPT_OUTPUT_PRICE": str(output_price),
                "WORLD_ADAPT_INPUT_LIMIT": str(input_limit),
                "WORLD_ADAPT_SPEND_AGENT": _safe_slug(model, fallback="model"),
                "WORLD_ADAPT_SPEND_STUDY": study_name,
            },
        }
    }
    agent_name = next(iter(generated["agents"]))
    generated["local_execution"] = {
        "mode": "local_scheduler",
        "endpoint_provider": provider,
        "endpoint_model": model,
        "deployment": "external endpoint; this command does not deploy cloud resources",
        "template": _relative(template_path),
        "agent_config": config_rel,
    }
    generated["budget_provenance"] = {
        "budget_usd": budget,
        "input_price_usd_per_million_tokens": input_price,
        "cached_input_price_usd_per_million_tokens": input_price,
        "output_price_usd_per_million_tokens": output_price,
        "input_limit_tokens": input_limit,
        "ledger": ledger_rel,
        "set_by": "world_adapt.runner.local prepare",
        "provider": provider,
        "model": model,
    }
    generated["sources"] = study.pin_sources(tuple(_pinned_paths(config_rel, template_path)))
    generated["status"] = "Local study prepared from the six-environment study template."
    study_path = out / "study.json"
    generated["source_snapshot"] = {
        "archive": "source-snapshot.zip",
        "receipt": "source-receipt.json",
        "format": "world_adapt-local-source-v1",
    }
    study_path.write_text(json.dumps(generated, indent=2) + "\n")
    snapshot = _write_source_snapshot(out, config_rel, template_path, study_path)
    generated["source_snapshot"] = snapshot
    study_path.write_text(json.dumps(generated, indent=2) + "\n")
    receipt_path = out / snapshot["receipt"]
    receipt = json.loads(receipt_path.read_text())
    receipt["study_sha256"] = hashlib.sha256(study_path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    # Validate the generated artifact before returning it. No credentials or provider requests are used.
    generated = study.load_study(study_path)
    _validate_prepared_manifest(generated, study_path, config)
    print(f"prepared {study_path}")
    print(f"agent {agent_name}; {provider}; {model}; budget ${budget:g}")
    return study_path


def _check_credentials(provider: str) -> None:
    load_env()
    required = [run_config.PROVIDERS[provider][1]] + (["MODAL_BASE_URL"] if provider == "modal" else [])
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError(f"missing required {provider} credential(s): {', '.join(missing)}")
    if provider == "modal":
        try:
            run_config.provider_base_url(provider)
        except SystemExit as error:
            raise ValueError(str(error)) from error


def _budget_history(manifest: dict[str, Any], study_path: Path) -> list[dict[str, Any]]:
    """The recorded top-ups, oldest first. They live beside the hashed study because the budget is accounting."""
    path = study_path.parent / BUDGET_FILE
    if not path.exists():
        return []
    try:
        history = json.loads(path.read_text())["history"]
        last = float(manifest["budget_provenance"]["budget_usd"])
        for entry in history:
            if entry["from"] != last or not entry["to"] > last or not math.isfinite(entry["to"]) \
                    or not isinstance(entry["at"], str):
                raise ValueError
            last = entry["to"]
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"budget history is invalid: {path}") from error
    return history


def current_budget(manifest: dict[str, Any], study_path: Path) -> float:
    history = _budget_history(manifest, study_path)
    return float(history[-1]["to"] if history else manifest["budget_provenance"]["budget_usd"])


def _raise_budget(manifest: dict[str, Any], study_path: Path, requested: float) -> None:
    """Record a higher budget (never a lower one; equal is a no-op) and raise the ledger cap to match."""
    requested = _finite_positive(requested, "--budget-usd")
    current = current_budget(manifest, study_path)
    if requested < current:
        raise ValueError(f"--budget-usd {requested:g} is below the current budget {current:g}; budgets only rise")
    if requested == current:
        return
    history = [*_budget_history(manifest, study_path),
               {"from": current, "to": requested, "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}]
    path = study_path.parent / BUDGET_FILE
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"history": history}, indent=2) + "\n")
    temporary.replace(path)
    print(f"budget raised from ${current:g} to ${requested:g}")


def _fetch_status(url: str, headers: dict[str, str]) -> int:
    """HTTP status of a GET, or 0 when the endpoint cannot be reached."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except OSError:
        return 0


def _wait_for_endpoint(provider: str, timeout: float) -> None:
    """Poll the Modal endpoint's /models until it answers 200; a cold container answers 503 for minutes."""
    url = run_config.provider_base_url(provider) + "/models"
    headers = {"Authorization": f"Bearer {os.environ[run_config.PROVIDERS[provider][1]]}", **run_config.modal_headers()}
    start = monotonic()
    while True:
        status = _fetch_status(url, headers)
        waited = monotonic() - start
        if status == 200:
            print(f"endpoint ready after {waited:.0f}s", flush=True)
            return
        if waited + ENDPOINT_POLL_SECONDS > timeout:
            raise ValueError(f"endpoint {url} was not ready after {waited:.0f}s (last status {status}); "
                             "nothing was launched; raise --endpoint-timeout or check the deployment")
        print(f"waiting for endpoint: HTTP {status or 'unreachable'}, {waited:.0f}s of {timeout:g}s", flush=True)
        sleep(ENDPOINT_POLL_SECONDS)


def _validate_existing_ledger(manifest: dict[str, Any], study_path: Path) -> None:
    _validate_prepared_manifest(manifest, study_path)
    provenance = manifest["budget_provenance"]
    ledger = _relative_repo_path(provenance["ledger"], "budget ledger")
    if ledger.exists():
        try:
            data = json.loads(ledger.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"spend ledger is not valid JSON: {ledger}") from error
        if data.get("limit_usd") != current_budget(manifest, study_path):
            raise ValueError("existing spend ledger limit differs from the study budget")
    if not study_path.exists():
        raise ValueError(f"study does not exist: {study_path}")


def run(args: argparse.Namespace) -> int:
    path = _inside_root(Path(args.study), "--study")
    manifest = study.load_study(path)
    if not manifest.get("local_execution"):
        raise ValueError("--study must be a study prepared by world_adapt.runner.local")
    if args.workers < 1 or args.max_attempts < 1:
        raise ValueError("--workers and --max-attempts must be positive")
    if args.budget_usd is not None and args.dry_run:
        raise ValueError("--budget-usd changes the study; it cannot be combined with --dry-run")
    _validate_prepared_manifest(manifest, path)
    drift = study.source_drift(manifest)
    if drift:
        changed = ", ".join(sorted(drift))
        raise ValueError(f"source pins do not match the prepared study: {changed}")
    if args.budget_usd is not None:
        _raise_budget(manifest, path, args.budget_usd)
    if not args.dry_run and _budget_history(manifest, path):  # Also repairs a top-up that stopped before the ledger.
        spending.raise_ledger_limit(_relative_repo_path(manifest["budget_provenance"]["ledger"], "budget ledger"),
                                    current_budget(manifest, path))
    _validate_existing_ledger(manifest, path)
    agent_names = list(manifest["agents"])
    provider = manifest["agents"][agent_names[0]]["backend"]
    if not args.dry_run:
        _check_credentials(provider)
        if provider == "modal":
            _wait_for_endpoint(provider, _finite_positive(args.endpoint_timeout, "--endpoint-timeout"))
    runs_dir = path.parent / "runs"
    logs_dir = path.parent / "logs"
    argv = [str(path), "--agent", agent_names[0], "--workers", str(args.workers),
            "--max-attempts", str(args.max_attempts), "--runs-dir", str(runs_dir),
            "--logs-dir", str(logs_dir), "--owner-lock", str(path.parent / ".local-run.lock")]
    if args.dry_run:
        argv.append("--dry-run")
    return study.main(argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", help="write a new source-pinned local study")
    prepare_parser.add_argument("--provider", choices=sorted(run_config.PROVIDERS), required=True)
    prepare_parser.add_argument("--model", required=True)
    prepare_parser.add_argument("--budget-usd", type=float, required=True,
                                help="positive inference budget; this is the scheduler ledger cap")
    prepare_parser.add_argument("--input-price", type=float, required=True,
                                help="positive dollars per million input tokens")
    prepare_parser.add_argument("--output-price", type=float, required=True,
                                help="positive dollars per million output tokens")
    prepare_parser.add_argument("--input-limit", type=int, required=True,
                                help="positive served input-token limit reserved per request")
    prepare_parser.add_argument("--out", type=Path, required=True,
                                help="new study directory inside the repository, for example outputs/my-study")
    prepare_parser.add_argument("--env", action="append", dest="env",
                                help="environment to include (repeatable; default: all six)")
    prepare_parser.add_argument("--condition", action="append",
                                help="condition slug to include (repeatable; default: all four)")
    prepare_parser.add_argument("--adaptation-tasks", type=int,
                                help="adaptation tasks per history (default: the template's 8)")
    prepare_parser.add_argument("--heldout-per-tier", type=int,
                                help="held-out tasks per tier (default: the template's 2)")
    prepare_parser.add_argument("--seed", action="append", type=int,
                                help="seed block to include (repeatable; default: the template's blocks)")
    prepare_parser.add_argument("--reasoning-effort", default="high",
                                help="none, low, medium, high, max, or off (default: high)")
    prepare_parser.add_argument("--agent-config", type=Path,
                                help=f"agent config to copy (default: {_relative(AGENT_CONFIG)})")
    prepare_parser.add_argument("--template", type=Path, default=TEMPLATE,
                                help=f"study template (default: {_relative(TEMPLATE)})")
    run_parser = sub.add_parser("run", help="run or resume a prepared study")
    run_parser.add_argument("--study", type=Path, required=True, help="the prepared study.json")
    run_parser.add_argument("--dry-run", action="store_true", help="print the plan and commands; no inference")
    run_parser.add_argument("--workers", type=int, default=1, help="concurrent runs (default: 1)")
    run_parser.add_argument("--max-attempts", type=int, default=3,
                            help="launches per history across all invocations; a spend-limit pause is not counted "
                                 "(default: 3)")
    run_parser.add_argument("--budget-usd", type=float,
                            help="raise the study's spend cap to this amount so paused histories resume; "
                                 "never lowers it, an equal value is a no-op (default: keep the current cap)")
    run_parser.add_argument("--endpoint-timeout", type=float, default=900,
                            help="Modal only: seconds to wait for /models to answer 200 before launching; "
                                 "a cold start takes 3 to 10 minutes (default: 900)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args)
            return 0
        return run(args)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    sys.exit(main())
