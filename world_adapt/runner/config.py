"""Run configuration: providers, config files, the command line, and turning flags into environment, agent and
budget settings.

Settings come from, in increasing priority: flag defaults, an agent config file (`--agent-config`), then flags
given on the command line. An agent config is a JSON object whose keys are flag names with underscores (e.g.
`"reasoning_effort": "high"`); see `configs/agents/example.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from world_adapt.agent import AgentConfig, Budgets
from world_adapt.agent.llm import REQUEST_TIMEOUT_SECONDS
from world_adapt.environments import registry
from world_adapt.environments.registry import EnvSpec, Level
from world_adapt.secrets import load_env

PROVIDERS = {
    # name: (base URL, key variable, default model). A base URL of the form "$NAME" is read from the environment
    # (after .env.local is loaded), for self-hosted endpoints whose URL is per deployment.
    "sail": ("https://api.sailresearch.com/v1", "SAIL_API_KEY", "moonshotai/Kimi-K3"),
    "modal": ("$MODAL_BASE_URL", "MODAL_API_KEY", ""),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY", "claude-opus-5-5"),
}

NOT_CONFIGURATION = ("agent_config", "resume_dir", "label", "runs_dir")
"""Flags that name files or folders rather than change what the run does."""


def provider_base_url(provider: str) -> str:
    """The provider's base URL, resolving environment-backed URLs when configured."""
    base_url = PROVIDERS[provider][0]
    if base_url.startswith("$"):
        load_env()
        base_url = os.environ.get(base_url[1:], "")
        if not base_url:
            raise SystemExit(f"{PROVIDERS[provider][0][1:]} is not set; add it to .env.local")
    if provider == "modal":
        endpoint = urlsplit(base_url)
        host = endpoint.hostname or ""
        if (endpoint.scheme != "https" or endpoint.username or endpoint.password
                or endpoint.query or endpoint.fragment
                or not (host == "api.modal.com" or host.endswith((".modal.direct", ".modal.run")))):
            raise SystemExit("MODAL_BASE_URL must be an HTTPS Modal endpoint without credentials, query or fragment")
    return base_url.rstrip("/")


def modal_headers() -> dict[str, str]:
    """Modal proxy-auth headers, needed in addition to the bearer token when the endpoint sits behind proxy auth."""
    proxy = {name: os.environ.get(name) for name in ("MODAL_PROXY_KEY", "MODAL_PROXY_SECRET")}
    if not any(proxy.values()):
        return {}
    if not all(proxy.values()):
        raise SystemExit("MODAL_PROXY_KEY and MODAL_PROXY_SECRET must be set together")
    return {"Modal-Key": proxy["MODAL_PROXY_KEY"], "Modal-Secret": proxy["MODAL_PROXY_SECRET"]}


def make_client(provider: str, base_url: str, key_env: str):
    """The provider's SDK client. SDK retries are off so every failure reaches the logged retry loop in `llm.py`.

    The timeout must stay below the launcher's 25-minute stall watchdog, so a hung request fails into that loop
    instead of leaving the run silent until it is killed.
    """
    load_env()
    key = os.environ.get(key_env)
    if not key:
        raise SystemExit(f"{key_env} is not set; add it to .env.local")
    if provider == "anthropic":
        from anthropic import Anthropic

        # The SDK appends /v1/messages itself.
        return Anthropic(api_key=key, base_url=base_url.removesuffix("/v1"), timeout=REQUEST_TIMEOUT_SECONDS,
                         max_retries=0)
    from openai import OpenAI

    headers = modal_headers() if provider == "modal" else {}
    return OpenAI(api_key=key, base_url=base_url, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0,
                  default_headers=headers or None)


def build_parser(description: str = "") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent-config", help="agent config JSON (configs/agents/); command-line flags override it")
    parser.add_argument("--provider", default="sail", choices=sorted(PROVIDERS))
    parser.add_argument("--model", help="provider model ID; required for Modal")
    parser.add_argument("--reasoning-effort", default="high", help="low | medium | high | max | none | off")
    parser.add_argument("--completion-window", default=None, help="Sail only: asap (default) or balanced (cheaper, slower)")
    parser.add_argument("--stream", action="store_true", help="assemble streamed chat replies before tool execution")
    parser.add_argument("--require-usage", action="store_true", help="require complete token accounting for play and notebook replies")
    parser.add_argument("--action-tool-choice", choices=["auto", "required"], default=None,
                        help="tool selection during action turns only; notebook requests stay unconstrained")
    parser.add_argument("--env", default="switchboard", choices=registry.names())
    parser.add_argument("--env-opt", action="append", default=[], metavar="KEY=VALUE", help="environment option, repeatable")
    parser.add_argument("--profile", default="identity",
                        help="predictor profile: identity (accurate) or the environment's faulty profile")
    parser.add_argument("--memory", default="preserve", choices=["preserve", "reset"],
                        help="preserve: saved notebook across adaptation tasks; reset: every task starts empty")
    parser.add_argument("--adaptation-tasks", type=int, default=8, help="protocol: adaptation tasks per run")
    parser.add_argument("--heldout-per-tier", type=int, default=2, help="protocol: held-out tasks per tier")
    parser.add_argument("--seed", type=int, default=0, help="run seed; the seed bundle is derived from it")
    parser.add_argument("--real-factor", type=float, default=3.0, help="real-action budget = factor x reference length")
    parser.add_argument("--prediction-factor", type=float, default=2.0, help="predicted-action budget as a multiple of real-action budget (0 disables prediction budget)")
    parser.add_argument("--max-chunk", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=40, help="cap on agent calls per task")
    parser.add_argument("--max-malformed", type=int, default=5)
    parser.add_argument("--max-task-tokens", type=int, default=1_500_000,
                        help="planning token cap (prompt + completion); writing has a separate allowance; 0 disables")
    parser.add_argument("--max-task-seconds", type=float, default=3600.0, help="planning wall-clock cap; writing has a separate deadline; 0 disables")
    parser.add_argument("--notebook-tokens", type=int, default=2000, help="notebook cap in tokens of the recorded tokenizer")
    parser.add_argument("--notebook-max-tokens", type=int, default=200_000,
                        help="writer estimated prompt+reply allowance per update")
    parser.add_argument("--notebook-max-seconds", type=float, default=900.0,
                        help="writer admission deadline and provider timeout per update")
    parser.add_argument("--notebook-max-completion-tokens", type=int, default=32_768)
    parser.add_argument("--writer-context-tokens", type=int, default=64_000,
                        help="writer prompt cap, independent of the planning prompt cap")
    parser.add_argument("--writer-evidence-tokens", type=int, default=56_000,
                        help="writer allowance for public tool records plus omission metadata")
    parser.add_argument("--max-retries", type=int, default=5, help="model API attempts per call before an infrastructure failure")
    parser.add_argument("--infra-retries", type=int, default=1,
                        help="fresh attempts of a task after an infrastructure failure, same configuration and seeds")
    parser.add_argument("--infrastructure-policy", choices=["stop", "skip"], default="stop",
                        help="after infrastructure retries, stop the history or mark the task missing and continue")
    parser.add_argument("--label", default="")
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument("--max-context-tokens", type=int, default=80_000,
                        help="drop the oldest tool exchanges before a call once the estimated prompt exceeds this "
                             "(world_adapt.agent.context)")
    parser.add_argument("--keep-recent-turns", type=int, default=4, help="exchanges never dropped by --max-context-tokens")
    parser.add_argument("--resume-dir", help="continue an interrupted run after its last fully completed task")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="root for new run records")
    return parser


def _load_json(path: str, parser: argparse.ArgumentParser) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"cannot read config {path}: {error}")
    if not isinstance(data, dict):
        parser.error(f"config {path} must be a JSON object")
    data.pop("_comment", None)
    return data


def parse_args(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> argparse.Namespace:
    """Parse flags on top of the agent config file."""
    first, _ = parser.parse_known_args(argv)
    agent_config_file = _load_json(first.agent_config, parser) if first.agent_config else {}
    known = {a.dest for a in parser._actions}
    unknown = sorted(set(agent_config_file) - known)
    if unknown:
        parser.error(f"unknown config keys: {', '.join(unknown)}")
    parser.set_defaults(**agent_config_file)
    return parser.parse_args(argv)


def check_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate flags and fill defaults that depend on other flags."""
    if args.prediction_factor < 0 or not math.isfinite(args.prediction_factor):
        parser.error("--prediction-factor must be finite and nonnegative")
    if args.env == "pusht" and args.max_completion_tokens is None:
        args.max_completion_tokens = 8192
    if args.adaptation_tasks < 0 or args.heldout_per_tier < 1:
        parser.error("--adaptation-tasks must be >= 0 and --heldout-per-tier >= 1")
    if args.infra_retries < 0 or args.max_retries < 1:
        parser.error("--infra-retries must be >= 0 and --max-retries >= 1")
    if args.keep_recent_turns < 1:
        parser.error("--keep-recent-turns must be >= 1")
    if args.provider == "anthropic":
        if not args.stream:
            parser.error("--provider anthropic streams every reply; set --stream")
        if args.max_completion_tokens is None:
            parser.error("--provider anthropic requires --max-completion-tokens")
        if args.action_tool_choice == "required":
            parser.error("--provider anthropic cannot force tool use while thinking")
        if args.completion_window:
            parser.error("--completion-window applies to Sail only")
    args.model = args.model or PROVIDERS[args.provider][2]
    if not args.model:
        parser.error("--model is required for this provider")
    try:
        agent_config(args, registry.get(args.env))
    except ValueError as error:
        parser.error(str(error))


def settings(args: argparse.Namespace) -> dict[str, Any]:
    """The flags that define the run (file and folder names excluded)."""
    return {k: v for k, v in sorted(vars(args).items()) if k not in NOT_CONFIGURATION}


def env_options(args: argparse.Namespace, spec: EnvSpec) -> dict[str, Any]:
    return registry.parse_options(args.env_opt, spec.default_options)


def agent_config(args: argparse.Namespace, spec: EnvSpec) -> AgentConfig:
    # "none" omits the parameter; "off" sends reasoning effort "none" explicitly (e.g. to disable Qwen thinking).
    effort = None if args.reasoning_effort == "none" else ("none" if args.reasoning_effort == "off" else args.reasoning_effort)
    return AgentConfig(model=args.model, reasoning_effort=effort, completion_window=args.completion_window,
                       end_on_dead_state=spec.has_solver, max_completion_tokens=args.max_completion_tokens,
                       notebook_tokens=args.notebook_tokens, writer_context_tokens=args.writer_context_tokens,
                       writer_evidence_tokens=args.writer_evidence_tokens,
                       notebook_max_tokens=args.notebook_max_tokens, notebook_max_seconds=args.notebook_max_seconds,
                       notebook_max_completion_tokens=args.notebook_max_completion_tokens,
                       max_retries=args.max_retries, max_context_tokens=args.max_context_tokens,
                       keep_recent_turns=args.keep_recent_turns, require_usage=args.require_usage,
                       action_tool_choice=args.action_tool_choice)


BUDGET_RULE = "real = level real_budget, else max(10, ceil(real_factor x reference length)); predicted = ceil(prediction_factor x real)"


def budgets_for(level: Level, args: argparse.Namespace) -> Budgets:
    """A level's own real_budget wins; otherwise a multiple of its reference solution length (at least 10)."""
    reference = level.features["solution_length"] or 0
    real = level.features.get("real_budget") or max(10, math.ceil(args.real_factor * reference))
    return Budgets(real_actions=real, predicted_transitions=math.ceil(args.prediction_factor * real),
                   max_chunk=args.max_chunk, max_turns=args.max_turns, max_malformed=args.max_malformed,
                   max_task_tokens=args.max_task_tokens or None, max_task_seconds=args.max_task_seconds or None)


def default_label(args: argparse.Namespace) -> str:
    return args.label or f"{args.env}-{args.profile}-{args.memory}"
