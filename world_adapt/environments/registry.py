"""Environment registry used by the pilot runner.

Each environment module exposes `SPEC: EnvSpec`. Entries are imported lazily so
an environment's dependencies load only when it is used.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable

from world_adapt.environments.base import Game


@dataclass(frozen=True)
class Level:
    """A solver-verified task instance."""

    state: Hashable
    solution: tuple
    features: dict[str, Any] = field(compare=False)
    """Must include `solution_length`; other structural features are optional."""


@dataclass(frozen=True)
class EnvSpec:
    name: str
    make_game: Callable[[dict[str, Any]], Game]
    make_levels: Callable[[Game, int, int, dict[str, Any]], list[Level]]
    """(game, n, seed, options) -> n distinct levels, deterministic in seed."""
    goal: str
    """Agent-facing goal. States the objective only, never the dynamics."""
    prune: Callable[[Any], bool] | None = None
    """Optional sound dead-state pruner for the solver."""
    default_options: dict[str, Any] = field(default_factory=dict)
    rules: str = ""
    """True mechanics for probes and opt-in `--describe-rules` runs.
    Descriptions must exclude private task parameters, predictor errors, and solution witnesses.
    """
    why: str = ""
    """Why a predictor should help in this environment (information or computation it supplies)."""
    has_solver: bool = True
    """False when no exact solver exists (continuous control): disables solver-based dead-state checks."""


_ENTRIES = {
    "depot": "world_adapt.environments.depot.spec:SPEC",
    "iceslide": "world_adapt.environments.iceslide.spec:SPEC",
    "patrol": "world_adapt.environments.patrol.spec:SPEC",
    "switchboard": "world_adapt.environments.switchboard.spec:SPEC",
    "portalmaze": "world_adapt.environments.portalmaze.spec:SPEC",
    "pusht": "world_adapt.environments.pusht.spec:SPEC",
}


def names() -> list[str]:
    return sorted(_ENTRIES)


def get(name: str) -> EnvSpec:
    module, _, attr = _ENTRIES[name].partition(":")
    return getattr(importlib.import_module(module), attr)


def parse_options(pairs: list[str], defaults: dict[str, Any]) -> dict[str, Any]:
    """Parse KEY=VALUE pairs, converting to the type of the default when one exists."""
    options = dict(defaults)
    for pair in pairs:
        key, _, raw = pair.partition("=")
        default = defaults.get(key)
        if isinstance(default, bool):
            value: Any = raw.lower() in ("1", "true", "yes")
        elif isinstance(default, int):
            value = int(raw)
        elif isinstance(default, float):
            value = float(raw)
        else:
            value = raw
        options[key] = value
    return options
