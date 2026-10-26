"""Switchboard: turn every lamp on using buttons whose wiring is hidden.

Pressing a button toggles a fixed subset of lamps. Some buttons are breakers:
pressing one also permanently disables one other button. A disabled button
does nothing when pressed. The agent sees lamps and which buttons are
disabled, never the wiring or which button a breaker will disable.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, replace
from typing import Any, Sequence

from world_adapt.environments.base import InvalidAction, InvalidState, Transition
from world_adapt.environments.registry import Level
from world_adapt.environments.search import solve

CURRICULUM = "calibration"
OPTIONS = {"buttons": 7, "max_expansions": 200_000, "curriculum": CURRICULUM, "tier": "mixed"}
"""Generation options. `tier` is easy, medium, hard, or mixed (contiguous balanced tiers)."""

# Per-tier bounds make structural claims that can be checked from a generated level rather than inferred from prose.
EXTENDED_TIERS: dict[str, dict[str, int]] = {
    "easy": {
        "rows": 3, "cols": 4, "breakers": 2, "toggle_min": 2, "toggle_max": 4,
        "reference_min": 4, "reference_max": 5, "unique_min": 4, "dependency_min": 1,
    },
    "medium": {
        "rows": 3, "cols": 5, "breakers": 3, "toggle_min": 3, "toggle_max": 5,
        "reference_min": 5, "reference_max": 6, "unique_min": 5, "dependency_min": 2,
    },
    "hard": {
        "rows": 4, "cols": 4, "breakers": 4, "toggle_min": 4, "toggle_max": 7,
        "reference_min": 6, "reference_max": 7, "unique_min": 6, "dependency_min": 3,
    },
}
EXTENDED_TIERS["easy"]["dependency_max"] = 2
EXTENDED_TIERS["medium"]["dependency_max"] = 3
EXTENDED_TIERS["hard"]["dependency_max"] = 4
_CURRICULUM_TIERS = tuple(EXTENDED_TIERS)


@dataclass(frozen=True)
class Wiring:
    rows: int
    cols: int
    toggles: tuple[int, ...]
    """Per button: bitmask of the lamps it toggles (bit r*cols+c)."""
    breaks: tuple[int, ...]
    """Per button: index of the button it permanently disables, or -1."""

    @property
    def n_lamps(self) -> int:
        return self.rows * self.cols

    @property
    def full(self) -> int:
        return (1 << self.n_lamps) - 1


@dataclass(frozen=True)
class SwitchState:
    wiring: Wiring
    lamps: int
    """Bitmask of lamps that are on."""
    disabled: int = 0
    """Bitmask of disabled buttons."""


class Switchboard:
    name = "switchboard"
    version = "0.1.0"

    def __init__(self, buttons: int = 7):
        self.buttons = buttons

    def actions(self) -> Sequence[str]:
        return tuple(f"press {i}" for i in range(1, self.buttons + 1))

    def parse_action(self, raw: str) -> str:
        m = re.fullmatch(r"(?:press\s*|p)?(\d+)", raw.strip().lower())
        if not m or not 1 <= int(m.group(1)) <= self.buttons:
            raise InvalidAction(f"unknown action {raw!r}; expected 'press N' with N in 1..{self.buttons}")
        return f"press {int(m.group(1))}"

    def step(self, state: SwitchState, action: str) -> Transition[SwitchState]:
        i = int(action.split()[1]) - 1
        if state.disabled >> i & 1:
            return Transition(state, ("disabled",))
        w = state.wiring
        disabled, events = state.disabled, ["press"]
        if w.breaks[i] >= 0:
            disabled |= 1 << w.breaks[i]
            events.append("break")
        return Transition(replace(state, lamps=state.lamps ^ w.toggles[i], disabled=disabled), tuple(events))

    def is_success(self, state: SwitchState) -> bool:
        return state.lamps == state.wiring.full

    def is_terminal(self, state: SwitchState) -> bool:
        return self.is_success(state)

    def observe(self, state: SwitchState) -> str:
        w = state.wiring
        on = lambda r, c: state.lamps >> (r * w.cols + c) & 1
        rows = ["".join("*" if on(r, c) else "." for c in range(w.cols)) for r in range(w.rows)]
        off = [(r, c) for r in range(w.rows) for c in range(w.cols) if not on(r, c)]
        disabled = [i + 1 for i in range(self.buttons) if state.disabled >> i & 1]
        status = "all lamps on" if self.is_success(state) else "in progress"
        return "\n".join([
            "Legend: '*' lamp on, '.' lamp off.",
            "Coordinates are (row, column); row 0 is the top.",
            *rows,
            f"Lamps off: {off}",
            f"Buttons: {list(range(1, self.buttons + 1))}",
            f"Disabled buttons: {disabled}",
            f"Status: {status}",
        ])

    def validate_state(self, state: SwitchState) -> None:
        w = state.wiring
        if not 0 <= state.lamps <= w.full:
            raise InvalidState("lamp mask out of range")
        if not 0 <= state.disabled < 1 << self.buttons or len(w.toggles) != self.buttons:
            raise InvalidState("disabled mask out of range")

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "deterministic": True, "fully_observed": False,
                "hidden": "button wiring and breaker targets", "actions": list(self.actions()),
                "termination": "success when all lamps are on; no terminal failure (state may become unsolvable)"}


def _graph(game: Switchboard, start: SwitchState) -> dict[SwitchState, list[SwitchState]]:
    graph, stack = {}, [start]
    while stack:
        s = stack.pop()
        if s in graph:
            continue
        graph[s] = [] if game.is_success(s) else list({game.step(s, a).state for a in game.actions()})
        stack.extend(graph[s])
    return graph


def solvable_states(game: Switchboard, start: SwitchState) -> tuple[set[SwitchState], set[SwitchState]]:
    """(all reachable states, those from which success is still reachable)."""
    graph = _graph(game, start)
    rev: dict[SwitchState, list[SwitchState]] = {s: [] for s in graph}
    for s, nxts in graph.items():
        for n in nxts:
            rev[n].append(s)
    good = {s for s in graph if game.is_success(s)}
    stack = list(good)
    while stack:
        for p in rev[stack.pop()]:
            if p not in good:
                good.add(p)
                stack.append(p)
    return set(graph), good


def _random_wiring(rng: random.Random, o: dict[str, Any]) -> Wiring:
    n, k = int(o["rows"]) * int(o["cols"]), int(o["buttons"])
    toggle_min = int(o["toggle_min"])
    toggle_max = int(o["toggle_max"])
    toggles = tuple(sum(1 << b for b in rng.sample(range(n), rng.randint(toggle_min, toggle_max))) for _ in range(k))
    breaks = [-1] * k
    for b in rng.sample(range(k), int(o["breakers"])):
        breaks[b] = rng.choice([j for j in range(k) if j != b])
    return Wiring(int(o["rows"]), int(o["cols"]), toggles, tuple(breaks))


def _tier(options: dict[str, Any], level_index: int) -> str:
    tier = options.get("tier", "mixed")
    if tier not in (*_CURRICULUM_TIERS, "mixed"):
        raise ValueError("switchboard tier must be one of easy, medium, hard, or mixed")
    return _CURRICULUM_TIERS[level_index % len(_CURRICULUM_TIERS)] if tier == "mixed" else tier


def _calibration_tier(options: dict[str, Any], level_index: int, total: int) -> str:
    """Return contiguous balanced tiers for the calibration progression."""
    tier = options.get("tier", "mixed")
    if tier != "mixed":
        return _tier(options, level_index)
    base, extra = divmod(total, len(_CURRICULUM_TIERS))
    counts = [base + int(index < extra) for index in range(len(_CURRICULUM_TIERS))]
    offset = 0
    for name, count in zip(_CURRICULUM_TIERS, counts):
        if level_index < offset + count:
            return name
        offset += count
    raise IndexError("calibration level index is outside the requested task pool")


def _dependency_count(wiring: Wiring, actions: Sequence[str]) -> int:
    """Count breaker targets that are also used by the reference witness."""
    used = {int(action.split()[1]) - 1 for action in actions}
    return sum(source in used and target in used for source, target in enumerate(wiring.breaks) if target >= 0)


def _retarget_enabled_opportunities(
    game: Switchboard, state: SwitchState, solution: Sequence[str]
) -> tuple[tuple[int, SwitchState, str], ...]:
    """Return reference steps where the enabled-target profile can change a prediction."""
    opportunities: list[tuple[int, SwitchState, str]] = []
    current = state
    for index, action in enumerate(solution):
        button = int(action.split()[1]) - 1
        target = current.wiring.breaks[button]
        has_enabled_alternate = any(
            candidate not in (button, target) and not (current.disabled >> candidate & 1)
            for candidate in range(len(current.wiring.breaks))
        )
        if (
            current.disabled
            and not (current.disabled >> button & 1)
            and target >= 0
            and has_enabled_alternate
        ):
            opportunities.append((index, current, action))
        current = game.step(current, action).state
    return tuple(opportunities)


def _recoverable_successors(game: Switchboard, state: SwitchState, good: set[SwitchState]) -> int:
    """Count distinct nonterminal state-changing actions that preserve solvability."""
    successors = set()
    for action in game.actions():
        successor = game.step(state, action).state
        if successor != state and successor in good and not game.is_terminal(successor):
            successors.add(successor)
    return len(successors)


def _calibration_levels(game: Switchboard, n: int, seed: int, options: dict[str, Any]) -> list[Level]:
    requested = options.get("tier", "mixed")
    _tier(options, 0)  # Validate even when a caller requests an empty pool.
    tier_scope = requested if requested in _CURRICULUM_TIERS else "mixed"
    required_buttons = max(
        EXTENDED_TIERS[tier]["unique_min"] for tier in _CURRICULUM_TIERS if tier_scope in ("mixed", tier)
    )
    required_breakers = max(
        EXTENDED_TIERS[tier]["breakers"] for tier in _CURRICULUM_TIERS if tier_scope in ("mixed", tier)
    )
    if game.buttons < required_buttons or game.buttons < required_breakers:
        raise ValueError(
            f"switchboard requires at least {max(required_buttons, required_breakers)} buttons; got {game.buttons}"
        )
    rng, levels, seen = random.Random(seed), [], set()
    attempts = 0
    max_attempts = max(200, n * 2500)
    while len(levels) < n and attempts < max_attempts:
        attempts += 1
        tier = _calibration_tier(options, len(levels), n)
        bounds = EXTENDED_TIERS[tier]
        local = {**options, **bounds, "buttons": game.buttons}
        wiring = _random_wiring(rng, local)
        lamps = wiring.full
        # The witness is sampled before the start state.  This makes breaker
        # decisions part of the task construction, while the solver still
        # verifies the shortest reference that the benchmark records.
        witness_length = rng.randint(bounds["reference_min"], bounds["reference_max"])
        witness = [f"press {i + 1}" for i in rng.sample(range(game.buttons), min(game.buttons, bounds["unique_min"]))]
        witness += [rng.choice(game.actions()) for _ in range(witness_length - len(witness))]
        for action in witness:
            lamps ^= wiring.toggles[int(action.split()[1]) - 1]
        state = SwitchState(wiring, lamps)
        if game.is_success(state) or game.observe(state) in seen:
            continue
        result = solve(game, state, int(options["max_expansions"]))
        if result.actions is None:
            continue
        solution = result.actions
        dependency_count = _dependency_count(wiring, solution)
        unique_buttons = len({int(action.split()[1]) for action in solution})
        if not bounds["reference_min"] <= len(solution) <= bounds["reference_max"]:
            continue
        if (
            unique_buttons < bounds["unique_min"]
            or not bounds["dependency_min"] <= dependency_count <= bounds["dependency_max"]
        ):
            continue
        _, good = solvable_states(game, state)
        opportunities = _retarget_enabled_opportunities(game, state, solution)
        if not opportunities:
            continue
        recoverable = 0
        for _, opportunity_state, action in opportunities:
            actual = game.step(opportunity_state, action).state
            if actual in good and _recoverable_successors(game, actual, good) >= 2:
                recoverable += 1
        if not recoverable:
            continue
        frontier = {state}
        unsafe3 = False
        for _ in range(3):
            frontier = {game.step(s, a).state for s in frontier for a in game.actions()}
            unsafe3 = unsafe3 or any(s not in good for s in frontier)
        if not unsafe3:
            continue
        seen.add(game.observe(state))
        levels.append(Level(state, solution, {
            "curriculum": CURRICULUM, "tier": tier, "solution_length": len(solution),
            "reference_range": (bounds["reference_min"], bounds["reference_max"]),
            "n_lamps": wiring.n_lamps, "n_buttons": game.buttons,
            "n_breakers": sum(b >= 0 for b in wiring.breaks),
            "toggle_min": min(bin(mask).count("1") for mask in wiring.toggles),
            "toggle_max": max(bin(mask).count("1") for mask in wiring.toggles),
            "unique_buttons": unique_buttons, "breaker_dependencies": dependency_count,
            "retarget_opportunities": len(opportunities), "retarget_recoverable": recoverable,
            "lamps_off": wiring.n_lamps - bin(lamps).count("1"), "unsafe_within_3": unsafe3, "seed": seed,
        }))
    if len(levels) != n:
        raise RuntimeError(f"could not generate {n} switchboard {options.get('tier', 'mixed')} levels after {attempts} attempts")
    return levels


def make_levels(game: Switchboard, n: int, seed: int, options: dict[str, Any]) -> list[Level]:
    o = {**OPTIONS, **options}
    if o["curriculum"] != CURRICULUM:
        raise ValueError(f"switchboard curriculum must be {CURRICULUM!r}")
    return _calibration_levels(game, n, seed, o)
