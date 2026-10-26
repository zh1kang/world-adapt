"""IceSlide (prototype): everything slides until something stops it.

The agent slides until the next cell is blocked or it enters a rough tile. If
its slide is stopped by a loose crate, that crate is launched and slides on,
stopping before an obstacle, on a rough tile, or by dropping into an empty
socket, where it is fixed for good. Goal: fill every socket.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any, Sequence

from world_adapt.environments.base import InvalidAction, InvalidState, Transition
from world_adapt.environments.registry import Level
from world_adapt.environments.search import solve

Pos = tuple[int, int]
DIRECTIONS: dict[str, Pos] = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
ALIASES = {"u": "up", "d": "down", "l": "left", "r": "right"}

CURRICULUM = "calibration"
CALIBRATION_VERSION = "iceslide-calibration-v1"
CALIBRATION_TIERS: dict[str, dict[str, int | float | str]] = {
    "easy": {
        "size": 6,
        "crates": 1,
        "rough": 3,
        "wall_density": 0.08,
        "min_solution": 4,
        "max_solution": 8,
        "min_launches": 1,
        "min_opportunities": 1,
        "max_initial_fatal": 0,
        "risk_band": "safe_intro",
    },
    "medium": {
        "size": 7,
        "crates": 2,
        "rough": 4,
        "wall_density": 0.10,
        "min_solution": 8,
        "max_solution": 16,
        "min_launches": 3,
        "min_opportunities": 2,
        "max_initial_fatal": 0,
        "risk_band": "safe_transfer",
    },
    "hard": {
        "size": 8,
        "crates": 3,
        "rough": 5,
        "wall_density": 0.12,
        "min_solution": 16,
        "max_solution": 28,
        "min_launches": 4,
        "min_opportunities": 2,
        "max_initial_fatal": 2,
        "risk_band": "later_risk",
    },
}
_CALIBRATION_TIERS = tuple(CALIBRATION_TIERS)
_CALIBRATION_MIXED_ORDER = ("easy", "easy", "medium", "medium", "hard", "hard")


def shift(p: Pos, d: str) -> Pos:
    return (p[0] + DIRECTIONS[d][0], p[1] + DIRECTIONS[d][1])


@dataclass(frozen=True)
class IceLayout:
    height: int
    width: int
    walls: frozenset[Pos]
    rough: frozenset[Pos]
    sockets: frozenset[Pos]


@dataclass(frozen=True)
class IceState:
    layout: IceLayout
    agent: Pos
    crates: frozenset[Pos]
    """Loose crates."""
    fixed: frozenset[Pos] = frozenset()
    """Crates dropped into sockets; they never move again and block like walls."""

    def blocked(self, p: Pos) -> bool:
        L = self.layout
        return not (0 <= p[0] < L.height and 0 <= p[1] < L.width) or p in L.walls or p in self.crates or p in self.fixed


class IceSlide:
    name = "iceslide"
    version = "0.1.0"

    def actions(self) -> Sequence[str]:
        return tuple(DIRECTIONS)

    def parse_action(self, raw: str) -> str:
        key = ALIASES.get(raw.strip().lower(), raw.strip().lower())
        if key not in DIRECTIONS:
            raise InvalidAction(f"unknown action {raw!r}; expected one of {', '.join(DIRECTIONS)}")
        return key

    def step(self, state: IceState, action: str) -> Transition[IceState]:
        L = state.layout
        pos, moved = state.agent, 0
        while not state.blocked(shift(pos, action)):
            pos = shift(pos, action)
            moved += 1
            if pos in L.rough:
                break
        events = ["slide"] if moved else []
        new = replace(state, agent=pos)
        target = shift(pos, action)
        if pos not in L.rough and target in state.crates:
            crate, travel = target, 0
            crates = state.crates - {target}
            fixed = state.fixed
            probe = replace(state, agent=pos, crates=crates)
            while not probe.blocked(shift(crate, action)):
                crate = shift(crate, action)
                travel += 1
                if crate in L.sockets and crate not in fixed:
                    fixed = fixed | {crate}
                    events.append("socket")
                    break
                if crate in L.rough:
                    break
            if travel:
                events.append("launch")
                new = replace(new, crates=crates if crate in fixed else crates | {crate}, fixed=fixed)
        return Transition(new, tuple(events) or ("blocked",))

    def is_success(self, state: IceState) -> bool:
        return len(state.fixed) == len(state.layout.sockets)

    def is_terminal(self, state: IceState) -> bool:
        return self.is_success(state)

    def observe(self, state: IceState) -> str:
        L = state.layout
        terrain, objects = [], []
        for r in range(L.height):
            t, o = [], []
            for c in range(L.width):
                p = (r, c)
                t.append("#" if p in L.walls else "o" if p in L.sockets else "r" if p in L.rough else ".")
                o.append("A" if p == state.agent else "F" if p in state.fixed else "C" if p in state.crates else ".")
            terrain.append("".join(t))
            objects.append("".join(o))
        return "\n".join([
            "Legend. Terrain: '#' wall, '.' floor, 'r' floor (type R), 'o' socket. Objects: 'A' agent, 'C' crate, 'F' crate in a socket.",
            "Coordinates are (row, column); row 0 is the top.",
            "Terrain:", *terrain, "Objects:", *objects,
            f"Agent: {state.agent}",
            f"Crates: {sorted(state.crates)}",
            f"Crates in sockets: {sorted(state.fixed)}",
            f"Empty sockets: {sorted(L.sockets - state.fixed)}",
            f"Type R floor: {sorted(L.rough)}",
        ])

    def validate_state(self, state: IceState) -> None:
        L = state.layout
        for p in (state.agent, *state.crates, *state.fixed):
            if not (0 <= p[0] < L.height and 0 <= p[1] < L.width) or p in L.walls:
                raise InvalidState(f"object at {p} is off the floor")
        if state.agent in state.crates or state.agent in state.fixed or state.crates & state.fixed:
            raise InvalidState("objects overlap")
        if not state.fixed <= L.sockets or len(state.crates) + len(state.fixed) != len(L.sockets):
            raise InvalidState("crate and socket counts disagree")

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "deterministic": True, "fully_observed": True,
                "actions": list(DIRECTIONS), "termination": "success when every socket holds a crate",
                "curriculum": {CURRICULUM: {"version": CALIBRATION_VERSION,
                    "tiers": {name: {key: value for key, value in bounds.items() if key != "risk_band"}
                              for name, bounds in CALIBRATION_TIERS.items()}}}}


def _calibration_tier(options: dict[str, Any], level_index: int) -> str:
    tier = options.get("tier", "mixed")
    if tier not in (*_CALIBRATION_TIERS, "mixed"):
        raise ValueError("iceslide calibration tier must be one of easy, medium, hard, or mixed")
    if tier != "mixed":
        return tier
    count = max(1, int(options.get("_calibration_count", len(_CALIBRATION_MIXED_ORDER))))
    quotient, remainder = divmod(count, len(_CALIBRATION_TIERS))
    counts = tuple(quotient + (index < remainder) for index in range(len(_CALIBRATION_TIERS)))
    if level_index < counts[0]:
        return "easy"
    if level_index < counts[0] + counts[1]:
        return "medium"
    return "hard"


def _initial_action_status(
    game: IceSlide, state: IceState, max_expansions: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (proven fatal, unverified) changing initial actions."""
    fatal: list[str] = []
    unverified: list[str] = []
    for action in game.actions():
        next_state = game.step(state, action).state
        if next_state == state or game.is_success(next_state):
            continue
        result = solve(game, next_state, max_expansions)
        if result.proved_unsolvable:
            fatal.append(action)
        elif result.exhausted:
            unverified.append(action)
    return tuple(fatal), tuple(unverified)


def _launch_opportunities(
    game: IceSlide, state: IceState, solution: Sequence[str], max_expansions: int
) -> tuple[int, int]:
    """Count changed launch-outcome branches around a private reference path.

    The branch actions are counterfactual probes, not actions from the reference
    witness.  This keeps admission independent of whether an agent chooses to
    query or follow the private witness.
    """
    from world_adapt.perturbations.base import PerturbedDynamics
    from world_adapt.perturbations.iceslide import PROFILES

    dynamics = PerturbedDynamics(game, PROFILES["launch_outcomes"])
    safe_successors: set[IceState] = set()
    reference_changed = 0
    current = state
    for action in (None, *solution):
        if not game.is_terminal(current):
            for probe in game.actions():
                nominal = game.step(current, probe)
                if nominal.state == current:
                    continue
                predicted = dynamics.step(current, probe)
                if game.observe(predicted.transition.state) == game.observe(nominal.state):
                    continue
                if solve(game, nominal.state, max_expansions).solved:
                    safe_successors.add(nominal.state)
        if action is not None:
            nominal = game.step(current, action)
            predicted = dynamics.step(current, action)
            if game.observe(predicted.transition.state) != game.observe(nominal.state):
                reference_changed += 1
            current = nominal.state
    return len(safe_successors), reference_changed


def _sample_calibration_state(rng: random.Random, options: dict[str, Any]) -> IceState | None:
    h = w = int(options["size"])
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    walls = {(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)}
    walls |= set(rng.sample(interior, round(float(options["wall_density"]) * len(interior))))
    floor = [p for p in interior if p not in walls]
    crates = int(options["crates"])
    rough = int(options["rough"])
    if len(floor) < 2 * crates + 1 + rough:
        return None
    rng.shuffle(floor)
    sockets, loose, agent = floor[:crates], floor[crates:2 * crates], floor[2 * crates]
    rough_tiles = floor[2 * crates + 1:2 * crates + 1 + rough]
    return IceState(
        IceLayout(h, w, frozenset(walls), frozenset(rough_tiles), frozenset(sockets)),
        agent,
        frozenset(loose),
    )


def _calibration_levels(game: IceSlide, n: int, seed: int, options: dict[str, Any]) -> list[Level]:
    _calibration_tier(options, 0)  # Validate even when n is zero.
    rng, levels, seen = random.Random(seed), [], set()
    attempts, max_attempts = 0, max(500, n * 5000)
    schedule_options = {**options, "_calibration_count": n}
    while len(levels) < n and attempts < max_attempts:
        attempts += 1
        tier = _calibration_tier(schedule_options, len(levels))
        bounds = CALIBRATION_TIERS[tier]
        local = {**options, **bounds}
        state = _sample_calibration_state(rng, local)
        if state is None or game.observe(state) in seen:
            continue
        max_expansions = int(local.get("max_expansions", 200_000))
        result = solve(game, state, max_expansions)
        if result.actions is None:
            continue
        solution = tuple(result.actions)
        if not int(bounds["min_solution"]) <= len(solution) <= int(bounds["max_solution"]):
            continue
        current, launches = state, 0
        for action in solution:
            transition = game.step(current, action)
            launches += "launch" in transition.events
            current = transition.state
        if not game.is_success(current) or launches < int(bounds["min_launches"]):
            continue
        fatal, unverified = _initial_action_status(game, state, max_expansions)
        if unverified:
            continue
        if len(fatal) > int(bounds["max_initial_fatal"]):
            continue
        opportunities, reference_changed = _launch_opportunities(game, state, solution, max_expansions)
        if opportunities < int(bounds["min_opportunities"]):
            continue
        seen.add(game.observe(state))
        levels.append(Level(state, solution, {
            "curriculum": "calibration",
            "curriculum_version": CALIBRATION_VERSION,
            "tier": tier,
            "risk_band": bounds["risk_band"],
            "solution_length": len(solution),
            "reference_range": (int(bounds["min_solution"]), int(bounds["max_solution"])),
            "reference_verified": True,
            "n_crates": int(bounds["crates"]),
            "n_rough": int(bounds["rough"]),
            "n_launches": launches,
            "initial_fatal_actions": fatal,
            "initial_fatal_count": len(fatal),
            "initial_unverified_actions": unverified,
            "initial_successors_verified": True,
            "launch_outcome_opportunities": opportunities,
            "reference_prefix_changed": reference_changed,
            "seed": seed,
        }))
    if len(levels) != n:
        raise RuntimeError(
            f"could not generate {n} IceSlide calibration levels after {attempts} attempts"
        )
    return levels


def make_levels(game: IceSlide, n: int, seed: int, options: dict[str, Any]) -> list[Level]:
    if options.get("curriculum", CURRICULUM) != CURRICULUM:
        raise ValueError(f"iceslide curriculum must be {CURRICULUM!r}")
    if n <= 0:
        return []
    return _calibration_levels(game, n, seed, options)
