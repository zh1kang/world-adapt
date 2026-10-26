"""Patrol (prototype): reach the exit without being caught by guards on hidden routes.

Each guard walks a fixed back-and-forth route, one cell per action. The agent
sees where guards are now, never their routes. Ending a step on a guard's cell,
or swapping cells with a guard, is a terminal failure.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Sequence

from world_adapt.environments.base import InvalidAction, InvalidState, Transition
from world_adapt.environments.registry import Level
from world_adapt.environments.search import solve

Pos = tuple[int, int]
DIRECTIONS: dict[str, Pos] = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1), "wait": (0, 0)}
ALIASES = {"u": "up", "d": "down", "l": "left", "r": "right", "w": "wait", "stay": "wait"}

CURRICULUM = "calibration"
OPTIONS = {
    "curriculum": CURRICULUM,
    "calibration_size": 10, "calibration_guards": 5, "calibration_route_min": 4,
    "calibration_route_max": 8, "calibration_wall_density": 0.18,
    "calibration_easy_min_solution": 16, "calibration_medium_min_solution": 17,
    "calibration_hard_min_solution": 18, "calibration_easy_min_detour": 4,
    "calibration_medium_min_detour": 4, "calibration_hard_min_detour": 5,
    "calibration_max_solution": 32, "calibration_min_persistent_witnesses": 1,
    "calibration_min_decision_witnesses": 1, "calibration_max_attempts": 4000,
    "calibration_profile": "adjacent_capture", "calibration_persistent_profile": "adjacent_capture",
}
"""Generation options. Every generated task must expose the faulty predictor's systematic discrepancy."""
MAX_EXPANSIONS = 200_000


@dataclass(frozen=True)
class PatrolLayout:
    height: int
    width: int
    walls: frozenset[Pos]
    exit: Pos
    routes: tuple[tuple[Pos, ...], ...]
    """Each route is the cyclic sequence of cells a guard occupies, one per tick."""

    @property
    def period(self) -> int:
        return math.lcm(*(len(r) for r in self.routes)) if self.routes else 1

    def guards(self, tick: int) -> tuple[Pos, ...]:
        return tuple(r[tick % len(r)] for r in self.routes)

    def is_open(self, p: Pos) -> bool:
        return 0 <= p[0] < self.height and 0 <= p[1] < self.width and p not in self.walls


@dataclass(frozen=True)
class PatrolState:
    layout: PatrolLayout
    agent: Pos
    tick: int = 0
    """Time modulo the layout period (keeps the state space finite)."""
    status: str = "playing"


class Patrol:
    name = "patrol"
    version = "0.1.0"

    def actions(self) -> Sequence[str]:
        return tuple(DIRECTIONS)

    def parse_action(self, raw: str) -> str:
        key = ALIASES.get(raw.strip().lower(), raw.strip().lower())
        if key not in DIRECTIONS:
            raise InvalidAction(f"unknown action {raw!r}; expected one of {', '.join(DIRECTIONS)}")
        return key

    def step(self, state: PatrolState, action: str) -> Transition[PatrolState]:
        if state.status != "playing":
            return Transition(state, ("over",))
        L = state.layout
        dr, dc = DIRECTIONS[action]
        nxt = (state.agent[0] + dr, state.agent[1] + dc)
        events = ["move" if action != "wait" else "wait"]
        if not L.is_open(nxt):
            nxt, events = state.agent, ["blocked"]
        before, after = L.guards(state.tick), L.guards(state.tick + 1)
        caught = nxt in after or any(b == nxt and a == state.agent for b, a in zip(before, after))
        status = "caught" if caught else "escaped" if nxt == L.exit else "playing"
        return Transition(replace(state, agent=nxt, tick=(state.tick + 1) % L.period, status=status), (*events, status))

    def is_success(self, state: PatrolState) -> bool:
        return state.status == "escaped"

    def is_terminal(self, state: PatrolState) -> bool:
        return state.status != "playing"

    def observe(self, state: PatrolState) -> str:
        L = state.layout
        guards = L.guards(state.tick)
        rows = []
        for r in range(L.height):
            row = []
            for c in range(L.width):
                p = (r, c)
                row.append("#" if p in L.walls else "A" if p == state.agent else "G" if p in guards else "E" if p == L.exit else ".")
            rows.append("".join(row))
        status = {"playing": "in progress", "caught": "CAUGHT (task failed)", "escaped": "reached the exit"}[state.status]
        return "\n".join([
            "Legend: '#' wall, '.' floor, 'E' exit, 'A' agent, 'G' guard.",
            "Coordinates are (row, column); row 0 is the top.",
            *rows,
            f"Agent: {state.agent}",
            f"Exit: {L.exit}",
            "Guards: " + ", ".join(f"g{i + 1} {g}" for i, g in enumerate(guards)),
            f"Status: {status}",
        ])

    def validate_state(self, state: PatrolState) -> None:
        if not state.layout.is_open(state.agent):
            raise InvalidState("agent is off the floor")
        if state.status not in ("playing", "caught", "escaped"):
            raise InvalidState("unknown status")

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "deterministic": True, "fully_observed": False,
                "hidden": "guard routes", "actions": list(DIRECTIONS),
                "termination": "success at the exit; failure when caught"}


def _random_route(rng: random.Random, floor: set[Pos], length: int) -> tuple[Pos, ...] | None:
    path = [rng.choice(sorted(floor))]
    for _ in range(length - 1):
        r, c = path[-1]
        options = [p for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if p in floor and p not in path]
        if not options:
            return None
        path.append(rng.choice(options))
    return tuple(path + path[-2:0:-1])


def _naive_path(L: PatrolLayout, start: Pos) -> list[str] | None:
    """Shortest path to the exit ignoring guards."""
    prev, queue = {start: None}, deque([start])
    while queue:
        p = queue.popleft()
        if p == L.exit:
            out = []
            while prev[p]:
                p, a = prev[p]
                out.append(a)
            return out[::-1]
        for a in ("up", "down", "left", "right"):
            q = (p[0] + DIRECTIONS[a][0], p[1] + DIRECTIONS[a][1])
            if L.is_open(q) and q not in prev:
                prev[q] = (p, a)
                queue.append(q)
    return None


def _calibration_witnesses(
    game: Patrol, state: PatrolState, solution: list[str], profile_name: str, persistent_profile_name: str
) -> tuple[int, int, int]:
    """Count persistent changes and safe decision-relevant false alarms.

    The persistent profile proves that the task exposes a systematic discrepancy.
    The decision profile must change the predicted outcome of an alternative action
    while that action remains safe in the real environment and the true suffix is
    still a winning recovery.
    """
    from world_adapt.perturbations.base import PerturbedDynamics
    from world_adapt.perturbations.patrol import PROFILES

    try:
        dynamics = PerturbedDynamics(game, PROFILES[profile_name])
        persistent_dynamics = PerturbedDynamics(game, PROFILES[persistent_profile_name])
    except KeyError as exc:
        raise ValueError(f"unknown calibration profile {exc.args[0]!r}") from exc
    persistent_witnesses = decision_witnesses = recoverable = 0
    current = state
    for index, action in enumerate(solution):
        if game.is_terminal(current):
            break
        nominal = game.step(current, action)
        if persistent_dynamics.step(current, action).changed:
            persistent_witnesses += 1
        for alternative in game.actions():
            real = game.step(current, alternative)
            predicted = dynamics.step(current, alternative)
            if (real.state.status == "playing" and predicted.transition.state.status == "caught"
                    and predicted.changed):
                suffix = real.state
                for suffix_action in solution[index + 1:]:
                    suffix = game.step(suffix, suffix_action).state
                if game.is_success(suffix):
                    decision_witnesses += 1
                    recoverable += 1
                    break
        current = nominal.state
    return persistent_witnesses, decision_witnesses, recoverable


def make_levels(game: Patrol, n: int, seed: int, options: dict[str, Any]) -> list[Level]:
    o = {**OPTIONS, **options}
    if o["curriculum"] != CURRICULUM:
        raise ValueError(f"patrol curriculum must be {CURRICULUM!r}")
    if n <= 0:
        return []
    rng, levels, seen = random.Random(seed), [], set()
    max_attempts = max(int(o["calibration_max_attempts"]), n)
    tiers = ("easy", "medium", "hard")
    tier_schedule = [tiers[min(i * len(tiers) // n, len(tiers) - 1)] for i in range(n)]
    for _ in range(max_attempts):
        if len(levels) >= n:
            break
        tier = tier_schedule[len(levels)]
        minimum = int(o[f"calibration_{tier}_min_solution"])
        h = w = int(o["calibration_size"])
        interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        walls = {(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)}
        walls |= set(rng.sample(interior, round(float(o["calibration_wall_density"]) * len(interior))))
        floor = {p for p in interior if p not in walls}
        if len(floor) < 12:
            continue
        routes = [_random_route(rng, floor, rng.randint(int(o["calibration_route_min"]), int(o["calibration_route_max"])))
                  for _ in range(int(o["calibration_guards"]))]
        if any(r is None for r in routes):
            continue
        guard_cells = {p for r in routes for p in r}
        free = sorted(floor - guard_cells)
        if len(free) < 2:
            continue
        start, exit_ = rng.sample(free, 2)
        layout = PatrolLayout(h, w, frozenset(walls), exit_, tuple(routes))
        state = PatrolState(layout, start)
        if game.observe(state) in seen:
            continue
        naive = _naive_path(layout, start)
        if naive is None:
            continue
        s = state
        for a in naive:
            s = game.step(s, a).state
        if s.status == "escaped":
            continue  # guards must matter
        result = solve(game, state, MAX_EXPANSIONS)
        if result.actions is None or not minimum <= len(result.actions) <= int(o["calibration_max_solution"]):
            continue
        detour = len(result.actions) - len(naive)
        if detour < int(o[f"calibration_{tier}_min_detour"]):
            continue  # guards must force waiting or a detour
        persistent, decision, recoverable = _calibration_witnesses(
            game, state, result.actions, str(o["calibration_profile"]), str(o["calibration_persistent_profile"]),
        )
        if (persistent < int(o["calibration_min_persistent_witnesses"])
                or decision < int(o["calibration_min_decision_witnesses"]) or recoverable < 1):
            continue
        seen.add(game.observe(state))
        waits = sum(a == "wait" for a in result.actions)
        levels.append(Level(state, result.actions, {
            "solution_length": len(result.actions), "naive_length": len(naive),
            "waits": waits, "n_guards": len(routes), "period": layout.period, "seed": seed,
            "curriculum": CURRICULUM, "tier": tier, "guards": len(routes), "detour": detour,
            "prediction_witnesses": persistent, "decision_witnesses": decision,
            "recoverable_witnesses": recoverable, "witness_profile": str(o["calibration_profile"]),
            "persistent_profile": str(o["calibration_persistent_profile"])}))
    if len(levels) != n:
        raise RuntimeError(f"could only generate {len(levels)} of {n} patrol levels after {max_attempts} attempts")
    return levels
