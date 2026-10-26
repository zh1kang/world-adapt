"""PortalMaze: reach the exit through a maze with keys, doors and hidden portals.

Doors are impassable without the matching key; keys are picked up by stepping
on them and never consumed. Every portal pad looks the same. Stepping onto a
pad teleports the agent to that pad's hidden destination pad (arriving does not
trigger the destination; step off and back on to teleport again). Some links
are one-way and one leads into a sealed pocket, an irreversible trap.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Sequence

from world_adapt.environments.base import InvalidAction, InvalidState, Transition
from world_adapt.environments.registry import Level
from world_adapt.environments.search import solve

Pos = tuple[int, int]
DIRECTIONS: dict[str, Pos] = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
ALIASES = {"u": "up", "d": "down", "l": "left", "r": "right"}
CURRICULUM = "extended"
OPTIONS = {"size": 9, "pads": 6, "colors": 2, "wall_density": 0.12, "min_solution": 14, "max_expansions": 200_000,
           "curriculum": CURRICULUM}
"""Generation options. Each tier in `EXTENDED_TIERS` overrides the size, pad, color and length bounds."""
COLOR_NAMES = "abcd"

# These bounds describe observable structure in generated levels: maze size, prerequisite count, portal use,
# and one-way decisions all increase with the tier.
EXTENDED_TIERS: dict[str, dict[str, int]] = {
    "easy": {"size": 9, "pads": 6, "colors": 2, "min_solution": 14, "reference_max": 28,
             "keys_min": 2, "teleports_min": 1, "one_way_min": 1},
    "medium": {"size": 11, "pads": 7, "colors": 3, "min_solution": 18, "reference_max": 45,
                "keys_min": 3, "teleports_min": 2, "one_way_min": 2},
    "hard": {"size": 13, "pads": 8, "colors": 3, "min_solution": 18, "reference_max": 70,
             "keys_min": 3, "teleports_min": 2, "one_way_min": 2},
}
_CURRICULUM_TIERS = tuple(EXTENDED_TIERS)


@dataclass(frozen=True)
class MazeLayout:
    height: int
    width: int
    walls: frozenset[Pos]
    exit: Pos
    keys: tuple[tuple[Pos, str], ...]
    doors: tuple[tuple[Pos, str], ...]
    pads: tuple[Pos, ...]
    links: tuple[int, ...]
    """links[i] is the index of the pad that pad i teleports to (hidden from the agent)."""

    def in_bounds(self, p: Pos) -> bool:
        return 0 <= p[0] < self.height and 0 <= p[1] < self.width

    @property
    def door_at(self) -> dict[Pos, str]:
        return dict(self.doors)

    @property
    def key_at(self) -> dict[Pos, str]:
        return dict(self.keys)


@dataclass(frozen=True)
class MazeState:
    layout: MazeLayout
    agent: Pos
    held: frozenset[str] = frozenset()
    status: str = "playing"


class PortalMaze:
    name = "portalmaze"
    version = "0.1.0"

    def actions(self) -> Sequence[str]:
        return tuple(DIRECTIONS)

    def parse_action(self, raw: str) -> str:
        key = ALIASES.get(raw.strip().lower(), raw.strip().lower())
        if key not in DIRECTIONS:
            raise InvalidAction(f"unknown action {raw!r}; expected one of {', '.join(DIRECTIONS)}")
        return key

    def step(self, state: MazeState, action: str) -> Transition[MazeState]:
        if state.status != "playing":
            return Transition(state, ("over",))
        L = state.layout
        nxt = (state.agent[0] + DIRECTIONS[action][0], state.agent[1] + DIRECTIONS[action][1])
        door = L.door_at.get(nxt)
        if not L.in_bounds(nxt) or nxt in L.walls or (door is not None and door not in state.held):
            return Transition(state, ("blocked",))
        events, held = ["move"], state.held
        key = L.key_at.get(nxt)
        if key is not None and key not in held:
            held = held | {key}
            events.append("key")
        if nxt in L.pads:
            nxt = L.pads[L.links[L.pads.index(nxt)]]
            events.append("teleport")
        status = "escaped" if nxt == L.exit else "playing"
        return Transition(replace(state, agent=nxt, held=held, status=status), tuple(events))

    def is_success(self, state: MazeState) -> bool:
        return state.status == "escaped"

    def is_terminal(self, state: MazeState) -> bool:
        return state.status != "playing"

    def observe(self, state: MazeState) -> str:
        L = state.layout
        keys = [(p, c) for p, c in L.keys if c not in state.held]
        marks: dict[Pos, str] = {p: "o" for p in L.pads}
        marks.update({p: c.upper() for p, c in L.doors})
        marks.update({p: c for p, c in keys})
        marks[L.exit] = "X"
        marks[state.agent] = "@"
        rows = ["".join("#" if (r, c) in L.walls else marks.get((r, c), ".") for c in range(L.width)) for r in range(L.height)]
        fmt = lambda items, noun: "; ".join(f"{noun} {c} at {p}" for p, c in sorted(items)) or "none"
        status = "reached the exit" if state.status == "escaped" else "in progress"
        return "\n".join([
            "Legend: '#' wall, '.' floor, '@' agent, 'X' exit, lowercase letter = key, uppercase letter = door, 'o' pad.",
            "Coordinates are (row, column); row 0 is the top.",
            *rows,
            f"Agent: {state.agent}",
            f"Exit: {L.exit}",
            f"Keys: {fmt(keys, 'key')}",
            f"Doors: {fmt(L.doors, 'door')}",
            f"Pads: {sorted(L.pads)}",
            f"Keys held: {', '.join(sorted(state.held)) or 'none'}",
            f"Status: {status}",
        ])

    def validate_state(self, state: MazeState) -> None:
        L = state.layout
        if not L.in_bounds(state.agent) or state.agent in L.walls:
            raise InvalidState("agent is off the floor")
        if not state.held <= {c for _, c in L.keys}:
            raise InvalidState("holding a key that does not exist")
        if state.status not in ("playing", "escaped"):
            raise InvalidState("unknown status")

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "deterministic": True, "fully_observed": False,
                "hidden": "portal destinations", "actions": list(DIRECTIONS),
                "termination": "success at the exit; no terminal failure (traps make the task unsolvable)"}


def naive_path(L: MazeLayout, start: Pos) -> tuple[str, ...] | None:
    """Shortest path treating pads as plain floor but respecting keys and doors."""
    key_at, door_at = L.key_at, L.door_at
    init = (start, frozenset({key_at[start]}) if start in key_at else frozenset())
    prev, queue = {init: None}, deque([init])
    while queue:
        node = queue.popleft()
        p, held = node
        if p == L.exit:
            out = []
            while prev[node]:
                node, a = prev[node]
                out.append(a)
            return tuple(reversed(out))
        for a, (dr, dc) in DIRECTIONS.items():
            q = (p[0] + dr, p[1] + dc)
            if not L.in_bounds(q) or q in L.walls or (q in door_at and door_at[q] not in held):
                continue
            nxt = (q, held | {key_at[q]} if q in key_at else held)
            if nxt not in prev:
                prev[nxt] = (node, a)
                queue.append(nxt)
    return None


def _build(rng: random.Random, o: dict[str, Any]) -> tuple[MazeLayout, Pos] | None:
    h = w = int(o["size"])
    n_colors, n_main = int(o["colors"]), int(o["pads"]) - 2
    walls = {(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)}
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    walls |= set(rng.sample(interior, round(o["wall_density"] * len(interior))))
    # Sealed two-pad pocket that only a one-way link can enter.
    horizontal = rng.random() < 0.5
    pr, pc = rng.randint(1, h - 3), rng.randint(1, w - 3)
    pocket = [(pr, pc), (pr, pc + 1)] if horizontal else [(pr, pc), (pr + 1, pc)]
    pocket_ring = {(p[0] + dr, p[1] + dc) for p in pocket for dr, dc in DIRECTIONS.values()} - set(pocket)
    walls |= pocket_ring
    walls -= set(pocket)
    # Full divider walls; each gap holds a door of one color. A gap must not sit in the pocket's ring: the ring
    # stays solid so the pocket is reachable only through its one-way link, and a door that is also a wall would be
    # listed to the agent yet impassable.
    cut = rng.randint(3, w - 4)
    if n_colors <= 2:
        # Two-color tiers keep their original random stream so task pools stay reproducible.
        dividers = [("col", cut)] + ([("row", rng.randint(3, h - 4))] if n_colors > 1 else [])
    else:
        dividers = [("col", cut)]
        used_cuts = {cut}
        for index in range(1, n_colors):
            axis = "row" if index % 2 else "col"
            limit = h if axis == "row" else w
            choices = [k for k in range(3, limit - 3) if k not in used_cuts]
            if not choices:
                return None
            divider = rng.choice(choices)
            used_cuts.add(divider)
            dividers.append((axis, divider))
    doors: list[tuple[Pos, str]] = []
    for color, (axis, k) in zip(COLOR_NAMES, dividers):
        cells = [(r, k) if axis == "col" else (k, c) for r in range(1, h - 1) for c in range(1, w - 1) if (axis == "col" and c == k) or (axis == "row" and r == k)]
        cells = sorted(set(cells))
        placed = {d for d, _ in doors}
        free = [p for p in cells if p not in pocket and p not in pocket_ring and p not in placed]
        if not free:
            return None
        gap = rng.choice(free)
        for p in cells:
            if p not in pocket and p != gap and p not in placed:
                walls.add(p)
        doors.append((gap, color))
    if any(d in walls for d, _ in doors):
        return None
    floor = [p for p in interior if p not in walls and p not in pocket and p not in {d for d, _ in doors}]
    need = 2 + n_main + n_colors
    if len(floor) < need + 4:
        return None
    left = [p for p in floor if p[1] < cut]
    right = [p for p in floor if p[1] > cut]
    if len(left) < 3 or len(right) < 3:
        return None
    start = rng.choice(left)
    exit_ = rng.choice(right)
    rest = [p for p in floor if p not in (start, exit_)]
    rng.shuffle(rest)
    keys = tuple((rest.pop(), c) for c in COLOR_NAMES[:n_colors])
    main_pads = [rest.pop() for _ in range(n_main)]
    pads = tuple(main_pads) + tuple(pocket)
    links = [0] * len(pads)
    links[0] = len(main_pads)  # pad 0 is the trap entrance: one-way into the pocket
    for i in range(1, n_main):
        links[i] = rng.choice([j for j in range(n_main) if j != i])
    links[n_main], links[n_main + 1] = n_main + 1, n_main
    layout = MazeLayout(h, w, frozenset(walls), exit_, keys, tuple(doors), pads, tuple(links))
    return layout, start


def _tier(options: dict[str, Any], level_index: int) -> str:
    tier = options.get("tier", "mixed")
    if tier not in (*_CURRICULUM_TIERS, "mixed"):
        raise ValueError("portalmaze tier must be one of easy, medium, hard, or mixed")
    return _CURRICULUM_TIERS[level_index % len(_CURRICULUM_TIERS)] if tier == "mixed" else tier


def _one_way_links(layout: MazeLayout) -> int:
    n_main = len(layout.pads) - 2
    return sum(
        1 for i in range(n_main)
        if layout.links[i] >= n_main or layout.links[layout.links[i]] != i
    )


def _extended_levels(game: PortalMaze, n: int, seed: int, options: dict[str, Any]) -> list[Level]:
    _tier(options, 0)  # Validate even when a caller requests an empty pool.
    rng, levels, seen = random.Random(seed), [], set()
    attempts = 0
    max_attempts = max(300, n * 3500)
    while len(levels) < n and attempts < max_attempts:
        attempts += 1
        tier = _tier(options, len(levels))
        bounds = EXTENDED_TIERS[tier]
        local = {**options, **bounds}
        built = _build(rng, local)
        if built is None:
            continue
        layout, start = built
        state = MazeState(layout, start)
        if game.observe(state) in seen:
            continue
        one_way = _one_way_links(layout)
        result = solve(game, state, min(int(options.get("max_expansions", 200_000)), 200_000))
        if result.actions is None or not bounds["min_solution"] <= len(result.actions) <= bounds["reference_max"]:
            continue
        s, events = state, []
        for action in result.actions:
            transition = game.step(s, action)
            s, events = transition.state, events + list(transition.events)
        if not game.is_success(s) or events.count("key") < bounds["keys_min"]:
            continue
        if events.count("teleport") < bounds["teleports_min"] or one_way < bounds["one_way_min"]:
            continue
        naive = naive_path(layout, start)
        naive_state, hits_trap = state, False
        for action in naive or ():
            naive_state = game.step(naive_state, action).state
            hits_trap = hits_trap or naive_state.agent in layout.pads[-2:]
        if naive is not None and game.is_success(naive_state):
            continue
        seen.add(game.observe(state))
        levels.append(Level(state, result.actions, {
            "curriculum": "extended", "tier": tier, "solution_length": len(result.actions),
            "reference_range": (bounds["min_solution"], bounds["reference_max"]),
            "keys_used": events.count("key"), "teleports_used": events.count("teleport"),
            "naive_length": None if naive is None else len(naive), "naive_fails": True,
            "naive_hits_trap": hits_trap, "n_pads": len(layout.pads), "n_main_pads": len(layout.pads) - 2,
            "n_doors": len(layout.doors), "n_keys": len(layout.keys), "one_way_links": one_way, "seed": seed,
        }))
    if len(levels) != n:
        raise RuntimeError(f"could not generate {n} portalmaze {options.get('tier', 'mixed')} levels after {attempts} attempts")
    return levels


def make_levels(game: PortalMaze, n: int, seed: int, options: dict[str, Any]) -> list[Level]:
    o = {**OPTIONS, **options}
    if o["curriculum"] != CURRICULUM:
        raise ValueError(f"portalmaze curriculum must be {CURRICULUM!r}")
    return _extended_levels(game, n, seed, o)
