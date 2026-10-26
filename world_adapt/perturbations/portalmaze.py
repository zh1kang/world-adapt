"""PortalMaze discrepancy atoms and the `door_and_portal_prerequisite` profile.

Pad indices follow the layout: the first `len(pads) - 2` pads are the main pads
and the last two form the sealed pocket; the trap entrance is the main pad
whose link leads into the pocket. Predicted teleports land on a pad (a valid
cell) and, as in the real game, do not retrigger on arrival.

| atom | family | type | direction | structure | motivation |
|---|---|---|---|---|---|
| door_wrong_key | prerequisite | avoidable | optimistic | systematic | Prerequisite error: a locked door is predicted to open while an unrelated key is held (state-prerequisite mismatch). |
| reverse_portal | motion | avoidable | neutral | systematic | Systematic link error: a safe pad is predicted to lead back along an incoming link instead of to its hidden destination. |
"""

from __future__ import annotations

from dataclasses import replace

from world_adapt.environments.base import Transition
from world_adapt.perturbations.base import Atom, Condition, Context, Profile

DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def _target(ctx: Context):
    d = DIRS[ctx.action]
    return (ctx.state.agent[0] + d[0], ctx.state.agent[1] + d[1])


def _is_floor(ctx: Context, pos) -> bool:
    """A cell the predicted agent may stand on: inside the maze and not a wall (`validate_state`'s rule)."""
    L = ctx.state.layout
    return L.in_bounds(pos) and pos not in L.walls


def _entered_pad(ctx: Context) -> int | None:
    """Index of the pad the real move steps onto, if it teleports."""
    if "teleport" not in ctx.nominal.events:
        return None
    return ctx.state.layout.pads.index(_target(ctx))


def _n_main(ctx: Context) -> int:
    return len(ctx.state.layout.pads) - 2


def _is_trap(ctx: Context, i: int) -> bool:
    return _n_main(ctx) >= 2 and i < _n_main(ctx) and ctx.state.layout.links[i] >= _n_main(ctx)


def _enters_reversible_pad(c: Context) -> bool:
    i = _entered_pad(c)
    return i is not None and i < _n_main(c) and not _is_trap(c, i)


locked_door = Condition(
    "blocked_by_locked_door",
    lambda c: "blocked" in c.nominal.events
    and c.state.layout.door_at.get(_target(c)) is not None
    and c.state.layout.door_at[_target(c)] not in c.state.held,
)
holds_unrelated_key = Condition("holds_unrelated_key", lambda c: bool(c.state.held))
enters_reversible_pad = Condition("enters_reversible_pad", _enters_reversible_pad)


def _land(ctx: Context, index: int) -> Transition:
    """Predicted teleport onto pad `index`; the prediction is left unchanged if that cell is not floor."""
    s = ctx.nominal.state
    pad = ctx.state.layout.pads[index]
    if not _is_floor(ctx, pad):
        return ctx.current
    return Transition(replace(s, agent=pad, status="playing"), ctx.nominal.events)


def open_door(ctx: Context) -> Transition:
    """Door predicted to open without its key. Never places the agent on a wall (a sealed door stays blocked)."""
    target = _target(ctx)
    if not _is_floor(ctx, target):
        return ctx.current
    return Transition(replace(ctx.state, agent=target), ("move",))


def reverse_portal(ctx: Context) -> Transition:
    """Predict a safe pad's incoming portal instead of its hidden destination."""
    i = _entered_pad(ctx)
    n = _n_main(ctx)
    predecessors = [j for j in range(n) if ctx.state.layout.links[j] == i]
    if not predecessors:
        return ctx.current
    return _land(ctx, predecessors[0])


ATOMS: dict[str, Atom] = {
    "door_wrong_key": Atom("door_wrong_key", "prerequisite", locked_door & holds_unrelated_key, open_door,
                           "A locked door is predicted to open while an unrelated key is held."),
    "reverse_portal": Atom("reverse_portal", "motion", enters_reversible_pad, reverse_portal,
                           "A safe pad is predicted to lead to a pad that links into it."),
}

PROFILES: dict[str, Profile] = {
    "door_and_portal_prerequisite": Profile("door_and_portal_prerequisite", "1",
                                            (ATOMS["door_wrong_key"], ATOMS["reverse_portal"])),
}
