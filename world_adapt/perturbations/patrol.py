"""Patrol discrepancy atom and the `adjacent_capture` profile.

The real game captures the agent only when it ends a step on a guard's cell or swaps cells with a guard.

| atom             | family  | type      | direction   | structure  | motivation |
|------------------|---------|-----------|-------------|------------|------------|
| adjacent_capture | contact | avoidable | pessimistic | systematic | False alarm from an over-wide contact radius: safe passes look fatal (Dixon et al. 2007 [5]); a mis-sized collision shape is a standard physics-engine fault (Peng et al. 2018 [11]). |
"""

from __future__ import annotations

from dataclasses import replace

from world_adapt.environments.base import Transition
from world_adapt.perturbations.base import Atom, Condition, Context, Profile

ends_next_to_guard = Condition(
    "agent_ends_orthogonally_adjacent_to_guard",
    lambda c: c.current.state.status == "playing"
    and any(
        abs(g[0] - c.current.state.agent[0]) + abs(g[1] - c.current.state.agent[1]) == 1
        for g in c.state.layout.guards(c.state.tick + 1)
    ),
)


def adjacent_is_capture(ctx: Context) -> Transition:
    t = ctx.current
    return Transition(replace(t.state, status="caught"), t.events[:-1] + ("caught",))


ATOMS: dict[str, Atom] = {
    "adjacent_capture": Atom(
        "adjacent_capture",
        "contact",
        ends_next_to_guard,
        adjacent_is_capture,
        "Ending a step orthogonally adjacent to a guard is predicted as a capture (the real rule needs the same cell).",
    ),
}

PROFILES: dict[str, Profile] = {"adjacent_capture": Profile("adjacent_capture", "1", (ATOMS["adjacent_capture"],))}
