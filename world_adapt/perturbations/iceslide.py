"""IceSlide discrepancy atoms and the `launch_outcomes` profile.

Everything here changes only the predicted transition; predicted states keep
the true layout, so `observe()` looks normal and `validate_state` holds. A
launched crate ends in exactly one of three ways (obstacle, type-R tile, empty
socket); three atoms key on that outcome, so they never overlap.

| atom                   | family         | type        | direction                     | structure    | motivation |
|------------------------|----------------|-------------|-------------------------------|--------------|------------|
| crate_stops_early      | motion         | avoidable   | optimistic                    | systematic   | Underestimated travel distance, the intuitive-physics error of mis-simulating sliding objects (McCloskey 1983 [9]; Battaglia et al. 2013 [10]); hides crates reaching walls and corners, which cannot be undone (a miss, Dixon & Wickens 2006 [4]). |
| crate_slides_past_rough| state         | avoidable   | mixed, measured mostly pessimistic (a crate that really stops looks like it slides on) | systematic | Mis-specified terrain property: the predictor does not know type R changes dynamics, like an unmodelled friction coefficient (dynamics randomization, Peng et al. 2018 [11]). |
| socket_vertical_no_capture | contact  | avoidable   | pessimistic (false alarm)     | systematic   | A crate dropping into a socket is not captured when moving up or down: the agent believes the goal is unreachable that way (false alarms erode reliance, Dixon et al. 2007 [5]; algorithm aversion, Dietvorst et al. 2015 [12]). |
"""

from __future__ import annotations

from dataclasses import replace

from world_adapt.environments.base import Transition
from world_adapt.environments.iceslide.game import IceState, shift
from world_adapt.perturbations.base import Atom, Condition, Context, Profile


def _launch(ctx: Context):
    """(origin, destination, dropped_in_socket) of the real launch, or None."""
    if "launch" not in ctx.nominal.events:
        return None
    before, after = ctx.state, ctx.nominal.state
    origin = next(iter(before.crates - after.crates))
    if after.fixed != before.fixed:
        return origin, next(iter(after.fixed - before.fixed)), True
    return origin, next(iter(after.crates - before.crates)), False


def _travel(origin, dest) -> int:
    return abs(origin[0] - dest[0]) + abs(origin[1] - dest[1])


# Conditions -----------------------------------------------------------------


def _stops_at_obstacle(ctx: Context) -> bool:
    launch = _launch(ctx)
    return bool(launch) and not launch[2] and launch[1] not in ctx.state.layout.rough and _travel(*launch[:2]) >= 2


crate_hits_obstacle_after_2 = Condition("crate_slides_2+_and_stops_at_obstacle", _stops_at_obstacle)

crate_stops_on_rough = Condition(
    "crate_stops_on_type_R",
    lambda c: bool(_launch(c)) and not _launch(c)[2] and _launch(c)[1] in c.state.layout.rough,
)

crate_drops_into_socket = Condition("crate_drops_into_socket", lambda c: bool(_launch(c)) and _launch(c)[2])

moves_vertically = Condition("action_is_vertical", lambda c: c.action in ("up", "down"))

# Effects --------------------------------------------------------------------


def stop_one_cell_early(ctx: Context) -> Transition:
    """The launched crate is predicted one cell short of its real stopping cell."""
    launch = _launch(ctx)
    state: IceState = ctx.current.state
    if not launch or launch[1] not in state.crates:
        return ctx.current
    back = shift(launch[1], {"up": "down", "down": "up", "left": "right", "right": "left"}[ctx.action])
    return Transition(replace(state, crates=(state.crates - {launch[1]}) | {back}), ctx.current.events)


def crate_ignores_rough(ctx: Context) -> Transition:
    """Predict the whole move on a layout without type R tiles, then restore the true layout."""
    plain = replace(ctx.state, layout=replace(ctx.state.layout, rough=frozenset()))
    t = ctx.game.step(plain, ctx.action)
    return Transition(replace(t.state, layout=ctx.state.layout), t.events)


def socket_does_not_capture(ctx: Context) -> Transition:
    """The crate stops in the socket cell but stays loose."""
    launch = _launch(ctx)
    state: IceState = ctx.current.state
    if not launch or launch[1] not in state.fixed:
        return ctx.current
    dest = launch[1]
    return Transition(
        replace(state, fixed=state.fixed - {dest}, crates=state.crates | {dest}),
        tuple(e for e in ctx.current.events if e != "socket"),
    )


# Atom catalog ---------------------------------------------------------------

ATOMS: dict[str, Atom] = {
    atom.name: atom
    for atom in [
        Atom(
            "crate_stops_early",
            "motion",
            crate_hits_obstacle_after_2,
            stop_one_cell_early,
            "A launched crate that really slides 2+ cells and stops at a wall or another crate is predicted to stop one cell "
            "earlier, hiding crates that reach walls and corners.",
        ),
        Atom(
            "crate_slides_past_rough",
            "prerequisite",
            crate_stops_on_rough,
            crate_ignores_rough,
            "A launched crate that really stops on a type R tile is predicted to slide on as if the tile were plain floor.",
        ),
        Atom(
            "socket_vertical_no_capture",
            "contact",
            crate_drops_into_socket & moves_vertically,
            socket_does_not_capture,
            "A crate launched up or down into an empty socket is predicted to stop there loose instead of being fixed.",
        ),
    ]
}


PROFILES: dict[str, Profile] = {
    "launch_outcomes": Profile("launch_outcomes", "1", (ATOMS["crate_stops_early"], ATOMS["crate_slides_past_rough"],
                                                         ATOMS["socket_vertical_no_capture"])),
}
