"""Controlled PushT prediction errors.

PushT uses the native simulator for both controls.  ``target_gain`` changes
only the requested future target displacement in the predictor's private
replay schedule; the real environment and its action history stay nominal.
"""

from __future__ import annotations

from world_adapt.perturbations.base import Atom, Condition, Context, Profile


def _live(ctx: Context) -> bool:
    return not ctx.game.is_terminal(ctx.state)


def _target_gain(ctx: Context):
    target = ctx.game.transformed_target(ctx.state, ctx.action, gain=0.8)
    return ctx.game.simulate(ctx.state, ctx.action, target=target)


target_gain = Atom(
    "target_gain",
    "motion",
    Condition("active", _live),
    _target_gain,
    "The predictor moves the pusher 80% of the requested target displacement.",
)


ATOMS = {"target_gain": target_gain}
PROFILES = {"target_gain": Profile("target_gain", "1", (target_gain,))}
