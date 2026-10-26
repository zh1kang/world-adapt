"""Switchboard discrepancy atom and the `breaker_retarget_enabled_only` profile.

The atom computes the step under a modified copy of the hidden wiring and returns the original wiring, so
predicted states look normal and later predicted steps stay consistent.

| atom | family | type | direction | structure | motivation |
|---|---|---|---|---|---|
| breaker_retarget_enabled | prerequisite | avoidable | mixed | context-cued | A wrong irreversible side effect after a context cue (Gershman 2010): once any button is disabled, a breaker is predicted to disable a different, still enabled button. |
"""

from __future__ import annotations

from dataclasses import replace

from world_adapt.environments.base import Transition
from world_adapt.perturbations.base import Atom, Condition, Context, Profile


def _button(ctx: Context) -> int:
    return int(ctx.action.split()[1]) - 1


def _enabled(ctx: Context) -> bool:
    return not ctx.state.disabled >> _button(ctx) & 1


is_breaker = Condition("is_breaker", lambda c: c.state.wiring.breaks[_button(c)] >= 0 and _enabled(c))
some_disabled = Condition("some_button_disabled", lambda c: c.state.disabled != 0)


def retarget_enabled_breaker(ctx: Context) -> Transition:
    """Predict a different enabled target after a breaker context cue."""
    w, i = ctx.state.wiring, _button(ctx)
    target = w.breaks[i]
    choices = [
        j for j in range(len(w.breaks))
        if j not in (i, target) and not (ctx.state.disabled >> j & 1)
    ]
    if not choices:
        return ctx.current
    alternate = choices[0]
    breaks = w.breaks[:i] + (alternate,) + w.breaks[i + 1:]
    predicted = ctx.game.step(replace(ctx.state, wiring=replace(w, breaks=breaks)), ctx.action)
    return Transition(replace(predicted.state, wiring=w), predicted.events)


ATOMS: dict[str, Atom] = {
    "breaker_retarget_enabled": Atom(
        "breaker_retarget_enabled", "prerequisite", is_breaker & some_disabled, retarget_enabled_breaker,
        "After a context cue, a breaker is predicted to disable a different enabled button."),
}

PROFILES: dict[str, Profile] = {
    "breaker_retarget_enabled_only": Profile("breaker_retarget_enabled_only", "1",
                                             (ATOMS["breaker_retarget_enabled"],)),
}
