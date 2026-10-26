from __future__ import annotations

from world_adapt.environments.registry import EnvSpec
from world_adapt.environments.switchboard.game import OPTIONS, Switchboard, make_levels

SPEC = EnvSpec(
    name="switchboard",
    make_game=lambda options: Switchboard(int({**OPTIONS, **options}["buttons"])),
    make_levels=make_levels,
    goal="Turn on every lamp on the panel.",
    prune=None,
    default_options=dict(OPTIONS),
    rules=(
        "Each numbered button toggles a fixed hidden subset of lamps (subsets may overlap). Some buttons are "
        "breakers: pressing one also permanently disables one specific other button (hidden until it happens). "
        "Pressing a disabled button does nothing but still costs the action. Success when all lamps are on; "
        "there is no terminal failure, but disabling a button that is needed can make the task unsolvable."
    ),
    why="Wiring and breaker targets are invisible; only the predictor reveals what a press does before spending budget.",
)
