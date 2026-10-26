from __future__ import annotations

from world_adapt.environments.portalmaze.game import OPTIONS, PortalMaze, make_levels
from world_adapt.environments.registry import EnvSpec

SPEC = EnvSpec(
    name="portalmaze",
    make_game=lambda options: PortalMaze(),
    make_levels=make_levels,
    goal="Reach the exit (X).",
    prune=None,
    default_options=dict(OPTIONS),
    rules=(
        "Walls block movement. A door is impassable until the agent has picked up the key of the same letter; "
        "keys are picked up by stepping on them and are never consumed. Every pad teleports the agent, on stepping "
        "onto it, to a fixed hidden destination pad (links may be one-way). Arriving on a destination pad does not "
        "trigger it; the agent must step off and back on. One pad leads one-way into a sealed pocket of two pads that "
        "link only to each other, an irreversible trap. Success on reaching the exit; no terminal failure."
    ),
    why="Portal destinations (including one-way traps) are invisible; the predictor shows where a pad leads before committing.",
)
