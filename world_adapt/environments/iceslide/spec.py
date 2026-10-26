from world_adapt.environments.iceslide.game import CURRICULUM, IceSlide, make_levels
from world_adapt.environments.registry import EnvSpec

SPEC = EnvSpec(
    name="iceslide", make_game=lambda options: IceSlide(), make_levels=make_levels,
    goal="Get every crate into a socket.",
    default_options={"curriculum": CURRICULUM},
    rules=(
        "The agent slides until the next cell is a wall or crate, or it enters a type R tile. If its slide is stopped by a "
        "loose crate (and it is not on type R), the crate is launched and slides until blocked, entering type R, or "
        "entering an empty socket, where it is fixed forever."
    ),
    why="Long slides and launches are laborious to trace exactly on a text grid; the predictor gives exact outcomes.",
)
