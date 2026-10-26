from world_adapt.environments.pusht.game import CURRICULUM, PushT, make_levels
from world_adapt.environments.registry import EnvSpec

SPEC = EnvSpec(
    name="pusht",
    make_game=lambda options: PushT(),
    make_levels=make_levels,
    goal="Push the T-shaped block so that it overlaps the goal pose by more than 95%.",
    default_options={"curriculum": CURRICULUM, "budget": 40},
    rules="2D rigid-body pushing (pymunk): a circular pusher under PD control pushes a T block with friction-free planar contact.",
    why="Contact dynamics (where to push to rotate and translate the T) are hard to compute mentally.",
    has_solver=False,
)
