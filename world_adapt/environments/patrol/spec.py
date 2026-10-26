from world_adapt.environments.patrol.game import OPTIONS, Patrol, make_levels
from world_adapt.environments.registry import EnvSpec

SPEC = EnvSpec(
    name="patrol", make_game=lambda options: Patrol(), make_levels=make_levels,
    goal="Reach the exit cell without being caught.",
    default_options=dict(OPTIONS),
    rules=(
        "One cell per action or wait. After the agent moves, each guard advances one step along a fixed hidden "
        "back-and-forth route. Ending on a guard's cell or swapping with a guard is capture (failure)."
    ),
    why="Guard routes are hidden; the predictor shows where guards will be, which decides whether a path is safe.",
)
