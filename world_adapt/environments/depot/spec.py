from world_adapt.environments.depot.game import CURRICULUM, Depot, make_levels
from world_adapt.environments.registry import EnvSpec

SPEC = EnvSpec(name="depot", make_game=lambda o: Depot(), make_levels=make_levels,
               goal="Arrange every crate so that all goal facts hold.", has_solver=False,
               default_options={"curriculum": CURRICULUM},
               rules="Standard Depot PDDL: clear-stack lifting, one crate per hoist, load/unload at the truck's place.",
               why="Coupled stack, hoist, and transport prerequisites; predicts consequences of long resource plans.")
