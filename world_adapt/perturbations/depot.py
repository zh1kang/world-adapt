"""False prerequisite beliefs in Depot; real rules and task goals stay fixed."""
from world_adapt.environments.base import Transition
from world_adapt.environments.depot.game import _ground
from world_adapt.perturbations.base import Atom, Condition, Profile


def _loaded(c):
    parts = c.action.split()
    return any(f.startswith("(in ") and f.endswith(f" {parts[3]})") for f in c.state.facts)


LOAD = Condition("load", lambda c: c.action.startswith("load "))
CARGO = Condition("truck_already_loaded", _loaded)
DROP = Condition("drop", lambda c: c.action.startswith("drop "))
STACK = Condition("support_is_crate", lambda c: _ground(c.state.problem)[2].objects[c.action.split()[3]].name == "crate")

ATOMS = {
    "single_cargo": Atom("single_cargo", "prerequisite", LOAD & CARGO,
                         lambda c: Transition(c.state, ("predicted_capacity_block",)),
                         "Predictor incorrectly blocks loading a second crate into a truck."),
    "no_stacking": Atom("no_stacking", "prerequisite", DROP & STACK,
                         lambda c: Transition(c.state, ("predicted_stack_block",)),
                         "Predictor incorrectly blocks dropping a crate onto another crate."),
}
PROFILES = {"cargo_and_stacks": Profile("cargo_and_stacks", "1", tuple(ATOMS.values()))}
