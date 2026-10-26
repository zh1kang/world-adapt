"""Standard Depot PDDL executed by Pyperplan; no planner or rules reach the agent."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
from pathlib import Path
import random

from world_adapt.environments.base import InvalidAction, InvalidState, Transition
from world_adapt.environments.registry import Level

DOMAIN = Path(__file__).with_name("domain.pddl")
CURRICULUM = "structural-v2"
"""Generated stack-and-transport tasks, deduplicated by structural task fingerprint."""
ARITIES = {"drive": 3, "lift": 4, "drop": 4, "load": 4, "unload": 4}


@dataclass(frozen=True)
class DepotState:
    problem: str
    facts: frozenset[str]


@lru_cache(maxsize=64)
def _ground(problem: str):
    try:
        from pyperplan.grounding import ground
        from pyperplan.pddl.parser import Parser
    except ImportError as error:
        raise RuntimeError("Install world_adapt[depot] to use the native PDDL adapter") from error
    parser = Parser(str(DOMAIN))
    domain = parser.parse_domain()
    parser.probInput = problem
    parsed = parser.parse_problem(domain, read_from_file=False)
    task = ground(parsed, remove_statics_from_initial_state=False, remove_irrelevant_operators=False)
    return task, {op.name[1:-1]: op for op in task.operators}, parsed


class Depot:
    name = "depot"
    version = "1.0.0"

    def prompt_notes(self, max_chunk: int) -> str:
        """Text appended to the system prompt."""
        return ("\nDepot: the listed actions are examples, not an exhaustive list of bindings. "
                "Use object names from the current observation with this syntax: " + self.action_syntax() + ".\n")

    def actions(self):
        # Examples are valid in every generated object set; the tool schema accepts other bindings.
        return ("drive t0 l0 l1", "lift h0 c0 p0 l0", "drop h0 c0 p0 l0",
                "load h0 c0 t0 l0", "unload h0 c0 t0 l0")

    @staticmethod
    def action_syntax():
        return ("drive <truck> <from> <to>; lift <hoist> <crate> <support> <place>; "
                "drop <hoist> <crate> <support> <place>; load <hoist> <crate> <truck> <place>; "
                "unload <hoist> <crate> <truck> <place>")

    def parse_action(self, raw):
        if not isinstance(raw, str):
            raise InvalidAction("Depot actions must be strings")
        parts = raw.lower().split()
        if not parts or parts[0] not in ARITIES or len(parts) != ARITIES[parts[0]] + 1:
            raise InvalidAction("expected " + self.action_syntax())
        return " ".join(parts)

    def action_item_schema(self):
        return {"type": "string", "description": self.action_syntax() + ". Use object names in the observation."}

    def step(self, state, action):
        action = self.parse_action(action)
        _, operators, _ = _ground(state.problem)
        if action not in operators:
            raise InvalidAction("action arguments do not match the task's objects or fixed locations")
        if self.is_terminal(state):
            return Transition(state, ("over",))
        op = operators[action]
        if not op.applicable(state.facts):
            return Transition(state, ("blocked",))
        nxt = replace(state, facts=frozenset(op.apply(state.facts)))
        self.validate_state(nxt)
        return Transition(nxt, (action.split()[0],))

    def is_success(self, state):
        return _ground(state.problem)[0].goal_reached(state.facts)

    def is_terminal(self, state):
        return self.is_success(state)

    def observe(self, state):
        task, _, parsed = _ground(state.problem)
        objects = ", ".join(f"{name}:{kind.name}" for name, kind in sorted(parsed.objects.items()))
        return (f"Depot. Objects: {objects}\nCurrent facts:\n" + "\n".join(sorted(state.facts))
                + "\nGoal facts:\n" + "\n".join(sorted(task.goals)))

    def validate_state(self, state):
        task, _, parsed = _ground(state.problem)
        # Static facts are not in task.facts, so include the native initial state.
        if not state.facts <= (task.facts | task.initial_state):
            raise InvalidState("unknown Depot facts")
        atoms = [f[1:-1].split() for f in state.facts]
        for relation in ("at", "on", "in", "lifting"):
            keys = [a[1] for a in atoms if a[0] == relation]
            if len(set(keys)) != len(keys):
                raise InvalidState(f"duplicate {relation} ownership")
        at = {a[1]: a[2] for a in atoms if a[0] == "at"}
        fixed = {k for k, t in parsed.objects.items() if t.name in ("truck", "hoist", "pallet")}
        if not fixed <= at.keys():
            raise InvalidState("every truck, hoist, and pallet needs exactly one place")
        for obj in fixed:
            if parsed.objects[obj].name in ("hoist", "pallet") and f"(at {obj} {at[obj]})" not in task.initial_state:
                raise InvalidState("hoists and pallets cannot change places")
        on = {a[1]: a[2] for a in atoms if a[0] == "on"}
        held = {a[2]: a[1] for a in atoms if a[0] == "lifting"}
        loaded = {a[1]: a[2] for a in atoms if a[0] == "in"}
        crates = {k for k, t in parsed.objects.items() if t.name == "crate"}
        hoists = {k for k, t in parsed.objects.items() if t.name == "hoist"}
        surfaces = crates | {k for k, t in parsed.objects.items() if t.name == "pallet"}
        for crate in crates:
            if sum(crate in x for x in (on, held, loaded)) != 1:
                raise InvalidState("each crate needs exactly one support, hoist, or truck")
            if (crate in at) != (crate in on):
                raise InvalidState("only supported crates have a place")
            if crate in on and at.get(crate) != at.get(on[crate]):
                raise InvalidState("crate and support must share a place")
            seen, cursor = set(), crate
            while cursor in on:
                if cursor in seen:
                    raise InvalidState("cyclic stack")
                seen.add(cursor)
                cursor = on[cursor]
        if len(set(on.values())) != len(on):
            raise InvalidState("a support holds at most one crate")
        for h in hoists:
            count = sum(a[0] == "lifting" and a[1] == h for a in atoms)
            if count > 1 or ((f"(available {h})" in state.facts) != (count == 0)):
                raise InvalidState("hoist availability disagrees with cargo")
        for surface in surfaces:
            clear = surface in at and surface not in on.values()
            if (f"(clear {surface})" in state.facts) != clear:
                raise InvalidState("clear status disagrees with stack")

    def spec(self):
        return {"name": self.name, "version": self.version, "deterministic": True,
                "fully_observed": True, "actions": list(self.actions()),
                "domain_sha256": hashlib.sha256(DOMAIN.read_bytes()).hexdigest(),
                "engine": "pyperplan 2.1 native STRIPS operators", "termination": "all goal facts true",
                "limitations": "Standard Depot: no fuel limit or truck capacity. Generated tasks; witness plans are not shortest plans."}


def _instance(rng, tier):
    rank = ("easy", "medium", "hard").index(tier)
    places, crates = rank + 2, 5 + rank * 3
    names = [f"c{i}" for i in range(crates)]
    rng.shuffle(names)
    initial = [[] for _ in range(places)]
    target = [[] for _ in range(places)]
    for i, c in enumerate(names):
        initial[i % 2].append(c)
    rng.shuffle(names)
    # Goal stacks use a different random order and distribution.
    for i, c in enumerate(names):
        target[i % places].append(c)
    truck_at = rng.randrange(places)
    facts = {f"(at t0 l{truck_at})"}
    goals = set()
    for p in range(places):
        facts |= {f"(at h{p} l{p})", f"(available h{p})", f"(at p{p} l{p})"}
        support = f"p{p}"
        for c in initial[p]:
            facts |= {f"(at {c} l{p})", f"(on {c} {support})"}
            support = c
        facts.add(f"(clear {support})")
        support = f"p{p}"
        for c in target[p]:
            goals.add(f"(on {c} {support})")
            support = c
    objects = (" ".join(f"l{i}" for i in range(places)) + " - depot\n"
               + " ".join(f"p{i}" for i in range(places)) + " - pallet\n"
               + " ".join(f"h{i}" for i in range(places)) + " - hoist\n"
               + " ".join(f"c{i}" for i in range(crates)) + " - crate\nt0 - truck")
    problem = (f"(define (problem generated) (:domain depot) (:objects {objects}) "
               f"(:init {' '.join(sorted(facts))}) (:goal (and {' '.join(sorted(goals))})))")
    plan = []
    for p in range(places):
        if truck_at != p:
            plan.append(f"drive t0 l{truck_at} l{p}")
            truck_at = p
        for i in reversed(range(len(initial[p]))):
            c = initial[p][i]
            support = initial[p][i-1] if i else f"p{p}"
            plan += [f"lift h{p} {c} {support} l{p}", f"load h{p} {c} t0 l{p}"]
    for p in range(places):
        if truck_at != p:
            plan.append(f"drive t0 l{truck_at} l{p}")
            truck_at = p
        support = f"p{p}"
        for c in target[p]:
            plan += [f"unload h{p} {c} t0 l{p}", f"drop h{p} {c} {support} l{p}"]
            support = c
    return DepotState(problem, frozenset(facts)), plan, places, crates


def make_levels(game, n, seed, options):
    if n < 0:
        raise ValueError("n must be nonnegative")
    curriculum = options.get("curriculum", CURRICULUM)
    if curriculum != CURRICULUM:
        raise ValueError(f"Depot curriculum must be {CURRICULUM}")
    from world_adapt.environments.depot.structure import generation_options, task_fingerprint
    maximum, excluded = generation_options(options, n)
    rng, levels, seen = random.Random(seed), [], set(excluded)
    duplicates = 0
    for attempt in range(maximum):
        if len(levels) == n:
            return levels
        tier = ("easy", "medium", "hard")[len(levels) % 3]
        state, plan, places, crates = _instance(rng, tier)
        identity = task_fingerprint(state)
        if identity in seen or game.is_success(state):
            duplicates += 1
            continue
        game.validate_state(state)
        cursor, witness = state, []
        for action in plan:
            cursor = game.step(cursor, action).state
            witness.append(action)
            if game.is_success(cursor):
                break
        if not game.is_success(cursor):
            raise RuntimeError("Depot generator produced an invalid reference plan")
        seen.add(identity)
        features = {"tier": tier, "solution_length": len(witness),
                    "reference_verified": True, "reference_optimal": False,
                    "places": places, "crates": crates, "max_initial_stack": (crates + 1) // 2,
                    "task_family": "depot-stack-transport-v1", "curriculum": curriculum, "task_fingerprint": identity,
                    "generation_attempt": attempt + 1, "duplicate_rejections": duplicates}
        levels.append(Level(state, tuple(witness), features))
    if len(levels) == n:
        return levels
    raise ValueError(f"could not produce {n} distinct Depot tasks after {maximum} attempts "
                     f"({len(levels)} admitted, {duplicates} duplicate/solved rejections)")
