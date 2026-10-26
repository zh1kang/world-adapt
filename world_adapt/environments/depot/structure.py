"""Task identity for generated Depot, independent of typed object names.

The generated family has one pallet and hoist per place, one truck, and all
crates initially in pallet-rooted stacks. Enumerate place permutations and
name crates by initial stack order, then canonicalize every initial/goal fact.
This avoids factorial enumeration of crate names without ignoring goals.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re

from world_adapt.environments.depot.game import DOMAIN, DepotState, _ground


def task_fingerprint(state: DepotState) -> str:
    """Identify the task in ``problem``, not a later execution snapshot.

    This is deliberately limited to the generated family, not general PDDL
    graph isomorphism. Domain bytes and object types are part of the identity.
    """
    task, _, parsed = _ground(state.problem)
    objects = parsed.objects
    places = sorted(name for name, kind in objects.items() if kind.name in ("depot", "distributor"))
    trucks = [name for name, kind in objects.items() if kind.name == "truck"]
    crates = {name for name, kind in objects.items() if kind.name == "crate"}
    if not 2 <= len(places) <= 4 or len(trucks) != 1:
        raise ValueError("structural Depot identity requires 2-4 places and one truck")
    atoms = [fact[1:-1].split() for fact in task.initial_state]
    at = {a[1]: a[2] for a in atoms if a[0] == "at"}
    children = {a[2]: a[1] for a in atoms if a[0] == "on"}
    pallets, hoists = {}, {}
    for kind, target in (("pallet", pallets), ("hoist", hoists)):
        for name, typ in objects.items():
            if typ.name == kind:
                place = at.get(name)
                if place not in places or place in target:
                    raise ValueError("structural Depot identity requires one pallet and hoist per place")
                target[place] = name
        if set(target) != set(places):
            raise ValueError("structural Depot identity requires one pallet and hoist per place")
    stacks = {}
    visited = set()
    for place in places:
        row, cursor = [], pallets[place]
        while cursor in children:
            cursor = children[cursor]
            if cursor in visited or cursor not in crates:
                raise ValueError("structural Depot identity requires pallet-rooted crate stacks")
            visited.add(cursor)
            row.append(cursor)
        stacks[place] = row
    if visited != crates:
        raise ValueError("structural Depot identity requires all crates initially stacked")

    def facts_key(facts, names):
        atoms = [fact[1:-1].split() for fact in facts]
        return tuple(sorted((words[0], *(names.get(word, word) for word in words[1:])) for words in atoms))

    candidates = []
    for order in itertools.permutations(places):
        names = {trucks[0]: "truck0"}
        for i, place in enumerate(order):
            names.update({place: f"place{i}", pallets[place]: f"pallet{i}", hoists[place]: f"hoist{i}"})
        names.update({crate: f"crate{i}" for i, crate in enumerate(c for p in order for c in stacks[p])})
        if set(names) != set(objects):
            raise ValueError("structural Depot identity encountered unsupported object types")
        candidates.append((tuple(sorted((names[name], kind.name) for name, kind in objects.items())),
                           facts_key(task.initial_state, names), facts_key(task.goals, names)))
    payload = ("depot-typed-task-v1", hashlib.sha256(DOMAIN.read_bytes()).hexdigest(), min(candidates))
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def generation_options(options: dict, n: int) -> tuple[int, set[str]]:
    """Parse explicit bounds and prior-inventory exclusions at the option boundary."""
    maximum = options.get("max_generation_attempts", max(100, n * 100))
    if isinstance(maximum, str) and re.fullmatch(r"[1-9][0-9]*", maximum):
        maximum = int(maximum)
    if type(maximum) is not int or maximum < 1:
        raise ValueError("max_generation_attempts must be a positive integer")
    excluded = options.get("exclude_fingerprints", ())
    if isinstance(excluded, str):
        excluded = excluded.split(",") if excluded else ()
    if not isinstance(excluded, (tuple, list, set, frozenset)) or any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in excluded
    ):
        raise ValueError("exclude_fingerprints must contain lowercase SHA-256 task identities")
    return maximum, set(excluded)
