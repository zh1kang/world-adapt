"""The memory protocol: which tasks a run plays, in what order, from which notebook.

A run plays `adaptation` tasks in a fixed, tier-balanced order (each starts from
the notebook the previous one left, or from nothing under reset memory), saves
the notebook, then plays every held-out task as an independent branch from a
copy of that saved notebook. Notebooks written after held-out tasks are kept
for the record and never reach another task.

Tiers come from each level's own `tier` feature, which every environment's generator sets.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from world_adapt.environments.registry import Level

TIERS = ("easy", "medium", "hard")
SEED_STREAMS = ("task_generation", "task_order", "environment", "predictor", "perturbation", "agent")


def seed_bundle(seed: int) -> dict[str, int]:
    """Independent seeds for every random stream, derived from one run seed.

    Task generation keeps the run seed itself, so a seed block regenerates the same levels as before.
    """
    bundle = {"task_generation": seed}
    for stream in SEED_STREAMS[1:]:
        bundle[stream] = int.from_bytes(hashlib.sha256(f"{seed}/{stream}".encode()).digest()[:4], "big") % 2**31
    return bundle


@dataclass(frozen=True)
class PlannedTask:
    task_id: str
    phase: str
    tier: str | None
    level_index: int
    """Index into the generated level pool."""


def task_record(task: PlannedTask) -> dict:
    return {"task_id": task.task_id, "phase": task.phase, "tier": task.tier, "level_index": task.level_index}


def assign_tiers(levels: list[Level]) -> list[str]:
    tiers = [lv.features.get("tier") for lv in levels]
    if any(tier not in TIERS for tier in tiers):
        raise ValueError(f"every level needs a 'tier' feature in {TIERS}")
    return tiers


def _interleave(groups: list[list[int]], rng: random.Random) -> list[int]:
    """Round-robin over tiers in a random tier order per round, so every prefix stays as balanced as possible."""
    out, groups = [], [list(g) for g in groups]
    while any(groups):
        live = [g for g in groups if g]
        rng.shuffle(live)
        # Larger groups first within a round keeps the remainder balanced.
        live.sort(key=len, reverse=True)
        out += [g.pop(0) for g in live]
    return out


def pool_size(adaptation: int, heldout_per_tier: int) -> int:
    """Levels to generate for the adaptation and held-out pools."""
    per_tier = heldout_per_tier + -(-adaptation // len(TIERS))
    return per_tier * len(TIERS)


def plan_protocol(levels: list[Level], adaptation: int, heldout_per_tier: int, order_seed: int) -> list[PlannedTask]:
    """Split generated levels into disjoint adaptation and held-out pools; order the adaptation tasks."""
    tiers = assign_tiers(levels)
    rng = random.Random(order_seed)
    by_tier = {t: [i for i, tier in enumerate(tiers) if tier == t] for t in TIERS}
    for t in TIERS:
        rng.shuffle(by_tier[t])
        if len(by_tier[t]) < heldout_per_tier:
            raise ValueError(f"only {len(by_tier[t])} {t} levels in the pool; need held-out tasks")
    heldout = {t: by_tier[t][:heldout_per_tier] for t in TIERS}
    rest = {t: by_tier[t][heldout_per_tier:] for t in TIERS}
    take = {t: 0 for t in TIERS}
    for _ in range(adaptation):
        options = [t for t in TIERS if take[t] < len(rest[t])]
        if not options:
            raise ValueError(f"pool too small for {adaptation} adaptation tasks")
        take[min(options, key=lambda t: (take[t], TIERS.index(t)))] += 1
    order = _interleave([rest[t][:take[t]] for t in TIERS], rng)
    adaptation_tasks = [PlannedTask(f"adapt-{k}", "adaptation", tiers[i], i) for k, i in enumerate(order)]
    heldout_tasks = [PlannedTask(f"heldout-{t}-{k}", "heldout", t, i) for t in TIERS for k, i in enumerate(heldout[t])]
    return adaptation_tasks + heldout_tasks
