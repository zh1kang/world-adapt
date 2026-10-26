"""Shared pieces of the study analysis: condition and contrast names, the seed-block bootstrap and formatting.

Intervals are 95% percentile bootstraps over seed blocks. Each interval draws from its own generator seeded by a
string key (bootstrap seed, environment, quantity), so a cell's interval does not depend on which other
environments, conditions or models are analysed alongside it.
"""

from __future__ import annotations

import random
from typing import Any

from world_adapt.runner.study import CONDITIONS, condition_name

CONDS = tuple(CONDITIONS)
"""Condition slugs in reporting order."""
TIERS = ("easy", "medium", "hard")
PAIRS = {
    "faulty-predictor-memory-benefit": ("faulty-predictor-saved-notebook", "faulty-predictor-reset-notebook"),
    "accurate-predictor-memory-benefit": ("accurate-predictor-saved-notebook", "accurate-predictor-reset-notebook"),
    "predictor-error-cost-saved-notebook": ("accurate-predictor-saved-notebook", "faulty-predictor-saved-notebook"),
    "predictor-error-cost-reset-notebook": ("accurate-predictor-reset-notebook", "faulty-predictor-reset-notebook"),
}
"""Paired contrast slug -> (minuend condition, subtrahend condition), within one seed block."""
INTERACTION = "memory-benefit-difference"
"""Faulty-predictor memory benefit minus accurate-predictor memory benefit."""
DIFFS = ("faulty-predictor-memory-benefit", "accurate-predictor-memory-benefit", INTERACTION,
         "predictor-error-cost-saved-notebook", "predictor-error-cost-reset-notebook")
"""Paired differences in reporting order."""
DIFFERENCE_LABELS = {
    "faulty-predictor-memory-benefit": "Faulty predictor: saved minus reset notebook",
    "accurate-predictor-memory-benefit": "Accurate predictor: saved minus reset notebook",
    INTERACTION: "Memory benefit difference (faulty minus accurate predictor)",
    "predictor-error-cost-saved-notebook": "Saved notebook: accurate minus faulty predictor",
    "predictor-error-cost-reset-notebook": "Reset notebook: accurate minus faulty predictor",
}


def condition_label(slug: str) -> str:
    """Human-readable condition label for reports and exports."""
    return condition_name(slug)


def difference_label(slug: str) -> str:
    """Human-readable label of a paired contrast."""
    return DIFFERENCE_LABELS[slug]


BOOT = 10_000


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _percentiles(draws: list[float]) -> tuple[float, float]:
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]


def block_ci(values_by_block: dict[int, float], key: str, boot: int = BOOT) -> tuple[float, float] | None:
    """Percentile interval of the block mean; None with fewer than two blocks."""
    blocks = list(values_by_block.values())
    if len(blocks) < 2:
        return None
    rng = random.Random(key)
    return _percentiles([sum(rng.choices(blocks, k=len(blocks))) / len(blocks) for _ in range(boot)])


def stratified_ci(values: dict[str, dict[int, dict[str, float]]], quantities: tuple[str, ...], key: str,
                  boot: int = BOOT) -> dict[str, tuple[float, float] | None]:
    """Intervals of the equally weighted mean over environments of per-environment block means.

    `values[env][block][quantity]` must hold every quantity for every block. Each draw resamples blocks within
    each environment (never across environments) and evaluates every quantity on the same resampled blocks, so
    the pairing of conditions within a seed block is preserved. None when no environment has two blocks.
    """
    envs = [e for e in values if values[e]]
    if not envs or max(len(values[e]) for e in envs) < 2:
        return {q: None for q in quantities}
    rng = random.Random(key)
    blocks = {e: [values[e][b] for b in sorted(values[e])] for e in envs}
    draws: dict[str, list[float]] = {q: [] for q in quantities}
    for _ in range(boot):
        totals = dict.fromkeys(quantities, 0.0)
        for e in envs:
            sample = rng.choices(blocks[e], k=len(blocks[e]))
            for q in quantities:
                totals[q] += sum(b[q] for b in sample) / len(sample)
        for q in quantities:
            draws[q].append(totals[q] / len(envs))
    return {q: _percentiles(draws[q]) for q in quantities}


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def pp(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:+.1f} pp"


def with_ci(x: float | None, ci: tuple[float, float] | None, fmt) -> str:
    if x is None:
        return "n/a"
    return fmt(x) if ci is None else f"{fmt(x)} [{fmt(ci[0])}, {fmt(ci[1])}]"


def usage_cell(u: dict[str, Any]) -> str:
    """Mean predict calls / zero-predict share / prediction-budget share."""
    if not u["episodes"]:
        return "n/a"
    return f"{u['mean_predict_calls']:.1f} / {pct(u['zero_predict_share'])} / {pct(u['prediction_budget_share'])}"


def exposure_cell(x: dict[str, Any]) -> str:
    """Share of episodes with a discrepant prediction / share of predictions that were discrepant."""
    if not x["episodes"]:
        return "n/a"
    return f"{pct(x['exposed_episode_share'])} / {pct(x['exposed_prediction_share'])}"


def tokens(x: float | None) -> str:
    return "unknown" if x is None else f"{x:,.0f}"

