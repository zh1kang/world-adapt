"""Aggregate a prepared study's run folders into per-environment, per-condition metrics.

    python -m world_adapt.runner.analyze --study outputs/my-study/study.json [--agent NAME] [--replay]

Writes results.json, runs.csv, episodes.csv and summary.md to `<study dir>/analysis` (or `--out`). Runs are read
from `<study dir>/runs` (or `--runs`). Headline metrics use each run's summary.json: the held-out success rate per
seed block, paired contrasts per seed block (saved minus reset notebook under each predictor, their difference,
and accurate minus faulty predictor under each notebook condition) and a percentile block bootstrap (seed blocks
resampled within an environment, 10,000 draws) for 95% intervals. The pooled core average weights environments
equally (mean of per-environment block means). Environments with a single seed block, or marked `"pooled": false`
in the study manifest, are reported separately and never pooled.

Next to success, every cell reports how the agent used the predictor: mean `predict` calls per episode, the share
of episodes with no `predict` call at all, and the share of the prediction budget spent. These separate "the
predictor did not help" from "the agent stopped asking".

Runs are checked against the study plan (missing, unfinished, incomplete, extra, superseded runs), and every
run's summary is checked against its event log (`world_adapt.runner.audit`); `--replay` also replays every run in
the real environment (`world_adapt.runner.rescore`, slow).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from world_adapt.runner import audit
from world_adapt.runner.report import (BOOT, CONDS, DIFFS, INTERACTION, PAIRS, TIERS, block_ci, condition_label,
                                       difference_label, exposure_cell, mean, pct, pp, stratified_ci, tokens,
                                       usage_cell, with_ci)
from world_adapt.runner.rescore import rescore
from world_adapt.runner.study import load_study

COUNT_KEYS = ("real_actions", "predicted_transitions", "predict_calls", "execute_calls", "turns", "malformed", "seconds")
TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def load_runs(runs: Path, prefix: str, envs: list[str] | None = None, replay: bool = False) -> list[dict[str, Any]]:
    """One record per (env, condition, seed) of the prefix, from its newest folder with a summary and a manifest:
    label parts, summary, per-task budgets, event counts, log-check mismatches and discrepancy exposure."""
    out = []
    for folder, older in audit.latest(audit.run_folders(runs, prefix, envs)).values():
        d = folder.path
        if not (d / "summary.json").exists() or not (d / "manifest.json").exists():
            continue
        summary = json.loads((d / "summary.json").read_text())
        manifest = json.loads((d / "manifest.json").read_text())
        scan = audit.scan_events(d)
        mismatches = audit.check_logs(summary, manifest, scan)
        if replay:
            mismatches += [f"replay: {m}" for m in rescore(d)["mismatches"]]
        out.append({"env": folder.env, "cond": folder.cond, "seed": folder.seed, "dir": d.name, "summary": summary,
                    "task_plan": {p["task_id"]: p for p in manifest["tasks"]["plan"]},
                    "task_hashes": manifest["tasks"]["observation_sha256"],
                    "budgets": {b["task_id"]: b for b in manifest["budgets"]["per_task"]},
                    "api_errors": scan["api_errors"], "context_trims": scan["context_trims"],
                    "reply_cutoffs": scan["reply_cutoffs"], "reply_cutoffs_all": scan["reply_cutoffs_all"],
                    "exposure": scan["exposure"], "log_mismatches": mismatches, "superseded": older,
                    "settings": manifest["settings"],
                    "missing_tasks": [{k: event[k] for k in ("task_id", "phase", "tier", "level_index", "attempt", "reason")}
                                      for event in scan["missing"].values()]})
    return sorted(out, key=lambda r: r["dir"])


def predictor_usage(episodes: list[dict[str, Any]]) -> dict[str, float | None]:
    """Mean predict calls, share of episodes with zero predict calls, share of the prediction budget spent."""
    if not episodes:
        return {"mean_predict_calls": None, "zero_predict_share": None, "prediction_budget_share": None, "episodes": 0}
    calls = [e["predict_calls"] for e in episodes]
    shares = [e["predicted_transitions"] / e["prediction_budget"] for e in episodes if e.get("prediction_budget")]
    return {"mean_predict_calls": mean(calls), "zero_predict_share": sum(c == 0 for c in calls) / len(calls),
            "prediction_budget_share": mean(shares), "episodes": len(episodes)}


def exposure(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Discrepancy exposure from the evaluator log: share of episodes with at least one prediction the faulty profile
    visibly changed, share of predictions it changed, and changed predicted steps per episode."""
    rec = [e["exposure"] for e in episodes if e.get("exposure") is not None]
    predictions = sum(x["predictions"] for x in rec)
    return {"episodes": len(rec),
            "exposed_episode_share": mean([float(x["exposed_predictions"] > 0) for x in rec]),
            "exposed_prediction_share": sum(x["exposed_predictions"] for x in rec) / predictions if predictions else None,
            "changed_steps_per_episode": mean([x["changed_steps"] for x in rec])}


def feedback(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Only executed actions that match an earlier prediction provide observable error feedback."""
    rows = [e["exposure"]["feedback"] for e in episodes if e.get("exposure") is not None]
    return {"episodes": len(rows),
            "matched_episode_share": mean([float(r["matched_prediction_queries"] > 0) for r in rows]),
            "mismatch_episode_share": mean([float(r["observed_mismatch_queries"] > 0) for r in rows]),
            "matched_steps": sum(r["matched_prediction_steps"] for r in rows),
            "mismatch_steps": sum(r["observed_mismatch_steps"] for r in rows)}


def per_episode(episodes: list[dict[str, Any]]) -> dict[str, float | None]:
    """Mean resources per episode; tokens over episodes with complete usage only."""
    known = [e for e in episodes if e["usage_known"]]
    return {**{k: mean([e.get(k) or 0 for e in episodes]) for k in COUNT_KEYS},
            **{k: mean([e[k] for e in known]) for k in TOKEN_KEYS},
            "requests": mean([e["usage"].get("planning_requests", 0) + e["usage"].get("writer_requests", 0)
                              for e in episodes]),
            "unknown_usage_episodes": len(episodes) - len(known)}


def token_totals(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Token sums over scored episodes; unknown usage is counted, never summed as zero. Tokens of attempts lost to
    infrastructure failures are not in any summary (`unrecorded_attempts`)."""
    episodes = [ep for r in runs for ep in _episodes(r)]
    known = [e for e in episodes if e["usage_known"]]
    return {"episodes": len(episodes), "unknown_usage_episodes": len(episodes) - len(known),
            **{k: sum(e[k] for e in known) if known else None for k in TOKEN_KEYS},
            "unrecorded_attempts": sum(len(r["summary"]["infrastructure_failures"]) for r in runs)}


def _episodes(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in run["summary"]["results"]:
        usage = row.get("usage", {})
        known = (all(type(usage.get(k)) is int and usage[k] >= 0 for k in TOKEN_KEYS)
                 and usage["total_tokens"] >= usage["prompt_tokens"] + usage["completion_tokens"]
                 and not usage.get("writer_unknown_requests", 0)
                 and not usage.get("writer_unknown_reserved_tokens", 0)
                 and not usage.get("planning_unknown_requests", 0)
                 and not usage.get("usage_unknown_requests", 0))
        rows.append({**row, "usage": usage,
                     "prediction_budget": run["budgets"].get(row["task_id"], {}).get("predicted_transitions"),
                     "usage_known": known, **{k: usage.get(k) if known else None for k in TOKEN_KEYS},
                     "exposure": run["exposure"].get(f"{row['task_id']}/attempt-{row['attempt']}")})
    return rows


def _cell_counts(runs: list[dict[str, Any]], episodes: list[dict[str, Any]]) -> dict[str, int]:
    return {"runs": len(runs), "episodes": len(episodes),
            "infrastructure_failures": sum(len(r["summary"]["infrastructure_failures"]) for r in runs),
            "api_errors": sum(r["api_errors"] for r in runs), "context_trims": sum(r["context_trims"] for r in runs),
            "reply_cutoffs": sum(r["reply_cutoffs"] for r in runs),
            "reply_cutoffs_all": sum(r["reply_cutoffs_all"] for r in runs),
            "malformed": sum(e["malformed"] for e in episodes)}


def _history_identity(run: dict[str, Any]) -> str:
    heldout = sorted(row["task_id"] for row in run["summary"]["results"] if row.get("phase") == "heldout")
    return json.dumps({"heldout": heldout, "plan": run["task_plan"],
                       "observations": run["task_hashes"], "budgets": run["budgets"]}, sort_keys=True)


def paired_blocks(cell: dict[tuple[str, int], dict[str, Any]], seeds: list[int]
                  ) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Per seed block, the held-out rates and every paired contrast whose arms are complete and comparable.

    Arms are compared only when both histories are valid and have identical task plans, observation hashes and
    budgets. `_complete` marks blocks in which all four conditions pair.
    """
    blocks, issues = {}, []
    identities = {key: _history_identity(run) for key, run in cell.items()}
    for seed in seeds:
        values: dict[str, Any] = {slug: run["summary"]["heldout_success_rate"]
                                  for (slug, block), run in cell.items() if block == seed}

        def matches(slugs: tuple[str, ...]) -> bool:
            if not all((slug, seed) in cell for slug in slugs):
                return False
            if any(cell[(slug, seed)]["log_mismatches"] for slug in slugs):
                issues.append({"seed": seed, "conditions": list(slugs), "reason": "invalid event/summary reconciliation"})
                return False
            if len({identities[(slug, seed)] for slug in slugs}) != 1:
                issues.append({"seed": seed, "conditions": list(slugs), "reason": "task identity or budget mismatch"})
                return False
            return True

        values["_complete"] = matches(CONDS)
        for name, (left, right) in PAIRS.items():
            if matches((left, right)):
                values[name] = values[left] - values[right]
        if values["_complete"]:
            values[INTERACTION] = values["faulty-predictor-memory-benefit"] - values["accurate-predictor-memory-benefit"]
        blocks[seed] = values
    return blocks, issues


def aggregate(runs: list[dict[str, Any]], seed: int = 0, *, excluded: dict[str, str | None] | None = None,
              boot: int = BOOT) -> dict[str, Any]:
    """Per-environment, per-condition metrics and the pooled core average.

    `excluded` maps environments to the reason they are not pooled (None when pooled), normally from the study
    manifest (`audit.exclusions`); without it an environment is pooled iff it has two or more seed blocks.
    """
    envs = sorted({r["env"] for r in runs})
    by: dict[str, dict[tuple[str, int], dict]] = defaultdict(dict)
    for r in runs:
        key = (r["cond"], r["seed"])
        if key in by[r["env"]]:
            raise ValueError(f"duplicate run identity: {r['env']}/{key}")
        by[r["env"]][key] = r
    complete_envs = [e for e in envs if any(r["summary"]["status"] == "complete" for r in by[e].values())]
    valid_runs = [r for r in runs if not r["log_mismatches"]]

    totals: Counter = Counter()
    end_reasons: Counter = Counter()
    for r in valid_runs:
        totals.update({k: r["summary"]["totals"].get(k, 0) for k in COUNT_KEYS})
        for row in r["summary"]["results"]:
            end_reasons[(r["env"], row.get("end_reason") or "none")] += 1
    all_episodes = [ep for r in runs for ep in _episodes(r)]
    missing_tasks = [{"run": r["dir"], "environment": r["env"], "condition": r["cond"], **task}
                     for r in runs for task in r["missing_tasks"]]
    results: dict[str, Any] = {
        "condition_labels": {slug: condition_label(slug) for slug in CONDS},
        "difference_labels": {slug: difference_label(slug) for slug in DIFFS},
        "counts": {**_cell_counts(runs, all_episodes), "missing_task_count": len(missing_tasks),
                   "missing_tasks": missing_tasks,
                   "incomplete_runs": [r["dir"] for r in runs if r["summary"]["status"] != "complete"],
                   "superseded_runs": sorted(d for r in runs for d in r["superseded"]),
                   "log_mismatches": {r["dir"]: r["log_mismatches"] for r in runs if r["log_mismatches"]},
                   "invalid_runs_excluded_from_metrics": [r["dir"] for r in runs if r["log_mismatches"]],
                   "episodes_by_cell": {f"{e}/{c}": sum(r["summary"]["tasks_completed"] for r in runs
                                                        if r["env"] == e and r["cond"] == c)
                                        for e in envs for c in CONDS}},
        "resources": dict(totals), "tokens": token_totals(valid_runs),
        "incomplete_environments": {e: "no complete histories" for e in envs if e not in complete_envs},
        "environments": {}, "pooled": {}, "pairing_issues": {},
        "end_reasons": {f"{e}/{reason}": n for (e, reason), n in sorted(end_reasons.items())},
    }

    complete_blocks: dict[str, dict[int, dict[str, Any]]] = {}
    for e in complete_envs:
        # Protocol metrics use complete runs only: a partial run's held-out set is not the planned one.
        cell = {k: r for k, r in by[e].items() if r["summary"]["status"] == "complete"}
        seeds = sorted({s for (_, s) in cell})
        s_by = {c: {s: cell[(c, s)]["summary"]["heldout_success_rate"] for s in seeds
                    if (c, s) in cell and not cell[(c, s)]["log_mismatches"]} for c in CONDS}
        reason = None if excluded is None else excluded.get(e, "not in the study manifest")
        if reason is None and len(seeds) < 2:
            reason = "single seed block"
        env_res: dict[str, Any] = {"blocks": len(seeds), "pooled": reason is None, "excluded": reason,
                                   "conditions": {}, "differences": {}}
        for c in CONDS:
            c_runs = [r for (cc, _), r in cell.items() if cc == c and not r["log_mismatches"]]
            episodes = [ep for r in c_runs for ep in _episodes(r)]
            heldout = [ep for ep in episodes if ep.get("phase") == "heldout"]
            summaries = [r["summary"] for r in c_runs]
            env_res["conditions"][c] = {
                "heldout_success": mean(list(s_by[c].values())), "ci95": block_ci(s_by[c], f"{seed}/{e}/{c}", boot),
                "blocks": len(s_by[c]), "per_block": s_by[c],
                "adaptation_success": mean([s["adaptation_success_rate"] for s in summaries
                                            if s["adaptation_success_rate"] is not None]),
                "heldout_by_tier": {t: mean([s["heldout_by_tier"][t] for s in summaries
                                             if s["heldout_by_tier"].get(t) is not None]) for t in TIERS},
                "predictor": predictor_usage(episodes), "predictor_heldout": predictor_usage(heldout),
                "exposure": exposure(episodes), "exposure_heldout": exposure(heldout),
                "feedback_heldout": feedback(heldout),
                "per_episode": per_episode(episodes), "counts": _cell_counts(c_runs, episodes),
                "end_reasons": dict(sorted(Counter(ep.get("end_reason") or "none" for ep in episodes).items())),
            }
        blocks, issues = paired_blocks(cell, seeds)
        if issues:
            results["pairing_issues"][e] = issues
        diffs = {name: {s: row[name] for s, row in blocks.items() if name in row} for name in DIFFS}
        env_res["differences"] = {name: {"mean": mean(list(d.values())), "ci95": block_ci(d, f"{seed}/{e}/{name}", boot),
                                         "blocks": len(d)} for name, d in diffs.items()}
        results["environments"][e] = env_res
        if reason is None:
            complete_blocks[e] = blocks
    results["pooled"] = pooled(results["environments"], complete_blocks, seed, boot, valid_runs)
    for e, reason in results["pooled"]["excluded"].items():
        results["environments"][e].update(pooled=False, excluded=reason)
    return results


def pooled(environments: dict[str, Any], blocks: dict[str, dict[int, dict[str, Any]]], seed: int,
           boot: int, runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Equal environment weights with a separate paired cohort for every quantity.

    Condition rates share the four-arm cohort; each contrast requires only its own arms. Environments need two
    complete paired blocks for each pooled quantity.
    """
    base_excluded = {e: r["excluded"] for e, r in environments.items() if r["excluded"]}
    coverage, cohorts, point, ci = {}, {}, {}, {}
    for quantity in (*CONDS, *DIFFS):
        available = {e: {s: row for s, row in rows.items()
                         if quantity in row and (quantity not in CONDS or row["_complete"])}
                     for e, rows in blocks.items()}
        cohorts[quantity] = {e: rows for e, rows in available.items() if len(rows) >= 2}
        excluded = {**base_excluded, **{e: "single complete paired seed block" if rows else
                    "no seed block with every required condition complete" for e, rows in available.items()
                    if len(rows) < 2}}
        coverage[quantity] = {"blocks": {e: len(rows) for e, rows in cohorts[quantity].items()},
                              "excluded": excluded}
        point[quantity] = mean([mean([row[quantity] for row in rows.values()])
                                for rows in cohorts[quantity].values()])
        ci[quantity] = stratified_ci(cohorts[quantity], (quantity,), f"{seed}/pooled/{quantity}", boot)[quantity]
    primary = cohorts["faulty-predictor-saved-notebook"]
    use_keys = ("mean_predict_calls", "zero_predict_share", "prediction_budget_share")

    def usage(c: str, key: str) -> dict[str, Any]:
        # Descriptive usage is not a causal reliance or compensation classification.
        cells = []
        for e, paired in cohorts[c].items():
            episodes = [ep for r in runs if r["env"] == e and r["cond"] == c and r["seed"] in paired
                        for ep in _episodes(r) if key != "predictor_heldout" or ep.get("phase") == "heldout"]
            cells.append(predictor_usage(episodes))
        return {"episodes": sum(u["episodes"] for u in cells),
                **{k: mean([u[k] for u in cells if u[k] is not None]) for k in use_keys}}

    return {"environments": sorted(primary), "excluded": coverage["faulty-predictor-saved-notebook"]["excluded"],
            "blocks": coverage["faulty-predictor-saved-notebook"]["blocks"], "quantity_coverage": coverage,
            "conditions": {c: {"heldout_success": point[c], "ci95": ci[c], "predictor": usage(c, "predictor"),
                               "predictor_heldout": usage(c, "predictor_heldout")} for c in CONDS},
            "differences": {k: {"mean": point[k], "ci95": ci[k]} for k in DIFFS}}


def _plan_lines(plan: dict[str, Any]) -> list[str]:
    md = [f"- Plan ({plan['agent']}): {plan['complete_runs']}/{plan['planned_runs']} runs complete, "
          f"{plan['episodes']}/{plan['planned_episodes']} episodes; missing {len(plan['missing'])}, unfinished "
          f"{len(plan['unfinished'])}, incomplete {len(plan['incomplete'])}, extra {len(plan['extra'])}, "
          f"superseded {len(plan['superseded'])}, settings mismatches {len(plan['mismatched'])}"]
    short = [c for c, v in plan["cells"].items() if v["episodes"] != v["planned_episodes"]]
    if short:
        md.append("- Cells short of the planned episodes: " + ", ".join(
            f"{c.split('/')[0]} / {condition_label(c.split('/')[1])} "
            f"{plan['cells'][c]['episodes']}/{plan['cells'][c]['planned_episodes']}" for c in short))
    return md


def _success_rows(environments: dict[str, Any]) -> list[str]:
    md = ["| Environment | Blocks | " + " | ".join(f"{condition_label(c)}: success | {condition_label(c)}: predictor use"
                                                for c in CONDS) + " |",
          "|---|---:|" + "---:|---:|" * len(CONDS)]
    for e, env_res in environments.items():
        cells = []
        for c in CONDS:
            r = env_res["conditions"][c]
            cells += [with_ci(r["heldout_success"], r["ci95"], pct), usage_cell(r["predictor"])]
        md.append(f"| {e} | {env_res['blocks']} | " + " | ".join(cells) + " |")
    return md


def _difference_rows(environments: dict[str, Any]) -> list[str]:
    md = ["| Environment | " + " | ".join(difference_label(k) for k in DIFFS) + " |", "|---|" + "---:|" * len(DIFFS)]
    for e, env_res in environments.items():
        md.append(f"| {e} | " + " | ".join(with_ci(env_res["differences"][k]["mean"], env_res["differences"][k]["ci95"], pp)
                                         for k in DIFFS) + " |")
    return md


def render(results: dict[str, Any]) -> str:
    md: list[str] = []
    counts = results["counts"]
    md.append("## Episode counts\n")
    if "plan" in results:
        md += _plan_lines(results["plan"])
    md.append(f"- Runs: {counts['runs']} ({counts['episodes']} episodes); incomplete runs: {len(counts['incomplete_runs'])}; "
              f"superseded run folders: {len(counts['superseded_runs'])}")
    md.append(f"- Infrastructure failures (task attempts abandoned): {counts['infrastructure_failures']}; model API error "
              f"events (retried calls): {counts['api_errors']}; malformed calls: {counts['malformed']}; "
              f"reply cutoffs in scored attempts: {counts['reply_cutoffs']} (all attempts: {counts['reply_cutoffs_all']}); "
              f"context trims: {counts['context_trims']}")
    md.append(f"- Explicitly missing tasks: {counts['missing_task_count']} (unscored; IDs and reasons in JSON)")
    md.append(f"- Log check: {len(counts['log_mismatches'])} runs whose summary.json disagrees with their event log"
              + "".join(f"\n  - {d}: {'; '.join(m[:3])}" for d, m in counts["log_mismatches"].items()) + "\n")
    envs = sorted({k.split("/")[0] for k in counts["episodes_by_cell"]})
    md.append("| Environment | " + " | ".join(condition_label(c) for c in CONDS) + " |")
    md.append("|---|" + "---:|" * len(CONDS))
    for e in envs:
        md.append(f"| {e} | " + " | ".join(str(counts["episodes_by_cell"][f"{e}/{c}"]) for c in CONDS) + " |")

    if results["incomplete_environments"]:
        md.append("\n## Incomplete environments\n")
        md.extend(f"- {e}: {reason}" for e, reason in results["incomplete_environments"].items())
    md.append("\nInvalid histories remain in audit counts but are excluded from success, resource, and token metrics.")
    if results["pairing_issues"]:
        md.append("\n## Pairing exclusions\n")
        for e, issues in results["pairing_issues"].items():
            md.extend(f"- {e}, seed {issue['seed']}, {', '.join(map(condition_label, issue['conditions']))}: "
                      f"{issue['reason']}" for issue in issues)

    core = {e: r for e, r in results["environments"].items() if r["pooled"]}
    separate = {e: r for e, r in results["environments"].items() if not r["pooled"]}
    p = results["pooled"]
    if core:
        md.append("\nPooled quantities can use different contrast-specific cohorts. Coverage is listed below.")
        for quantity, coverage in p["quantity_coverage"].items():
            md.append(f"- {difference_label(quantity) if quantity in DIFFS else condition_label(quantity)}: "
                      f"blocks {coverage['blocks']}; excluded {coverage['excluded']}")
        md.append("\n## Held-out success per condition (complete runs; mean over seed blocks, 95% block-bootstrap CI) "
                  "with predictor usage\n")
        md.append("Predictor usage per cell = mean `predict` calls per episode / share of episodes with zero `predict` calls / "
                  "share of the prediction budget spent (all episodes of the cell). The pooled row weights environments "
                  "equally; its interval resamples seed blocks within each environment.\n")
        md += _success_rows(core)
        md.append("| **Pooled core (equally weighted)** | " + ", ".join(f"{e} {n}" for e, n in p["blocks"].items()) + " | "
                  + " | ".join(f"{with_ci(r['heldout_success'], r['ci95'], pct)} | {usage_cell(r['predictor'])}"
                               for r in p["conditions"].values()) + " |")
        md.append("\n## Paired differences per seed block (held-out success)\n")
        md += _difference_rows(core)
        md.append("| **Pooled core** | " + " | ".join(with_ci(p["differences"][k]["mean"], p["differences"][k]["ci95"], pp)
                                                    for k in DIFFS) + " |")
    if p["excluded"]:
        md.append("\nExcluded from the pooled core average: " + "; ".join(f"{e} ({r})" for e, r in p["excluded"].items()))
    if separate:
        md.append("\n## Separately reported environments (not pooled; a single seed block has point estimates only)\n")
        md += _success_rows(separate)
        md.append("")
        md += _difference_rows(separate)

    if results["environments"]:
        md.append("\n## Matched prediction feedback (held-out, evaluator-only)\n")
        md.append("A mismatch counts only when the executed action prefix matches a prediction from the same real state. "
                  "Changed hypothetical predictions alone are not observed feedback.\n")
        md += ["| Environment | Condition | Episodes | Matched episode share | Mismatch episode share |",
               "|---|---|---:|---:|---:|"]
        for env, cell in results["environments"].items():
            for condition, row in cell["conditions"].items():
                f = row["feedback_heldout"]
                md.append(f"| {env} | {condition_label(condition)} | {f['episodes']} | "
                          f"{pct(f['matched_episode_share'])} | {pct(f['mismatch_episode_share'])} |")
        md.append("\n## Held-out success by tier, adaptation-phase success, held-out predictor usage and discrepancy exposure\n")
        md.append("Discrepancy exposure = share of episodes with a visibly changed prediction / share of changed "
                  "predictions. The accurate predictor's exposure is zero.\n")
        md.append("| Environment | Condition | Easy | Medium | Hard | Adaptation success | Held-out predictor use | "
                  "Exposure (all) | Exposure (held-out) |")
        md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for e, env_res in results["environments"].items():
            for c in CONDS:
                r = env_res["conditions"][c]
                md.append(f"| {e} | {condition_label(c)} | " + " | ".join(pct(r["heldout_by_tier"][t]) for t in TIERS)
                          + f" | {pct(r['adaptation_success'])} | {usage_cell(r['predictor_heldout'])} | "
                          f"{exposure_cell(r['exposure'])} | {exposure_cell(r['exposure_heldout'])} |")

    md.append("\n## Resources (all runs)\n")
    t = results["tokens"]
    md.append(f"- Tokens: prompt {tokens(t['prompt_tokens'])}, completion {tokens(t['completion_tokens'])}, total "
              f"{tokens(t['total_tokens'])} over {t['episodes'] - t['unknown_usage_episodes']} episodes with recorded usage; "
              f"{t['unknown_usage_episodes']} episodes with unknown usage; {t['unrecorded_attempts']} failed attempts "
              f"whose tokens no summary records")
    md.append("")
    md.append("| Metric | Total |")
    md.append("|---|---:|")
    for k in COUNT_KEYS:
        v = results["resources"].get(k, 0)
        md.append(f"| {k} | {v:,.0f}" + (f" ({v / 3600:.1f} agent-hours)" if k == "seconds" else "") + " |")
    md.append("\nPer environment and condition (mean per episode; tokens over episodes with recorded usage; requests = "
              "planning plus notebook requests):\n")
    md.append("| Environment | Condition | Prompt tokens | Completion tokens | Total tokens | Requests | Seconds | "
              "Real actions | Predict calls | Unknown usage | Infrastructure failures | API errors | Malformed | "
              "Reply cutoffs | Context trims |")
    md.append("|---|---|" + "---:|" * 13)
    for e, env_res in results["environments"].items():
        for c, r in env_res["conditions"].items():
            q = r["per_episode"]
            if q["real_actions"] is None:
                continue
            n = r["counts"]
            md.append(f"| {e} | {condition_label(c)} | {tokens(q['prompt_tokens'])} | {tokens(q['completion_tokens'])} | "
                      f"{tokens(q['total_tokens'])} | {q['requests']:.1f} | {q['seconds']:.0f} | "
                      f"{q['real_actions']:.1f} | {q['predict_calls']:.1f} | {q['unknown_usage_episodes']} | "
                      + " | ".join(str(n[k]) for k in ("infrastructure_failures", "api_errors", "malformed",
                                                         "reply_cutoffs", "context_trims")) + " |")

    reasons = sorted({k.split("/", 1)[1] for k in results["end_reasons"]})
    if reasons and results["environments"]:
        md.append("\n## Episode end reasons\n")
        md.append("| Environment | Condition | " + " | ".join(reasons) + " |")
        md.append("|---|---|" + "---:|" * len(reasons))
        for e, env_res in results["environments"].items():
            for c in CONDS:
                er = env_res["conditions"][c]["end_reasons"]
                md.append(f"| {e} | {condition_label(c)} | " + " | ".join(str(er.get(r, 0)) for r in reasons) + " |")
    return "\n".join(md) + "\n"


def tables(runs: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Flat per-run and per-episode rows for runs.csv and episodes.csv."""
    run_rows, episode_rows = [], []
    for r in runs:
        s = r["summary"]
        run_rows.append({"env": r["env"], "condition": condition_label(r["cond"]), "condition_slug": r["cond"],
                         "seed": r["seed"], "run": r["dir"],
                         "status": s["status"], "tasks": s["tasks_completed"], "heldout_success": s["heldout_success_rate"],
                         "adaptation_success": s["adaptation_success_rate"],
                         **{f"heldout_{t}": s["heldout_by_tier"].get(t) for t in TIERS},
                         **{k: s["totals"].get(k) for k in COUNT_KEYS},
                         **token_totals([r]),
                         "infra_failures": len(s["infrastructure_failures"]), "api_errors": r["api_errors"],
                         "reply_cutoffs": r["reply_cutoffs"], "reply_cutoffs_all": r["reply_cutoffs_all"],
                         "context_trims": r["context_trims"], "log_mismatches": len(r["log_mismatches"])})
        for ep in _episodes(r):
            x = ep["exposure"] or {}
            episode_rows.append({"env": r["env"], "condition": condition_label(r["cond"]), "condition_slug": r["cond"],
                                 "seed": r["seed"], **{
                k: ep.get(k) for k in ("task_id", "phase", "tier", "attempt", "success", "end_reason", "real_actions",
                                       "predicted_transitions", "prediction_budget", "predict_calls", "turns", "malformed",
                                       "seconds", *TOKEN_KEYS)},
                **{k: x.get(k) for k in ("predictions", "exposed_predictions", "changed_steps", "state_changed_steps")}})
    return run_rows, episode_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_outputs(out: Path, runs: list[dict[str, Any]], results: dict[str, Any]) -> str:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    md = render(results)
    (out / "summary.md").write_text(md)
    for name, rows in zip(("runs.csv", "episodes.csv"), tables(runs)):
        write_csv(out / name, rows)
    return md


def analyze_agent(runs_dir: Path, study: dict[str, Any], agent: str, *, replay: bool = False, seed: int = 0,
                  boot: int = BOOT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and aggregate one agent's runs of a study, with the plan check."""
    prefix = f"{study['name']}-{agent}"
    runs = load_runs(runs_dir, prefix, list(study["environments"]), replay)
    enrolled = audit.plan(study, agent)
    selected = [r for r in runs if (r["env"], r["cond"], r["seed"]) in enrolled]
    results = aggregate(selected, seed, excluded=audit.exclusions(study), boot=boot)
    results["plan"] = audit.verify_plan(study, agent, audit.run_folders(runs_dir, prefix), runs)
    return runs, results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", type=Path, required=True, help="prepared study manifest (study.json)")
    ap.add_argument("--agent", help="agent name from the study manifest (default: its only agent)")
    ap.add_argument("--runs", type=Path, help="run folders (default: <study dir>/runs)")
    ap.add_argument("--out", type=Path, help="output folder (default: <study dir>/analysis)")
    ap.add_argument("--replay", action="store_true", help="also replay every run in the real environment (slow)")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    a = ap.parse_args(argv)
    study = load_study(a.study)
    agent = a.agent
    if agent is None:
        if len(study["agents"]) != 1:
            ap.error(f"--agent is required; the manifest knows {', '.join(study['agents'])}")
        [agent] = study["agents"]
    if agent not in study["agents"]:
        ap.error(f"unknown agent {agent!r}; the manifest knows {', '.join(study['agents'])}")
    runs_dir = a.runs or a.study.parent / "runs"
    runs, results = analyze_agent(runs_dir, study, agent, replay=a.replay, seed=a.seed)
    if not runs:
        print(f"no runs of {study['name']}-{agent} under {runs_dir}", file=sys.stderr)
        return 1
    print(write_outputs(a.out or a.study.parent / "analysis", runs, results), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
