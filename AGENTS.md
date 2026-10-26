# World-Adapt agent instructions

World-Adapt is a benchmark for LLM agents that plan with a predictor that may be wrong.
Read `README.md` for what the benchmark measures and how to run it; this file covers how to change the code safely.

## Setup and checks

- Python 3.12 and uv. Install with `uv sync --locked`; run everything with `uv run --locked ...` from the repository root.
- Run `uv run --locked python -m pytest -q` before reporting any code change as done. It takes about two minutes and makes no network calls; tests must stay offline and use the scripted clients in `tests/fakes.py`.
- For changes to study preparation or scheduling, also run `local prepare` into a new `outputs/<name>` directory and `local run --study outputs/<name>/study.json --dry-run`. Both make no model calls.
- Change dependencies only with `uv add` or `uv lock`, never by editing `uv.lock`.

## Layout

- `world_adapt/environments/`: the six environments, task generators, and reference solvers.
- `world_adapt/perturbations/`: perturbation atoms and the one primary profile per environment that the study uses.
- `world_adapt/predictors/`: the simulator behind the `predict` tool.
- `world_adapt/agent/`: agent loop, tools, prompts, notebook writer, provider clients, and the spend ledger (`spending.py`).
- `world_adapt/runner/`: `local` (prepare and run), `study` (scheduler), `run` (one history), `preflight`, `analyze`, `rescore`, and `judge` (the Notebook Judge).
- `configs/study.json`: the study protocol. `configs/agents/example.json`: agent settings.
- `scripts/`: task export, GIF rendering for the docs, and the Modal vLLM server. `docs/`: environment documentation.
- `data/`: the study's task instances and environment definitions, written by `scripts/export_tasks.py`; regenerate them instead of editing them.
- `results/`: the published per-task outcomes and Notebook Judge labels of a finished study. Treat them as read-only records.

## Invariants

- The protocol is pinned. `tests/golden/study.json` fixes the generated tasks, prompts, tool schemas, and budgets of the full study. If that test fails, you changed the benchmark; do not regenerate the golden file unless the protocol change is intended, and say so in your report.
- Prepared studies pin a hash of `world_adapt/`, `README.md`, `pyproject.toml`, `uv.lock`, and the Depot domain file. Editing any of them while someone is running a study breaks resuming it. Ask before editing these if a study may be in progress.
- Conditions have exactly four names: `accurate-predictor-saved-notebook`, `accurate-predictor-reset-notebook`, `faulty-predictor-saved-notebook`, `faulty-predictor-reset-notebook` (displayed as "Accurate predictor + saved notebook" and so on). Never show short codes such as `UM` or `PM` in output, flags, or folder names.
- The real environment is never perturbed; only the predictor is. Keep that distinction in code, docs, and labels.
- The spend ledger must never under-count. A request the provider refused before doing work (4xx except 408, 503, 529) is released; anything that may have been billed (timeouts, cut-off streams, 408, 500, 502, 504) stays reserved as unknown. Budgets can be raised with `local run --budget-usd`, never lowered.
- Tasks lost to provider or infrastructure errors are missing, not failures. Do not add code that scores them as failures or silently skips them.

## Data and credentials

- Never commit `outputs/`, run folders, raw logs, or other run artifacts; they are ignored by Git and stay local. Add to `results/` only summary tables and labels with no local paths or credentials.
- API keys live only in `.env.local` (template: `.env.example`). Never print, log, or commit key values.
- Any command that calls a model spends real money, including `runner.judge`. Run `--dry-run` and `preflight` first (or `judge --sample`), and get the user's approval and a budget before a live run.
