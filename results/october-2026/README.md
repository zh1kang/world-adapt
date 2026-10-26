# October 2026 study results

Nine models, six environments, four conditions, and one seed block (3000): 216 histories of 14 tasks.
Each model ran 24 histories (336 tasks); Kimi K2.6 has 332 tasks scored, because four held-out tasks of its Patrol history under the faulty predictor with a saved notebook are missing.
Missing tasks are absent from these files; they are not failures.
Claude Opus 5.5 ran at high reasoning effort and the other models at low effort.

| File | One row per | Contents |
|---|---|---|
| `tasks.csv` | scored task (3,020) | model, environment, condition, task id, phase (`adaptation` or `heldout`), tier, `success` (1 or 0), `predict` and `execute` calls, how the task ended, total tokens, seconds |
| `histories.csv` | history (216) | the run id and the adaptation and held-out tasks scored and solved |
| `notebook-judge/notebooks.csv` | notebook written after an adaptation task (1,672) | the majority label of three judges and each judge's labels |
| `notebook-judge/histories.csv` | history | first task whose notebook names or locates the perturbed rule, final diagnosis and predictor policy, false-alarm and false-belief counts |
| `notebook-judge/judgments.jsonl` | judge and notebook | each judge's full judgment: labels, quoted evidence (checked against the notebook), false beliefs, rationale, and token usage |
| `notebook-judge/disagreements.csv` | notebook without a majority or with split labels | the labels of all three judges |
| `notebook-judge/agreement.json` | label | Fleiss' kappa over the three judges and pairwise Cohen's kappa with confusion matrices |
| `notebook-judge/cost.json` | judge | tokens and list-price cost of the judging run |

The judges were Claude Opus 5.5 (Anthropic Batch API), DeepSeek V4.1 Flash and Kimi K3 (both on SAIL).
`end_reason` values: `success`; `dead_state` (the goal became unreachable); `conceded` (the agent gave up); `terminal_failure` (the environment ended the task, such as a capture in Patrol); `real_budget` (the real-step budget ran out); `malformed_limit` (too many malformed tool calls); `token_limit`.
The task instances themselves are in [data/tasks/seed-3000.jsonl](../../data/tasks/seed-3000.jsonl), keyed by environment and `task_id`.
