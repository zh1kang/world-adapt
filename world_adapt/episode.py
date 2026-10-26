"""One task played under budgets: the rules every agent harness plays by.

`TaskSession` enforces the information boundary (the agent sees observations,
never states). `Episode` adds the benchmark's accounting on top: action and
prediction budgets, chunk limits, how a task ends, and the logs. Any harness
(ours in `world_adapt.agent`, or another one) calls `predict`, `execute` and
`concede` here, so budgets and endings are counted identically for all of them.

Two logs: `log` receives what the agent saw and the resource accounting;
`evaluator_log` receives hidden data (perturbation activations, real-state
events, solver checks, diagnostics). Without an `evaluator_log`, both go to `log`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

from world_adapt.environments.base import Game, InvalidAction, Task
from world_adapt.environments.real import Execution, TrajectoryRecord
from world_adapt.environments.search import solve
from world_adapt.predictors.simulator import Prediction, SimulatorPredictor
from world_adapt.session import TaskSession
from world_adapt.feedback import PredictionFeedback

Log = Callable[[dict[str, Any]], None]

ENDINGS = ("conceded", "dead_state", "terminal_failure")
"""End reasons set by the episode itself; others (turn_limit, malformed_limit, token_limit, time_limit) belong to
the harness."""


@dataclass(frozen=True)
class Budgets:
    real_actions: int
    predicted_transitions: int
    max_chunk: int = 8
    max_turns: int = 40
    max_malformed: int = 5
    max_task_tokens: int | None = None
    """Cap on model tokens (prompt + completion, summed over calls) while playing one task; None means no cap."""
    max_task_seconds: float | None = None
    """Cap on wall-clock seconds while playing one task; None means no cap."""


@dataclass
class TaskResult:
    task_id: str
    success: bool
    real_actions: int
    predicted_transitions: int
    predict_calls: int
    execute_calls: int
    turns: int
    malformed: int
    end_reason: str
    usage: dict[str, int] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    """Wall-clock time of the task, notebook writing included."""


@dataclass(frozen=True)
class ExecuteOutcome:
    execution: Execution
    records: tuple[TrajectoryRecord, ...]
    """Evaluator-only records of the transitions just executed."""
    cut_to: int | None
    """Set when the chunk was cut to the remaining real-action budget."""
    task_over: str | None
    """Agent-facing message when this execution ended the task."""
    execution_id: str = ""


class Episode:
    def __init__(self, game: Game, task: Task, predictor: SimulatorPredictor, budgets: Budgets, *,
                 end_on_dead_state: bool = True, dead_check_expansions: int = 300_000,
                 prune: Callable[[Any], bool] | None = None, log: Log = lambda event: None,
                 evaluator_log: Log | None = None, episode_id: str | None = None):
        self.session = TaskSession(game, task, predictor)
        self.game, self.task, self.budgets = game, task, budgets
        self.end_on_dead_state = end_on_dead_state
        self.dead_check_expansions = dead_check_expansions
        self.prune = prune
        self.log = log
        self.evaluator_log = evaluator_log or log
        self.episode_id = episode_id or task.task_id
        self.result = TaskResult(task.task_id, False, 0, 0, 0, 0, 0, 0, "turn_limit")
        self.feedback = PredictionFeedback()
        self.last_query_id: str | None = None
        self._executions: dict[str, ExecuteOutcome] = {}
        self._started = time.monotonic()

    # Agent-visible -------------------------------------------------------------

    def observe(self) -> str:
        return self.session.observe()

    def check_chunk(self, actions: Any) -> list[str]:
        if not isinstance(actions, list) or not actions:
            raise InvalidAction("'actions' must be a nonempty list")
        if len(actions) > self.budgets.max_chunk:
            raise InvalidAction(f"at most {self.budgets.max_chunk} actions per call")
        if any(not isinstance(action, str) for action in actions):
            raise InvalidAction("every action must be a string")
        return actions

    def predict(self, actions: Sequence[str]) -> Prediction:
        remaining = self.budgets.predicted_transitions - self.result.predicted_transitions
        if len(actions) > remaining:
            raise InvalidAction(f"prediction budget allows only {remaining} more predicted actions")
        prediction = self.session.predict(actions)
        self.result.predict_calls += 1
        self.result.predicted_transitions += len(actions)
        self.last_query_id = f"{self.episode_id}/predict-{self.result.predict_calls}"
        self.feedback.predict(self.last_query_id, self.result.real_actions,
                              [self.game.parse_action(a) for a in actions],
                              [step.observation for step in prediction.steps])
        return prediction

    def log_prediction(self, actions: Sequence[str]) -> None:
        trace = self.session.prediction_traces[-1]
        self.log({"type": "predict", "task_id": self.task.task_id, "query_id": self.last_query_id,
                  "actions": list(actions), "charged": len(actions), "simulated": len(trace.states),
                  "prediction_seed": None})
        self.evaluator_log({"type": "predict_trace", "task_id": self.task.task_id, "query_id": self.last_query_id,
                            "changed": [list(c) for c in trace.changed],
                            "activated": [list(a) for a in trace.activated],
                            "rejected": [list(r) for r in trace.rejected],
                            "visible_changed": list(trace.visible_changed),
                            "transitions_requested": trace.transitions_requested,
                            "visible_change_recording_version": 1})

    def execute(self, actions: Sequence[str], execution_id: str | None = None) -> ExecuteOutcome:
        """Commit a chunk. A repeated `execution_id` returns the first outcome and executes nothing."""
        if execution_id is not None and execution_id in self._executions:
            self.log({"type": "execute_duplicate", "task_id": self.task.task_id, "execution_id": execution_id})
            return self._executions[execution_id]
        execution_id = execution_id or f"{self.episode_id}/execute-{self.result.execute_calls + 1}"
        remaining = self.budgets.real_actions - self.result.real_actions
        execution = self.session.execute(actions[:remaining])
        self.result.execute_calls += 1
        self.result.real_actions += execution.transitions_executed
        self.result.success = execution.success
        trajectory = self.session.real.trajectory
        records = trajectory[len(trajectory) - execution.transitions_executed:]
        for feedback in self.feedback.execute(self.game, trajectory):
            self.evaluator_log({"type": "prediction_feedback", "task_id": self.task.task_id,
                                "execution_id": execution_id, **feedback})
        task_over = None
        if execution.terminal and not execution.success:
            self.result.end_reason = "terminal_failure"
            task_over = "The task has ended in failure."
        elif not execution.success and self.end_on_dead_state and self._is_dead():
            self.result.end_reason = "dead_state"
            task_over = "The puzzle can no longer be solved from the real state. The task has ended."
        self.log({"type": "execute", "task_id": self.task.task_id, "execution_id": execution_id,
                  "actions": list(actions), "executed": execution.transitions_executed, "success": execution.success,
                  "terminal": execution.terminal, "real_actions_used": self.result.real_actions})
        self.evaluator_log({"type": "execute_trace", "task_id": self.task.task_id, "execution_id": execution_id,
                            "events": [list(r.events) for r in records]})
        outcome = ExecuteOutcome(execution, records, remaining if len(actions) > remaining else None, task_over,
                                 execution_id)
        self._executions[execution_id] = outcome
        return outcome

    def concede(self) -> None:
        self.result.end_reason = "conceded"
        self.log({"type": "concede", "task_id": self.task.task_id})

    def record_malformed(self) -> None:
        self.result.malformed += 1

    def budget_remaining(self) -> dict[str, int]:
        return {"real_actions": self.budgets.real_actions - self.result.real_actions,
                "predicted_actions": self.budgets.predicted_transitions - self.result.predicted_transitions}

    @property
    def ended(self) -> bool:
        r = self.result
        return r.success or r.real_actions >= self.budgets.real_actions or r.end_reason in ENDINGS

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    @property
    def deadline(self) -> float | None:
        seconds = self.budgets.max_task_seconds
        return self._started + seconds if seconds is not None else None

    # Evaluator-side ------------------------------------------------------------

    def settle_end_reason(self) -> None:
        """Call once the episode has ended: success, or the budget running out."""
        if self.result.success:
            self.result.end_reason = "success"
        elif self.result.end_reason not in ENDINGS:
            self.result.end_reason = "real_budget"

    def close(self) -> TaskResult:
        """Record diagnostics and the task_end events (public accounting; hidden trajectory and diagnostics)."""
        diagnose = getattr(self.game, "diagnostics", None)
        if diagnose is not None:
            self.result.diagnostics = diagnose(self.session.real.snapshot)
        self.evaluator_log({"type": "task_end_private", "task_id": self.task.task_id, "episode_id": self.episode_id,
                            "diagnostics": self.result.diagnostics, "prediction_feedback": self.feedback.summary(),
                            "trajectory": [[r.action, list(r.events)] for r in self.session.real.trajectory],
                            "prediction_changed": [list(map(list, t.changed)) for t in self.session.prediction_traces],
                            "prediction_visible_changed": [list(t.visible_changed) for t in self.session.prediction_traces],
                            "prediction_requested_transitions": [t.transitions_requested for t in self.session.prediction_traces],
                            "visible_change_recording_version": 1})
        public = {k: v for k, v in asdict(self.result).items() if k not in ("diagnostics", "seconds")}
        self.log({"type": "task_end", "episode_id": self.episode_id, "result": public})
        return self.result

    def _is_dead(self) -> bool:
        """True only when the solver proves the real state unsolvable; unknown (cap hit) counts as alive."""
        search = solve(self.game, self.session.real.snapshot, self.dead_check_expansions, prune=self.prune)
        self.evaluator_log({"type": "dead_check", "task_id": self.task.task_id, "dead": search.proved_unsolvable,
                            "exhausted": search.exhausted, "expansions": search.expansions})
        return search.proved_unsolvable
