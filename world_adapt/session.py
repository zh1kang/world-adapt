"""One task episode: a real environment plus a predictor conditioned on its current state.

The runner owns budgets and logging; this only enforces the information
boundary. The agent sees observations, predictions and executions; the real
state, trajectory and prediction traces stay evaluator-only.
"""

from __future__ import annotations

from typing import Generic, Sequence

from world_adapt.environments.base import A, Game, S, Task
from world_adapt.environments.real import Execution, RealEnvironment
from world_adapt.predictors.simulator import Prediction, PredictionTrace, SimulatorPredictor


class TaskSession(Generic[S, A]):
    def __init__(self, game: Game[S, A], task: Task[S], predictor: SimulatorPredictor[S, A]):
        if predictor.game is not game:
            raise ValueError("predictor and environment must share the same game instance")
        self.real = RealEnvironment(game, task)
        self.predictor = predictor
        self.prediction_traces: list[PredictionTrace[S]] = []

    def observe(self) -> str:
        return self.real.observe()

    def predict(self, raw_actions: Sequence[str]) -> Prediction:
        prediction, trace = self.predictor.predict(self.real.snapshot, raw_actions)
        self.prediction_traces.append(trace)
        return prediction

    def execute(self, raw_actions: Sequence[str]) -> Execution:
        return self.real.execute(raw_actions)
