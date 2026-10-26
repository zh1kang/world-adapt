"""Simulator predictor: rolls out T_phi from a private clone of the real state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Sequence

from world_adapt.environments.base import A, Game, InvalidAction, S
from world_adapt.perturbations.base import IDENTITY, PerturbedDynamics, Profile


@dataclass(frozen=True)
class PredictedStep:
    index: int
    action: str
    observation: str
    predicted_success: bool


@dataclass(frozen=True)
class Prediction:
    """Agent-visible prediction. Hypothetical: never evidence of real success."""

    steps: tuple[PredictedStep, ...]
    transitions_requested: int
    transitions_simulated: int
    predicted_terminal: bool


@dataclass(frozen=True)
class PredictionTrace(Generic[S]):
    """Evaluator-only record: predicted states and which atoms activated per step."""

    states: tuple[S, ...]
    activated: tuple[tuple[str, ...], ...]
    changed: tuple[tuple[str, ...], ...]
    rejected: tuple[tuple[str, ...], ...] = ()
    """Per step, atoms whose effect the game rejected as an invalid state (see `PerturbedTransition.rejected`)."""
    visible_changed: tuple[bool, ...] = ()
    """Per aligned predicted or identity step, whether the agent-visible result differs."""
    transitions_requested: int = 0


class SimulatorPredictor(Generic[S, A]):
    """Predictor backed by the game's own dynamics plus a discrepancy profile.

    Errors propagate: each step applies T_phi to the previous *predicted* state.
    The predictor keeps no memory between calls, so it cannot repair itself.
    """

    def __init__(self, game: Game[S, A], profile: Profile = IDENTITY, max_horizon: int | None = None):
        self.game = game
        self.profile = profile
        self.max_horizon = max_horizon
        self._dynamics = PerturbedDynamics(game, profile)

    def predict(self, start: S, raw_actions: Sequence[str]) -> tuple[Prediction, PredictionTrace[S]]:
        if not raw_actions:
            raise InvalidAction("action sequence must be nonempty")
        if self.max_horizon is not None and len(raw_actions) > self.max_horizon:
            raise InvalidAction(f"query horizon {len(raw_actions)} exceeds maximum {self.max_horizon}")
        actions = [self.game.parse_action(raw) for raw in raw_actions]
        state = start
        steps: list[PredictedStep] = []
        states: list[S] = []
        activated: list[tuple[str, ...]] = []
        changed: list[tuple[str, ...]] = []
        rejected: list[tuple[str, ...]] = []
        for raw, action in zip(raw_actions, actions):
            if self.game.is_terminal(state):
                break
            result = self._dynamics.step(state, action)
            state = result.transition.state
            states.append(state)
            activated.append(result.activated)
            changed.append(result.changed)
            rejected.append(result.rejected)
            steps.append(PredictedStep(len(steps) + 1, raw, self.game.observe(state), self.game.is_success(state)))
        if self.profile.is_identity:
            # Identity is already the reference rollout.  Avoid replaying native
            # simulators a second time, which can be expensive and can allocate
            # external simulator state even though the result is known equal.
            visible_changed = [False] * len(states)
        else:
            identity = self._identity_rollout(start, actions)
            predicted = [
                (step.observation, self.game.is_success(state), self.game.is_terminal(state))
                for step, state in zip(steps, states)
            ]
            visible_changed = []
            for index in range(max(len(predicted), len(identity))):
                if index >= len(predicted) or index >= len(identity):
                    # A profile that terminates early, or stays alive after the
                    # nominal terminal state, has an observable cardinality
                    # mismatch even when the shared prefix looks identical.
                    visible_changed.append(True)
                else:
                    visible_changed.append(predicted[index] != identity[index])
        prediction = Prediction(
            steps=tuple(steps),
            transitions_requested=len(raw_actions),
            transitions_simulated=len(steps),
            predicted_terminal=self.game.is_terminal(state),
        )
        return prediction, PredictionTrace(tuple(states), tuple(activated), tuple(changed), tuple(rejected),
                                           tuple(visible_changed), len(raw_actions))

    def _identity_rollout(self, start: S, actions: Sequence[A]) -> list[tuple[str, bool, bool]]:
        """Return the full agent-visible reference rollout for one query."""
        state = start
        result: list[tuple[str, bool, bool]] = []
        for action in actions:
            if self.game.is_terminal(state):
                break
            transition = self.game.step(state, action)
            state = transition.state
            result.append((self.game.observe(state), self.game.is_success(state), self.game.is_terminal(state)))
        return result
