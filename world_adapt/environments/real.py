"""The real environment: holds private state and executes committed action chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Sequence

from world_adapt.environments.base import A, Game, InvalidAction, S, Task


@dataclass(frozen=True)
class RealStep:
    index: int
    action: str
    observation: str


@dataclass(frozen=True)
class Execution:
    """Agent-visible result of executing an action chunk."""

    steps: tuple[RealStep, ...]
    transitions_executed: int
    success: bool
    terminal: bool


@dataclass(frozen=True)
class TrajectoryRecord(Generic[S, A]):
    """Evaluator-only record of one real transition."""

    index: int
    state: S
    action: A
    next_state: S
    events: tuple[str, ...]


class RealEnvironment(Generic[S, A]):
    """Real environment for one task. State is private to the runner and scorer."""

    def __init__(self, game: Game[S, A], task: Task[S]):
        self.game = game
        self.task = task
        self._state = task.initial_state
        self._trajectory: list[TrajectoryRecord[S, A]] = []
        game.validate_state(self._state)

    @property
    def snapshot(self) -> S:
        """Private copy of the current state (states are immutable, so no copy is needed)."""
        return self._state

    @property
    def trajectory(self) -> tuple[TrajectoryRecord[S, A], ...]:
        return tuple(self._trajectory)

    @property
    def steps_taken(self) -> int:
        return len(self._trajectory)

    def observe(self) -> str:
        return self.game.observe(self._state)

    def execute(self, raw_actions: Sequence[str]) -> Execution:
        """Execute an open-loop chunk, stopping early at real termination.

        All actions are parsed before any is executed, so an invalid action
        never leaves the environment partially advanced.
        """
        if not raw_actions:
            raise InvalidAction("action sequence must be nonempty")
        actions = [self.game.parse_action(raw) for raw in raw_actions]
        steps: list[RealStep] = []
        state = self._state
        records: list[TrajectoryRecord] = []
        for raw, action in zip(raw_actions, actions):
            if self.game.is_terminal(state):
                break
            transition = self.game.step(state, action)
            records.append(
                TrajectoryRecord(len(self._trajectory) + len(records), state, action, transition.state, transition.events)
            )
            state = transition.state
            steps.append(RealStep(len(self._trajectory) + len(records), raw, self.game.observe(state)))
        # State-dependent action validation can fail after an earlier valid action.
        # Commit only after the whole chunk is accepted.
        self._trajectory.extend(records)
        self._state = state
        return Execution(
            steps=tuple(steps),
            transitions_executed=len(steps),
            success=self.game.is_success(self._state),
            terminal=self.game.is_terminal(self._state),
        )
