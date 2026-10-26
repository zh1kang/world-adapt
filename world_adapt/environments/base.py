"""Environment contract shared by every game in the benchmark.

A game is a pure, deterministic transition function over immutable, hashable
states. Purity is what makes the rest of the benchmark simple: cloning a state
is free, a prediction can never mutate the real environment, and two candidate
rollouts from the same state are independent by construction.

To add an environment:
1. Define an immutable, hashable state type (e.g. a frozen dataclass).
2. Implement the `Game` protocol below.
3. Register a fixture in `tests/contracts/fixtures.py` so the shared contract
   tests (identity predictor, isolation, branching, termination) run on it.
4. Write environment-specific perturbation atoms in `world_adapt/perturbations/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Hashable, Protocol, Sequence, TypeVar

S = TypeVar("S", bound=Hashable)
A = TypeVar("A", bound=Hashable)


class InvalidAction(ValueError):
    """Raised when an agent-supplied action cannot be parsed or is not allowed."""


class InvalidState(ValueError):
    """Raised when a state violates the game's schema (e.g. after a faulty perturbation)."""


@dataclass(frozen=True)
class Transition(Generic[S]):
    """Result of one primitive step.

    `events` are evaluator-only diagnostics (e.g. "push", "slide"); they are
    never shown to the agent.
    """

    state: S
    events: tuple[str, ...] = ()


@dataclass(frozen=True)
class Task(Generic[S]):
    """One task instance: an identifier, an initial state, and free-form metadata."""

    task_id: str
    initial_state: S
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


class Game(Protocol[S, A]):
    """The true dynamics `T` plus everything needed to observe and score states."""

    name: str
    version: str

    def actions(self) -> Sequence[A]:
        """All primitive actions."""
        ...

    def parse_action(self, raw: str) -> A:
        """Convert agent-facing text to an action; raise InvalidAction if invalid."""
        ...

    def step(self, state: S, action: A) -> Transition[S]:
        """The real transition function T(s, a). Must be pure and deterministic."""
        ...

    def is_success(self, state: S) -> bool: ...

    def is_terminal(self, state: S) -> bool:
        """True if no further actions should be applied (success or terminal failure)."""
        ...

    def observe(self, state: S) -> str:
        """Agent-facing observation. Used identically for real and predicted states."""
        ...

    def validate_state(self, state: S) -> None:
        """Raise InvalidState if the state is malformed."""
        ...

    def spec(self) -> dict[str, Any]:
        """Adapter declaration (modality, actions, termination, limitations)."""
        ...
