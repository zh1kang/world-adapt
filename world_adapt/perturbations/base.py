"""Composable discrepancy rules: T_phi = atom_k ∘ ... ∘ atom_1 ∘ T.

An *atom* is a condition plus an effect. The condition decides whether the atom
activates on a transition; the effect rewrites the predicted next state. A
*profile* is an ordered tuple of atoms and is fixed for a whole run.
The empty profile is the identity control.

Atoms apply in order, and each sees the transition produced by the previous
atoms, so composition is not commutative. Prefer atoms whose conditions do not
overlap; `PerturbedDynamics` records every atom that activates so overlaps can
be detected in diagnostics.

Conditions combine with `&`, `|` and `~`. A profile's `complexity` counts leaf
conditions, the number of conditions controlling activation, and is recorded
with the profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic

from world_adapt.environments.base import A, Game, InvalidState, S, Transition


@dataclass(frozen=True)
class Context(Generic[S, A]):
    """Everything an atom may inspect for one transition."""

    game: Game[S, A]
    state: S
    action: A
    nominal: Transition[S]
    """The real transition T(s, a)."""
    current: Transition[S]
    """The transition after all earlier atoms in the profile."""


class Condition:
    """A named predicate over a `Context`, composable with &, | and ~."""

    def __init__(self, name: str, fn: Callable[[Context], bool], leaves: int = 1):
        self.name = name
        self._fn = fn
        self.leaves = leaves

    def __call__(self, ctx: Context) -> bool:
        return bool(self._fn(ctx))

    def __and__(self, other: Condition) -> Condition:
        return Condition(f"({self.name} & {other.name})", lambda c: self(c) and other(c), self.leaves + other.leaves)

    def __or__(self, other: Condition) -> Condition:
        return Condition(f"({self.name} | {other.name})", lambda c: self(c) or other(c), self.leaves + other.leaves)

    def __invert__(self) -> Condition:
        return Condition(f"~{self.name}", lambda c: not self(c), self.leaves)

    def __repr__(self) -> str:
        return f"Condition({self.name})"


Effect = Callable[[Context], Transition]


@dataclass(frozen=True)
class Atom:
    name: str
    family: str
    """Semantic error family: "motion", "contact" or "prerequisite"."""
    condition: Condition
    effect: Effect
    description: str = ""


@dataclass(frozen=True)
class Profile:
    name: str
    version: str
    atoms: tuple[Atom, ...] = ()

    @property
    def is_identity(self) -> bool:
        return not self.atoms

    @property
    def complexity(self) -> int:
        return sum(atom.condition.leaves for atom in self.atoms)

    def describe(self) -> dict[str, Any]:
        """Evaluator-only description; never show this to the agent."""
        return {
            "name": self.name,
            "version": self.version,
            "complexity": self.complexity,
            "atoms": [
                {"name": a.name, "family": a.family, "condition": a.condition.name, "description": a.description}
                for a in self.atoms
            ],
        }


IDENTITY = Profile("identity", "1")


@dataclass(frozen=True)
class PerturbedTransition(Generic[S]):
    transition: Transition[S]
    activated: tuple[str, ...]
    """Atoms whose condition held."""
    changed: tuple[str, ...]
    """Atoms whose effect changed the predicted state."""
    rejected: tuple[str, ...] = ()
    """Atoms whose effect produced a state the game rejects; their effect was discarded (an atom bug, recorded
    so it is auditable rather than fatal to the run)."""


class PerturbedDynamics(Generic[S, A]):
    """T_phi: the game's true dynamics followed by the profile's atoms."""

    def __init__(self, game: Game[S, A], profile: Profile = IDENTITY):
        self.game = game
        self.profile = profile

    def step(self, state: S, action: A) -> PerturbedTransition[S]:
        nominal = self.game.step(state, action)
        current = nominal
        activated: list[str] = []
        changed: list[str] = []
        rejected: list[str] = []
        for atom in self.profile.atoms:
            ctx = Context(self.game, state, action, nominal, current)
            if not atom.condition(ctx):
                continue
            activated.append(atom.name)
            result = atom.effect(ctx)
            if result.state != current.state:
                try:
                    self.game.validate_state(result.state)
                except InvalidState:
                    rejected.append(atom.name)
                    continue
                changed.append(atom.name)
            current = result
        return PerturbedTransition(current, tuple(activated), tuple(changed), tuple(rejected))
