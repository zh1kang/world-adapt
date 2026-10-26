"""Perturbation profiles per environment.

Each environment's module (`world_adapt/perturbations/<env>.py`) exposes `ATOMS` and `PROFILES`: the faulty
profile the benchmark uses for that environment. The exact `identity` profile is shared by every environment.
"""

from __future__ import annotations

import importlib

from world_adapt.environments import registry as environments
from world_adapt.perturbations.base import IDENTITY, Profile


def profiles(env: str) -> dict[str, Profile]:
    """Profiles for a registered environment: `identity` plus its faulty profile."""
    if env not in environments.names():
        raise KeyError(env)
    found = importlib.import_module(f"world_adapt.perturbations.{env}").PROFILES
    return {IDENTITY.name: IDENTITY, **found}
