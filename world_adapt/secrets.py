"""Load API keys from `.env.local` without logging or committing them."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / ".env.local"


def load_env(path: Path = DEFAULT_PATH, override: bool = False) -> list[str]:
    """Set environment variables from KEY=VALUE lines; returns the names that were set (never values)."""
    if not path.exists():
        return []
    loaded = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if value and (override or key not in os.environ):
            os.environ[key] = value
            loaded.append(key)
    return loaded
