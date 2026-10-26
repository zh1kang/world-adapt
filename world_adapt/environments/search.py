"""Generic breadth-first solver for any deterministic `Game`."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Hashable


@dataclass(frozen=True)
class SearchResult:
    """Outcome of `solve`.

    `actions` is a shortest solution, or None. With `actions is None`,
    `exhausted=False` means the reachable space was fully explored (unsolvable
    under the given prune) and `exhausted=True` means the expansion cap was hit.
    """

    actions: tuple | None
    expansions: int
    exhausted: bool

    @property
    def solved(self) -> bool:
        return self.actions is not None

    @property
    def proved_unsolvable(self) -> bool:
        return self.actions is None and not self.exhausted


def solve(
    game: Any,
    state: Hashable,
    max_expansions: int = 200_000,
    prune: Callable[[Any], bool] | None = None,
) -> SearchResult:
    """Shortest action sequence from `state` to a success state, by BFS over `game.step`."""
    if game.is_success(state):
        return SearchResult((), 0, False)
    actions = tuple(game.actions())
    parent: dict[Hashable, tuple[Hashable, Any] | None] = {state: None}
    queue = deque([state])
    expansions = 0
    while queue:
        if expansions >= max_expansions:
            return SearchResult(None, expansions, True)
        current = queue.popleft()
        if prune is not None and prune(current):
            continue
        expansions += 1
        for action in actions:
            nxt = game.step(current, action).state
            if nxt == current or nxt in parent:
                continue
            parent[nxt] = (current, action)
            if game.is_success(nxt):
                path = []
                node = nxt
                while parent[node] is not None:
                    node, act = parent[node]
                    path.append(act)
                return SearchResult(tuple(reversed(path)), expansions, False)
            queue.append(nxt)
    return SearchResult(None, expansions, False)
