"""Cross-task memory: which notebook a task starts from, and the tokenizer that caps it.

Memory is the experiment's manipulated variable (saved vs reset), so it is kept in one place. How the agent
writes its notebook is `world_adapt.agent.writer`; which task starts from which notebook (adaptation chain,
held-out branches) is the runner's protocol (`world_adapt.runner.protocol`).
"""

from __future__ import annotations

from functools import lru_cache

MEMORY_MODES = ("preserve", "reset")
TOKENIZER = "tiktoken/o200k_base"
"""The recorded tokenizer for the notebook cap and context estimates. It is not the agent model's own tokenizer."""


def carry_over(mode: str, notebook: str) -> str:
    """What the next task starts with: the latest notebook (preserve) or nothing (reset)."""
    if mode not in MEMORY_MODES:
        raise ValueError(f"memory mode must be one of {MEMORY_MODES}")
    return notebook if mode == "preserve" else ""


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding(TOKENIZER.split("/", 1)[1])


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))
