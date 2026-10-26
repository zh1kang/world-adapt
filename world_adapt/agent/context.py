"""Keeping a task's conversation inside a model's context window.

Some models have small windows (Gemma 4 31B: 30,720 tokens including the reply), and a long IceSlide or Patrol
episode overflows them, which the provider reports as HTTP 400 and the harness as an infrastructure failure. When
`AgentConfig.max_context_tokens` is set, `fit` drops the oldest completed tool exchanges before every model call
until the estimate fits. What is always kept: the opening (system prompt, task prompt with the notebook and the
starting observation, game notes), the most recent `keep_recent_turns` exchanges, and a note in place of what was
dropped that carries the current real observation, so the model is never without the present state. Dropping
whole exchanges keeps every assistant tool call paired with its tool results, which providers require.

The policy depends only on the configuration and the conversation, never on the condition, so it applies
identically across the accurate/faulty predictor and saved/reset notebook conditions; every trim is logged as a `context_trim` event.
Token counts use the recorded notebook tokenizer (`memory.TOKENIZER`), not the model's own, so the cap should
leave a margin below the provider's limit.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from world_adapt.agent.memory import count_tokens

PER_MESSAGE_OVERHEAD = 4
"""Rough per-message framing cost in tokens (role, separators)."""
NOTE_PREFIX = "[Context window: "
"""Marks the user message that stands in for dropped exchanges."""


def message_tokens(message: dict[str, Any]) -> int:
    n = PER_MESSAGE_OVERHEAD
    content = message.get("content")
    if isinstance(content, str):
        n += count_tokens(content)
    for key in ("reasoning_content", "reasoning", "name"):
        value = message.get(key)
        if isinstance(value, str):
            n += count_tokens(value)
    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        n += count_tokens(function.get("name", "")) + count_tokens(function.get("arguments") or "") + PER_MESSAGE_OVERHEAD
    return n


def conversation_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
    n = sum(message_tokens(m) for m in messages)
    if tools:
        n += count_tokens(json.dumps(tools))
    return n


def _opening_length(messages: list[dict[str, Any]]) -> int:
    """Messages before the first assistant turn: system prompt, task prompt (notebook + start), game notes."""
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            return i
    return len(messages)


def _turns(messages: list[dict[str, Any]], start: int) -> list[tuple[int, int]]:
    """[start, end) index ranges of exchanges: an assistant message with everything up to the next assistant one."""
    bounds = [i for i in range(start, len(messages)) if messages[i].get("role") == "assistant"]
    return [(a, b) for a, b in zip(bounds, bounds[1:] + [len(messages)])]


def note(dropped: int, observation: str) -> dict[str, Any]:
    return {"role": "user", "content": f"{NOTE_PREFIX}{dropped} earlier tool exchange(s) were removed to fit the "
                                       f"model's context window. Your notebook (above) and the exchanges below are "
                                       f"intact. The current REAL observation is:]\n{observation}"}


def fit(messages: list[dict[str, Any]], cap: int, keep_recent_turns: int, observe: Callable[[], str],
        tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """Trim `messages` in place until it is estimated to fit in `cap` tokens. Returns an event dict describing the
    trim, or None when nothing was dropped. A trailing non-assistant message (e.g. the notebook prompt) is part of
    the last exchange and is never dropped."""
    before = conversation_tokens(messages, tools)
    if before <= cap:
        return None
    opening = _opening_length(messages)
    already = 0
    if opening and str(messages[opening - 1].get("content", "")).startswith(NOTE_PREFIX):
        opening -= 1
        already = int(messages[opening]["content"][len(NOTE_PREFIX):].split()[0])
        del messages[opening]
    turns = _turns(messages, opening)
    dropped = 0
    while len(turns) - dropped > max(1, keep_recent_turns):
        candidate = messages[:opening] + [note(already + dropped + 1, observe())] + messages[turns[dropped][1]:]
        dropped += 1
        if conversation_tokens(candidate, tools) <= cap:
            break
    if dropped:
        del messages[opening:turns[dropped - 1][1]]
    if already + dropped:
        messages.insert(opening, note(already + dropped, observe()))
    after = conversation_tokens(messages, tools)
    if not dropped:
        return None
    return {"type": "context_trim", "dropped_turns": dropped, "dropped_total": already + dropped,
            "tokens_before": before, "tokens_after": after, "cap": cap, "kept_turns": len(turns) - dropped}
