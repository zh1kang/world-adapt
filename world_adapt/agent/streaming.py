"""Fail-closed assembly for chat-completions streaming responses.

The agent loop consumes one complete response.  This adapter buffers content, reasoning, and indexed tool-call
fragments until the provider sends a terminal finish reason and a final usage record, so an interrupted stream can
never execute a partially assembled tool call.
"""

from __future__ import annotations

from types import SimpleNamespace
import time
from typing import Any, Iterable

from world_adapt.agent.llm import InfrastructureError


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _text(value: Any, label: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise InfrastructureError(f"streamed {label} must be a string")
    return value


def _usage(value: Any) -> SimpleNamespace | None:
    if value is None:
        return None
    usage = SimpleNamespace(**{key: _get(value, key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")})
    details = _get(value, 'prompt_tokens_details')
    if details is not None:
        usage.prompt_tokens_details = SimpleNamespace(cached_tokens=_get(details, 'cached_tokens'))
    return usage


class _Call(SimpleNamespace):
    """Small SDK-compatible tool call used by the chat-completions loop."""

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {"id": self.id, "type": "function",
                "function": {"name": self.function.name, "arguments": self.function.arguments}}


class _Message(SimpleNamespace):
    """Small SDK-compatible message used by ``llm.assistant_message`` and history logging."""

    def model_dump(self, exclude_none: bool = False, **_: Any) -> dict[str, Any]:
        data: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.reasoning_content is not None:
            data["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            data["tool_calls"] = [call.model_dump() for call in self.tool_calls]
        return {key: value for key, value in data.items() if value is not None} if exclude_none else data


def assemble(chunks: Iterable[Any], *, require_usage: bool = True, deadline: float | None = None) -> Any:
    """Assemble a complete provider stream or raise without returning a partial response."""
    content: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    finish_reason: str | None = None
    final_usage: Any = None
    saw_choice = False
    terminal_seen = False
    early_usage = False

    try:
        iterator = iter(chunks)
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise InfrastructureError("stream exceeded the request deadline; remote completion is uncertain")
            try:
                chunk = next(iterator)
            except StopIteration:
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise InfrastructureError("stream exceeded the request deadline; remote completion is uncertain")
            error = _get(chunk, "error")
            if error is not None:
                raise InfrastructureError(f"stream provider error: {error!r}")
            choices = _get(chunk, "choices")
            if choices is None:
                choices = []
            if not isinstance(choices, list):
                raise InfrastructureError("stream choices must be a list")
            if len(choices) > 1:
                raise InfrastructureError("stream emitted multiple choices")
            if terminal_seen and choices:
                raise InfrastructureError("stream emitted a choice after the terminal finish reason")
            for choice in choices:
                saw_choice = True
                if terminal_seen:
                    raise InfrastructureError("stream emitted a second terminal choice")
                index = _get(choice, "index", 0)
                if isinstance(index, bool) or not isinstance(index, int) or index != 0:
                    raise InfrastructureError("stream emitted a nonzero choice index")
                choice_finish = _get(choice, "finish_reason")
                if choice_finish is not None and not isinstance(choice_finish, str):
                    raise InfrastructureError("stream finish_reason must be a string")
                finish_reason = choice_finish or finish_reason
                delta = _get(choice, "delta")
                if delta is None:
                    if choice_finish is not None:
                        terminal_seen = True
                    continue
                content.append(_text(_get(delta, "content"), "content"))
                reasoning.append(_text(_get(delta, "reasoning_content"), "reasoning_content") or
                                 _text(_get(delta, "reasoning"), "reasoning"))
                tool_deltas = _get(delta, "tool_calls")
                if tool_deltas is None:
                    tool_deltas = []
                if not isinstance(tool_deltas, list):
                    raise InfrastructureError("stream tool_calls must be a list")
                for call in tool_deltas:
                    index = _get(call, "index", 0)
                    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                        raise InfrastructureError("streamed tool call had an invalid index")
                    item = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    item["id"] += _text(_get(call, "id"), "tool call id")
                    function = _get(call, "function")
                    item["name"] += _text(_get(function, "name"), "tool name")
                    item["arguments"] += _text(_get(function, "arguments"), "tool arguments")
                if choice_finish is not None:
                    terminal_seen = True
            usage = _get(chunk, "usage")
            if usage is not None:
                if terminal_seen:
                    final_usage = usage
                else:
                    early_usage = True
    except InfrastructureError:
        raise
    except Exception as error:
        raise InfrastructureError(f"stream failed before a complete response: {error!r}") from error

    if not saw_choice or not terminal_seen or finish_reason is None:
        raise InfrastructureError("stream ended without a terminal choice")
    if require_usage and final_usage is None:
        detail = " after an early usage record" if early_usage else ""
        raise InfrastructureError(f"stream ended without final usage{detail}")
    if final_usage is not None:
        values = {key: _get(final_usage, key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values.values()):
            raise InfrastructureError("stream usage must contain nonnegative integer token counts")
        if values["total_tokens"] < values["prompt_tokens"] + values["completion_tokens"]:
            raise InfrastructureError("stream total_tokens is smaller than prompt_tokens + completion_tokens")

    if calls and sorted(calls) != list(range(len(calls))):
        raise InfrastructureError("stream tool call indices were not contiguous")
    if calls and finish_reason not in ("length", "incomplete"):
        ids = [item["id"] for _, item in sorted(calls.items())]
        if any(not item["id"] or not item["name"] for item in calls.values()):
            raise InfrastructureError("complete streamed tool calls require nonempty ids and names")
        if len(ids) != len(set(ids)):
            raise InfrastructureError("complete streamed tool calls require unique ids")
    tool_calls = [_Call(id=item["id"], function=SimpleNamespace(name=item["name"], arguments=item["arguments"]))
                  for _, item in sorted(calls.items())]
    message = _Message(content="".join(content) or None, reasoning_content="".join(reasoning) or None,
                       tool_calls=tool_calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
                           usage=_usage(final_usage))


def create(client: Any, *, require_usage: bool = True, **kwargs: Any) -> Any:
    """Request and assemble a streamed chat completion with final usage enabled."""
    options = dict(kwargs)
    options["stream"] = True
    stream_options = dict(options.get("stream_options") or {})
    stream_options["include_usage"] = True
    options["stream_options"] = stream_options
    timeout = options.get("timeout")
    deadline = time.monotonic() + timeout if isinstance(timeout, (int, float)) else None
    stream = client.chat.completions.create(**options)
    try:
        return assemble(stream, require_usage=require_usage, deadline=deadline)
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()


class StreamingChatAdapter:
    """Expose a streaming provider through the chat-completions interface used by ``PilotAgent``."""

    def __init__(self, client: Any, *, require_usage: bool = True):
        self.client = client
        self.require_usage = require_usage
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: Any) -> Any:
        if "tools" not in kwargs and kwargs.get("timeout") is not None:
            # V1 writers use a finite non-streaming request, not an unbounded stream assembly.
            return self.client.chat.completions.create(**kwargs)
        return create(self.client, require_usage=self.require_usage, **kwargs)
