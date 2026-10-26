"""Claude through the Anthropic Messages API, behind the chat-completions interface the agent loop uses.

`AnthropicChatAdapter` turns chat messages and tool schemas into one streamed Messages request and the reply back
into a chat response with token usage, so the loop, writer and spend ledger need no Anthropic-specific code.

Claude's thinking blocks must be sent back unchanged during a tool-use exchange, but chat messages cannot carry
them. The adapter therefore keeps the native blocks of every reply it produced, keyed by a hash of the public
message, and restores them when that message returns in the history. They never reach the logs or the notebook
writer. A request with no assistant message starts a new conversation and forgets them.
"""

from __future__ import annotations

import hashlib
import json
import time
from types import SimpleNamespace
from typing import Any

from world_adapt.agent.errors import InfrastructureError
from world_adapt.agent.llm import REQUEST_TIMEOUT_SECONDS
from world_adapt.agent.streaming import _Call, _Message

EFFORTS = ("low", "medium", "high", "xhigh", "max")
"""Values of `output_config.effort`."""
THINKING_BINDING_BETA = "thinking-binding-controls-2026-08-01"
"""Lets a request drop a thinking block that fails the conversation check instead of failing with HTTP 400."""
FINISH_REASONS = {"max_tokens": "length", "tool_use": "tool_calls", "end_turn": "stop"}


def message_key(message: dict[str, Any]) -> str:
    public = {key: message[key] for key in ("role", "content", "tool_calls") if key in message}
    return hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()


def native_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"name": tool["function"]["name"], "description": tool["function"]["description"],
             "input_schema": tool["function"]["parameters"]} for tool in tools]


class AnthropicChatAdapter:
    def __init__(self, client: Any):
        self.client = client
        self.history: dict[str, list[dict[str, Any]]] = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def translate(self, messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """The system prompt and the alternating user/assistant messages of the Messages API."""
        if not any(message["role"] == "assistant" for message in messages):
            self.history.clear()
        system: list[str] = []
        native: list[dict[str, Any]] = []
        for message in messages:
            role = message["role"]
            if role == "system":
                if native or not isinstance(message["content"], str):
                    raise ValueError("only leading text system messages are supported")
                system.append(message["content"])
                continue
            if role == "tool":
                role = "user"
                blocks = [{"type": "tool_result", "tool_use_id": message["tool_call_id"], "content": message["content"]}]
            elif role == "assistant":
                blocks = self.history.get(message_key(message))
                if blocks is None:
                    if message.get("tool_calls"):
                        raise ValueError("the native blocks of an assistant tool call are missing")
                    blocks = [{"type": "text", "text": message.get("content") or "[empty assistant reply]"}]
            elif role == "user" and isinstance(message.get("content"), str):
                blocks = [{"type": "text", "text": message["content"]}]
            else:
                raise ValueError(f"unsupported {role} message")
            if native and native[-1]["role"] == role:
                native[-1]["content"].extend(blocks)  # Tool results and a following prompt share one user turn.
            else:
                native.append({"role": role, "content": list(blocks)})
        return "\n\n".join(system), native

    def options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if kwargs.get("seed") is not None or kwargs.get("metadata") or kwargs.get("tool_choice") not in (None, "auto"):
            raise ValueError("seed, completion windows and forced tool choice are not supported")
        if not kwargs.get("max_completion_tokens"):
            raise ValueError("the Messages API requires max_completion_tokens")
        system, messages = self.translate(kwargs["messages"])
        options: dict[str, Any] = {"model": kwargs["model"], "messages": messages, "system": system,
                                   "max_tokens": kwargs["max_completion_tokens"]}
        effort = kwargs.get("reasoning_effort")
        if effort == "none":
            options["thinking"] = {"type": "disabled"}
        elif effort in EFFORTS:
            options["thinking"] = {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}}
            options["output_config"] = {"effort": effort}
            options["betas"] = [THINKING_BINDING_BETA]
        elif effort is not None:
            raise ValueError(f"reasoning effort must be none or one of {', '.join(EFFORTS)}")
        if kwargs.get("tools"):
            options["tools"] = native_tools(kwargs["tools"])
        return options

    def create(self, **kwargs: Any) -> Any:
        options = self.options(kwargs)
        timeout = kwargs.get("timeout") or REQUEST_TIMEOUT_SECONDS
        deadline = time.monotonic() + timeout
        # Errors raised while opening the stream carry the HTTP status; anything later is a cut-off stream.
        with self.client.beta.messages.stream(**options, timeout=timeout) as stream:
            try:
                for _ in stream:
                    if time.monotonic() >= deadline:
                        raise InfrastructureError("stream exceeded the request deadline; remote completion is uncertain")
                reply = stream.get_final_message()
            except InfrastructureError:
                raise
            except Exception as error:
                raise InfrastructureError(f"stream failed before a complete response: {error!r}") from error
        if reply.model != kwargs["model"]:
            raise InfrastructureError(f"served model {reply.model!r} differs from the requested {kwargs['model']!r}")
        usage = self.usage(reply.usage)
        reason = FINISH_REASONS.get(reply.stop_reason)
        if reason is None:
            raise InfrastructureError(f"unsupported stop reason {reply.stop_reason!r}")
        calls, texts, blocks = [], [], []
        for block in reply.content:
            blocks.append(block.model_dump(exclude_none=True))
            if block.type == "tool_use":
                calls.append(_Call(id=block.id, function=SimpleNamespace(name=block.name,
                                                                         arguments=json.dumps(block.input))))
            elif block.type == "text":
                texts.append(block.text)
            elif block.type not in ("thinking", "redacted_thinking"):
                raise InfrastructureError(f"unsupported response block {block.type!r}")
        message = _Message(content="".join(texts) or None, tool_calls=calls or None, reasoning_content=None)
        self.history[message_key(message.model_dump(exclude_none=True))] = blocks
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=reason)], usage=usage)

    @staticmethod
    def usage(usage: Any) -> SimpleNamespace:
        """Chat-style usage. Cache reads count as prompt tokens, and are also reported as cached tokens."""
        if getattr(usage, "cache_creation_input_tokens", 0):
            raise InfrastructureError("cache writes are billed at a surcharge and are not accounted")
        cached = getattr(usage, "cache_read_input_tokens", 0) or 0
        prompt = getattr(usage, "input_tokens", None)
        completion = getattr(usage, "output_tokens", None)
        if any(type(value) is not int or value < 0 for value in (prompt, completion, cached)):
            raise InfrastructureError("response usage must contain nonnegative integer token counts")
        prompt += cached
        return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion,
                               prompt_tokens_details=SimpleNamespace(cached_tokens=cached))
