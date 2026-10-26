"""Talking to the model: one chat-completions call with retries, plus usage accounting.

Any client exposing `chat.completions.create(**kwargs)` works: the OpenAI SDK
pointed at a provider, `StreamingChatAdapter` around it, or a scripted fake in tests.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Callable, Protocol

from world_adapt.agent.errors import InfrastructureError
from world_adapt.agent.spending import SpendLimitExceeded


class ChatClient(Protocol):
    """The subset of the OpenAI SDK client the loop uses (also satisfied by test fakes)."""

    chat: Any


class PlanningDeadlineExceeded(InfrastructureError):
    """No further planning submission is allowed within the task's deadline."""


REQUEST_TIMEOUT_SECONDS = 900.0
"""Shared transport ceiling; a longer task allowance must not lengthen one request."""

RETRYABLE_STATUSES = (408, 429, 500, 502, 503, 504, 529)
"""HTTP statuses worth another identical request; 529 is Anthropic's overloaded error."""

MAX_RETRY_AFTER_SECONDS = 60.0
"""Longest server-requested `Retry-After` wait that is honoured."""

REPLY_CUTOFF_RECORDING_VERSION = 1
"""Version of the per-task cutoff event contract written by the current harness."""


def retry_delay(error: BaseException, default: float) -> float:
    """Seconds to wait before retrying: the server's `Retry-After` for 429 and 5xx replies, bounded, else `default`."""
    headers = getattr(getattr(error, "response", None), "headers", None)
    status = getattr(error, "status_code", None)
    if headers is None or status not in (429, 503, 529):
        return default
    try:
        seconds = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return default
    return min(seconds, MAX_RETRY_AFTER_SECONDS) if math.isfinite(seconds) and seconds >= 0 else default


def transient_before_reply(error: BaseException | None) -> bool:
    """A 429 or 5xx status, or a connection failure, so no reply was produced and a new request is safe.

    Timeouts and a stream cut off after it started are not: the provider may still have generated and billed a reply.
    """
    status = getattr(error, "status_code", None)
    if type(status) is int:
        return status == 429 or 500 <= status < 600
    return isinstance(error, ConnectionError) or type(error).__name__ == "APIConnectionError"


def request(client: ChatClient, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, *, model: str,
            reasoning_effort: str | None, max_completion_tokens: int | None, completion_window: str | None,
            max_retries: int, log: Callable[[dict[str, Any]], None], seed: int | None = None,
            retry_wait: float = 10.0, require_usage: bool = False,
            action_tool_choice: str | None = None, timeout: float | None = None,
            deadline: float | None = None, planning_usage: dict[str, int] | None = None) -> Any:
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        if action_tool_choice is not None:
            kwargs["tool_choice"] = action_tool_choice
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if max_completion_tokens:
        kwargs["max_completion_tokens"] = max_completion_tokens
    if completion_window:
        kwargs["metadata"] = {"completion_window": completion_window}
    if seed is not None:
        kwargs["seed"] = seed
    if timeout is not None:
        kwargs["timeout"] = timeout
    if planning_usage is not None:
        planning_usage.setdefault("planning_requests", 0)
    last_error = None
    for attempt in range(max_retries):
        options = dict(kwargs)
        if planning_usage is not None and timeout is None:
            options["timeout"] = REQUEST_TIMEOUT_SECONDS
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log({"type": "planning_request_not_admitted", "attempt": attempt, "reason": "time_limit"})
                if last_error is not None:
                    raise InfrastructureError("planning infrastructure failure exhausted the task deadline") from last_error
                raise PlanningDeadlineExceeded("planning task deadline reached")
            ceiling = timeout if timeout is not None else REQUEST_TIMEOUT_SECONDS
            options["timeout"] = min(ceiling, remaining)
        if planning_usage is not None:
            planning_usage["planning_requests"] += 1
        try:
            response = client.chat.completions.create(**options)
        except SpendLimitExceeded:
            raise
        except Exception as error:  # Network and rate-limit errors are infrastructure failures.
            last_error = error
            if planning_usage is not None:
                # A missing reply does not prove that remote computation was cancelled or free.
                planning_usage["planning_unknown_requests"] = planning_usage.get("planning_unknown_requests", 0) + 1
            log({"type": "api_error", "attempt": attempt, "error": repr(error)})
            status = getattr(error, "status_code", None)
            if isinstance(status, int) and status not in RETRYABLE_STATUSES:
                raise InfrastructureError(f"provider rejected the request with HTTP {status}; no identical retry") from error
            if deadline is not None and time.monotonic() >= deadline:
                raise InfrastructureError("planning response unknown at the task deadline") from error
            if attempt == max_retries - 1:
                raise InfrastructureError(f"model API failed {max_retries} times: {error!r}") from error
            wait = retry_delay(error, min(120, 2**attempt * retry_wait))
            if deadline is not None:
                wait = min(wait, max(0.0, deadline - time.monotonic()))
            time.sleep(wait)
            continue
        try:
            return validate_response(response, require_usage=require_usage)
        except InfrastructureError as error:
            log({"type": "response_validation_error", "error": str(error), "usage_required": require_usage})
            raise


def add_usage(usage: dict[str, int], response: Any) -> None:
    record = getattr(response, "usage", None)
    values = {key: getattr(record, key, None) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    if (any(type(value) is not int or value < 0 for value in values.values())
            or values["total_tokens"] < values["prompt_tokens"] + values["completion_tokens"]):
        usage["usage_unknown_requests"] = usage.get("usage_unknown_requests", 0) + 1
        return
    for key, value in values.items():
        usage[key] = usage.get(key, 0) + value


def validate_response(response: Any, *, require_usage: bool = False) -> Any:
    """Reject incomplete provider responses before the agent can execute any tool call.

    Legacy callers may accept a response without usage and will keep recording unknown usage.  A paid run opts in
    with ``require_usage`` so missing or inconsistent accounting becomes an infrastructure failure for the task.
    """
    choices = getattr(response, "choices", None)
    if not isinstance(choices, (list, tuple)) or not choices:
        raise InfrastructureError("model response did not contain a choice")
    if getattr(choices[0], "message", None) is None:
        raise InfrastructureError("model response choice did not contain a message")
    calls = getattr(choices[0].message, "tool_calls", None)
    if calls is not None:
        if not isinstance(calls, (list, tuple)):
            raise InfrastructureError("model tool calls must be a list")
        for call in calls:
            function = getattr(call, 'function', None)
            if (not isinstance(getattr(call, 'id', None), str) or not call.id
                    or not isinstance(getattr(function, 'name', None), str) or not function.name):
                raise InfrastructureError("model tool call identity must contain nonempty text fields")
            try:
                json.dumps(getattr(function, 'arguments', None), allow_nan=False)
            except (ValueError, TypeError) as error:
                raise InfrastructureError("model tool call arguments are not JSON-compatible") from error
    if not require_usage:
        return response
    usage = getattr(response, "usage", None)
    if usage is None:
        raise InfrastructureError("model response did not contain usage for required accounting")
    values = {key: getattr(usage, key, None) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values.values()):
        raise InfrastructureError("model response usage must contain nonnegative integer token counts")
    if values["total_tokens"] < values["prompt_tokens"] + values["completion_tokens"]:
        raise InfrastructureError("model response total_tokens is smaller than prompt_tokens + completion_tokens")
    return response


CUT_OFF_FINISH_REASONS = ("length", "incomplete")
"""Chat-completions `length`, and `incomplete`, which some OpenAI-compatible servers report instead."""


def cut_off(choice: Any) -> bool:
    """The reply stopped at the completion-token limit and its entire tool-call batch is incomplete."""
    return getattr(choice, "finish_reason", None) in CUT_OFF_FINISH_REASONS


INVALID_ARGUMENTS_PREVIEW = 400
"""Characters of a malformed `arguments` string kept in the sanitized history."""


def invalid_arguments(call: Any) -> str | None:
    """Why arguments are not an encoded object; None also permits an absent empty argument object."""
    if call.function.arguments is not None and not isinstance(call.function.arguments, str):
        return f"not a JSON string ({type(call.function.arguments).__name__})"
    try:
        parsed = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError as error:
        return f"not valid JSON ({error.msg} at position {error.pos})"
    return None if isinstance(parsed, dict) else f"a JSON {type(parsed).__name__}, not an object"


def assistant_message(message: Any) -> dict[str, Any]:
    """The complete assistant message, including reasoning_content, as Kimi requires for multi-turn tool use.

    Tool calls whose arguments are not valid JSON are rewritten to a valid object that quotes the original text:
    providers validate the history on every request, so echoing the raw string back would make the whole
    conversation unusable (HTTP 400 on every retry). The call itself is still a malformed call for the episode.
    """
    data = message.model_dump(exclude_none=True)
    for call, dumped in zip(message.tool_calls or [], data.get("tool_calls") or []):
        reason = invalid_arguments(call)
        if call.function.arguments is None:
            dumped["function"]["arguments"] = "{}"
        elif reason is not None:
            raw = call.function.arguments
            preview = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            dumped["function"]["arguments"] = json.dumps({"invalid_arguments": preview[:INVALID_ARGUMENTS_PREVIEW],
                                                          "problem": reason})
    return data
