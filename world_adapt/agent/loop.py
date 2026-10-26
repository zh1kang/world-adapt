"""The agent loop: ask the model, run its tool calls on the episode, repeat; then rewrite the notebook.

Everything else lives in its own module: prompt text (`prompts`), tool schemas and payloads (`tools`), model
calls (`llm`), context trimming (`context`), the notebook writer (`writer`), memory (`memory`), and budgets and
endings (`world_adapt.episode`).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable

from world_adapt.agent import context, llm, writer
from world_adapt.agent.prompts import (CUT_OFF_PLACEHOLDER, MISSING_TOOL_CALL, add_game_notes, cut_off_prompt,
                                       system_prompt, task_prompt)
from world_adapt.agent.tools import handle_call, tool_schemas
from world_adapt.environments.base import Game, Task
from world_adapt.episode import Budgets, Episode, TaskResult
from world_adapt.predictors.simulator import SimulatorPredictor


@dataclass(frozen=True)
class AgentConfig:
    model: str
    reasoning_effort: str | None = "high"
    notebook_tokens: int = 2000
    """Notebook cap in tokens of `memory.TOKENIZER`."""
    notebook_max_tokens: int = 200_000
    notebook_max_seconds: float = 900.0
    notebook_max_completion_tokens: int = 32_768
    """The writer's own estimated prompt+reply allowance, deadline and reply cap per update, separate from planning."""
    writer_context_tokens: int = 64_000
    writer_evidence_tokens: int = 56_000
    """Writer prompt cap, and the part of it available to public tool records plus omission metadata."""
    max_retries: int = 5
    retry_wait: float = 10.0
    """Seconds before the first API retry; doubles on each further retry (at most 120)."""
    completion_window: str | None = None
    """Sail scheduling hint sent as metadata.completion_window."""
    seed: int | None = None
    """Decoding seed sent to the provider; None sends none (not every provider supports one)."""
    max_completion_tokens: int | None = None
    """Cap on tokens per response (reasoning included); None means the provider default."""
    end_on_dead_state: bool = True
    """End the task once the solver proves the real state unsolvable (a terminal failure)."""
    dead_check_expansions: int = 300_000
    max_context_tokens: int = 80_000
    """Estimated planning prompt size above which the oldest tool exchanges are dropped before a call
    (`agent.context`). Set it with a margin below the model's window minus its reply cap."""
    keep_recent_turns: int = 4
    """Exchanges never dropped by context trimming (the most recent ones)."""
    require_usage: bool = False
    """Require complete token accounting from the provider."""
    action_tool_choice: str | None = None
    """Optional tool-selection policy for action turns; never applied to notebook requests."""

    def __post_init__(self) -> None:
        if self.notebook_tokens < 1:
            raise ValueError("notebook_tokens must be positive")
        if (self.notebook_max_tokens < 1 or self.notebook_max_completion_tokens < 1
                or not math.isfinite(self.notebook_max_seconds) or self.notebook_max_seconds <= 0):
            raise ValueError("writer resource limits must be positive and finite")
        if (self.max_context_tokens < 1000 or self.writer_context_tokens < 1000
                or not 32 <= self.writer_evidence_tokens <= self.writer_context_tokens):
            raise ValueError("planning and writer context caps must be at least 1000 tokens, and the writer "
                             "evidence allowance between 32 tokens and the writer context cap")


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _malformed_limit_reason(result: TaskResult, cutoff_replies: int) -> str:
    """Return the reason for the shared malformed-call budget boundary."""
    if cutoff_replies == result.malformed and cutoff_replies:
        return "reply_cutoff_limit"
    if cutoff_replies:
        return "response_limit"
    return "malformed_limit"


class PilotAgent:
    def __init__(self, client: llm.ChatClient, config: AgentConfig,
                 log: Callable[[dict[str, Any]], None] = lambda event: None,
                 evaluator_log: Callable[[dict[str, Any]], None] | None = None):
        self.client = client
        self.config = config
        self.log = log
        self.evaluator_log = evaluator_log
        self._writer_budget: writer.WriterBudget | None = None
        self._writer_call_ids: list[str] = []
        self._context: tuple[str, Callable[[], str], Episode] | None = None
        """Task identity, current observation, and owned episode while the task is active."""

    def _complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> Any:
        c = self.config
        if self._context is not None and self._writer_budget is None:
            task_id, observe, _ = self._context
            trimmed = context.fit(messages, c.max_context_tokens, c.keep_recent_turns, observe, tools)
            if trimmed:
                self.log({"task_id": task_id, **trimmed})
            if context.conversation_tokens(messages, tools) > c.max_context_tokens:
                self.log({"type": "planning_context_not_admitted", "task_id": task_id,
                          "cap": c.max_context_tokens, "estimated_tokens": context.conversation_tokens(messages, tools)})
                raise llm.InfrastructureError("planning request exceeds the declared context cap after fitting")
        planning = self._context[2] if self._context is not None and tools is not None else None

        def invoke(reply=c.max_completion_tokens, timeout=None):
            return llm.request(self.client, messages, tools, model=c.model, reasoning_effort=c.reasoning_effort,
                               max_completion_tokens=reply, completion_window=c.completion_window,
                               max_retries=1 if self._writer_budget is not None else c.max_retries,
                               log=self.log, seed=c.seed, retry_wait=c.retry_wait, timeout=timeout,
                               require_usage=c.require_usage, action_tool_choice=c.action_tool_choice,
                               deadline=planning.deadline if planning is not None else None,
                               planning_usage=planning.result.usage if planning is not None else None)

        if self._writer_budget is None:
            return invoke()
        audit = {"type": "notebook_input", "task_id": self._context[0] if self._context is not None else None,
                 "request_sha256": digest(messages), "estimated_prompt_tokens": context.conversation_tokens(messages),
                 "input_version": writer.INPUT_VERSION, "messages": json.loads(json.dumps(messages)),
                 "available_tool_call_ids": self._writer_call_ids,
                 "evidence_access": {**writer.access(messages),
                                     "estimated_evidence_tokens": writer.evidence_tokens(messages)}}
        if writer.evidence_tokens(messages) > c.writer_evidence_tokens:
            raise writer.WriterNotAdmitted("evidence_limit")
        if max(self._writer_budget.prompt_floor, context.conversation_tokens(messages)) > c.writer_context_tokens:
            raise writer.WriterNotAdmitted("context_limit")
        return self._writer_budget.request(messages, invoke, lambda: self.log(audit))

    def _opening(self, game: Game, episode: Episode, budgets: Budgets, notebook: str, goal: str,
                 rules: str | None = None) -> list[dict[str, Any]]:
        c = self.config
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt(budgets, c.end_on_dead_state, goal, tuple(game.actions()),
                                                         rules=rules)},
            {"role": "user", "content": task_prompt(episode.observe(), notebook)},
        ]
        add_game_notes(messages, game, budgets.max_chunk)
        return messages

    def _over_cap(self, episode: Episode, budgets: Budgets) -> str | None:
        if budgets.max_task_tokens is not None and episode.result.usage.get("total_tokens", 0) >= budgets.max_task_tokens:
            return "token_limit"
        if budgets.max_task_seconds is not None and episode.elapsed >= budgets.max_task_seconds:
            return "time_limit"
        return None

    @staticmethod
    def _time_up(episode: Episode, budgets: Budgets) -> bool:
        return episode.deadline is not None and episode.elapsed >= budgets.max_task_seconds

    def run_task(
        self, game: Game, task: Task, predictor: SimulatorPredictor, budgets: Budgets, notebook: str, goal: str,
        prune: Callable[[Any], bool] | None = None, episode_id: str | None = None, rules: str | None = None,
        commit_outcome: Callable[[TaskResult], None] | None = None,
    ) -> tuple[TaskResult, str]:
        """Play one task; return its result and the notebook the agent wrote afterwards.

        `commit_outcome` persists the scored result before the notebook request starts. Raises
        `llm.InfrastructureError` when the model API keeps failing; the task is then unscored.
        """
        c = self.config
        episode = Episode(game, task, predictor, budgets, end_on_dead_state=c.end_on_dead_state,
                          dead_check_expansions=c.dead_check_expansions, prune=prune, log=self.log,
                          evaluator_log=self.evaluator_log, episode_id=episode_id)
        schema_factory = getattr(game, "action_item_schema", None)
        tools = tool_schemas(budgets.max_chunk, tuple(game.actions()), schema_factory() if schema_factory else None)
        messages = self._opening(game, episode, budgets, notebook, goal, rules)
        # The writer sees the public instructions with an empty notebook; the previous notebook is a separate field.
        public_instructions = writer.public_messages(messages)
        public_instructions[1] = self._opening(game, episode, budgets, "", goal, rules)[1]
        public_records = []
        result = episode.result
        self._context = (task.task_id, episode.observe, episode)
        self.log({"type": "task_start", "task_id": task.task_id, "episode_id": episode.episode_id, "notebook": notebook,
                  "budgets": asdict(budgets), "messages": list(messages), "tools": tools,
                  "prompt_hash": digest(messages[0]["content"]), "tools_hash": digest(tools),
                  "reply_cutoff_recording_version": llm.REPLY_CUTOFF_RECORDING_VERSION})

        cutoff_replies = 0
        while result.turns < budgets.max_turns:
            cap = self._over_cap(episode, budgets)
            if cap:
                result.end_reason = cap
                break
            result.turns += 1
            try:
                response = self._complete(messages, tools)
            except llm.PlanningDeadlineExceeded:
                result.end_reason = "time_limit"
                break
            llm.add_usage(result.usage, response)
            if self._time_up(episode, budgets):
                self.log({"type": "planning_response_rejected", "task_id": task.task_id,
                          "reason": "time_limit", "turn": result.turns})
                result.end_reason = "time_limit"
                break
            choice = response.choices[0]
            message, truncated = choice.message, llm.cut_off(choice)
            # A cut-off reply is mostly unfinished reasoning; keeping it would only fill the context.
            messages.append({"role": "assistant", "content": CUT_OFF_PLACEHOLDER} if truncated
                            else llm.assistant_message(message))
            self.log({"type": "assistant", "task_id": task.task_id, "turn": result.turns, "content": message.content,
                      "finish_reason": getattr(choice, "finish_reason", None),
                      "tool_calls": [call.model_dump() for call in message.tool_calls or []]})
            if truncated:
                cutoff_replies += 1
                self.log({"type": "reply_cutoff", "task_id": task.task_id, "turn": result.turns,
                          "finish_reason": getattr(choice, "finish_reason", None),
                          "rejected_calls": len(message.tool_calls or [])})
                episode.record_malformed()
                messages.append({"role": "user", "content": cut_off_prompt(c.max_completion_tokens)})
                if result.malformed >= budgets.max_malformed:
                    result.end_reason = _malformed_limit_reason(result, cutoff_replies)
                    break
                continue
            if not message.tool_calls:
                episode.record_malformed()
                messages.append({"role": "user", "content": MISSING_TOOL_CALL})
                if result.malformed >= budgets.max_malformed:
                    result.end_reason = _malformed_limit_reason(result, cutoff_replies)
                    break
                continue
            ended = False
            dispatch = True
            for call in message.tool_calls:
                ref = None
                if ended:
                    payload: dict[str, Any] = {"error": "task already ended"}
                elif not dispatch:
                    payload = {"error": "tool call not executed because malformed-call limit was reached"}
                elif self._time_up(episode, budgets):
                    result.end_reason = "time_limit"
                    ended = True
                    payload = {"error": "tool call not executed because the task time limit was reached"}
                else:
                    malformed_before = result.malformed
                    payload, ref = handle_call(episode, call)
                    ended = episode.ended
                    if result.malformed >= budgets.max_malformed and result.malformed > malformed_before:
                        dispatch = False
                content = json.dumps(payload)
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.function.name, "content": content})
                public_records.append(writer.record(episode.episode_id, len(public_records) + 1,
                                                    call_id=call.id, name=call.function.name,
                                                    arguments=call.function.arguments, content=content))
                self.log({"type": "tool_result", "task_id": task.task_id, "turn": result.turns, "call_id": call.id,
                          "name": call.function.name, "arguments": call.function.arguments, "ref": ref,
                          "content": content, "invalid_arguments": llm.invalid_arguments(call)})
            if ended:
                if result.end_reason != "time_limit":
                    episode.settle_end_reason()
                break
            if result.malformed >= budgets.max_malformed:
                result.end_reason = _malformed_limit_reason(result, cutoff_replies)
                break

        episode.close()
        result.seconds = round(episode.elapsed, 3)
        if commit_outcome is not None:
            commit_outcome(result)
        experience = {"input_version": writer.INPUT_VERSION, "task_id": task.task_id,
                      "messages": public_instructions, "current_observation": episode.observe(),
                      "outcome": {k: v for k, v in asdict(result).items() if k not in ("diagnostics", "usage", "seconds")},
                      "public_records": public_records,
                      "information_limits": {"context_tokens": c.writer_context_tokens,
                                             "evidence_tokens": c.writer_evidence_tokens}}
        self.log({"type": "writer_experience", "task_id": task.task_id, "experience": experience})
        self._writer_budget = writer.WriterBudget(
            task.task_id, result.usage, self.log, c.notebook_max_tokens, c.notebook_max_seconds,
            c.notebook_max_completion_tokens)
        messages, self._writer_call_ids, _, self._writer_budget.prompt_floor = writer.prepare_input(
            experience, notebook, c.notebook_tokens)
        self.log({"type": "writer_evidence_selection", "task_id": task.task_id,
                  **writer.access(messages), "estimated_evidence_tokens": writer.evidence_tokens(messages),
                  "evidence_cap": c.writer_evidence_tokens, "context_cap": c.writer_context_tokens,
                  "estimated_prompt_tokens": context.conversation_tokens(messages)})
        try:
            new_notebook = writer.rewrite(messages, result, self._complete, self.log, cap=c.notebook_tokens,
                                          previous=notebook, retry_wait=c.retry_wait)
        finally:
            self._writer_budget = None
            self._writer_call_ids = []
            self._context = None
        result.seconds = round(episode.elapsed, 3)
        return result, new_notebook
