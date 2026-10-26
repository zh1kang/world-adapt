"""The agent's tools: their JSON schemas, and turning one tool call into the JSON the model sees.

Budgets, endings and logging are the episode's job (`world_adapt.episode`); this
module only parses calls and formats results.
"""

from __future__ import annotations

import json
from typing import Any

from world_adapt.agent.llm import invalid_arguments
from world_adapt.environments.base import InvalidAction
from world_adapt.episode import Episode


def tool_schemas(max_chunk: int, actions_list: tuple[str, ...],
                 action_item_schema: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    actions = {
        "type": "array",
        "items": action_item_schema if action_item_schema is not None else {"type": "string", "enum": list(actions_list)},
        "minItems": 1,
        "maxItems": max_chunk,
        "description": f"1 to {max_chunk} actions, applied in order.",
    }
    return [
        {
            "type": "function",
            "function": {
                "name": "predict",
                "description": (
                    "Simulate actions from the current real state WITHOUT changing the real environment. "
                    "Returns hypothetical observations. The prediction tool may be imperfect."
                ),
                "parameters": {"type": "object", "properties": {"actions": actions}, "required": ["actions"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute",
                "description": "Perform actions in the REAL environment, open loop. Returns the real observation after each action.",
                "parameters": {"type": "object", "properties": {"actions": actions}, "required": ["actions"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "concede",
                "description": "End this task now because you believe it can no longer be completed. Counts as a failure.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def handle_call(episode: Episode, call: Any) -> tuple[dict[str, Any], str | None]:
    """Run one tool call against the episode; return the payload shown to the model and its query/execution ID."""
    name = call.function.name
    ref = None
    reason = invalid_arguments(call)
    if reason is not None:
        episode.record_malformed()
        return {"error": f"tool arguments were {reason}; send a JSON object such as "
                         "{\"actions\": [\"...\"]}", "budget_remaining": episode.budget_remaining()}, ref
    if name == "concede":
        episode.concede()
        return {"kind": "task over", "note": "You conceded; the task ended as a failure."}, ref
    try:
        actions = episode.check_chunk(json.loads(call.function.arguments or "{}").get("actions"))
        if name == "predict":
            prediction = episode.predict(actions)
            payload = {
                "kind": "HYPOTHETICAL prediction (real environment unchanged)",
                "steps": [{"step": s.index, "action": s.action, "observation": s.observation} for s in prediction.steps],
                "predicted_goal_reached": bool(prediction.steps and prediction.steps[-1].predicted_success),
            }
            episode.log_prediction(actions)
            ref = episode.last_query_id
        elif name == "execute":
            outcome = episode.execute(actions)
            ref = outcome.execution_id
            execution = outcome.execution
            payload = {
                "kind": "REAL execution",
                "steps": [{"step": s.index, "action": s.action, "observation": s.observation} for s in execution.steps],
                "goal_reached": execution.success,
            }
            if outcome.cut_to is not None:
                payload["note"] = f"only the first {outcome.cut_to} actions were executed: real-action budget reached"
            if outcome.task_over:
                payload["task_over"] = outcome.task_over
        else:
            raise InvalidAction(f"unknown tool {name!r}")
    except (InvalidAction, json.JSONDecodeError) as error:
        episode.record_malformed()
        payload = {"error": str(error)}
    payload["budget_remaining"] = episode.budget_remaining()
    return payload, ref
