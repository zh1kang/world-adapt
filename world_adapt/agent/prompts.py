"""All text the agent harness sends to the model while it plays a task.

Environment-specific additions come from the game's optional `prompt_notes(max_chunk)` hook, so this module never
names a particular environment. The notebook writer's text lives in `world_adapt.agent.writer`.
"""

from __future__ import annotations

from world_adapt.episode import Budgets


def system_prompt(budgets: Budgets, end_on_dead_state: bool, goal: str, actions_list: tuple[str, ...],
                  rules: str | None = None) -> str:
    mechanics = (f"Environment mechanics: {rules}\n\n" if rules
                 else "How things move and interact is not described; infer it from what you observe.\n\n")
    return (
        f"You are playing a turn-based puzzle. Goal: {goal}\n"
        f"Available actions: {', '.join(actions_list)}.\n"
        f"{mechanics}"
        "Tools:\n"
        "- predict(actions): simulates a hypothetical action sequence from the current real state and returns "
        "predicted observations. It never changes the real environment. The prediction tool may be imperfect.\n"
        "- execute(actions): performs the actions in the real environment, in order, without interruption, "
        "and returns the real observation after each action.\n"
        "- concede(): ends the task as a failure, if you are certain it can no longer be completed.\n\n"
        "Budgets:\n"
        f"- Real actions: {budgets.real_actions} per task (every action counts, including ones that move nothing).\n"
        f"- Predicted actions: {budgets.predicted_transitions} per task (each requested action counts).\n"
        f"- At most {budgets.max_chunk} actions per tool call.\n"
        "The task ends when the goal is reached, when the real-action budget is spent, or when the environment ends the task"
        + (", including as soon as the goal can no longer be reached from the real state.\n" if end_on_dead_state else ".\n")
        + "Always respond with a tool call."
    )


def task_prompt(observation: str, notebook: str) -> str:
    memory = notebook.strip() or "(empty)"
    return f"Your notebook from previous tasks (written by you):\n{memory}\n\nNew task. Current real observation:\n{observation}"


MISSING_TOOL_CALL = "Respond with a tool call."
CUT_OFF_PLACEHOLDER = "[reply cut off at the token limit before any tool call]"
"""Stands in for a cut-off reply in the history: chat templates such as Gemma's require alternating roles."""


def cut_off_prompt(limit: int | None) -> str:
    size = f"the {limit}-token reply limit" if limit else "the reply token limit"
    return (f"Your reply reached {size} before any tool call, so it was discarded. Think more briefly, "
            "then respond with a tool call.")


def add_game_notes(messages: list, game, max_chunk: int) -> None:
    """Append the game's own notes, if it has any, to the system prompt."""
    hook = getattr(game, "prompt_notes", None)
    if hook is not None:
        messages[0]["content"] += hook(max_chunk)
