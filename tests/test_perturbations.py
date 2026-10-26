"""Faulty predictors: atom composition, each environment's fixed error, visible-change tracking, and matched feedback."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Sequence

import pytest

from tests.test_environments import generated, maze
from world_adapt.environments.base import InvalidState, Task, Transition
from world_adapt.environments.iceslide.game import IceLayout, IceSlide, IceState
from world_adapt.environments.patrol.game import Patrol, PatrolLayout, PatrolState
from world_adapt.environments.portalmaze.game import PortalMaze
from world_adapt.environments.switchboard.game import Switchboard, SwitchState, Wiring
from world_adapt.episode import Budgets, Episode
from world_adapt.feedback import PredictionFeedback
from world_adapt.perturbations import registry
from world_adapt.perturbations.base import IDENTITY, Atom, Condition, PerturbedDynamics, Profile
from world_adapt.predictors.simulator import SimulatorPredictor


# A toy game with a latent counter -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ToyState:
    latent: int = 0
    visible: int = 0
    done: bool = False


class ToyGame:
    name, version = "toy", "test"

    def actions(self) -> Sequence[str]:
        return ("advance", "finish")

    def parse_action(self, raw: str) -> str:
        if raw not in self.actions():
            raise ValueError(raw)
        return raw

    def step(self, state: ToyState, action: str) -> Transition[ToyState]:
        if state.done:
            return Transition(state, ("terminal",))
        if action == "advance":
            return Transition(ToyState(state.latent + 1, state.visible + state.latent, False))
        return Transition(ToyState(state.latent, state.visible, True))

    def is_success(self, state: ToyState) -> bool:
        return state.done and state.visible >= 2

    def is_terminal(self, state: ToyState) -> bool:
        return state.done

    def observe(self, state: ToyState) -> str:
        return f"visible={state.visible}"

    def validate_state(self, state: ToyState) -> None:
        if not isinstance(state, ToyState) or state.latent < 0:
            raise InvalidState("invalid toy state")

    def spec(self) -> dict:
        return {"name": self.name}


TOY = ToyGame()
ALWAYS = Condition("always", lambda c: True, leaves=0)


def kick(amount: int):
    def effect(ctx):
        s = ctx.current.state
        return Transition(ToyState(s.latent + amount, s.visible, s.done), ctx.current.events)
    return effect


LATENT_KICK = Atom("latent_kick", "motion", Condition("initial_latent", lambda c: c.state.latent == 0), kick(1))


def test_conditions_compose_and_every_environment_has_one_faulty_profile():
    yes, no = Condition("yes", lambda c: True), Condition("no", lambda c: False)
    assert (yes & no)(None) is False and (yes | no)(None) is True and (~no)(None) is True
    assert (yes & no | yes).leaves == 3
    assert IDENTITY.is_identity and IDENTITY.complexity == 0
    for env in ("depot", "iceslide", "patrol", "portalmaze", "pusht", "switchboard"):
        profiles = registry.profiles(env)
        assert list(profiles)[0] == "identity" and len(profiles) == 2
        faulty = list(profiles.values())[1]
        assert faulty.describe()["atoms"] and faulty.complexity >= 1


def test_atoms_apply_in_profile_order_and_each_sees_the_previous_result():
    then = Atom("then", "motion", Condition("kicked", lambda c: c.current.state.latent == 2), kick(10))
    first = Atom("first", "motion", ALWAYS, kick(1))
    forward = PerturbedDynamics(TOY, Profile("a", "1", (first, then))).step(ToyState(), "advance")
    reverse = PerturbedDynamics(TOY, Profile("b", "1", (then, first))).step(ToyState(), "advance")
    assert forward.transition.state.latent == 12 and forward.activated == ("first", "then")
    assert reverse.activated == ("first",)


def test_an_effect_that_produces_an_invalid_state_is_discarded_and_recorded():
    broken = Atom("broken", "motion", ALWAYS, kick(-5))
    result = PerturbedDynamics(TOY, Profile("broken", "1", (broken,))).step(ToyState(), "advance")
    assert result.rejected == ("broken",) and result.changed == ()
    assert result.transition.state == TOY.step(ToyState(), "advance").state


def test_visible_change_tracks_latent_errors_only_when_the_agent_could_see_them():
    profile = Profile("latent-kick", "1", (LATENT_KICK,))
    prediction, trace = SimulatorPredictor(TOY, profile).predict(ToyState(), ["advance", "advance"])
    assert trace.activated == trace.changed == (("latent_kick",), ())
    assert trace.visible_changed == (False, True)
    assert [s.observation for s in prediction.steps] == ["visible=0", "visible=2"]
    _, accurate = SimulatorPredictor(TOY, IDENTITY).predict(ToyState(), ["advance", "finish"])
    assert accurate.visible_changed == (False, False)


def test_extra_or_missing_predicted_steps_count_as_visible_changes():
    def finish(ctx):
        return Transition(replace(ctx.current.state, done=True), ctx.current.events)

    def keep_going(ctx):
        return Transition(replace(ctx.current.state, done=False), ctx.current.events)

    early = Profile("early", "1", (Atom("early", "motion", ALWAYS, finish),))
    assert SimulatorPredictor(TOY, early).predict(ToyState(), ["advance", "advance"])[1].visible_changed == (True, True)
    late = Profile("late", "1", (Atom("late", "motion", ALWAYS, keep_going),))
    assert SimulatorPredictor(TOY, late).predict(ToyState(), ["finish", "advance"])[1].visible_changed == (True, True)


# Each environment's fixed error -------------------------------------------------------------------------------------


def faulty_step(env: str, game, state, action):
    profile = list(registry.profiles(env).values())[1]
    return PerturbedDynamics(game, profile).step(state, action)


def test_depot_predicts_false_cargo_and_stacking_prerequisites_without_touching_reality():
    game, levels = generated("depot")
    for level in levels:
        state, changes = level.state, set()
        for action in level.solution:
            before = hash(state)
            result = faulty_step("depot", game, state, action)
            assert not result.rejected and hash(state) == before
            changes.update(result.changed)
            game.validate_state(result.transition.state)
            state = game.step(state, action).state
        assert changes == {"single_cargo", "no_stacking"} and game.is_success(state)


def test_switchboard_predicts_a_breaker_disabling_a_different_enabled_button():
    game = Switchboard(buttons=7)
    wiring = Wiring(rows=1, cols=4, toggles=(1, 2, 4, 8, 1, 2, 4), breaks=(3, -1, -1, -1, -1, -1, -1))
    cued = SwitchState(wiring=wiring, lamps=0, disabled=1 << 1)
    result = faulty_step("switchboard", game, cued, "press 1")
    assert result.changed == ("breaker_retarget_enabled",)
    assert result.transition.state.disabled == (1 << 1) | (1 << 2)
    assert game.step(cued, "press 1").state.disabled == (1 << 1) | (1 << 3)
    assert faulty_step("switchboard", game, replace(cued, disabled=0), "press 1").changed == ()


@pytest.mark.parametrize("held,changed", [(frozenset(), False), (frozenset({"b"}), True), (frozenset({"a"}), False)])
def test_portalmaze_predicts_a_locked_door_opening_while_an_unrelated_key_is_held(held, changed):
    state = replace(maze("########\n#@ABXab#\n########"), held=held)
    result = faulty_step("portalmaze", PortalMaze(), state, "right")
    assert bool(result.changed) == changed and not result.rejected


def test_portalmaze_predicts_a_safe_pad_leading_back_along_an_incoming_link():
    state = maze("#############\n#@0.1..2X#34#\n#############", links=(1, 2, 0, 4, 3))
    real = PortalMaze().step(state, "right").state
    predicted = faulty_step("portalmaze", PortalMaze(), state, "right")
    assert real.agent == (1, 4) and predicted.changed == ("reverse_portal",)
    assert predicted.transition.state.agent == (1, 7)


def test_patrol_predicts_capture_beside_a_guard_as_a_false_alarm():
    walls = frozenset((r, c) for r in range(5) for c in range(9) if r in (0, 4) or c in (0, 8))
    state = PatrolState(PatrolLayout(5, 9, walls, (3, 7), (((2, 3), (2, 2)),)), (2, 1), 0)
    assert Patrol().step(state, "wait").state.status == "playing"
    result = faulty_step("patrol", Patrol(), state, "wait")
    assert result.changed == ("adjacent_capture",) and result.transition.state.status == "caught"
    far = PatrolState(PatrolLayout(5, 9, walls, (3, 7), (((2, 5), (2, 4)),)), (2, 1), 0)
    assert faulty_step("patrol", Patrol(), far, "wait").changed == ()


def ice(text: str) -> IceState:
    """'#' wall, '.' floor, 'r' rough, 'o' socket, 'A' agent, 'C' crate."""
    walls, rough, sockets, crates, agent = set(), set(), set(), set(), None
    rows = text.split("\n")
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            walls |= {(r, c)} if ch == "#" else set()
            rough |= {(r, c)} if ch == "r" else set()
            sockets |= {(r, c)} if ch == "o" else set()
            crates |= {(r, c)} if ch == "C" else set()
            agent = (r, c) if ch == "A" else agent
    return IceState(IceLayout(len(rows), max(map(len, rows)), frozenset(walls), frozenset(rough), frozenset(sockets)),
                    agent, frozenset(crates))


@pytest.mark.parametrize("text,action,atom", [
    ("########\n#AC....#\n#o.....#\n########", "right", "crate_stops_early"),
    ("#########\n#AC..r..#\n#o......#\n#########", "right", "crate_slides_past_rough"),
    ("####\n#A.#\n#C.#\n#..#\n#o.#\n####", "down", "socket_vertical_no_capture"),
    ("########\n#AC.#..#\n#o.....#\n########", "right", None),
])
def test_iceslide_mispredicts_only_launch_outcomes(text, action, atom):
    game, state = IceSlide(), ice(text)
    nominal = game.step(state, action)
    result = faulty_step("iceslide", game, state, action)
    if atom is None:
        assert result.changed == () and result.transition.state == nominal.state
    else:
        assert result.changed == (atom,) and result.transition.state != nominal.state
        game.validate_state(result.transition.state)
    assert game.step(state, action) == nominal


# Matched prediction feedback ----------------------------------------------------------------------------------------


class Observer:
    def observe(self, state):
        return str(state)


def record(action, state):
    return SimpleNamespace(action=action, next_state=state)


def test_feedback_matches_executed_prefixes_across_chunks_and_stops_at_divergence():
    f = PredictionFeedback()
    f.predict("q1", 0, ["a", "b", "c"], ["1", "WRONG", "3"])
    f.predict("unexecuted", 0, ["x"], ["bad"])
    history = [record("a", 1)]
    assert f.execute(Observer(), history)[0]["visible_mismatch"] is False
    history.append(record("b", 2))
    assert f.execute(Observer(), history)[0]["visible_mismatch"] is True
    history.append(record("x", 3))
    assert f.execute(Observer(), history) == []
    assert f.summary() == dict(matched_prediction_queries=1, matched_prediction_steps=2, observed_mismatch_queries=1,
                               observed_mismatch_steps=1, first_mismatch_real_action=2)


def test_feedback_from_a_later_state_never_matches_earlier_actions():
    f = PredictionFeedback()
    f.predict("q", 1, ["a"], ["2"])
    assert f.execute(Observer(), [record("a", 1)]) == []
    f.execute(Observer(), [record("a", 1), record("a", 2)])
    assert f.summary()["matched_prediction_steps"] == 1 and f.summary()["observed_mismatch_steps"] == 0


def test_the_episode_keeps_visible_changes_and_feedback_out_of_the_public_log():
    game, levels = generated("switchboard")
    events, evaluator = [], []
    profile = list(registry.profiles("switchboard").values())[1]
    episode = Episode(game, Task("t", levels[0].state), SimulatorPredictor(game, profile), Budgets(20, 20),
                      log=events.append, evaluator_log=evaluator.append)
    action = levels[0].solution[0]
    episode.predict([action])
    episode.log_prediction([action])
    episode.execute([action])
    episode.close()
    public = next(e for e in events if e["type"] == "predict")
    assert "visible_changed" not in public and "changed" not in public
    trace = next(e for e in evaluator if e["type"] == "predict_trace")
    end = next(e for e in evaluator if e["type"] == "task_end_private")
    assert trace["visible_change_recording_version"] == end["visible_change_recording_version"] == 1
    assert end["prediction_feedback"]["matched_prediction_queries"] == 1
    assert not any(e["type"] in ("prediction_feedback", "task_end_private") for e in events)
