"""The six environments: rules, study task generation, and the contracts every game keeps.

Contracts: deterministic and pure steps, accurate predictions equal to real execution, isolation of predictions from
the real environment, terminal handling, atomic rejection of invalid actions, and an information boundary that never
shows hidden structure or predictor errors to the agent.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import re
from dataclasses import dataclass, replace
from functools import lru_cache

import pytest

from tests.fakes import TEMPLATE
from world_adapt.environments import registry
from world_adapt.environments.base import Game, InvalidAction, InvalidState, Task
from world_adapt.environments.depot.game import Depot, _ground
from world_adapt.environments.depot.structure import task_fingerprint
from world_adapt.environments.portalmaze.game import DIRECTIONS, MazeLayout, MazeState, PortalMaze, naive_path
from world_adapt.environments.pusht.game import PushT, PushTState
from world_adapt.environments.real import RealEnvironment
from world_adapt.environments.search import solve
from world_adapt.environments.switchboard.game import SwitchState, Switchboard, Wiring, solvable_states
from world_adapt.perturbations import registry as perturbations
from world_adapt.perturbations.base import IDENTITY, PerturbedDynamics
from world_adapt.predictors.simulator import SimulatorPredictor
from world_adapt.runner.protocol import plan_protocol, pool_size, seed_bundle
from world_adapt.session import TaskSession

ENVIRONMENTS = ("depot", "iceslide", "patrol", "portalmaze", "pusht", "switchboard")
STUDY = json.loads(TEMPLATE.read_text())


def options(env: str) -> dict:
    """The environment options the study template passes to the runner."""
    return registry.parse_options(STUDY["environments"][env]["runner_args"].get("env_opt", []),
                                  registry.get(env).default_options)


@lru_cache(maxsize=None)
def generated(env: str, n: int = 3, seed: int = 3000):
    spec = registry.get(env)
    game = spec.make_game(options(env))
    return game, tuple(spec.make_levels(game, n, seed, options(env)))


def faulty(env: str):
    return perturbations.profiles(env)[STUDY["environments"][env]["primary_profile"]]


@dataclass(frozen=True)
class Fixture:
    env: str
    game: Game
    tasks: list[Task]
    sequences: list[list[str]]
    actions: tuple[str, ...]
    """A concrete action universe for exhaustive checks (Depot and PushT actions are parameterized)."""
    depth: int = 3


@lru_cache(maxsize=None)
def fixture(env: str) -> Fixture:
    game, levels = generated(env)
    if env == "depot":
        levels = levels[:1]
        actions = tuple(dict.fromkeys(levels[0].solution))
    elif env == "pusht":
        actions = ("move_to 256 320", "move_to 200 180")
    else:
        actions = tuple(game.actions())
    tasks = [Task(f"{env}/{i}", level.state, {"solution": list(level.solution)}) for i, level in enumerate(levels)]
    sequences = [list(levels[0].solution), [actions[i % len(actions)] for i in range(6)],
                 list(reversed(actions))[:5]]
    return Fixture(env, game, tasks, sequences, actions, depth=2 if env == "pusht" else 3)


@pytest.fixture(params=ENVIRONMENTS)
def fx(request) -> Fixture:
    return fixture(request.param)


def reachable_states(fx: Fixture, task: Task, cap: int = 120) -> list:
    seen, frontier = {task.initial_state}, [task.initial_state]
    for _ in range(fx.depth):
        nxt = []
        for state in frontier:
            for action in fx.actions:
                s = fx.game.step(state, action).state
                if s not in seen and len(seen) < cap:
                    seen.add(s)
                    nxt.append(s)
        frontier = nxt
    return list(seen)


# Contracts every environment keeps -------------------------------------------------------------------------------


def test_accurate_predictions_equal_real_execution_from_the_start_and_midway(fx):
    for task in fx.tasks:
        for seq in fx.sequences:
            env = RealEnvironment(fx.game, task)
            prediction, trace = SimulatorPredictor(fx.game).predict(env.snapshot, seq)
            execution = env.execute(seq)
            assert prediction.transitions_simulated == execution.transitions_executed == len(execution.steps)
            assert prediction.transitions_requested == len(seq) and prediction.predicted_terminal == execution.terminal
            assert [s.observation for s in prediction.steps] == [s.observation for s in execution.steps]
            assert list(trace.states) == [r.next_state for r in env.trajectory]
            assert all(a == () and c == () for a, c in zip(trace.activated, trace.changed))
            env = RealEnvironment(fx.game, task)
            env.execute(seq[:2])
            prediction, _ = SimulatorPredictor(fx.game).predict(env.snapshot, seq[2:] or seq[:1])
            assert [s.observation for s in prediction.steps] == [s.observation for s in
                                                                  env.execute(seq[2:] or seq[:1]).steps]


@pytest.mark.parametrize("profile", ["identity", "faulty"])
def test_predictions_never_change_the_real_environment_and_repeat_exactly(fx, profile):
    profile = IDENTITY if profile == "identity" else faulty(fx.env)
    task = fx.tasks[0]
    env = RealEnvironment(fx.game, task)
    env.execute(fx.sequences[1][:1])
    state, trajectory, observation = env.snapshot, env.trajectory, env.observe()
    predictor = SimulatorPredictor(fx.game, profile)
    first = [predictor.predict(env.snapshot, seq) for seq in fx.sequences]
    assert env.snapshot is state and env.trajectory == trajectory and env.observe() == observation
    assert [SimulatorPredictor(fx.game, profile).predict(env.snapshot, seq) for seq in reversed(fx.sequences)] == \
        list(reversed(first))


def test_the_identity_profile_is_a_no_op_on_reachable_states(fx):
    dynamics = PerturbedDynamics(fx.game, IDENTITY)
    for state in reachable_states(fx, fx.tasks[0]):
        for action in fx.actions:
            result = dynamics.step(state, action)
            assert result.transition == fx.game.step(state, action) and result.activated == result.changed == ()


def test_faulty_predictions_stay_valid_and_report_only_profile_atoms(fx):
    profile = faulty(fx.env)
    names = {a.name for a in profile.atoms}
    dynamics = PerturbedDynamics(fx.game, profile)
    for state in reachable_states(fx, fx.tasks[0]):
        for action in fx.actions:
            result = dynamics.step(state, action)
            fx.game.validate_state(result.transition.state)
            assert set(result.changed) <= set(result.activated) <= names and not result.rejected


def test_the_faulty_profile_visibly_changes_some_prediction_on_study_tasks(fx):
    profile, changed = faulty(fx.env), 0
    for task in fx.tasks:
        state = task.initial_state
        for action in task.metadata["solution"]:
            for probe in dict.fromkeys([action, *fx.actions]):
                accurate, _ = SimulatorPredictor(fx.game).predict(state, [probe])
                predicted, _ = SimulatorPredictor(fx.game, profile).predict(state, [probe])
                changed += accurate != predicted
            state = fx.game.step(state, action).state
    assert changed > 0


def test_execution_and_prediction_stop_at_success(fx):
    for task in fx.tasks:
        solution = task.metadata["solution"]
        env = RealEnvironment(fx.game, task)
        execution = env.execute(solution + [solution[0]] * 2)
        assert execution.transitions_executed == len(solution) and execution.success and execution.terminal
        again = env.execute([fx.actions[0]])
        assert again.transitions_executed == 0 and again.success and env.steps_taken == len(solution)
        prediction, trace = SimulatorPredictor(fx.game).predict(task.initial_state, solution + [solution[0]])
        assert prediction.transitions_simulated == len(solution) == len(trace.states)
        assert prediction.steps[-1].predicted_success and not any(s.predicted_success for s in prediction.steps[:-1])
        from_terminal, _ = SimulatorPredictor(fx.game).predict(env.snapshot, [fx.actions[0]])
        assert from_terminal.steps == () and from_terminal.predicted_terminal


@pytest.mark.parametrize("bad", [[], ["definitely-not-an-action"], ["", "x"], ["valid", "bogus"]])
def test_invalid_actions_raise_and_change_nothing(fx, bad):
    bad = [fx.actions[0] if a == "valid" else a for a in bad]
    env = RealEnvironment(fx.game, fx.tasks[0])
    env.execute([fx.actions[0]])
    state, trajectory = env.snapshot, env.trajectory
    with pytest.raises(InvalidAction):
        env.execute(bad)
    with pytest.raises(InvalidAction):
        SimulatorPredictor(fx.game).predict(state, bad)
    assert env.snapshot == state and env.trajectory == trajectory


def test_the_predictor_enforces_its_horizon(fx):
    predictor = SimulatorPredictor(fx.game, max_horizon=2)
    predictor.predict(fx.tasks[0].initial_state, list(fx.actions[:1]) * 2)
    with pytest.raises(InvalidAction):
        predictor.predict(fx.tasks[0].initial_state, list(fx.actions[:1]) * 3)


def test_agent_visible_outputs_hide_profile_details(fx):
    profile = faulty(fx.env)
    forbidden = {profile.name, "profile", "atom", "activated", "perturb", "faulty"} | {a.name for a in profile.atoms}
    env = RealEnvironment(fx.game, fx.tasks[0])
    prediction, _ = SimulatorPredictor(fx.game, profile).predict(env.snapshot, fx.sequences[0])
    execution = env.execute(fx.sequences[0])
    for text in [repr(prediction), repr(execution), *(s.observation for s in prediction.steps)]:
        assert not any(word.lower() in text.lower() for word in forbidden), text
    assert {f.name for f in dataclasses.fields(prediction)} == {
        "steps", "transitions_requested", "transitions_simulated", "predicted_terminal"}
    assert {f.name for f in dataclasses.fields(execution)} == {"steps", "transitions_executed", "success", "terminal"}


def test_steps_are_deterministic_pure_and_declared(fx):
    for state in reachable_states(fx, fx.tasks[0])[:30]:
        for action in fx.actions:
            assert fx.game.step(state, action) == fx.game.step(state, action)
    for task in fx.tasks:
        fx.game.validate_state(task.initial_state)
        assert fx.game.observe(task.initial_state) == fx.game.observe(task.initial_state)
    spec = fx.game.spec()
    assert spec["deterministic"] is True and spec["name"] == fx.game.name and spec["version"] == fx.game.version
    json.dumps(spec)


def test_a_session_predicts_from_the_current_real_state_and_records_traces_privately(fx):
    task = fx.tasks[0]
    predictor = SimulatorPredictor(fx.game, faulty(fx.env))
    session = TaskSession(fx.game, task, predictor)
    seq = fx.sequences[0]
    before = session.predict(seq)
    session.execute(seq[:2])
    assert session.predict(seq) == predictor.predict(session.real.snapshot, seq)[0]
    assert before == predictor.predict(task.initial_state, seq)[0]
    assert len(session.prediction_traces) == 2 and not hasattr(before, "activated")
    with pytest.raises(ValueError):
        TaskSession(fx.game, task, SimulatorPredictor(copy.copy(fx.game)))


# Study task generation --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("env", ENVIRONMENTS)
def test_study_tasks_are_deterministic_distinct_tiered_and_solved_by_their_witness(env):
    game, levels = generated(env)
    spec = registry.get(env)
    again = spec.make_levels(game, 3, 3000, options(env))
    assert [(lv.state, lv.solution) for lv in levels] == [(lv.state, lv.solution) for lv in again]
    assert [lv.features["tier"] for lv in levels] == ["easy", "medium", "hard"]
    assert len({game.observe(lv.state) for lv in levels}) == 3
    other = spec.make_levels(game, 3, 3001, options(env))
    assert {game.observe(lv.state) for lv in other}.isdisjoint(game.observe(lv.state) for lv in levels)
    for level in levels:
        state = level.state
        for action in level.solution:
            state = game.step(state, action).state
        assert game.is_success(state) and level.features["solution_length"] == len(level.solution)


@pytest.mark.parametrize("env", ENVIRONMENTS)
def test_generation_rejects_retired_curricula(env):
    game, _ = generated(env)
    with pytest.raises(ValueError, match="curriculum"):
        registry.get(env).make_levels(game, 1, 0, {**options(env), "curriculum": "legacy"})


def test_the_full_depot_pool_is_structurally_distinct_and_excludes_prior_inventories():
    game, excluded, seen = Depot(), [], set()
    for seed in (3000, 3001):
        opts = {**options("depot"), "exclude_fingerprints": ",".join(excluded)}
        levels = registry.get("depot").make_levels(game, pool_size(8, 2), seed, opts)
        fingerprints = [lv.features["task_fingerprint"] for lv in levels]
        assert len(set(fingerprints)) == 15 and seen.isdisjoint(fingerprints)
        assert all(task_fingerprint(lv.state) == lv.features["task_fingerprint"] for lv in levels)
        plan = plan_protocol(levels, 8, 2, seed_bundle(seed)["task_order"])
        assert len({levels[p.level_index].features["task_fingerprint"] for p in plan}) == 14
        seen.update(fingerprints)
        excluded.extend(fingerprints)


# Rules: Switchboard -----------------------------------------------------------------------------------------------

W = Wiring(1, 4, (0b0011, 0b0110, 0b1100), (-1, 2, -1))
"""B1 toggles lamps 0,1; B2 toggles 1,2 and disables B3; B3 toggles 2,3."""
BOARD = Switchboard(3)


def lamps(mask, disabled=0):
    return SwitchState(W, mask, disabled)


def test_switchboard_buttons_toggle_hidden_lamp_sets_and_breakers_disable_for_good():
    assert BOARD.step(lamps(0), "press 1").state.lamps == 0b0011
    assert BOARD.step(BOARD.step(lamps(0), "press 1").state, "press 3").state.lamps == 0b1111
    broken = BOARD.step(lamps(0), "press 2").state
    assert broken.lamps == 0b0110 and broken.disabled == 0b100
    pressed = BOARD.step(broken, "press 3")
    assert pressed.state == broken and pressed.events == ("disabled",)
    assert solve(BOARD, broken).proved_unsolvable and not BOARD.is_terminal(broken)
    assert broken not in solvable_states(BOARD, lamps(0))[1]


def test_switchboard_parsing_validation_and_hidden_wiring():
    for raw in ("press 2", "2", "p2", " Press2 "):
        assert BOARD.parse_action(raw) == "press 2"
    for bad in ("press 0", "press 4", "", "up", "press 2 3"):
        with pytest.raises(InvalidAction):
            BOARD.parse_action(bad)
    with pytest.raises(InvalidState):
        BOARD.validate_state(lamps(0b10000))
    observation = BOARD.observe(lamps(0b0101, 0b100))
    assert "*.*." in observation and "Disabled buttons: [3]" in observation
    other = SwitchState(Wiring(1, 4, (0b1000, 0b0001, 0b0010), (1, -1, -1)), 0b0101, 0b100)
    assert BOARD.observe(other) == observation


# Rules: PortalMaze ------------------------------------------------------------------------------------------------

MAZE = PortalMaze()


def maze(text: str, links: tuple[int, ...] = ()) -> MazeState:
    """'#' wall, '@' start, 'X' exit, lowercase key, uppercase door, digit = pad with that index."""
    rows = text.split("\n")
    walls, keys, doors, pads, start, exit_ = set(), [], [], {}, None, None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            p = (r, c)
            if ch == "#":
                walls.add(p)
            elif ch == "@":
                start = p
            elif ch == "X":
                exit_ = p
            elif ch.islower():
                keys.append((p, ch))
            elif ch.isupper():
                doors.append((p, ch.lower()))
            elif ch.isdigit():
                pads[int(ch)] = p
    layout = MazeLayout(len(rows), len(rows[0]), frozenset(walls), exit_, tuple(keys), tuple(doors),
                        tuple(pads[i] for i in sorted(pads)), links)
    return MazeState(layout, start)


def walk(state, *actions):
    for a in actions:
        state = MAZE.step(state, a).state
    return state


def test_portalmaze_keys_open_matching_doors_and_are_never_consumed():
    s = maze("#######\n#@.aAX#\n#######")
    assert MAZE.step(walk(s, "right"), "right").state.held == {"a"}
    assert walk(s, "right", "right", "right", "right").status == "escaped"
    assert MAZE.step(replace(s, agent=(1, 3)), "right").events == ("blocked",)
    wrong = walk(maze("#####\n#@bAX#\n#####"), "right", "right")
    assert MAZE.step(wrong, "right").state == wrong


def test_portalmaze_pads_teleport_once_per_entry_and_one_way_links_can_trap():
    s = maze("#########\n#@0.#.1X#\n#########", links=(1, 0))
    assert MAZE.step(s, "right").state.agent == (1, 6) and MAZE.step(s, "right").events == ("move", "teleport")
    assert walk(s, "right", "left").agent == (1, 5) and walk(s, "right", "left", "right").agent == (1, 2)
    trap = walk(maze("#########\n#@0.X#12#\n#########", links=(1, 2, 1)), "right")
    assert trap.agent == (1, 6) and solve(MAZE, trap).proved_unsolvable
    done = walk(maze("####\n#@X#\n####"), "right")
    assert MAZE.is_terminal(done) and MAZE.step(done, "left").events == ("over",)


def test_portalmaze_observations_list_pads_but_never_their_links():
    s = maze("#########\n#@0.aAX1#\n#########", links=(1, 0))
    observation = MAZE.observe(s)
    for line in ("Agent: (1, 1)", "Exit: (1, 6)", "Keys: key a at (1, 4)", "Pads: [(1, 2), (1, 7)]"):
        assert line in observation
    assert MAZE.observe(replace(s, layout=replace(s.layout, links=(0, 1)))) == observation
    assert naive_path(s.layout, s.agent) == ("right",) * 5 and not MAZE.is_success(walk(s, *("right",) * 5))


def test_portalmaze_study_levels_keep_doors_open_and_the_trap_pocket_sealed():
    _, levels = generated("portalmaze")
    for level in levels:
        layout = level.state.layout
        assert all(pos not in layout.walls for pos, _ in layout.doors)
        pocket = set(layout.pads[-2:])
        ring = {(p[0] + dr, p[1] + dc) for p in pocket for dr, dc in DIRECTIONS.values()} - pocket
        assert ring <= layout.walls and level.features["one_way_links"] >= 1


# Rules: Depot -----------------------------------------------------------------------------------------------------


def test_depot_enforces_native_preconditions_and_rejects_invalid_states():
    game, levels = generated("depot")
    state = levels[0].state
    _, operators, _ = _ground(state.problem)
    blocked = next(name for name, op in operators.items() if not op.applicable(state.facts))
    assert game.step(state, blocked).state == state
    with pytest.raises(InvalidAction):
        game.step(state, "drive imaginary l0 l1")
    with pytest.raises(InvalidState):
        game.validate_state(replace(state, facts=state.facts | {"(at t0 l0)", "(at t0 l1)"}))
    assert ":effect" not in game.observe(state) and ":precondition" not in game.observe(state)


def test_depot_task_identity_ignores_object_names_but_not_goals():
    _, levels = generated("depot")
    state = levels[0].state
    names = {f"{prefix}{i}": f"{prefix}{1 - i}" for prefix in ("l", "h", "p") for i in range(2)}
    names.update({f"c{i}": f"box{4 - i}" for i in range(5)})
    pattern = re.compile(r"\b(" + "|".join(names) + r")\b")
    rename = lambda text: pattern.sub(lambda m: names[m[0]], text)  # noqa: E731
    renamed = replace(state, problem=rename(state.problem), facts=frozenset(map(rename, state.facts)))
    assert task_fingerprint(renamed) == task_fingerprint(state)
    goals = " ".join(sorted(f for f in state.facts if f.startswith("(on ")))
    changed = replace(state, problem=state.problem.split("(:goal")[0] + f"(:goal (and {goals})))")
    assert task_fingerprint(changed) != task_fingerprint(state)


def test_depot_prompt_notes_explain_parameterized_actions():
    game = Depot()
    notes = game.prompt_notes(8)
    assert "examples, not an exhaustive" in notes and Depot.action_syntax() in notes
    assert "predictor" not in notes.lower() and "enum" not in game.action_item_schema()


# Rules: PushT -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["right 25", "move_to -1 10", "move_to 513 10", "move_to nan 10", "move_to x 10", None])
def test_pusht_accepts_only_finite_absolute_targets(raw):
    with pytest.raises(InvalidAction):
        PushT().parse_action(raw)
    assert PushT().parse_action(" MOVE_TO 256 320.0 ") == "move_to 256 320"


def test_pusht_replay_is_exact_cache_independent_and_absorbing_after_success():
    game, levels = generated("pusht")
    level = levels[0]
    state = level.state
    for action in level.solution:
        state = game.step(state, action).state
    assert game.is_success(state)
    fresh = PushT(cache_size=1)
    assert {k: v for k, v in fresh.physical(state).items() if k != "coverage"} == \
        {k: v for k, v in game.physical(state).items() if k != "coverage"}
    after = game.step(state, "move_to 80 80")
    assert after.state == state and after.events == ("terminal",)
    extended = replace(state, actions=state.actions + ("move_to 0 0",), native_targets=state.native_targets + (None,))
    fresh.close()
    with pytest.raises(InvalidState, match="after native termination"):
        fresh.physical(extended)
    fresh.close()


def test_pusht_rejects_overlapping_starts_and_study_starts_are_at_rest():
    game, levels = generated("pusht")
    with pytest.raises(InvalidState, match="overlapping native shapes"):
        PushT().physical(PushTState(3, initial_pose=(205.0, 236.0, 204.18, 227.82, math.pi / 4)))
    for level in levels:
        physical = game.physical(level.state)
        assert physical["agent_velocity"] == physical["block_velocity"] == (0.0, 0.0)
        assert physical["coverage"] < 0.5 and level.features["real_budget"] == 40


def test_pusht_target_gain_changes_only_the_predicted_future_target():
    game, _ = generated("pusht")
    state = PushTState(7, initial_pose=(150.0, 150.0, 224.0, 243.0, math.pi / 4 + 0.05))
    result = PerturbedDynamics(game, faulty("pusht")).step(state, "move_to 190 300")
    predicted = result.transition.state
    assert result.changed == ("target_gain",) and state.actions == () and state.native_targets == ()
    origin = game.physical(state)["agent"]
    expected = tuple(origin[i] + 0.8 * ((190.0, 300.0)[i] - origin[i]) for i in range(2))
    assert predicted.native_targets[0] == pytest.approx(expected)
    nominal = game.step(state, "move_to 190 300").state
    assert nominal.native_targets == (None,) and game.physical(predicted) != game.physical(nominal)
    assert "target_gain" not in game.observe(predicted)


def test_pusht_prompt_notes_state_geometry_and_control_without_a_plan():
    notes = PushT().prompt_notes(8)
    assert "local bar x[-60, 60]" in notes and "held for 4 native control steps" in notes
    assert "up to 8 actions per tool call" in notes and "move_to 2" not in notes
