"""PushT (gym-pusht, LeRobot, Apache-2.0) behind the benchmark's Game protocol.

A pusher (circle) pushes a T-shaped block until it covers more than 95% of a
fixed goal pose. The physics (pymunk) is deterministic, so a state is just
(seed, action history): replaying it reproduces the real simulator exactly.
That replay is the perfect-clone predictor. A small cache of live simulators
keeps replay incremental.

Agent-facing actions are absolute pusher targets ("move_to 256 320") held for
four native control steps. Coordinates are the simulator's pixels, x to the
right, y downward. Tasks are seeded variations of verified reference-controller
templates (`make_levels`).
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from world_adapt.environments.base import InvalidAction, InvalidState, Transition
from world_adapt.environments.registry import Level

HOLD_STEPS = 4
"""Native control steps per action."""
CURRICULUM = "seeded-v1"


@dataclass(frozen=True)
class PushTState:
    seed: int
    actions: tuple[str, ...] = ()
    # ``native_targets`` is evaluator-only replay state.  A None entry means
    # that the action was applied through the nominal action interface; a
    # coordinate entry is a transformed future target used by a predictor.
    # Keeping this separate from ``actions`` leaves the real action history
    # unchanged when a perturbation simulates a different target.
    native_targets: tuple[tuple[float, float] | None, ...] = ()
    initial_pose: tuple[float, float, float, float, float] | None = None


def _schedule(state: PushTState) -> tuple[tuple[float, float] | None, ...]:
    """Return a complete native-target schedule; states built from actions alone use nominal targets."""
    if not state.native_targets and state.actions:
        return (None,) * len(state.actions)
    if len(state.actions) != len(state.native_targets):
        raise InvalidState("PushT action and native-target histories must have equal length")
    return state.native_targets


class PushT:
    name = "pusht"
    version = "0.3.0"

    def __init__(self, hold_steps: int = HOLD_STEPS, cache_size: int = 256):
        if hold_steps < 1 or cache_size < 1:
            raise ValueError("hold_steps and cache_size must be positive")
        self.hold_steps = hold_steps
        self._cache: OrderedDict[PushTState, Any] = OrderedDict()
        self._cache_size = cache_size

    # Simulation --------------------------------------------------------------

    def _fresh(self, state: PushTState):
        import gymnasium as gym
        import gym_pusht  # noqa: F401  (registers the environment)

        env = gym.make("gym_pusht/PushT-v0", obs_type="state").unwrapped
        options = {"reset_to_state": list(state.initial_pose)} if state.initial_pose is not None else None
        env.reset(seed=state.seed, options=options)
        if state.initial_pose is not None:
            query_shapes = (*env.agent.shapes, *env.block.shapes)
            for query_shape in query_shapes:
                for info in env.space.shape_query(query_shape):
                    contact = info.contact_point_set
                    if any(point.distance < -1e-6 for point in contact.points):
                        self._close(env)
                        raise InvalidState("PushT initial pose has overlapping native shapes")
        return env

    @staticmethod
    def _close(env: Any) -> None:
        close = getattr(env, "close", None)
        if close is not None:
            close()

    @staticmethod
    def _target(action: str) -> np.ndarray:
        _, x, y = action.split()
        return np.clip(np.array([float(x), float(y)], dtype=np.float64), 0, 512)

    def _apply(self, env: Any, action: str, target: tuple[float, float] | None = None) -> bool:
        target_array = self._target(action) if target is None else np.asarray(target, dtype=np.float64)
        if target_array.shape != (2,) or not np.isfinite(target_array).all():
            raise InvalidState("PushT native targets must be finite two-dimensional coordinates")
        target_array = np.clip(target_array, 0, 512)
        for _ in range(self.hold_steps):
            _, _, terminated, truncated, _ = env.step(target_array)
            if terminated or truncated:
                return True
        return False

    def _env(self, state: PushTState):
        """A live simulator at `state`, taken out of the cache (the caller owns it)."""
        history = _schedule(state)
        if state in self._cache:
            return self._cache.pop(state)
        # Resume from the longest cached prefix, else replay from reset.
        for k in range(len(state.actions) - 1, -1, -1):
            prefix = PushTState(state.seed, state.actions[:k], history[:k], state.initial_pose)
            if prefix in self._cache:
                env = self._cache.pop(prefix)
                break
        else:
            env, k = self._fresh(state), 0
        try:
            remaining = tuple(zip(state.actions[k:], history[k:]))
            for index, (action, target) in enumerate(remaining):
                if float(env._get_coverage()) > 0.95:
                    raise InvalidState("PushT history contains an action after native termination")
                if self._apply(env, action, target) and index + 1 < len(remaining):
                    raise InvalidState("PushT history contains an action after native termination")
        except Exception:
            self._close(env)
            raise
        return env

    def _store(self, state: PushTState, env) -> None:
        previous = self._cache.get(state)
        if previous is not None and previous is not env:
            self._close(previous)
        self._cache[state] = env
        self._cache.move_to_end(state)
        while len(self._cache) > self._cache_size:
            _, evicted = self._cache.popitem(last=False)
            if evicted is not env:
                self._close(evicted)

    def close(self) -> None:
        """Close every cached native simulator owned by this game."""
        cached = tuple(self._cache.values())
        self._cache.clear()
        for env in cached:
            self._close(env)

    def physical(self, state: PushTState) -> dict[str, Any]:
        env = self._env(state)
        try:
            coverage = float(env._get_coverage())
            info = {
                "agent": tuple(float(v) for v in env.agent.position),
                "agent_velocity": tuple(float(v) for v in env.agent.velocity),
                "agent_angular_velocity": float(env.agent.angular_velocity),
                "block": (float(env.block.position[0]), float(env.block.position[1]), float(env.block.angle)),
                "block_velocity": tuple(float(v) for v in env.block.velocity),
                "block_angular_velocity": float(env.block.angular_velocity),
                "goal": tuple(float(v) for v in env.goal_pose),
                "coverage": coverage,
                "contacts": int(getattr(env, "n_contact_points", 0)),
                "status": "success" if coverage > 0.95 else "active",
            }
        except Exception:
            self._close(env)
            raise
        self._store(state, env)
        return info

    def transformed_target(self, state: PushTState, action: str, gain: float) -> tuple[float, float]:
        """Scale only the requested future target displacement from the pusher."""
        if not math.isfinite(gain) or gain < 0:
            raise ValueError("gain must be finite and non-negative")
        action = self.parse_action(action)
        env = self._env(state)
        try:
            origin = np.asarray(env.agent.position, dtype=np.float64)
            target = self._target(action)
            transformed = np.clip(origin + gain * (target - origin), 0, 512)
        except Exception:
            self._close(env)
            raise
        self._store(state, env)
        return (float(transformed[0]), float(transformed[1]))

    # Game protocol -------------------------------------------------------------

    def actions(self) -> Sequence[str]:
        return (f"move_to <x> <y>: absolute pusher target in pixels, coordinates in [0, 512]; "
                f"held for {self.hold_steps} control steps; arrival is not guaranteed",)

    def action_item_schema(self) -> dict[str, Any]:
        return {"type": "string", "description": "Absolute pusher target: 'move_to 256 320'. "
                f"Finite coordinates in [0, 512]; held for {self.hold_steps} control steps, "
                "even if the pusher has not arrived. A chunk of targets forms a path."}

    def parse_action(self, raw: str) -> str:
        if not isinstance(raw, str):
            raise InvalidAction("action must be a string")
        parts = raw.strip().lower().replace("_", "-").split()
        if len(parts) == 3 and parts[0] == "move-to":
            try:
                x, y = map(float, parts[1:])
            except ValueError:
                pass
            else:
                if all(math.isfinite(v) and 0 <= v <= 512 for v in (x, y)):
                    return f"move_to {x:g} {y:g}"
        raise InvalidAction("expected 'move_to <x> <y>' with finite coordinates in [0, 512]")

    def step(self, state: PushTState, action: str) -> Transition[PushTState]:
        action = self.parse_action(action)
        if self.is_terminal(state):
            return Transition(state, ("terminal",))
        env = self._env(state)
        try:
            before = float(env._get_coverage())
            self._apply(env, action)
            after = float(env._get_coverage())
            history = _schedule(state)
            new = PushTState(state.seed, state.actions + (action,), history + (None,), state.initial_pose)
        except Exception:
            self._close(env)
            raise
        self._store(new, env)
        return Transition(new, ("move", "progress" if after > before + 1e-6 else "no_progress"))

    def simulate(self, state: PushTState, action: str, *, target: tuple[float, float]) -> Transition[PushTState]:
        """Apply a native target to the predicted future and preserve real history."""
        action = self.parse_action(action)
        target = tuple(float(v) for v in target)
        if len(target) != 2 or not all(math.isfinite(v) and 0 <= v <= 512 for v in target):
            raise InvalidState("PushT simulated targets must be finite coordinates in [0, 512]")
        if self.is_terminal(state):
            return Transition(state, ("terminal",))
        env = self._env(state)
        try:
            before = float(env._get_coverage())
            self._apply(env, action, target)
            after = float(env._get_coverage())
            history = _schedule(state)
            new = PushTState(state.seed, state.actions + (action,), history + (target,), state.initial_pose)
        except Exception:
            self._close(env)
            raise
        self._store(new, env)
        return Transition(new, ("move", "progress" if after > before + 1e-6 else "no_progress"))

    def is_success(self, state: PushTState) -> bool:
        return self.physical(state)["coverage"] > 0.95

    def is_terminal(self, state: PushTState) -> bool:
        return self.is_success(state)

    def observe(self, state: PushTState) -> str:
        p = self.physical(state)
        ax, ay = p["agent"]
        avx, avy = p["agent_velocity"]
        bx, by, ba = p["block"]
        bvx, bvy = p["block_velocity"]
        gx, gy, ga = p["goal"]
        return "\n".join([
            "Workspace: 512 x 512 pixels; x grows to the right, y grows downward.",
            "Object: a T-shaped block with local bar x[-60, 60], y[0, 30] and stem x[-15, 15], y[30, 120]; its pose is given at the native local origin.",
            f"Pusher (you): ({ax:.1f}, {ay:.1f})",
            f"Pusher velocity: ({avx:+.2f}, {avy:+.2f}); angular velocity {p['agent_angular_velocity']:+.2f}",
            f"T block: position ({bx:.1f}, {by:.1f}), angle {math.degrees(ba) % 360:.1f} degrees",
            f"T block velocity: ({bvx:+.2f}, {bvy:+.2f}); angular velocity {p['block_angular_velocity']:+.2f}",
            f"Goal pose: position ({gx:.1f}, {gy:.1f}), angle {math.degrees(ga) % 360:.1f} degrees",
            "Native body pose uses the T block's local origin; its center of mass is offset by local (0, 45). Positive angles rotate clockwise in this y-down view.",
            f"Each target is held for {self.hold_steps} native control steps by a proportional-derivative controller.",
            f"Native collision-callback count during the last control step: {p['contacts']}",
            f"Overlap with the goal: {100 * p['coverage']:.1f}% (need more than 95%)",
            f"Status: {p['status']}",
        ])

    def render_png(self, state: PushTState, size: int = 512) -> bytes:
        """The scene as the simulator draws it (no action marker), as PNG bytes."""
        import io

        from PIL import Image

        env = self._env(state)
        try:
            frame = env._get_img(env._draw(), size, size)
        except Exception:
            self._close(env)
            raise
        self._store(state, env)
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="PNG")
        return buffer.getvalue()

    def prompt_notes(self, max_chunk: int) -> str:
        """Text appended to the system prompt: the native geometry, control and chunking facts."""
        return (
            "\nPushT mechanics: the grey object is a T-shaped rigid block with local bar x[-60, 60], y[0, 30] and "
            "stem x[-15, 15], y[30, 120]. Its reported position is the native local origin, while its center of mass "
            "is offset by local (0, 45). Positive angles rotate clockwise in this y-down view. "
            f"Each move_to target is held for {self.hold_steps} native control steps under a proportional-derivative "
            "controller; the pusher may contact the block during that interval."
            f"\nYou may send up to {max_chunk} actions per tool call; each call runs them in order."
            "\nPredictions do not use real actions; they only count against the predicted-action budget."
        )

    def validate_state(self, state: PushTState) -> None:
        if not isinstance(state, PushTState) or not isinstance(state.seed, int):
            raise InvalidState("expected a PushT state")
        if state.initial_pose is not None:
            if len(state.initial_pose) != 5 or not all(math.isfinite(float(v)) for v in state.initial_pose):
                raise InvalidState("PushT initial_pose must contain five finite values")
            if not all(0 <= float(v) <= 512 for v in state.initial_pose[:4]):
                raise InvalidState("PushT initial positions must be within native bounds")
        history = _schedule(state)
        try:
            for action in state.actions:
                self.parse_action(action)
            for target in history:
                if target is not None and (len(target) != 2 or not all(math.isfinite(float(v)) and 0 <= float(v) <= 512 for v in target)):
                    raise InvalidState("PushT native target history is malformed")
        except (InvalidAction, InvalidState, TypeError, ValueError) as exc:
            raise InvalidState("malformed PushT state") from exc

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "deterministic": True, "fully_observed": True,
                "actions": list(self.actions()), "hold_control_steps": self.hold_steps,
                "action_mode": "primitives",
                "termination": "success when coverage of the goal pose exceeds 95%",
                "geometry": "T block local bar x[-60, 60], y[0, 30] and stem x[-15, 15], y[30, 120]; pose uses the native local origin; center of mass offset is local (0, 45)",
                "observation": "agent and block pose, linear and angular velocity, goal pose, coverage, native collision-callback count for the last control step, and status",
                "source": "gym-pusht 0.1.6 (Apache-2.0) with pymunk 6.x"}


# Reference-controller templates. Witnesses are private evaluator metadata; the agent sees only the initial
# state and the generic move_to interface.
TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "category": "translation", "band": "translation_approachable_a",
        "pose": (80.0, 80.0, 144.1802, 167.8198, math.pi / 4),
        "witness": ("move_to 180 300", "move_to 140 180", "move_to 220 300"),
    },
    {
        "category": "translation", "band": "translation_approachable_b",
        "pose": (100.0, 80.0, 144.1802, 167.8198, math.pi / 4),
        "witness": ("move_to 180 340", "move_to 180 100", "move_to 260 140", "move_to 300 300"),
    },
    {
        "category": "combo", "band": "combo_small_rotation_left",
        "pose": (120.0, 170.0, 194.0, 218.0, math.pi / 4 - 0.15),
        "witness": ("move_to 170 170", "move_to 170 310", "move_to 200 260", "move_to 215 275"),
    },
    {
        "category": "combo", "band": "combo_small_rotation_right",
        "pose": (130.0, 170.0, 194.0, 218.0, math.pi / 4 + 0.05),
        "witness": ("move_to 170 170", "move_to 170 310", "move_to 200 260", "move_to 215 275"),
    },
    {
        "category": "combo", "band": "combo_small_rotation_right_high",
        "pose": (130.0, 160.0, 194.0, 218.0, math.pi / 4 + 0.05),
        "witness": ("move_to 170 170", "move_to 170 310", "move_to 200 260", "move_to 215 275"),
    },
    {
        "category": "combo", "band": "combo_small_rotation_right_low",
        "pose": (130.0, 180.0, 194.0, 218.0, math.pi / 4 + 0.05),
        "witness": ("move_to 170 170", "move_to 170 310", "move_to 200 260", "move_to 215 275"),
    },
)

# Labels select structural strata before any candidate is simulated. They are
# not measured difficulty ranks or bins derived from reference solution length.
STRATA = (("easy", (0, 1)), ("medium", (2,)), ("hard", (3, 4, 5)))


def _finite(physical):
    vectors = ("agent", "block", "agent_velocity", "block_velocity", "goal")
    scalars = ("agent_angular_velocity", "block_angular_velocity", "coverage", "contacts")
    return (all(math.isfinite(v) for key in vectors for v in physical[key])
            and all(math.isfinite(physical[key]) for key in scalars)
            and all(0 <= v <= 512 for key in ("agent", "block") for v in physical[key][:2]))


def _witness(game, state, actions):
    initial = game.physical(state)
    if not _finite(initial) or initial["coverage"] >= 0.5:
        return None
    coverage, contact = [initial["coverage"]], False
    replay = state
    for index, action in enumerate(actions):
        replay = game.step(replay, action).state
        physical = game.physical(replay)
        if not _finite(physical):
            return None
        coverage.append(physical["coverage"])
        contact |= physical["contacts"] > 0
        if physical["coverage"] > 0.95:
            if index != len(actions) - 1:
                return None
            break
    if coverage[-1] <= 0.95 or not contact:
        return None
    return tuple(coverage)


def make_levels(game: PushT, n: int, seed: int, options: dict[str, Any]) -> list[Level]:
    """Seeded native variations around declared reference-controller templates.

    This is a bounded template-conditioned population, not an unrestricted PushT solver or evidence of transfer
    to unseen task families. Admission uses nominal physics only, never predictor errors or agent outcomes.
    """
    if options.get("curriculum", CURRICULUM) != CURRICULUM:
        raise ValueError(f"pusht curriculum must be {CURRICULUM!r}")
    attempts = options.get("max_attempts", 200)
    budget = options.get("budget", 40)
    if type(attempts) is not int or attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if type(budget) is not int or budget != 40:
        raise ValueError("seeded-v1 uses a fixed budget of 40")
    if type(n) is not int or n < 1 or type(seed) is not int:
        raise ValueError("seeded-v1 requires a positive task count and integer seed")
    levels, seen, observations = [], set(), set()
    for index in range(n):
        rng = random.Random(f"pusht/seeded-v1/{seed}/{index}")
        tier, templates = STRATA[index % len(STRATA)]
        candidate = TEMPLATES[rng.choice(templates)]
        failures = Counter()
        for attempt in range(attempts):
            pose = tuple(v + rng.uniform(-4, 4) for v in candidate["pose"][:4]) + (
                candidate["pose"][4] + rng.uniform(-0.005, 0.005),)
            state = PushTState(rng.randrange(2**31), initial_pose=pose)
            try:
                trace = _witness(game, state, candidate["witness"])
            except InvalidState:
                failures["invalid_native_start"] += 1
                continue
            if trace is None:
                failures["no_valid_nominal_witness"] += 1
                continue
            physical = game.physical(state)
            key = (*physical["agent"], *physical["block"])
            observation = game.observe(state)
            if key in seen or observation in observations:
                failures["duplicate_start"] += 1
                continue
            seen.add(key)
            observations.add(observation)
            identity = hashlib.sha256(repr(key).encode()).hexdigest()
            witness = tuple(candidate["witness"])
            levels.append(Level(state, witness, {
                "curriculum": "seeded-v1", "generator_version": 1,
                "seed": seed, "env_seed": state.seed, "task_id": f"seeded-v1-{identity}",
                "task_family": candidate["band"], "tier": tier,
                "tier_basis": "preassigned reference-controller stratum, not validated difficulty",
                "population": "nominal-witness-conditioned calibration-template variations",
                "initial_pose_input": pose, "initial_coverage": trace[0],
                "coverage_trace": trace, "real_budget": budget, "solution_length": len(witness),
                "reference_verified": True, "witness_terminal_success": True,
                "reference_controller": "bounded fixed-template native replay, not optimal",
                "contact_verified": True, "velocity_verified": True,
                "candidate_attempts": attempt + 1, "rejections": dict(failures),
            }))
            break
        else:
            raise ValueError(f"seeded-v1 exhausted {attempts} attempts for task {index} ({tier}): {dict(failures)}")
    return levels
