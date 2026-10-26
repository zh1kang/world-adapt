"""Render one README GIF per environment: the real rollout beside the faulty predictor's rollout.

For each environment the script rebuilds the task pool exactly as the study does (seed block 3000, the study's
environment options), takes one fixed task, and replays its reference solution twice from the same start: once
in the real environment and once through the simulator predictor with the environment's primary perturbation
profile. The left panel shows reality, the right panel shows the prediction for the same actions. From the first
step whose prediction differs from reality, the right panel gets a red border and the differing cells are outlined.

Run from the repository root:

    uv run --locked python scripts/render_environment_gifs.py            # all environments
    uv run --locked python scripts/render_environment_gifs.py --env iceslide

Output: docs/assets/<env>.gif. The output is deterministic.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from world_adapt.environments import registry
from world_adapt.perturbations import registry as perturbations
from world_adapt.predictors.simulator import SimulatorPredictor
from world_adapt.runner.protocol import pool_size

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "configs/study.json"

# One task per environment, chosen from the study pool for a short, readable rollout in which the primary
# profile changes the prediction early. Indices refer to the pool generated for seed block 3000.
TASK_INDEX = {"depot": 0, "portalmaze": 9, "patrol": 3, "switchboard": 0, "iceslide": 5, "pusht": 0}
TITLES = {"depot": "Depot", "portalmaze": "PortalMaze", "patrol": "Patrol", "switchboard": "Switchboard",
          "iceslide": "IceSlide", "pusht": "PushT"}
# Milliseconds per step; PushT shows each native control step, so it uses a shorter frame.
STEP_MS = {"depot": 110, "portalmaze": 150, "patrol": 170, "switchboard": 500, "iceslide": 230, "pusht": 120}
START_MS, DIVERGE_MS, END_MS = 500, 1000, 1500

# Layout (pixels).
WIDTH, PANEL, MARGIN = 480, 228, 8
HEADER, LABEL, STATUS, CAPTION, LEGEND = 24, 18, 18, 34, 22
HEIGHT = HEADER + LABEL + PANEL + STATUS + CAPTION + LEGEND + MARGIN

# Palette.
BG, CARD, INK, MUTED, GRID = "#f6f6f4", "#ffffff", "#1f2328", "#6b7280", "#e5e5e2"
WALL, AGENT, GOAL, HAZARD, ITEM, PORTAL = "#4b5058", "#2563eb", "#16a34a", "#ea7317", "#d9a21b", "#7c3aed"
CRATE, ROUGH, OFF, RED = "#a16207", "#efe3cf", "#d4d4d0", "#dc2626"


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


F_HEAD, F_TEXT, F_SMALL, F_CELL = font(14), font(12), font(11), font(10)


# Study configuration and rollouts ------------------------------------------------------------------------------


def study_task(env: str):
    """The game, the chosen task, and the primary profile, exactly as a study run would build them."""
    study = json.loads(STUDY.read_text())
    config = study["environments"][env]
    args = {**study["protocol"]["runner_args"], **config["runner_args"]}
    spec = registry.get(env)
    options = registry.parse_options(args.get("env_opt", []), spec.default_options)
    if args.get("action_primitives"):
        options["action_primitives"] = True
    game = spec.make_game(options)
    seed = study["protocol"]["seed_blocks"][0]
    levels = spec.make_levels(game, pool_size(args["adaptation_tasks"], args["heldout_per_tier"]), seed, options)
    profile = perturbations.profiles(env)[config["primary_profile"]]
    return game, levels[TASK_INDEX[env]], profile, seed


@dataclass
class Rollout:
    actions: list[str]
    real: list[Any]
    """States after 0..N actions."""
    predicted: list[Any]
    """Predicted states after 0..N actions (the last predicted state repeats if the prediction ended early)."""
    diverge: int | None
    """1-based step at which the agent-visible prediction first differs from reality."""
    atoms: tuple[str, ...]
    """Atoms that changed the prediction at the diverging step."""


def rollout(game, level, profile) -> Rollout:
    actions = [str(a) for a in level.solution]
    real = [level.state]
    for action in actions:
        real.append(game.step(real[-1], action).state)
    prediction, trace = SimulatorPredictor(game, profile).predict(level.state, actions)
    predicted = [level.state, *trace.states]
    predicted += [predicted[-1]] * (len(real) - len(predicted))
    diverge = next((i for i, changed in enumerate(trace.visible_changed) if changed), None)
    atoms = tuple(trace.changed[diverge]) if diverge is not None and diverge < len(trace.changed) else ()
    return Rollout(actions, real, predicted, None if diverge is None else diverge + 1, atoms)


# Grid painting shared by the discrete environments -------------------------------------------------------------

Cells = dict[tuple[int, int], tuple]


@dataclass
class Scene:
    rows: int
    cols: int
    cells: Cells
    square: bool = True


def paint_grid(scene: Scene, draw_token: Callable, diff: set[tuple[int, int]]) -> Image.Image:
    img = Image.new("RGB", (PANEL, PANEL), CARD)
    d = ImageDraw.Draw(img)
    cw, ch = PANEL // scene.cols, PANEL // scene.rows
    if scene.square:
        cw = ch = min(cw, ch)
    ox, oy = (PANEL - cw * scene.cols) // 2, (PANEL - ch * scene.rows) // 2
    for (r, c), token in sorted(scene.cells.items()):
        draw_token(d, (ox + c * cw, oy + r * ch, ox + (c + 1) * cw, oy + (r + 1) * ch), token)
    for r, c in diff:
        d.rectangle((ox + c * cw, oy + r * ch, ox + (c + 1) * cw - 1, oy + (r + 1) * ch - 1), outline=RED, width=2)
    return img


def inset(box, k):
    x0, y0, x1, y1 = box
    return (x0 + k, y0 + k, x1 - k, y1 - k)


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def text_center(d, box, text, fill, f=F_CELL):
    d.text(center(box), text, fill=fill, font=f, anchor="mm")


def tile(d, box, kind):
    """Terrain shared by maze-like grids."""
    fill = {"wall": WALL, "rough": ROUGH}.get(kind, CARD)
    d.rectangle(box, fill=fill, outline=GRID if kind != "wall" else WALL)


def agent_disc(d, box, caught=False):
    d.ellipse(inset(box, max(3, (box[2] - box[0]) // 5)), fill=AGENT)
    if caught:
        b = inset(box, 3)
        d.line((b[0], b[1], b[2], b[3]), fill=RED, width=3)
        d.line((b[0], b[3], b[2], b[1]), fill=RED, width=3)


def maze_cells(height, width, walls, default="floor") -> Cells:
    return {(r, c): ("wall" if (r, c) in walls else default,) for r in range(height) for c in range(width)}


# PortalMaze ------------------------------------------------------------------------------------------------------


def portalmaze_scene(state) -> Scene:
    L = state.layout
    cells = maze_cells(L.height, L.width, L.walls)
    for p in L.pads:
        cells[p] = ("floor", "pad")
    for p, color in L.doors:
        cells[p] = ("floor", "door", color, color in state.held)
    for p, color in L.keys:
        if color not in state.held:
            cells[p] = ("floor", "key", color)
    cells[L.exit] = ("floor", "exit")
    cells[state.agent] = cells[state.agent] + ("agent",)
    return Scene(L.height, L.width, cells)


def portalmaze_token(d, box, token):
    tile(d, box, token[0])
    kind = token[1] if len(token) > 1 else None
    if kind == "pad":
        d.ellipse(inset(box, 4), outline=PORTAL, width=2)
    elif kind == "door":
        d.rectangle(inset(box, 2), fill=CARD if token[3] else ITEM, outline=ITEM, width=2)
        text_center(d, box, token[2].upper(), INK)
    elif kind == "key":
        text_center(d, box, token[2], ITEM, F_TEXT)
    elif kind == "exit":
        d.rectangle(inset(box, 3), fill=GOAL)
    if token[-1] == "agent":
        agent_disc(d, box)


def portalmaze_status(state) -> str:
    keys = ", ".join(sorted(state.held)) or "none"
    return "exit reached" if state.status == "escaped" else f"keys held: {keys}"


# Patrol ----------------------------------------------------------------------------------------------------------


def patrol_scene(state) -> Scene:
    L = state.layout
    cells = maze_cells(L.height, L.width, L.walls)
    cells[L.exit] = ("floor", "exit")
    for g in L.guards(state.tick):
        cells[g] = ("floor", "guard")
    cells[state.agent] = cells[state.agent] + ("agent", state.status == "caught")
    return Scene(L.height, L.width, cells)


def patrol_token(d, box, token):
    tile(d, box, token[0])
    if "exit" in token:
        d.rectangle(inset(box, 3), fill=GOAL)
    if "guard" in token:
        d.regular_polygon((*center(box), (box[2] - box[0]) * 0.36), 4, rotation=45, fill=HAZARD)
    if "agent" in token:
        agent_disc(d, box, caught=token[-1])


def patrol_status(state) -> str:
    return {"playing": "in progress", "caught": "caught", "escaped": "exit reached"}[state.status]


# IceSlide --------------------------------------------------------------------------------------------------------


def iceslide_scene(state) -> Scene:
    L = state.layout
    cells = maze_cells(L.height, L.width, L.walls)
    for p in L.rough:
        cells[p] = ("rough",)
    for p in L.sockets:
        cells[p] = cells[p] + ("socket",)
    for p in state.crates:
        cells[p] = cells[p] + ("crate",)
    for p in state.fixed:
        cells[p] = cells[p] + ("fixed",)
    cells[state.agent] = cells[state.agent] + ("agent",)
    return Scene(L.height, L.width, cells)


def iceslide_token(d, box, token):
    tile(d, box, token[0])
    k = (box[2] - box[0]) // 6
    if "socket" in token:
        d.rectangle(inset(box, k - 1), outline=GOAL, width=2)
    if "crate" in token or "fixed" in token:
        d.rectangle(inset(box, k + 2), fill=CRATE, outline=GOAL if "fixed" in token else CRATE, width=2)
    if "agent" in token:
        agent_disc(d, box)


def iceslide_status(state) -> str:
    return f"sockets filled: {len(state.fixed)}/{len(state.layout.sockets)}"


# Switchboard -----------------------------------------------------------------------------------------------------


def switchboard_scene(state, pressed: int | None = None) -> Scene:
    w = state.wiring
    buttons = len(w.toggles)
    cols = max(w.cols, buttons)
    offset = (cols - w.cols) // 2
    cells: Cells = {}
    for r in range(w.rows):
        for c in range(w.cols):
            cells[(r, c + offset)] = ("lamp", bool(state.lamps >> (r * w.cols + c) & 1))
    for i in range(buttons):
        cells[(w.rows + 1, i)] = ("button", i + 1, bool(state.disabled >> i & 1), pressed == i + 1)
    return Scene(w.rows + 2, cols, cells)


def switchboard_token(d, box, token):
    if token[0] == "lamp":
        d.ellipse(inset(box, 5), fill=ITEM if token[1] else OFF, outline=ITEM if token[1] else MUTED)
        return
    _, number, disabled, pressed = token
    b = inset(box, 3)
    d.rounded_rectangle(b, radius=4, fill=OFF if disabled else CARD, outline=AGENT if pressed else MUTED,
                        width=3 if pressed else 1)
    text_center(d, box, str(number), MUTED if disabled else INK, F_TEXT)
    if disabled:
        d.line((b[0] + 2, b[3] - 2, b[2] - 2, b[1] + 2), fill=INK, width=2)


def switchboard_status(state) -> str:
    on = bin(state.lamps).count("1")
    disabled = [str(i + 1) for i in range(len(state.wiring.toggles)) if state.disabled >> i & 1]
    return f"lamps on: {on}/{state.wiring.n_lamps}; disabled: {', '.join(disabled) or 'none'}"


# Depot -----------------------------------------------------------------------------------------------------------


def _facts(state) -> list[list[str]]:
    return [fact[1:-1].split() for fact in state.facts]


def _depot_goals(state) -> list[str]:
    return re.findall(r"\(on \w+ \w+\)", state.problem.split("(:goal", 1)[1])


def _depot_counts(state) -> tuple[int, int]:
    """(places, crates) from the problem's object declarations."""
    places = len(re.findall(r"\bl\d+\b", state.problem.split("- depot")[0].split(":objects")[1]))
    crates = len(re.findall(r"\bc\d+\b", state.problem.split("- crate")[0].split("- hoist")[1]))
    return places, crates


def depot_scene(state) -> Scene:
    """Columns: one per place plus the truck's cargo. Rows: label, hoist, stack (top to bottom), pallet, truck."""
    places, crates = _depot_counts(state)
    atoms = _facts(state)
    on = {a[2]: a[1] for a in atoms if a[0] == "on"}
    lifting = {a[1]: a[2] for a in atoms if a[0] == "lifting"}
    truck_at = next(int(a[2][1:]) for a in atoms if a[0] == "at" and a[1] == "t0")
    cargo = sorted(a[1] for a in atoms if a[0] == "in")
    goals = set(_depot_goals(state))
    rows = crates + 4
    cells: Cells = {}
    for p in range(places):
        cells[(0, p)] = ("label", f"l{p}")
        cells[(1, p)] = ("hoist", lifting.get(f"h{p}"))
        stack, cursor = [], f"p{p}"
        while cursor in on:
            cursor = on[cursor]
            stack.append(cursor)
        for height in range(crates):
            crate = stack[height] if height < len(stack) else None
            below = (stack[height - 1] if height else f"p{p}") if crate else None
            placed = crate is not None and f"(on {crate} {below})" in goals
            cells[(rows - 3 - height, p)] = ("crate", crate, placed) if crate else ("empty",)
        cells[(rows - 2, p)] = ("pallet",)
        cells[(rows - 1, p)] = ("truck",) if truck_at == p else ("empty",)
    cells[(0, places)] = ("label", "truck")
    for i in range(crates + 2):
        cells[(1 + i, places)] = ("cargo", cargo[i] if i < len(cargo) else None)
    return Scene(rows, places + 1, cells, square=False)


def depot_token(d, box, token):
    kind = token[0]
    w = box[2] - box[0]
    pad = max(4, (w - 44) // 2)
    crate_box = (box[0] + pad, box[1] + 2, box[2] - pad, box[3] - 1)
    if kind == "label":
        text_center(d, box, token[1], MUTED, F_TEXT)
    elif kind == "hoist":
        x = center(box)[0]
        d.line((x, box[1], x, box[1] + 6), fill=MUTED, width=2)
        d.line((x - 10, box[1] + 6, x + 10, box[1] + 6), fill=MUTED, width=2)
        if token[1]:
            d.rectangle((crate_box[0], box[1] + 8, crate_box[2], box[3] - 1), fill=CRATE)
            text_center(d, (box[0], box[1] + 8, box[2], box[3]), token[1], CARD)
    elif kind == "crate" and token[1]:
        d.rectangle(crate_box, fill=CRATE, outline=GOAL if token[2] else CRATE, width=3)
        text_center(d, box, token[1], CARD)
    elif kind == "pallet":
        d.rectangle((box[0] + pad - 4, box[1] + 3, box[2] - pad + 4, box[1] + 9), fill=WALL)
    elif kind == "truck":
        d.rounded_rectangle((box[0] + pad - 4, box[1] + 3, box[2] - pad + 4, box[3] - 3), radius=4, fill=AGENT)
        text_center(d, box, "t0", CARD)
    elif kind == "cargo":
        d.rectangle(inset(crate_box, 0), outline=GRID)
        if token[1]:
            d.rectangle(crate_box, fill=CRATE)
            text_center(d, box, token[1], CARD)


def depot_status(state) -> str:
    goals = _depot_goals(state)
    return f"goal facts true: {sum(g in state.facts for g in goals)}/{len(goals)}"


# PushT -----------------------------------------------------------------------------------------------------------


def pusht_substeps(game, level, profile, run: Rollout):
    """Per-control-step states for both panels, replayed on a one-step-hold copy of the game.

    Returns that copy with the states: they must be rendered with it, since a state is (seed, action history).

    Holding an absolute target for four control steps is the same native computation as four one-step holds
    with the same target, so these frames are exact intermediate states of the rollouts in `run`. The predicted
    target is fixed once per action from the pusher position at the start of that action, as `target_gain` does.
    """
    from world_adapt.environments.pusht.game import PushT

    gain = 0.8 if profile.atoms else 1.0
    fine = PushT(hold_steps=1)
    real, pred = [level.state], [level.state]
    marks = [(None, None)]
    for action in run.actions:
        requested = tuple(float(v) for v in action.split()[1:])
        aimed = fine.transformed_target(pred[-1], action, gain=gain)
        for _ in range(game.hold_steps):
            real.append(fine.step(real[-1], action).state)
            pred.append(fine.simulate(pred[-1], action, target=aimed).state)
            marks.append((requested, aimed))
    for coarse, exact in ((run.real[-1], real[-1]), (run.predicted[-1], pred[-1])):
        a, b = game.physical(coarse), fine.physical(exact)
        if abs(a["coverage"] - b["coverage"]) > 1e-9 or a["block"] != b["block"]:
            raise RuntimeError("PushT sub-step replay does not reproduce the rollout")
    return fine, real, pred, marks


def pusht_panel(game, state, requested, aimed, predicted: bool) -> Image.Image:
    img = Image.open(io.BytesIO(game.render_png(state, size=512))).convert("RGB").resize((PANEL, PANEL), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    scale = PANEL / 512

    def cross(p, color):
        x, y = p[0] * scale, p[1] * scale
        d.line((x - 5, y, x + 5, y), fill=color, width=2)
        d.line((x, y - 5, x, y + 5), fill=color, width=2)

    if requested:
        cross(requested, INK)
        if predicted and aimed != requested:
            cross(aimed, RED)
    return img


def pusht_status(game, state) -> str:
    coverage = game.physical(state)["coverage"]
    return f"goal overlap: {100 * coverage:.0f}%" + (" (success)" if coverage > 0.95 else "")


# Frame composition -----------------------------------------------------------------------------------------------


def swatch(d, x, y, color, shape="square"):
    if shape == "disc":
        d.ellipse((x, y, x + 10, y + 10), fill=color)
    elif shape == "ring":
        d.ellipse((x, y, x + 10, y + 10), outline=color, width=2)
    elif shape == "cross":
        d.line((x, y + 5, x + 10, y + 5), fill=color, width=2)
        d.line((x + 5, y, x + 5, y + 10), fill=color, width=2)
    elif shape == "outline":
        d.rectangle((x, y, x + 10, y + 10), outline=color, width=2)
    else:
        d.rectangle((x, y, x + 10, y + 10), fill=color)


LEGENDS = {
    "depot": [(CRATE, "square", "crate"), (GOAL, "outline", "crate on its goal support"), (AGENT, "square", "truck"),
              (WALL, "square", "pallet")],
    "portalmaze": [(AGENT, "disc", "agent"), (GOAL, "square", "exit"), (ITEM, "square", "key / door"),
                   (PORTAL, "ring", "pad"), (WALL, "square", "wall")],
    "patrol": [(AGENT, "disc", "agent"), (HAZARD, "square", "guard"), (GOAL, "square", "exit"),
               (WALL, "square", "wall")],
    "switchboard": [(ITEM, "disc", "lamp on"), (OFF, "disc", "lamp off"), (AGENT, "outline", "pressed"),
                    (OFF, "square", "disabled button")],
    "iceslide": [(AGENT, "disc", "agent"), (CRATE, "square", "crate"), (GOAL, "outline", "socket"),
                 (ROUGH, "square", "type R floor"), (WALL, "square", "wall")],
    "pusht": [(AGENT, "disc", "pusher"), ("#90ee90", "square", "goal pose"), (INK, "cross", "requested target"),
              (RED, "cross", "predicted target")],
}


def compose(env, title, subtitle, step, total, action, left, right, left_status, right_status, run: Rollout,
            diverged: bool) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(img)
    d.text((MARGIN, 5), title, fill=INK, font=F_HEAD)
    d.text((WIDTH - MARGIN, 7), subtitle, fill=MUTED, font=F_SMALL, anchor="ra")
    y = HEADER
    xl, xr = MARGIN, WIDTH - MARGIN - PANEL
    d.text((xl, y + 2), "Real environment", fill=INK, font=F_TEXT)
    d.text((xr, y + 2), "Predictor (perturbed)", fill=RED if diverged else INK, font=F_TEXT)
    y += LABEL
    img.paste(left, (xl, y))
    img.paste(right, (xr, y))
    d.rectangle((xl - 1, y - 1, xl + PANEL, y + PANEL), outline=GRID)
    d.rectangle((xr - 2, y - 2, xr + PANEL + 1, y + PANEL + 1), outline=RED if diverged else GRID,
                width=3 if diverged else 1)
    y += PANEL + 3
    d.text((xl, y), left_status, fill=MUTED, font=F_SMALL)
    d.text((xr, y), right_status, fill=RED if diverged else MUTED, font=F_SMALL)
    y += STATUS
    d.text((MARGIN, y), f"step {step}/{total}: {action}" if action else f"step 0/{total}: start state",
           fill=INK, font=F_TEXT)
    if diverged:
        note = f"prediction differs from reality since step {run.diverge} ({', '.join(run.atoms)})"
        d.text((MARGIN, y + 16), note, fill=RED, font=F_TEXT)
    else:
        d.text((MARGIN, y + 16), "prediction matches reality so far", fill=MUTED, font=F_TEXT)
    y += CAPTION
    x = MARGIN
    for color, shape, label in LEGENDS[env]:
        swatch(d, x, y + 3, color, shape)
        d.text((x + 14, y + 1), label, fill=MUTED, font=F_SMALL)
        x += 14 + int(d.textlength(label, font=F_SMALL)) + 12
    return img


GRID_ENVS = {
    "portalmaze": (portalmaze_scene, portalmaze_token, portalmaze_status),
    "patrol": (patrol_scene, patrol_token, patrol_status),
    "iceslide": (iceslide_scene, iceslide_token, iceslide_status),
    "switchboard": (switchboard_scene, switchboard_token, switchboard_status),
    "depot": (depot_scene, depot_token, depot_status),
}


def frames_for(env: str) -> tuple[list[Image.Image], list[int]]:
    game, level, profile, seed = study_task(env)
    run = rollout(game, level, profile)
    tier = level.features.get("tier", "")
    subtitle = f"seed {seed}, task {TASK_INDEX[env]} ({tier}), profile {profile.name}"
    total = len(run.actions)
    frames, durations = [], []

    def add(step, action, left, right, ls, rs, ms, step_done=True):
        """Append a frame; pause on the frame that completes the diverging step."""
        diverged = run.diverge is not None and step >= run.diverge
        frames.append(compose(env, TITLES[env], subtitle, step, total, action, left, right, ls, rs, run, diverged))
        durations.append(ms + (DIVERGE_MS if step_done and run.diverge == step else 0))

    if env == "pusht":
        fine, real, pred, marks = pusht_substeps(game, level, profile, run)
        hold = game.hold_steps
        for k, (r, p, (requested, aimed)) in enumerate(zip(real, pred, marks)):
            step = -(-k // hold)
            action = run.actions[step - 1] if step else None
            add(step, action, pusht_panel(fine, r, requested, aimed, False),
                pusht_panel(fine, p, requested, aimed, True), pusht_status(fine, r), pusht_status(fine, p),
                START_MS if k == 0 else STEP_MS[env], step_done=k % hold == 0)
        fine.close()
        game.close()
    else:
        scene, token, status = GRID_ENVS[env]
        for step, (r, p) in enumerate(zip(run.real, run.predicted)):
            action = run.actions[step - 1] if step else None
            if env == "switchboard":
                pressed = int(action.split()[1]) if action else None
                rs, ps = scene(r, pressed), scene(p, pressed)
            else:
                rs, ps = scene(r), scene(p)
            diff = {c for c in rs.cells.keys() | ps.cells.keys() if rs.cells.get(c) != ps.cells.get(c)}
            add(step, action, paint_grid(rs, token, set()), paint_grid(ps, token, diff), status(r), status(p),
                START_MS if step == 0 else STEP_MS[env])
    durations[-1] += END_MS
    return frames, durations


def save_gif(frames: list[Image.Image], durations: list[int], path: Path) -> None:
    # One shared palette, built from a sample of frames, keeps colors stable across the animation.
    sample = frames[:: max(1, len(frames) // 8)]
    sheet = Image.new("RGB", (WIDTH, HEIGHT * len(sample)))
    for i, frame in enumerate(sample):
        sheet.paste(frame, (0, HEIGHT * i))
    palette = sheet.quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    indexed[0].save(path, save_all=True, append_images=indexed[1:], duration=durations, loop=0, optimize=True,
                    disposal=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", action="append", choices=sorted(TASK_INDEX), help="render only these (repeatable)")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/assets", help="output directory")
    args = parser.parse_args()
    for env in args.env or list(TASK_INDEX):
        frames, durations = frames_for(env)
        path = args.out / f"{env}.gif"
        save_gif(frames, durations, path)
        print(f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}: {len(frames)} frames, "
              f"{sum(durations) / 1000:.1f} s, {path.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
