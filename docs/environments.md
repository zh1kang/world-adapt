# Environments

The benchmark has six deterministic environments.
The real environment always follows its true dynamics T and never changes during a study.
Before committing actions, the agent can ask a predictor to roll out a sequence of actions from the current real state.
The accurate predictor rolls out T.
The faulty predictor rolls out T_phi: the true step followed by a fixed, ordered set of perturbation atoms, each a condition plus an effect on the predicted next state.
Errors compound, because every predicted step starts from the previous predicted state.
The study uses one faulty profile per environment, its primary profile, named below.

In every study run the agent receives a text observation, the goal, and a short statement of the true rules.
Hidden task parameters, such as portal links, guard routes, and button wiring, are never shown.
The real action budget is three times the reference solution length (at least 10), except in PushT, which has a fixed budget of 40.
The predicted-transition budget is twice the real budget.

## Reading the GIFs

Each GIF replays one task from the study's seed block 3000 using the task's reference solution.
The left panel is the real environment.
The right panel is the faulty predictor's rollout of the same actions from the same start state, without correction from reality.
From the first step at which the prediction differs from reality, the right panel has a red border, the cells that differ are outlined, and the caption names the atom that changed the prediction.
To regenerate them, run `uv run --locked python scripts/render_environment_gifs.py`.

## Depot

![Depot](assets/depot.gif)

Depot is the standard Depot planning domain in PDDL, executed with Pyperplan's grounded STRIPS operators.
The agent moves crates with five parameterized actions: `drive <truck> <from> <to>`, `lift <hoist> <crate> <support> <place>`, `drop <hoist> <crate> <support> <place>`, `load <hoist> <crate> <truck> <place>`, and `unload <hoist> <crate> <truck> <place>`.
An action whose preconditions do not hold leaves the state unchanged.
The observation is text: the typed object list, every current fact, and the goal facts.
The task succeeds when every goal fact holds, which means that each crate rests on its goal support; there is no failure state, fuel limit, or truck capacity.

In the study's `structural-v2` curriculum, easy, medium, and hard tasks have 2, 3, and 4 places with 5, 8, and 11 crates.
All crates start in stacks on the first two pallets, and the goal stacks use a new random order spread over all places.
Tasks are deduplicated up to renaming of objects.
The reference solution is the generator's plan, which loads every crate onto the truck and then unloads them into the goal stacks; it is valid but not the shortest.

**Perturbation.**
The profile `cargo_and_stacks` contains two prerequisite atoms.
`single_cargo` makes the predictor treat `load` as blocked whenever the truck already holds a crate, as if the truck could carry only one.
`no_stacking` makes the predictor treat `drop` as blocked whenever the support is a crate rather than a pallet.
In both cases the predicted state is unchanged, which is exactly how a real blocked action looks, so the faulty predictor reports that a legal action has no effect.
Both atoms fire on the reference plan, which loads several crates and builds stacks.

## PortalMaze

![PortalMaze](assets/portalmaze.gif)

PortalMaze is a grid maze in which the agent must reach the exit.
The actions are `up`, `down`, `left`, and `right`; walls block movement, and a door stays closed until the agent has stepped on the key of the same letter (keys are never used up).
Stepping onto a portal pad teleports the agent to a fixed destination pad that the observation does not show; links can be one-way, and one pad leads into a sealed two-pad pocket from which the exit cannot be reached.
The observation is a text grid with a legend, plus the coordinates of the agent, exit, keys, doors, and pads, the keys held, and the status.
The task succeeds when the agent reaches the exit; there is no terminal failure, but entering the pocket makes the task unsolvable.

In the study's `extended` curriculum, easy, medium, and hard mazes are 9x9, 11x11, and 13x13 with 6, 7, and 8 pads and 2, 3, and 3 door colors.
Their shortest solutions must be 14-28, 18-45, and 18-70 moves long and must collect at least 2, 3, and 3 keys and use at least 1, 2, and 2 teleports.
The shortest path that treats pads as plain floor must fail, so the agent has to learn where pads lead.

**Perturbation.**
The profile `door_and_portal_prerequisite` combines `door_wrong_key` and `reverse_portal`.
`door_wrong_key` fires when the agent walks into a locked door while it holds some other key: the predictor shows the door opening and the agent stepping into the doorway.
`reverse_portal` fires when the agent steps onto an ordinary pad (not the trap entrance or a pocket pad): the predictor sends the agent to a pad that links into this pad instead of to the pad's true destination.
If no pad links into it, the prediction stays correct.
The faulty predictor therefore gets portal directions backwards and is too lenient about which key opens a door.

## Patrol

![Patrol](assets/patrol.gif)

Patrol asks the agent to reach the exit cell without being caught by guards.
The actions are `up`, `down`, `left`, `right`, and `wait`; after each action, every guard advances one cell along a fixed back-and-forth route that the agent cannot see.
The observation is a text grid with the agent, guards, exit, and walls, plus the coordinates of each and the status.
Ending a step on a guard's new cell, or swapping cells with a guard, is a capture and ends the task in failure; reaching the exit is success.

The study's `calibration` curriculum uses a 10x10 grid with 5 guards whose routes cover 4 to 8 cells.
The shortest path that ignores guards must fail, and the reference solution must be between 16, 17, or 18 (easy, medium, hard) and 32 steps long and at least 4, 4, or 5 steps longer than that naive path.
A task is admitted only if the primary profile changes at least one prediction along the reference solution and makes at least one safe alternative action look fatal.

**Perturbation.**
The profile `adjacent_capture` has one contact atom of the same name.
When a step really ends with the agent alive and orthogonally next to a guard's new cell, the predictor reports a capture.
The real rule needs the same cell or a swap, so this is a false alarm: the predicted rollout ends in failure after moves that are safe.
The atom never hides a real capture.

## Switchboard

![Switchboard](assets/switchboard.gif)

Switchboard is a panel of lamps and seven numbered buttons, and the agent must turn every lamp on.
The action is `press N` for N from 1 to 7.
Each button toggles a fixed hidden subset of lamps, and some buttons are breakers that also permanently disable one specific other button; pressing a disabled button does nothing but still costs an action.
The observation shows the lamp grid, the list of lamps that are off, and the disabled buttons, but not the wiring or the breaker targets.
The task succeeds when all lamps are on; there is no terminal failure, but disabling a button that the solution needs can make the task unsolvable.

In the study's `calibration` curriculum, easy, medium, and hard panels have 3x4, 3x5, and 4x4 lamps and 2, 3, and 4 breakers, each button toggles 2-4, 3-5, or 4-7 lamps, and the shortest solution takes 4-5, 5-6, or 6-7 presses.
The shortest solution must press some breakers whose targets it also needs, so the order of presses matters.
From every start state, some sequence of at most three presses leads to an unsolvable state.

**Perturbation.**
The profile `breaker_retarget_enabled_only` has one prerequisite atom, `breaker_retarget_enabled`.
It fires when the agent presses an enabled breaker while at least one button is already disabled.
The predictor then shows the breaker disabling a different button: the lowest-numbered button that is still enabled and is neither the pressed button nor the true target.
Lamp changes are predicted correctly, so the error appears only in the list of disabled buttons.
Before the first button is disabled, the predictor is exact.

## IceSlide

![IceSlide](assets/iceslide.gif)

IceSlide is a sliding puzzle in which the agent must move every crate into a socket.
The actions are `up`, `down`, `left`, and `right`; the agent slides until the next cell is a wall or a crate, or until it enters a type R tile.
If a loose crate stops the agent's slide and the agent is not on a type R tile, the crate is launched and slides until it is blocked, enters a type R tile, or enters an empty socket, where it is fixed for good.
The observation is a terrain grid and an object grid in text, with coordinate lists for the agent, loose crates, filled and empty sockets, and type R tiles.
The task succeeds when every socket holds a crate; there is no terminal failure, but a crate pushed against a wall or into a corner can make the task unsolvable.

In the study's `calibration` curriculum, easy, medium, and hard boards are 6x6, 7x7, and 8x8 with 1, 2, and 3 crates and 3, 4, and 5 type R tiles.
Their shortest solutions take 4-8, 8-16, and 16-28 moves with at least 1, 3, and 4 launches.
No first move makes an easy or medium task unsolvable, and at most two first moves do so on a hard board.

**Perturbation.**
The profile `launch_outcomes` changes how a launched crate ends its slide, with one atom for each of the three outcomes.
`crate_stops_early`: a crate that slides two or more cells and stops against a wall or crate is predicted to stop one cell short.
`crate_slides_past_rough`: a crate that really stops on a type R tile is predicted to slide on as if type R tiles were plain floor.
`socket_vertical_no_capture`: a crate launched up or down into an empty socket is predicted to rest there loose instead of being fixed.
The agent's own slides are predicted correctly.

## PushT

![PushT](assets/pusht.gif)

PushT is the gym-pusht task: a circular pusher must push a T-shaped block onto a fixed goal pose, simulated with pymunk.
In the study each action is `move_to <x> <y>`, an absolute pusher target in pixels on the 512 x 512 workspace, which a proportional-derivative controller holds for four control steps, so the pusher may not arrive.
The observation is text: the pusher position and velocity, the block pose and velocity, the goal pose, a contact count for the last control step, and the percentage of the goal that the block covers.
The task succeeds when the block covers more than 95% of the goal pose; the real budget is fixed at 40 actions, and there is no exact solver.

The `seeded-v1` curriculum starts from six fixed templates, each with a reference controller of three or four targets.
It moves the pusher and block start positions by up to 4 pixels and the block angle by up to 0.005 radians, and it keeps a start only if the template's reference controller still succeeds with contact.
Tiers are assigned by template (the two translation templates are easy, a small left rotation is medium, and three right rotations are hard), not by measured difficulty.

**Perturbation.**
The profile `target_gain` has one motion atom of the same name, active on every step until the task ends.
The predictor moves the pusher toward a point 80% of the way from its current position to the requested target, so every predicted push falls short.
The shortened targets stay in the predicted rollout's history, so the error compounds over a multi-step prediction, while the real environment always uses the requested target.
