# g1_maze_lab — Week 7: Curriculum learning + G1 Maze Challenge

Isaac Lab 3 + Newton project. Two parts:

| Part | Tasks | Main files |
|---|---|---|
| Lab 7.1 · Curriculum | `G1-Curriculum-{Base,Reward,Push,Velocity,Full,ActionRate}` | `src/g1_maze_lab/tasks/curriculum/mdp/curriculums.py`, `env_cfg.py` |
| Lab 7.2 · Maze challenge | `G1-Maze-{Challenge,Random,Open}`, `G1-Maze-{Challenge,Open}-FlatPolicy` | `src/g1_maze_lab/terrains/maze.py`, `src/g1_maze_lab/autopilot.py`, `scripts/maze_autopilot.py` |

## Setup (first time)
Place this folder at `~/Downloads/Simulation_for_Humanoid/Week07_Curriculum_Maze/g1_maze_lab`
(`pyproject.toml` points to `~/IsaacLab` with a relative path), then:

```bash
cd ~/Downloads/Simulation_for_Humanoid/Week07_Curriculum_Maze/g1_maze_lab
uv sync
uv run isaaclab list_envs | grep G1-
```

## Pretrained checkpoints (`pretrained/`)
- `g1_rough/model_pretrained.pt` — Isaac Lab's `Isaac-Velocity-Rough-G1` (Newton). Walking policy for the maze, start point for fine-tuning.
- `g1_flat/model_pretrained.pt` — Isaac Lab's `Isaac-Velocity-Flat-G1` (Newton, Week 3). Only for Exercise 7.3.
- `curriculum/<run>/model_5298.pt` — the Lab 7.1 fine-tuning runs from the slides (300 iterations each).

## Lab 7.1 — train with a curriculum (fine-tune, ~6 min per run on an RTX 3060 Laptop)
```bash
uv run isaaclab train --rl_library rsl_rl --task G1-Curriculum-Full --num_envs 1024 \
    --checkpoint pretrained/g1_rough/model_pretrained.pt agent.max_iterations=300 agent.run_name=full300
uv run tensorboard --logdir logs/rsl_rl/g1_curriculum          # graphs Curriculum/*
uv run python scripts/eval_g1.py --task G1-Curriculum-Base --num_envs 64 \
    --checkpoint logs/rsl_rl/g1_curriculum/<run>/model_5298.pt   # speed / falls / air time
EVAL_PUSH=1.0 EVAL_CMDS="0.5,0,0" uv run python scripts/eval_g1.py ...   # push test
```

## Lab 7.2 — maze autopilot
```bash
# watch it (Newton viewer window)
uv run python scripts/maze_autopilot.py --task G1-Maze-Challenge --num_envs 1 \
    --checkpoint pretrained/g1_rough/model_pretrained.pt --viz newton
# score it headless: 16 robots in random mazes
MAZE_STRATEGY=flood uv run python scripts/maze_autopilot.py --task G1-Maze-Random --num_envs 16 \
    --checkpoint pretrained/g1_rough/model_pretrained.pt
```
Viewer window (`--viz newton`): the camera follows robot 0 from behind (key **C** = free camera / follow again),
a timer at the top shows TIME → **GOAL xx.x s**, cells walked and falls/crashes; a robot that reaches the goal stays there.

Video with a timer overlay: add `MAZE_REC_DIR=frames MAZE_REC_LABEL="my run"` (frames every `MAZE_REC_EVERY` steps),
then `scripts/make_video.py out.mp4 spec.json` (title cards + clips, see the docstring).
Several robots, one per maze, seen from above: `--task G1-Maze-Random --num_envs 4 MAZE_ONE_PER_TILE=1 MAZE_CAM=grid`
with `env.scene.terrain.terrain_generator.num_rows=2 env.scene.terrain.terrain_generator.num_cols=2 env.terminations.goal_reached=null`.

Settings: `MAZE_STRATEGY=left|right|flood`, `MAZE_SENSOR=height|lidar`, `MAZE_SPEED` (m/s), `MAZE_TIME` (s),
`MAZE_SHOW_SCAN=1`, `MAZE_CAM=chase|top`, `MAZE_SNAP_AT="5,20"` (save viewer frames).
Change the sensor from the command line, e.g. `env.scene.maze_scanner.pattern_cfg.resolution=0.2`.

## Challenge 7.5 — G1 Hazard Maze (extra)
Task `G1-Maze-Hard`: random 7 x 7 maze (seed), 300 s. Two ways out of the start: a SHORT route with hard
hazards (a trench 0.15-0.22 m, 2 steep ramps 0.24-0.30 m, 2 spike beds, ice) and a LONG, twisty route with easy ones
(2 low ramps 0.10-0.18 m, loose boxes, ice). The first hard hazard of the short route is 1-2 cells after the fork,
so walking in a little shows it. Spike beds: red cones 25-35 cm, slippery, on one side of a corridor (no extra rule,
just hard to walk on). A DNF is ranked by how close the robot got to the goal along the route it was on.
```bash
# try your policy / pilot (nothing goes to the leaderboard)
uv run python scripts/challenge.py --policy logs/rsl_rl/<exp>/<run>/model_XXXX.pt --pilot pilot.py --seed 1 --viz newton
# instructor: upload page + leaderboard (students http://<ip>:8000, projector http://<ip>:8000/board)
CHALLENGE_SEED=<seed> uv run python scripts/challenge_server.py
# instructor: when everyone has uploaded, run the queue on the round's seed (upload order; the pages show the queue)
uv run python scripts/challenge_runner.py --seed <seed>            # --all = everyone again, --no-replay = faster
# then play every run on the projector, last place first, the winner at the end: http://<ip>:8000/replay
uv run python scripts/challenge.py --team team_a --seed <seed> --viz newton   # or one team by hand
```
Submit `policy.pt` (required, same network as G1 Rough: checked on upload), `pilot.py` (optional, `class Pilot`, see
`DefaultPilot` in `scripts/challenge.py`) and up to 5 helper `.py` files the pilot imports (every `.py` must compile).
The leaderboard shows the round's seed only (`CHALLENGE_SEED`), best run per team; run logs: `submissions/_logs/`, replay videos (the latest run of each team, recorded from the Newton viewer: a run takes about 1.2x its maze time, `--no-replay` runs without a window, about 2x faster): `submissions/_replays/`. A fall, a crash or being stuck 10 s sends the robot back to the start; the clock keeps running.

## Tests
`uv run pytest tests` checks the maze generator and the planner without the simulator (a robot that jumps cell to cell).

Tested on an RTX 3060 Laptop 6 GB · Ubuntu · Isaac Lab 3 + Newton (Oct 2026).
