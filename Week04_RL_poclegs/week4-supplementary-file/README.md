# Week 4 supplementary file — build the poclegs project step by step with the slides

These files are ported from the original course's `week5-supplementary-file` (Isaac Lab 2) to **Isaac Lab 3 + Newton**.
Use them with the Week 4 slides, Step 0 – Assignment 4.3: start from an empty template, then copy the files in one part at a time.

In the commands below, `SUP` = this folder, `PRJ` = the project created in Step 0, `ISAACLAB_PATH` = where Isaac Lab is installed
(run the first line from the folder where you unzipped the course, or `cd` into this README's folder and use `SUP=$PWD`).

```bash
SUP=$(realpath Week04_RL_poclegs/week4-supplementary-file)
export ISAACLAB_PATH=~/IsaacLab   # change to where you installed Isaac Lab
PRJ=~/projects/poclegs_lab        # where to create the project (you can change it)
PKG=$PRJ/src/poclegs_lab
```

## Step 0 · Create the project from the template
```bash
cd $ISAACLAB_PATH
uv run isaaclab --new --non_interactive --project_path $(dirname $PRJ) --name poclegs_lab \
    --author "Group X" --task_name locomotion --robot_name poclegs
cd $PRJ && uv sync
uv run isaaclab list_envs | grep Poclegs     # for now only PoclegsLab-Locomotion-Poclegs (the template's example)
```
Use exactly `--name poclegs_lab`: the files in this folder import from `poclegs_lab.*`

## Step 1 · Assets (robot + motors)
```bash
cp $SUP/assets/poclegs.py $PKG/assets/
cp -r $SUP/assets/data/poclegs $PKG/assets/data/
```
- `poclegs.usd` + `configuration/` = the original files from Isaac Sim
- `poclegs_newton.usd` = a version fixed so Newton sees the collisions (made with `scripts/make_newton_usd.py`, see the Pitfall slide)
- Assignment 4.1: open `poclegs.py`, look at `init_state` and convert it to degrees

## Step 2 · Agent (PPO settings)
Open the `agents/rsl_rl_ppo_cfg.py` the template generated, compare it with `$SUP/agents/rsl_rl_ppo_cfg.py` and make it match
(class names `PoclegsRoughPPORunnerCfg` / `PoclegsFlatPPORunnerCfg`, `experiment_name`, network size), or copy over it:
```bash
cp $SUP/agents/rsl_rl_ppo_cfg.py $PKG/tasks/locomotion/config/poclegs/agents/
```

## Step 3 · Environment + MDP
1. Extra reward (`feet_air_time_range`)
   ```bash
   cp $SUP/mdp/rewards.py $SUP/mdp/__init__.pyi $PKG/tasks/locomotion/mdp/
   ```
2. Main env: copy the skeleton over the template's `env_cfg.py`
   ```bash
   cp $SUP/my_poclegs_env_cfg.py $PKG/tasks/locomotion/config/poclegs/env_cfg.py
   ```
   The skeleton has the imports, terrain and env classes (Rough/Flat), but **Observations and Rewards are still empty** (marked TODO)
3. Observations: open `env_cfg.py`, find `TODO (Step 3 · Observations)`, delete the empty `PoclegsObservationsCfg` class
   and paste the contents of `terms/observations.py` in its place
4. Rewards: find `TODO (Step 3 · Rewards)`, delete the empty `PoclegsRewardsCfg` class and paste `terms/rewards.py` in its place
5. (Exercise 4.3 / Week 6) paste `terms/tuned.py` at the end of the file, at `TODO (Exercise 4.3 / Week 6)`
6. Register the tasks
   ```bash
   cp $SUP/register/__init__.py $PKG/tasks/locomotion/config/poclegs/__init__.py
   cp $SUP/register/test_registration.py $PRJ/tests/
   uv run isaaclab list_envs | grep Poclegs   # Poclegs-Velocity-Flat / -Rough / -Flat-Tuned / -Flat-Urdf
   ```

## Step 4 · Zero agent → Train (Assignment 4.2 – 4.3)
```bash
cp $SUP/scripts/*.py $PRJ/scripts/
cd $PRJ
uv run isaaclab zero_agent --task Poclegs-Velocity-Flat --num_envs 16 --viz newton
uv run isaaclab train --rl_library rsl_rl --task Poclegs-Velocity-Flat --num_envs 2048 --max_iterations 1500 --run_name base1500
```

## Stuck?
The project with every step done is in `../poclegs_lab` (solution + checkpoints in `pretrained/`).
You can use it to go on with Exercise 4.1 – 4.4 and Week 6.
Note: the `-Flat-Urdf` task works only after Week 6 (it needs the `usd_from_urdf/` files)
