# spot_reward_lab — Week 5: customize the reward function (Isaac Lab 3 + Newton)

Tasks (registered through the `isaaclab.tasks` entry point):

| Task | What changes vs `IsaacContrib-Velocity-Flat-Spot` |
|---|---|
| `Spot-Reward-Exp1` | foot_clearance target 0.10 → 0.28 m, gait = front/rear pairs (Lab 5.2) |
| `Spot-Reward-Tripod` | gait → BrokenLegGaitReward + broken_leg_elevation (Lab 5.3) |
| `Spot-Reward-Pace` | gait pairs = left/right (Exercise 5.2) |
| `Spot-Reward-Tripod-Fix` | Tripod + the old foot rewards ignore fr_foot (Exercise 5.3 — write `air_time_reward_any_feet` first) |
| `Spot-Limp` | baseline rewards, fr motors with stiffness 0 (Exercise 5.4) |
| `Spot-Reward-Tripod-Limp` | Tripod rewards, fr motors with stiffness 0 (Exercise 5.4) |

Spot needs `physics=newton_mjwarp` (the default preset is PhysX).

The baseline policy (4096 envs × 1000 iterations, ~2 h on an RTX 3060 laptop) is provided:
`pretrained/spot_baseline/model_999.pt` (+ `params/`). `[tool.uv.sources]` in `pyproject.toml` finds
Isaac Lab through relative paths (`../../../../IsaacLab/...`), i.e. `IsaacLab` must sit four folders up from
this project (e.g. `~/IsaacLab` + `~/Downloads/Simulation_for_Humanoid/Week05_Reward_Function/spot_reward_lab`).
Otherwise, point them at your install once before `uv sync`:

```bash
export ISAACLAB_PATH=~/IsaacLab   # change to where you installed Isaac Lab
sed -i "s|\.\./\.\./\.\./\.\./IsaacLab|$ISAACLAB_PATH|g" pyproject.toml
```

Run everything from this folder (`cd Week05_Reward_Function/spot_reward_lab` from the course folder you unzipped/cloned):

```bash
uv sync
uv run isaaclab list_envs | grep Spot
# baseline (optional — or use pretrained/spot_baseline/model_999.pt)
uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-Flat-Spot --num_envs 4096 \
    physics=newton_mjwarp agent.max_iterations=1000 agent.run_name=baseline
# Lab 5.2: fine-tune from the baseline
uv run isaaclab train --rl_library rsl_rl --task Spot-Reward-Exp1 --num_envs 4096 physics=newton_mjwarp \
    --checkpoint pretrained/spot_baseline/model_999.pt agent.run_name=exp1 agent.max_iterations=300
# measure
uv run python scripts/eval_gait.py --task Spot-Reward-Exp1 --num_envs 64 physics=newton_mjwarp \
    --checkpoint logs/rsl_rl/spot_flat/<run>/model_1298.pt
```
