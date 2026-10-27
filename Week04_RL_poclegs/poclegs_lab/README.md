# poclegs_lab — Week 4 + Week 6 (Isaac Lab 3 + Newton)

The poclegs biped (BipedalRobotSim / original course supplementary file) as an external Isaac Lab project,
created with `uv run isaaclab --new` and ported from Isaac Lab 2 to Isaac Lab 3 + Newton.

## Setup

```bash
cd Week04_RL_poclegs/poclegs_lab   # from the course folder you unzipped/cloned
uv sync                     # project venv: Newton backend, rsl_rl, kit-less URDF importer
uv run isaaclab list_envs   # Poclegs-Velocity-Flat / -Rough / -Flat-Urdf
```

`pyproject.toml` points `[tool.uv.sources]` at Isaac Lab with relative paths (`../../../../IsaacLab/...`).
They only resolve if `IsaacLab` sits four folders up from this project (e.g. `~/IsaacLab` +
`~/Downloads/Simulation_for_Humanoid/Week04_RL_poclegs/poclegs_lab`). Otherwise, point them at your
install once before `uv sync`:

```bash
export ISAACLAB_PATH=~/IsaacLab   # change to where you installed Isaac Lab
sed -i "s|\.\./\.\./\.\./\.\./IsaacLab|$ISAACLAB_PATH|g" pyproject.toml
```

## Tasks

| Task | Robot |
|---|---|
| `Poclegs-Velocity-Flat` | original-course USD, made Newton-friendly (`poclegs_newton.usd`) |
| `Poclegs-Velocity-Rough` | same robot, original-course rough terrain + curriculum |
| `Poclegs-Velocity-Flat-Tuned` | same robot, H1-style rewards + y-axis commands + joint velocity limits (walks; Week 4 Ex 4.3) |
| `Poclegs-Velocity-Flat-Urdf` | robot rebuilt from URDF (Week 6), Tuned recipe; swap it with Hydra overrides |

## Pretrained checkpoints

`pretrained/<run>/model_600.pt` and `model_1499.pt` (+ `params/` with the exact env/agent config):

| Run | Task | Notes |
|---|---|---|
| `base1500` | `Poclegs-Velocity-Flat` | original-course rewards: stands/turns, does not walk |
| `tuned1500` | `Poclegs-Velocity-Flat-Tuned` | walks (vy ±0.4 -> ~±0.3 m/s) |
| `urdf1x` | `Poclegs-Velocity-Flat-Urdf` | x1.0 legs, Week 6 baseline |
| `long13` | `Poclegs-Velocity-Flat-Urdf` | x1.3 legs: needs the `usd_long` override and z 0.539 |

Pass the file directly, e.g. `--checkpoint pretrained/tuned1500/model_1499.pt`.

## Week 4

```bash
uv run isaaclab zero_agent --task Poclegs-Velocity-Flat --num_envs 16 --viz newton
uv run isaaclab train --rl_library rsl_rl --task Poclegs-Velocity-Flat --num_envs 2048 --max_iterations 1500
uv run isaaclab train --rl_library rsl_rl --task Poclegs-Velocity-Flat-Tuned --num_envs 2048 --max_iterations 1500
uv run isaaclab play  --rl_library rsl_rl --task Poclegs-Velocity-Flat-Tuned --num_envs 16 --checkpoint latest --viz newton
uv run python scripts/eval_policy.py --task Poclegs-Velocity-Flat --num_envs 64 --checkpoint latest
```

## Week 6

```bash
D=src/poclegs_lab/assets/data
uv run python scripts/usd_to_urdf.py $D/poclegs/poclegs_newton.usd $D/urdf      # USD -> URDF + meshes
uv run python scripts/check_urdf.py  $D/poclegs/poclegs_newton.usd $D/urdf/poclegs.urdf
mkdir -p $D/urdf_long && cp -r $D/urdf/meshes $D/urdf_long/
uv run python scripts/scale_leg.py   $D/urdf/poclegs.urdf $D/urdf_long/poclegs_long.urdf --thigh 1.3 --shank 1.3
uv run python scripts/urdf_to_usd.py $D/urdf_long/poclegs_long.urdf $D/usd_long  # URDF -> USD (kit-less)
uv run isaaclab train --rl_library rsl_rl --task Poclegs-Velocity-Flat-Urdf --num_envs 2048 \
    env.scene.robot.spawn.usd_path=$PWD/$D/usd_long/poclegs_long/poclegs_long.usda \
    "env.scene.robot.init_state.pos=[0.0,0.0,0.539]"
```

## Scripts

| Script | What it does |
|---|---|
| `make_newton_usd.py` | flattens an Isaac Sim USD, turns instancing off and moves the collision API onto the meshes (Newton needs it there) |
| `usd_to_urdf.py` | USD -> URDF + OBJ meshes (inertia from the collision convex hulls, colors from materials) |
| `check_urdf.py` | compares URDF forward kinematics and joint axes with the USD |
| `scale_leg.py` | lengthens thigh / shank (joint origin + mesh origin and scale) |
| `urdf_to_usd.py` | URDF -> USD with Isaac Lab's `UrdfConverter`, no Isaac Sim needed |
| `eval_policy.py` | fixed velocity commands, reports measured velocity and falls |
