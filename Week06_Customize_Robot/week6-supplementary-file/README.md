# Week 6 supplementary file — continuing the poclegs project from Week 4

Use it with the project you built in Week 4 (`~/projects/poclegs_lab`), or with the finished one in `../../Week04_RL_poclegs/poclegs_lab`
(the finished one already has everything on this page: skip to Lab 6.1).

| File | What it is for |
|---|---|
| `scripts/usd_to_urdf.py` | USD → URDF + OBJ meshes (Lab 6.1) |
| `scripts/check_urdf.py` | Checks that the URDF's joints match the USD (Lab 6.1) |
| `scripts/scale_leg.py` | Lengthen/shorten the thigh and shank, add mass, change one leg only (Lab 6.2, Exercises) |
| `scripts/urdf_to_usd.py` | URDF → USD without opening Isaac Sim (Lab 6.2) |
| `scripts/eval_policy.py` | Measures speed + number of falls (you already have it if you did Week 4 Step 4) |
| `terms/tuned.py` | Tuned + Urdf classes: paste at the end of env_cfg.py if you have not done Exercise 4.3 |
| `expected_output/*.urdf` | The files you should get: check your work with `diff` |
| `pretrained/urdf1x`, `pretrained/long13` | Already-trained policies (×1.0, ×1.3), to compare against in the Exercises |

Run the first line from the folder where you unzipped the course (or `cd` into this README's folder and use `SUP=$PWD`).
`PRJ` = your Week 4 poclegs project (change it to where you created it).

```bash
SUP=$(realpath Week06_Customize_Robot/week6-supplementary-file)
PRJ=~/projects/poclegs_lab
cd $PRJ
D=src/poclegs_lab/assets/data
```

## Step 0 · Prepare the project (once)
1. Open `pyproject.toml`, add the line `"isaaclab-dev[importers]",` under `dependencies = [`, then
   ```bash
   uv sync          # installs the URDF converter that works without Isaac Sim
   ```
2. Copy the scripts
   ```bash
   cp $SUP/scripts/*.py scripts/
   ```
3. No `Poclegs-Velocity-Flat-Urdf` task yet? Paste `$SUP/terms/tuned.py` at the end of
   `src/poclegs_lab/tasks/locomotion/config/poclegs/env_cfg.py` (at the Exercise 4.3 TODO)

## Lab 6.1 · USD → URDF, then check it
```bash
uv run python scripts/usd_to_urdf.py $D/poclegs/poclegs_newton.usd $D/urdf
uv run python scripts/check_urdf.py  $D/poclegs/poclegs_newton.usd $D/urdf/poclegs.urdf   # must print "all joints match"
diff $D/urdf/poclegs.urdf $SUP/expected_output/poclegs.urdf && echo OK
uv run python scripts/urdf_to_usd.py $D/urdf/poclegs.urdf $D/usd_from_urdf   # the ×1.0 robot through URDF = the -Flat-Urdf task's default
```

## Lab 6.2 · Legs ×1.3 → USD
```bash
mkdir -p $D/urdf_long && cp -r $D/urdf/meshes $D/urdf_long/
uv run python scripts/scale_leg.py $D/urdf/poclegs.urdf $D/urdf_long/poclegs_long.urdf --thigh 1.3 --shank 1.3
diff $D/urdf_long/poclegs_long.urdf $SUP/expected_output/poclegs_long.urdf && echo OK
uv run python scripts/urdf_to_usd.py $D/urdf_long/poclegs_long.urdf $D/usd_long
uv run isaaclab zero_agent --task Poclegs-Velocity-Flat-Urdf --viz newton \
    env.scene.robot.spawn.usd_path=$PWD/$D/usd_long/poclegs_long/poclegs_long.usda "env.scene.robot.init_state.pos=[0,0,0.539]"
```

## Lab 6.3 · Train the new robot
```bash
uv run isaaclab train --rl_library rsl_rl --task Poclegs-Velocity-Flat-Urdf --num_envs 2048 --max_iterations 1500 --run_name long13 \
    env.scene.robot.spawn.usd_path=$PWD/$D/usd_long/poclegs_long/poclegs_long.usda "env.scene.robot.init_state.pos=[0.0,0.0,0.539]"
# baseline with the original ×1.0 legs (no override needed):
uv run isaaclab train --rl_library rsl_rl --task Poclegs-Velocity-Flat-Urdf --num_envs 2048 --max_iterations 1500 --run_name urdf1x
# no time to train: use $SUP/pretrained/long13/model_1499.pt and $SUP/pretrained/urdf1x/model_1499.pt
```

## Exercise 6.1 – 6.4
The steps are on the slide "Exercise 6.1–6.3 · ขั้นตอนเดียวกันทุกข้อ" (the same steps for every exercise; only the name, the scale_leg options and z change)

Tested on 16 Oct 2026: following this page on a project created from the Week 4 template gives files identical to expected_output.
