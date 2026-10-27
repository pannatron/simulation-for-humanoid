"""Measure a G1 walking policy (headless): speed tracking, falls, step length and push recovery.

It reuses the ``isaaclab play`` pipeline (task config + checkpoint loading) and replaces the playback
loop: every robot gets the same command; optionally every robot is shoved sideways at a fixed interval.

Usage (from the project folder):
    uv run python scripts/eval_g1.py --task G1-Curriculum-Base --num_envs 64 \
        --checkpoint logs/rsl_rl/g1_curriculum/<run>/model_299.pt
Env vars:
    EVAL_STEPS       steps per command (default 500 = 10 s, the first 1 s is not measured)
    EVAL_CMDS        "vx,vy,wz;..." (default "1,0,0;1.5,0,0;2,0,0")
    EVAL_PUSH        push speed [m/s] added to the base in a random horizontal direction (default 0 = off)
    EVAL_PUSH_EVERY  seconds between pushes (default 2)
"""

import math
import os
import sys

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402

from isaaclab_rl.entrypoints.backends import play_rsl_rl  # noqa: E402

import g1_maze_lab.tasks  # noqa: E402, F401  (registers the G1-* tasks)

STEPS = int(os.environ.get("EVAL_STEPS", 500))
CMDS = [[float(v) for v in c.split(",")] for c in os.environ.get("EVAL_CMDS", "1,0,0;1.5,0,0;2,0,0").split(";")]
PUSH = float(os.environ.get("EVAL_PUSH", 0.0))
PUSH_EVERY = float(os.environ.get("EVAL_PUSH_EVERY", 2.0))
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]


def _find_env(step):
    for cell in step.__closure__ or ():
        obj = cell.cell_contents
        if hasattr(obj, "unwrapped") and hasattr(obj.unwrapped, "command_manager"):
            return obj.unwrapped
    raise RuntimeError("Could not find the environment in the play loop.")


def evaluate(step, *, dt, args_cli, env_cfg):
    env = _find_env(step)
    robot = env.scene["robot"]
    sensor = env.scene.sensors["contact_forces"]
    feet = sensor.find_sensors(FEET, preserve_order=True)[0]
    cmd = env.command_manager.get_term("base_velocity")
    cmd.cfg.heading_command = False
    cmd.is_heading_env[:] = False
    cmd.is_standing_env[:] = False
    env.cfg.episode_length_s = 1.0e6
    fall_term = "base_contact" if "base_contact" in env.termination_manager.active_terms else "body_contact"
    push_every = max(int(PUSH_EVERY / dt), 1)
    for command in CMDS:
        target = torch.tensor(command, device=env.device)
        env.reset()
        lin, ang, n, falls = torch.zeros(3, device=env.device), 0.0, 0, 0
        air_sum, air_n = 0.0, 0
        for i in range(STEPS):
            cmd.time_left[:] = 1.0e9
            cmd.vel_command_b[:] = target
            if PUSH > 0 and i > 0 and i % push_every == 0:
                # shove every robot: add PUSH m/s to the base velocity in a random horizontal direction
                theta = torch.rand(env.num_envs, device=env.device) * 2 * math.pi
                vel = robot.data.root_vel_w.torch.clone()
                vel[:, 0] += PUSH * torch.cos(theta)
                vel[:, 1] += PUSH * torch.sin(theta)
                robot.write_root_velocity_to_sim_index(root_velocity=vel)
            with torch.inference_mode():
                step()
            falls += int(env.termination_manager.get_term(fall_term).sum())
            if i >= 50:  # skip the first second (start-up)
                lin += robot.data.root_lin_vel_b.torch.mean(dim=0)
                ang += robot.data.root_ang_vel_b.torch[:, 2].mean().item()
                n += 1
                # air time of each step, read at touch-down
                touch = sensor.compute_first_contact(dt).torch[:, feet] > 0
                air = sensor.data.last_air_time.torch[:, feet]
                air_sum += float(air[touch].sum())
                air_n += int(touch.sum())
        v = (lin / n).tolist()
        print(
            f"[EVAL] cmd vx={command[0]:+.2f} vy={command[1]:+.2f} wz={command[2]:+.2f} -> "
            f"vx={v[0]:+.3f} vy={v[1]:+.3f} wz={ang / n:+.3f}  air={air_sum / max(air_n, 1):.3f}s  "
            f"falls={falls} ({env.num_envs} robots, {STEPS * dt:.0f} s"
            + (f", push {PUSH} m/s every {PUSH_EVERY} s)" if PUSH > 0 else ")"),
            flush=True,
        )


play_rsl_rl.run_playback = evaluate

if __name__ == "__main__":
    play_rsl_rl.run(sys.argv[1:])
