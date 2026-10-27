"""Measure how well a trained policy tracks fixed velocity commands (headless).

It reuses the ``isaaclab play`` pipeline (task config + checkpoint loading) and replaces the
playback loop: every env gets the same command, and we report the measured base velocity
and how many times robots fell (``base_contact`` terminations).

Usage (from the project folder):
    uv run python scripts/eval_policy.py --task Poclegs-Velocity-Flat --num_envs 64 \
        --checkpoint latest [Hydra overrides...]
Env vars: EVAL_STEPS (default 500 = 10 s), EVAL_CMDS ("vx,vy,wz;..." default "0.5,0,0;0,0.5,0;0,0,1")
"""

import os
import sys

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402

from isaaclab_rl.entrypoints.backends import play_rsl_rl  # noqa: E402

import poclegs_lab.tasks  # noqa: E402, F401  (registers the Poclegs-* tasks)

STEPS = int(os.environ.get("EVAL_STEPS", 500))
CMDS = [[float(v) for v in c.split(",")] for c in os.environ.get("EVAL_CMDS", "0.5,0,0;0,0.5,0;0,0,1").split(";")]


def _find_env(step):
    for cell in step.__closure__ or ():
        obj = cell.cell_contents
        if hasattr(obj, "unwrapped") and hasattr(obj.unwrapped, "command_manager"):
            return obj.unwrapped
    raise RuntimeError("Could not find the environment in the play loop.")


def evaluate(step, *, dt, args_cli, env_cfg):
    env = _find_env(step)
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("base_velocity")
    cmd.cfg.heading_command = False
    cmd.is_heading_env[:] = False
    cmd.is_standing_env[:] = False
    env.cfg.episode_length_s = 1.0e6
    for command in CMDS:
        target = torch.tensor(command, device=env.device)
        env.reset()
        lin, ang, n, falls = torch.zeros(3, device=env.device), 0.0, 0, 0
        for i in range(STEPS):
            cmd.time_left[:] = 1.0e9
            cmd.vel_command_b[:] = target
            with torch.inference_mode():
                step()
            falls += int(env.termination_manager.get_term("base_contact").sum())
            if i >= 50:  # skip the first second (start-up)
                lin += robot.data.root_lin_vel_b.torch.mean(dim=0)
                ang += robot.data.root_ang_vel_b.torch[:, 2].mean().item()
                n += 1
        v = (lin / n).tolist()
        print(
            f"[EVAL] cmd vx={command[0]:+.2f} vy={command[1]:+.2f} wz={command[2]:+.2f} -> "
            f"vx={v[0]:+.3f} vy={v[1]:+.3f} wz={ang / n:+.3f}  falls={falls} "
            f"({env.num_envs} robots, {STEPS * dt:.0f} s)",
            flush=True,
        )


play_rsl_rl.run_playback = evaluate

if __name__ == "__main__":
    play_rsl_rl.run(sys.argv[1:])
