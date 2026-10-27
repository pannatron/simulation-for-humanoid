"""Challenge 7.5 hazard maze (Week 7) with the RPO: Roboparty's policy.onnx walks, the Week 7 pilot steers.

    uv run python scripts/challenge_rpo.py --seed 7 --viz newton
    uv run python scripts/challenge_rpo.py --seed 7 --pilot my_pilot.py --viz newton

Same timer, HUD, rules and settings (CHALLENGE_TIME, MAZE_CAM, MAZE_REC_DIR) as g1_maze_lab's
scripts/challenge.py, which this script reuses. Nothing is written to the leaderboard.
The pilot's commands are clipped to what the RPO policy was trained for (deploy limits below).
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import warp as wp

wp.config.enable_backward = False

import gymnasium as gym  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

import g1_maze_lab  # noqa: E402
from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

import rpo_lab.tasks  # noqa: E402, F401  (registers RPO-Maze-Hard)
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMD_MIN = (-0.4, -0.4, -0.8)  # vx, vy [m/s], yaw [rad/s]: roboparty_deploy configs/default.yaml clip_cmd
CMD_MAX = (0.6, 0.4, 0.8)


def load_challenge():
    """Import g1_maze_lab/scripts/challenge.py (a script, not part of the package)."""
    path = Path(g1_maze_lab.__file__).resolve().parents[2] / "scripts" / "challenge.py"
    spec = importlib.util.spec_from_file_location("challenge", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--pilot", default="")
    p.add_argument("--onnx", default=os.path.join(PROJECT, "pretrained/roboparty/policy.onnx"))
    add_launcher_args(p)
    args = p.parse_args()
    challenge = load_challenge()
    challenge.RUN.update(team="RPO", seed=args.seed, pilot=args.pilot, policy=args.onnx, score=False)

    sys.argv = sys.argv[:1]  # resolve_task_config reads Hydra overrides from sys.argv
    env_cfg, _ = resolve_task_config("RPO-Maze-Hard", None, play_mode=True)
    env_cfg.scene.terrain.terrain_generator.seed = args.seed
    with launch_simulation(env_cfg, args):
        env = gym.make("RPO-Maze-Hard", cfg=env_cfg)
        cmd = env.unwrapped.command_manager.get_term("base_velocity")
        lo = torch.tensor(CMD_MIN, device=env.unwrapped.device)
        hi = torch.tensor(CMD_MAX, device=env.unwrapped.device)
        sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        obs, _ = env.reset()

        def step():
            nonlocal obs
            cmd.vel_command_b[:] = torch.clamp(cmd.vel_command_b, lo, hi)
            action = sess.run(None, {name: obs["policy"].cpu().numpy()})[0]
            obs, *_ = env.step(torch.from_numpy(action).to(env.unwrapped.device))

        challenge.challenge_playback(step, dt=env.unwrapped.step_dt, args_cli=args, env_cfg=env_cfg)
        env.close()


if __name__ == "__main__":
    main()
