"""Run Roboparty's pretrained RPO walking policy (policy.onnx) on Isaac Lab 3 + Newton.

    uv run python scripts/play_rpo.py --viz newton                      # drive with the keyboard
    uv run python scripts/play_rpo.py --plan "2:0,0,0 6:0.5,0,0"         # scripted commands, no keyboard

Keys (click the viewer window first):
    I / K   forward / backward        J / L   sidestep left / right
    U / O   turn left / right         C       follow camera on / off
WASD / QE / arrows move the free camera (viewer built-in), so the robot uses IJKL like Week 3.

--plan is a list of segments  seconds:vx,vy,yaw  played one after another.
--rec DIR saves every viewer frame as DIR/00000.jpg ... (needs --viz newton).

Experiments:
    --kp-scale 0.5   softer motors (kp and kd of every joint x 0.5)
    --no-remap       give the policy Newton's joint order instead of the order it was trained with
"""

import argparse
import math
import os
import sys

import warp as wp

wp.config.enable_backward = False

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

import rpo_lab.tasks  # noqa: E402, F401  (registers RPO-Velocity-Flat-Play)
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V_X, V_X_BACK, V_Y, W_Z = 0.5, 0.3, 0.3, 0.6  # keyboard command sizes, inside the deploy limits


def find_viewer():
    from isaaclab.sim import SimulationContext

    for viz in getattr(SimulationContext.instance(), "visualizers", []):
        viewer = getattr(viz, "_viewer", None)
        if viewer is not None and hasattr(viewer, "is_key_down"):
            return viewer
    return None


def parse_plan(text):
    plan = [(float(t), [float(v) for v in c.split(",")]) for t, c in (s.split(":") for s in text.split())]
    return plan, np.cumsum([t for t, _ in plan])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="RPO-Velocity-Flat-Play")
    p.add_argument("--onnx", default=os.path.join(PROJECT, "pretrained/roboparty/policy.onnx"))
    p.add_argument("--plan", default="")
    p.add_argument("--rec", default="")
    p.add_argument("--kp-scale", type=float, default=1.0)
    p.add_argument("--no-remap", action="store_true")
    add_launcher_args(p)
    args = p.parse_args()
    plan, ends = parse_plan(args.plan) if args.plan else (None, None)

    sys.argv = sys.argv[:1]  # resolve_task_config reads Hydra overrides from sys.argv
    env_cfg, _ = resolve_task_config(args.task, None, play_mode=True)
    for act in env_cfg.scene.robot.actuators.values():
        for gain in ("stiffness", "damping"):
            v = getattr(act, gain)
            setattr(act, gain, {k: g * args.kp_scale for k, g in v.items()} if isinstance(v, dict) else v * args.kp_scale)
    if args.no_remap:  # Newton order: left leg, right leg, torso, left arm, right arm
        env_cfg.actions.joint_pos.joint_names = [".*"]
        env_cfg.observations.policy.frame.params["asset_cfg"] = SceneEntityCfg("robot")
    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        robot = env.scene["robot"]
        cmd = env.command_manager.get_term("base_velocity")
        cmd.cfg.heading_command = False
        cmd.is_heading_env[:] = False
        cmd.is_standing_env[:] = False

        sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        obs, _ = env.reset()
        viewer = find_viewer()
        follow, c_was_down, t, frame = True, False, 0.0, 0
        if args.rec:
            os.makedirs(args.rec, exist_ok=True)
        print(f"[RPO] obs {tuple(obs['policy'].shape)}  joints {robot.joint_names}", flush=True)
        while viewer is None or viewer.is_running():
            if plan is not None:
                if t >= ends[-1]:
                    break
                command = plan[min(int(np.searchsorted(ends, t, side="right")), len(plan) - 1)][1]
            elif viewer is not None:
                key = viewer.is_key_down
                command = [V_X * key("i") - V_X_BACK * key("k"), V_Y * (key("j") - key("l")), W_Z * (key("u") - key("o"))]
                if key("c") and not c_was_down:
                    follow = not follow
                c_was_down = key("c")
            else:
                command = [0.0, 0.0, 0.0]
            cmd.time_left[:] = 1.0e9
            cmd.vel_command_b[:] = torch.tensor(command, device=env.device)

            x = obs["policy"].cpu().numpy().astype(np.float32)
            action = torch.from_numpy(sess.run(None, {name: x})[0]).to(env.device)
            with torch.inference_mode():
                obs, _, terminated, truncated, _ = env.step(action)
            t += env.step_dt

            pos = robot.data.root_pos_w.torch[0].tolist()
            if viewer is not None and follow:
                viewer.set_camera(wp.vec3(pos[0] - 1.5, pos[1] - 1.5, 1.2), -15.0, 45.0)
            if args.rec and viewer is not None:
                from PIL import Image

                img = np.asarray(viewer.get_frame(render_ui=False).numpy())
                Image.fromarray(img[..., :3]).save(os.path.join(args.rec, f"{frame:05d}.jpg"), quality=90)
                frame += 1
            if int(t / env.step_dt) % 50 == 0:
                print(f"[RPO] t={t:5.1f}s cmd={[round(c, 2) for c in command]} "
                      f"x={pos[0]:5.2f} y={pos[1]:5.2f} height={pos[2]:.2f}", flush=True)
            if bool(terminated[0]):
                print(f"[RPO] fell at t={t:.1f}s", flush=True)
                if plan is not None and args.rec == "":
                    break
        env.close()


if __name__ == "__main__":
    main()
