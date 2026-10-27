"""Drive a trained humanoid (G1 / H1) with the keyboard on Isaac Lab 3 + Newton.

Week 3 workshop script. It reuses the official ``isaaclab play`` pipeline (task config,
checkpoint loading, policy export) and replaces only the playback loop: every step it reads
keys from the Newton viewer window and writes them into the robot's velocity command.

Run from the Isaac Lab checkout (``~/IsaacLab``):

    uv run python ~/Downloads/Simulation_for_Humanoid/Week03_G1_Humanoid/code/play_interactive.py \
        --task Isaac-Velocity-Flat-G1 --checkpoint pretrained --num_envs 1 --viz newton

Keys (click the viewer window first):
    I / K   forward / backward          (v_x)
    J / L   strafe left / right          (v_y)
    U / O   turn left / right            (w_z)
    Space   pause the simulation (viewer built-in)
    C       toggle chase camera
    P       print the robot's position and heading (for the square-path exercise)
    WASD/QE/arrows move the free camera (viewer built-in), so robot control uses IJKL.

The old Isaac Sim version used ``carb.input`` from Omniverse Kit. Newton's viewer is a
pyglet window, so we poll ``viewer.is_key_down()`` instead.

Command sizes can be changed without editing the file, e.g. ``KBD_VX=2.0 uv run python ...``.
G1 Flat was trained with ``lin_vel_x`` in [0, 1] only, so backward (K) is a separate, smaller
step ``KBD_VX_BACK``: at 0.8 m/s backward the pretrained G1 falls within about a second.
"""

import math
import os
import sys

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402

from isaaclab_rl.entrypoints.backends import play_rsl_rl  # noqa: E402

# command step sizes [m/s, m/s, rad/s]
V_X = float(os.environ.get("KBD_VX", 0.8))
V_X_BACK = float(os.environ.get("KBD_VX_BACK", 0.3))  # backward was never trained; larger values fall
V_Y = float(os.environ.get("KBD_VY", 0.4))
W_Z = float(os.environ.get("KBD_WZ", 1.0))


def _find_env(step):
    """Return the unwrapped Isaac Lab env captured by the play loop's ``step`` closure."""
    for cell in step.__closure__ or ():
        obj = cell.cell_contents
        if hasattr(obj, "unwrapped") and hasattr(obj.unwrapped, "command_manager"):
            return obj.unwrapped
    raise RuntimeError("Could not find the environment in the play loop.")


def _find_viewer():
    """Return the Newton GL viewer (needs ``--viz newton``)."""
    from isaaclab.sim import SimulationContext

    for viz in getattr(SimulationContext.instance(), "visualizers", []):
        viewer = getattr(viz, "_viewer", None)
        if viewer is not None and hasattr(viewer, "is_key_down"):
            return viewer
    raise RuntimeError("No Newton viewer window. Add --viz newton to the command.")


def _print_pose(env):
    """Print env 0's root position (relative to its spawn origin) and yaw."""
    robot = env.scene["robot"]
    x, y, _ = (robot.data.root_pos_w.torch[0] - env.scene.env_origins[0]).tolist()
    qx, qy, qz, w = robot.data.root_quat_w.torch[0].tolist()  # Isaac Lab 3 quaternions are xyzw
    yaw = math.degrees(math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
    print(f"[POSE] x={x:+.2f} m  y={y:+.2f} m  yaw={yaw:+.0f} deg", flush=True)


def interactive_playback(step, *, dt, args_cli, env_cfg):
    env = _find_env(step)
    viewer = _find_viewer()
    cmd = env.command_manager.get_term("base_velocity")
    # keyboard owns the command: no heading control, no standing envs, never resample
    cmd.cfg.heading_command = False
    cmd.is_heading_env[:] = False
    cmd.is_standing_env[:] = False
    # training episodes time out after 20 s; a keyboard session should not reset the robot mid-drive
    env.cfg.episode_length_s = 1.0e6

    chase, c_was_down, p_was_down = True, False, False
    print(f"[INFO] command sizes: v_x={V_X} m/s (back {V_X_BACK}), v_y={V_Y} m/s, w_z={W_Z} rad/s")
    print("[INFO] I/K forward-back, J/L strafe, U/O turn, C chase camera, P print pose. Ctrl+C to quit.")
    try:
        while viewer.is_running():
            key = viewer.is_key_down
            command = torch.tensor(
                [
                    V_X * key("i") - V_X_BACK * key("k"),
                    V_Y * (key("j") - key("l")),
                    W_Z * (key("u") - key("o")),
                ],
                device=env.device,
            )
            cmd.time_left[:] = 1.0e9
            cmd.vel_command_b[:] = command

            if key("c") and not c_was_down:
                chase = not chase
            c_was_down = key("c")
            if key("p") and not p_was_down:
                _print_pose(env)
            p_was_down = key("p")
            if chase:
                p = env.scene["robot"].data.root_pos_w[0].tolist()
                viewer.set_camera(wp.vec3(p[0] - 3.0, p[1] - 3.0, 1.8), -15.0, 45.0)

            with torch.inference_mode():
                step()
    except KeyboardInterrupt:
        pass


play_rsl_rl.run_playback = interactive_playback

if __name__ == "__main__":
    play_rsl_rl.run(sys.argv[1:])
