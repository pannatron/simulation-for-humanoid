"""Measure *how* a trained Spot policy walks (headless): speed, falls, foot height and gait pattern.

It reuses the ``isaaclab play`` pipeline (task config + checkpoint loading) and replaces the
playback loop: every robot gets the same command and we record, per foot,
  * contact %   - share of time the foot is on the ground
  * lift (m)    - average highest point of each swing, minus the foot height when on the ground
and, per foot pair, how often both feet are in the same state (on ground / in air):
  trot = diagonal pairs agree, bound = front/rear pairs agree, pace = left/right pairs agree.

Usage (from the project folder):
    uv run python scripts/eval_gait.py --task Spot-Reward-Exp1 --num_envs 64 \
        --checkpoint logs/rsl_rl/spot_flat/<run>/model_299.pt physics=newton_mjwarp
Env vars: EVAL_STEPS (default 500 = 10 s), EVAL_CMDS ("vx,vy,wz;..." default "1,0,0;0,0.5,0;0,0,1"),
          EVAL_FEET (default "fl_foot,fr_foot,hl_foot,hr_foot")
"""

import os
import sys

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402

from isaaclab_rl.entrypoints.backends import play_rsl_rl  # noqa: E402

import spot_reward_lab.tasks  # noqa: E402, F401  (registers the Spot-Reward-* tasks)

STEPS = int(os.environ.get("EVAL_STEPS", 500))
CMDS = [[float(v) for v in c.split(",")] for c in os.environ.get("EVAL_CMDS", "1,0,0;0,0.5,0;0,0,1").split(";")]
# order: front-left, front-right, hind-left, hind-right (Go2: "FL_foot,FR_foot,RL_foot,RR_foot")
FEET = os.environ.get("EVAL_FEET", "fl_foot,fr_foot,hl_foot,hr_foot").split(",")
PAIRS = {"trot": ((0, 3), (1, 2)), "bound": ((0, 1), (2, 3)), "pace": ((0, 2), (1, 3))}


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
    body_ids = robot.find_bodies(FEET, preserve_order=True)[0]
    sensor_ids = sensor.find_sensors(FEET, preserve_order=True)[0]
    cmd = env.command_manager.get_term("base_velocity")
    cmd.cfg.heading_command = False
    cmd.is_heading_env[:] = False
    cmd.is_standing_env[:] = False
    env.cfg.episode_length_s = 1.0e6
    for command in CMDS:
        target = torch.tensor(command, device=env.device)
        env.reset()
        lin, ang, n, falls = torch.zeros(3, device=env.device), 0.0, 0, 0
        contact_sum = torch.zeros(4, device=env.device)
        agree = {k: 0.0 for k in PAIRS}
        apex_now = torch.zeros(env.num_envs, 4, device=env.device)
        apex_sum, apex_cnt = torch.zeros(4, device=env.device), torch.zeros(4, device=env.device)
        stance_z, stance_n = torch.zeros(4, device=env.device), torch.zeros(4, device=env.device)
        z_sum = torch.zeros(4, device=env.device)
        prev = None
        for i in range(STEPS):
            cmd.time_left[:] = 1.0e9
            cmd.vel_command_b[:] = target
            with torch.inference_mode():
                step()
            if i < 50:  # skip the first second (start-up after reset)
                continue
            for name in ("body_contact", "base_contact"):
                if name in env.termination_manager.active_terms:
                    falls += int(env.termination_manager.get_term(name).sum())
            contact = sensor.data.current_contact_time.torch[:, sensor_ids] > 0.0
            foot_z = robot.data.body_pos_w.torch[:, body_ids, 2]
            lin += robot.data.root_lin_vel_b.torch.mean(dim=0)
            ang += robot.data.root_ang_vel_b.torch[:, 2].mean().item()
            contact_sum += contact.float().mean(dim=0)
            stance_z += torch.where(contact, foot_z, 0.0).sum(dim=0)
            stance_n += contact.float().sum(dim=0)
            z_sum += foot_z.mean(dim=0)
            for k, pairs in PAIRS.items():
                agree[k] += sum((contact[:, a] == contact[:, b]).float().mean().item() for a, b in pairs) / 2
            # swing apex: running max of z while in the air, recorded at touch-down
            apex_now = torch.where(contact, apex_now, torch.maximum(apex_now, foot_z))
            if prev is not None:
                touchdown = contact & ~prev
                apex_sum += torch.where(touchdown, apex_now, 0.0).sum(dim=0)
                apex_cnt += touchdown.float().sum(dim=0)
                apex_now = torch.where(touchdown, torch.zeros_like(apex_now), apex_now)
            prev = contact
            n += 1
        v = (lin / n).tolist()
        duty = (100 * contact_sum / n).tolist()
        # lift = swing apex minus the foot height while standing on the ground
        ground = stance_z.sum() / stance_n.sum().clamp(min=1)
        apex = (apex_sum / apex_cnt.clamp(min=1) - ground).tolist()
        print(
            f"[EVAL] cmd vx={command[0]:+.2f} vy={command[1]:+.2f} wz={command[2]:+.2f} -> "
            f"vx={v[0]:+.3f} vy={v[1]:+.3f} wz={ang / n:+.3f}  falls={falls} "
            f"({env.num_envs} robots, {(STEPS - 50) * dt:.0f} s after 1 s warm-up)",
            flush=True,
        )
        print(
            "[EVAL]   contact %  " + "  ".join(f"{f[:2]} {d:5.1f}" for f, d in zip(FEET, duty, strict=True)),
            "| lift m " + "  ".join(f"{f[:2]} {a:.3f}" for f, a in zip(FEET, apex, strict=True)),
            "| steps " + " ".join(f"{int(c)}" for c in apex_cnt.tolist()),
            flush=True,
        )
        print(
            "[EVAL]   mean foot height above ground m "
            + "  ".join(f"{f[:2]} {z:.3f}" for f, z in zip(FEET, (z_sum / n - ground).tolist(), strict=True)),
            flush=True,
        )
        print("[EVAL]   pair agree % " + "  ".join(f"{k} {100 * a / n:5.1f}" for k, a in agree.items()), flush=True)


play_rsl_rl.run_playback = evaluate

if __name__ == "__main__":
    play_rsl_rl.run(sys.argv[1:])
