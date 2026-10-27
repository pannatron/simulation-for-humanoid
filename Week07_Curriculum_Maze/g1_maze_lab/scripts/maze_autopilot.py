"""Walk the G1 through the maze by itself: height scanner -> walls -> next cell -> velocity command.

It reuses the ``isaaclab play`` pipeline (task config + checkpoint loading) and replaces only the playback
loop, like ``play_interactive.py`` in Week 3 -- but the velocity command now comes from
``g1_maze_lab.autopilot.MazeAutopilot`` instead of the keyboard.

Usage (from the project folder):
    uv run python scripts/maze_autopilot.py --task G1-Maze-Challenge --num_envs 1 \
        --checkpoint pretrained/g1_flat/model_pretrained.pt --viz newton

Settings (environment variables):
    MAZE_STRATEGY  left | right | flood          (default left)
    MAZE_SENSOR    height | lidar                (default height)
    MAZE_SPEED     forward speed [m/s]           (default 0.6)
    MAZE_TIME      give up after this many s     (default 150)
    MAZE_ACC       "v,w" command rate limits [m/s^2, rad/s^2] (default "0.8,2.0"; "1e9,1e9" = off)
    MAZE_SHOW_SCAN 1 = draw the height-scan hits in the viewer
    MAZE_CAM       chase | top (robot 0) | grid (every maze) -- camera of the viewer window
    MAZE_ONE_PER_TILE 1 = put robot i in maze tile i (use --num_envs = number of mazes)
    MAZE_SNAP_AT   "5,20,40" = save viewer frames at these times [s] into MAZE_SNAP_DIR (default snapshots/)
    MAZE_REC_DIR   folder: save a 1280x720 frame every MAZE_REC_EVERY steps (default 2 = 25 fps) for a video
"""

import contextlib
import math
import os
import sys
import time

# Load imgui_bundle (the viewer's UI, used for the timer) before Isaac Lab: another GLFW library loaded later
# by the simulator stack would otherwise hide the symbols it needs ("undefined symbol: glfwGetX11Window").
with contextlib.suppress(ImportError):
    import imgui_bundle  # noqa: F401

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402

from isaaclab_rl.entrypoints.backends import play_rsl_rl  # noqa: E402

import g1_maze_lab.tasks  # noqa: E402, F401  (registers the G1-Maze-* tasks)
from g1_maze_lab.autopilot import MazeAutopilot  # noqa: E402
from g1_maze_lab.tasks.maze.env_cfg import CELL_SIZE, MAZE_CELLS  # noqa: E402

STRATEGY = os.environ.get("MAZE_STRATEGY", "left")
SENSOR = os.environ.get("MAZE_SENSOR", "height")
SPEED = float(os.environ.get("MAZE_SPEED", 0.6))
TIME_LIMIT = float(os.environ.get("MAZE_TIME", 150))
SHOW_SCAN = os.environ.get("MAZE_SHOW_SCAN", "0") == "1"
CAMERA = os.environ.get("MAZE_CAM", "chase")  # chase | top
SNAP_AT = os.environ.get("MAZE_SNAP_AT", "")  # e.g. "5,20,40": save viewer frames at these times [s]
SNAP_DIR = os.environ.get("MAZE_SNAP_DIR", "snapshots")
REC_DIR = os.environ.get("MAZE_REC_DIR", "")  # record a frame sequence (for a video) into this folder
REC_EVERY = int(os.environ.get("MAZE_REC_EVERY", 2))  # every N policy steps (2 = 25 fps real time)
REC_LABEL = os.environ.get("MAZE_REC_LABEL", "")  # text in the overlay, e.g. "iteration 200"
REC_HOLD = float(os.environ.get("MAZE_REC_HOLD", 0.0))  # keep recording this many s after the goal (robot respawns!)


def _find_env(step):
    for cell in step.__closure__ or ():
        obj = cell.cell_contents
        if hasattr(obj, "unwrapped") and hasattr(obj.unwrapped, "command_manager"):
            return obj.unwrapped
    raise RuntimeError("Could not find the environment in the play loop.")


def _find_viewer():
    from isaaclab.sim import SimulationContext

    for viz in getattr(SimulationContext.instance(), "visualizers", []):
        viewer = getattr(viz, "_viewer", None)
        if viewer is not None and hasattr(viewer, "is_key_down"):
            return viewer
    return None


def _yaw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quat_xyzw.unbind(-1)  # Isaac Lab 3 quaternions are xyzw
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# live HUD in the viewer window (drawn by _hud every frame from these values)
HUD = {"label": "", "t": 0.0, "extra": "", "crashes": 0, "goal": None, "flash": False}
_CAM = {"yaw": None, "follow": True, "c_down": False}


def _hud(imgui):
    """Newton viewer 'free' UI callback: a big timer at the top centre of the window."""
    io = imgui.get_io()
    # top, to the left of the stats panel (the model panel sits on the left)
    imgui.set_next_window_pos(
        imgui.ImVec2(io.display_size.x - 0.18 * io.display_size.x, 12.0), imgui.Cond_.always, imgui.ImVec2(1.0, 0.0)
    )
    imgui.set_next_window_bg_alpha(0.8)
    flags = (
        imgui.WindowFlags_.no_decoration
        | imgui.WindowFlags_.always_auto_resize
        | imgui.WindowFlags_.no_move
        | imgui.WindowFlags_.no_saved_settings
        | imgui.WindowFlags_.no_focus_on_appearing
        | imgui.WindowFlags_.no_nav
    )
    imgui.begin("maze timer", None, flags)
    imgui.push_font(None, imgui.get_font_size())  # imgui >= 1.92: (font, size); about 2x the panel text (DPI)
    green = imgui.ImVec4(0.48, 0.79, 0.26, 1.0)
    if HUD["goal"] is not None:
        imgui.text_colored(green, HUD["goal"])
    else:
        imgui.text(f"TIME {HUD['t']:5.1f} s")
    imgui.pop_font()
    if HUD["label"]:
        imgui.text_colored(green, HUD["label"])
    imgui.text(f"{HUD['extra']}   falls/crashes {HUD['crashes']}")
    if HUD["flash"]:
        imgui.text_colored(imgui.ImVec4(0.95, 0.3, 0.3, 1.0), "FELL / CRASH -> restart")
    imgui.end()


def _set_camera(viewer, robot, origins):
    if CAMERA == "chase":  # behind robot 0, turning with it (C toggles a free camera)
        c_down = viewer.is_key_down("c")
        if c_down and not _CAM["c_down"]:
            _CAM["follow"] = not _CAM["follow"]
        _CAM["c_down"] = c_down
        if not _CAM["follow"]:
            return
        p = robot.data.root_pos_w.torch[0].tolist()
        x, y, z, w = robot.data.root_quat_w.torch[0].tolist()
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        if _CAM["yaw"] is None:
            _CAM["yaw"] = yaw
        # low-pass the heading so the camera does not shake with every step
        _CAM["yaw"] += 0.05 * math.atan2(math.sin(yaw - _CAM["yaw"]), math.cos(yaw - _CAM["yaw"]))
        cy = _CAM["yaw"]
        eye = wp.vec3(p[0] - 3.2 * math.cos(cy), p[1] - 3.2 * math.sin(cy), 2.6)
        viewer.set_camera(eye, -28.0, math.degrees(cy))
    elif CAMERA == "grid":  # every maze from above
        c = (origins[:, :2].mean(dim=0) + 0.5 * (MAZE_CELLS - 1) * CELL_SIZE).tolist()
        ext = origins[:, :2].max(dim=0).values - origins[:, :2].min(dim=0).values
        if float(ext[1]) > 2.0 * float(ext[0]):  # a row of mazes along y: look along -x so it fills the frame
            h = 0.52 * float(ext[1]) + 6.0
            viewer.set_camera(wp.vec3(c[0] + 0.18 * h, c[1], h), -80.0, 180.0)
        else:
            h = 0.95 * (float(ext.max()) + 10.0)
            viewer.set_camera(wp.vec3(c[0], c[1] - 0.18 * h, h), -80.0, 90.0)
    elif CAMERA == "top":  # whole maze of robot 0 from above
        c = (origins[0, :2] + 0.5 * (MAZE_CELLS - 1) * CELL_SIZE).tolist()
        viewer.set_camera(wp.vec3(c[0], c[1] - 3.2, 12.5), -75.0, 90.0)
    else:  # "diag": follow robot 0 from behind-left, fixed direction
        p = robot.data.root_pos_w.torch[0].tolist()
        viewer.set_camera(wp.vec3(p[0] - 2.5, p[1] - 2.5, 3.2), -40.0, 45.0)


def _save_frame(viewer, k, dt):
    import numpy as np
    from PIL import Image

    os.makedirs(SNAP_DIR, exist_ok=True)
    img = np.asarray(viewer.get_frame(render_ui=os.environ.get("MAZE_SNAP_UI") == "1").numpy())
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    path = os.path.join(SNAP_DIR, f"maze_{CAMERA}_{k * dt:05.1f}s.png")
    Image.fromarray(img[..., :3]).save(path)
    print(f"[MAZE] saved {path}", flush=True)


def _goal_markers(env):
    """Green pads on the goal cells and orange ones on the start cells (viewer only)."""
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/MazeGoal",
        markers={
            name: sim_utils.CuboidCfg(
                size=(1.2, 1.2, 0.02), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)
            )
            for name, color in (("goal", (0.48, 0.79, 0.26)), ("start", (0.95, 0.65, 0.35)))
        },
    )
    markers = VisualizationMarkers(cfg)
    origins = env.scene.env_origins
    goal = origins.clone()
    goal[:, :2] += (MAZE_CELLS - 1) * CELL_SIZE
    pos = torch.cat((goal, origins)) + torch.tensor((0.0, 0.0, 0.01), device=env.device)
    idx = torch.cat((torch.zeros(len(origins)), torch.ones(len(origins)))).long().to(env.device)
    markers.visualize(translations=pos, marker_indices=idx)
    return markers


def _robot_markers():
    """Orange balls above the robots' heads, so they stand out from the white floor in wide shots."""
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/MazeRobot",
        markers={
            "ball": sim_utils.SphereCfg(
                radius=0.3, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.45, 0.1))
            )
        },
    )
    return VisualizationMarkers(cfg)


def _record_frame(viewer, index, info):
    """Save one 1280x720 frame with a timer overlay (robot 0): label, time, cells, falls/crashes, GOAL."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(REC_DIR, exist_ok=True)
    img = np.asarray(viewer.get_frame(render_ui=False).numpy())
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(img[..., :3]).resize((1280, 720))
    d = ImageDraw.Draw(im, "RGBA")
    try:
        big = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
        small = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        big = small = ImageFont.load_default()
    d.rounded_rectangle((16, 16, 380, 150 if REC_LABEL else 112), 14, fill=(14, 22, 33, 210))
    y = 26
    if REC_LABEL:
        d.text((32, y), REC_LABEL, font=small, fill=(122, 201, 67))
        y += 36
    d.text((32, y), f"{info['t']:5.1f} s", font=big, fill=(244, 243, 238))
    d.text((200, y + 8), info["extra"], font=small, fill=(184, 196, 208))
    d.text(
        (32, y + 46),
        f"falls/crashes {info['crashes']}",
        font=small,
        fill=(242, 166, 90) if info["crashes"] else (184, 196, 208),
    )
    if info["flash"]:
        d.rounded_rectangle((440, 16, 840, 80), 14, fill=(184, 40, 40, 220))
        d.text((462, 26), "FELL / CRASH -> restart", font=small, fill=(255, 255, 255))
    if info["goal"] is not None:
        d.rounded_rectangle((440, 16, 840, 80), 14, fill=(61, 122, 18, 230))
        d.text((462, 24), info["goal"], font=big, fill=(255, 255, 255))
    im.save(os.path.join(REC_DIR, f"{index:05d}.jpg"), quality=90)


def autopilot_playback(step, *, dt, args_cli, env_cfg):
    env = _find_env(step)
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("base_velocity")
    cmd.cfg.heading_command = False
    cmd.is_heading_env[:] = False
    cmd.is_standing_env[:] = False
    scanner = env.scene["maze_scanner"]
    lidar = env.scene["wall_lidar"]
    if SHOW_SCAN:
        scanner.set_debug_vis(True)

    n = env.num_envs
    if os.environ.get("MAZE_ONE_PER_TILE") == "1":  # robot i -> maze tile i (no two robots share a maze)
        terrain = env.scene.terrain
        rows, cols = terrain.terrain_origins.shape[:2]
        ids = torch.arange(n, device=env.device) % (rows * cols)
        terrain.terrain_levels[:] = ids // cols
        terrain.terrain_types[:] = ids % cols
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        env.reset()
        if os.environ.get("MAZE_LAYOUTS") == "1":  # print each random maze (generation order = tile order)
            from g1_maze_lab.terrains import GENERATED, maze_stats, maze_to_ascii

            for k, m in enumerate(GENERATED):
                st = maze_stats(m["walls"])
                print(
                    f"[MAZE] tile {k}: open_ratio {m['open_ratio']:.2f}  shortest {st['shortest']} cells  "
                    f"dead ends {st['dead_ends']}\n" + maze_to_ascii(m["walls"]),
                    flush=True,
                )
        if os.environ.get("MAZE_DEBUG") == "1":
            print("[MAZE] tile origins", terrain.terrain_origins.reshape(-1, 3).tolist(), flush=True)
            print("[MAZE] env origins", env.scene.env_origins.tolist(), flush=True)
    pilot = MazeAutopilot(n, MAZE_CELLS, CELL_SIZE, env.device, strategy=STRATEGY, speed=SPEED)
    pilot.debug = os.environ.get("MAZE_DEBUG") == "1"
    if "MAZE_ACC" in os.environ:  # "v,w" rate limits [m/s^2, rad/s^2]; "1e9,1e9" = no smoothing
        pilot.max_acc, pilot.max_ang_acc = (float(v) for v in os.environ["MAZE_ACC"].split(","))
    done = torch.zeros(n, dtype=torch.bool, device=env.device)
    finish_time = torch.full((n,), float("nan"), device=env.device)
    crashes = torch.zeros(n, dtype=torch.long, device=env.device)
    cells_at_finish = torch.zeros(n, dtype=torch.long, device=env.device)
    viewer = _find_viewer()
    if viewer is not None:
        goal_markers = _goal_markers(env)  # noqa: F841  (keep a reference while the viewer runs)
        robot_markers = _robot_markers() if CAMERA == "grid" else None
        HUD["label"] = REC_LABEL or f"{STRATEGY} / {SENSOR}"
        viewer.register_ui_callback(_hud, position="free")
        # watching: robots that reach the goal stay there (no reset), the window stays open
        if "goal_reached" in env.termination_manager.active_terms:
            env.termination_manager.get_term_cfg("goal_reached").func = lambda env, **kw: torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )
    snap_steps = {int(round(float(t) / dt)) for t in SNAP_AT.split(",")} if SNAP_AT else set()

    print(f"[MAZE] strategy={STRATEGY} sensor={SENSOR} speed={SPEED} m/s robots={n}", flush=True)
    steps = int(TIME_LIMIT / dt)
    last_crash_t, hold_until = -10.0, None
    wall_t0 = time.time()
    for k in range(steps):
        origins = env.scene.env_origins
        pos_xy = robot.data.root_pos_w.torch[:, :2] - origins[:, :2]
        yaw = _yaw(robot.data.root_quat_w.torch)
        cell = pilot.cell_of(pos_xy)

        # 1. perception: which sides of the current cell are walls?
        if SENSOR == "height":
            hits = torch.nan_to_num(scanner.data.ray_hits_w.torch, nan=0.0, posinf=0.0, neginf=0.0)
            walls_now = pilot.walls_from_height_scan(
                hits[..., :2] - origins[:, None, :2], hits[..., 2] - origins[:, None, 2], cell
            )
        else:
            # ray_alignment="yaw" rotates only the ray start points (it was made for height scanners),
            # so we take each ray's world angle from its hit point; rays that hit nothing are ignored
            rel = lidar.data.ray_hits_w.torch[..., :2] - lidar.data.pos_w.torch[:, None, :2]
            dist = torch.nan_to_num(torch.linalg.norm(rel, dim=-1), nan=1e6, posinf=1e6)
            angles_w = torch.atan2(rel[..., 1], rel[..., 0])
            walls_now = pilot.walls_from_lidar(dist, angles_w, pos_xy - pilot.centre_of(cell))

        # 2 + 3. plan and turn the plan into a velocity command
        command = pilot.step(pos_xy, yaw, walls_now, dt)
        command[done] = 0.0
        cmd.time_left[:] = 1.0e9
        cmd.vel_command_b[:] = command

        if viewer is not None:
            if not viewer.is_running():
                break
            _set_camera(viewer, robot, origins)
            if robot_markers is not None:
                robot_markers.visualize(
                    translations=robot.data.root_pos_w.torch + torch.tensor((0.0, 0.0, 0.9), device=env.device)
                )

        with torch.inference_mode():
            step()
        if viewer is not None and k + 1 in snap_steps:
            _save_frame(viewer, k + 1, dt)
        if viewer is not None:
            if n == 1:
                goal0 = None if torch.isnan(finish_time[0]) else float(finish_time[0])
                info = {
                    "t": (k + 1) * dt if goal0 is None else goal0,
                    "extra": f"cells {int(cells_at_finish[0] if done[0] else pilot.cells_walked[0])}",
                    "crashes": int(crashes[0]),
                    "goal": None if goal0 is None else f"GOAL  {goal0:.1f} s",
                    "flash": (k + 1) * dt - last_crash_t < 1.0,
                }
            else:
                ok = int(done.sum())
                info = {
                    "t": (k + 1) * dt,
                    "extra": f"goal {ok}/{n}",
                    "crashes": int(crashes.sum()),
                    "goal": f"ALL {n} GOAL" if ok == n else None,
                    "flash": False,
                }
            HUD.update(info)
            if REC_DIR and k % REC_EVERY == 0:
                _record_frame(viewer, k // REC_EVERY, info)

        # bookkeeping: goal reached / crashed into a wall or fell (the env resets those robots)
        if "goal_reached" in env.termination_manager.active_terms and viewer is None:
            reached = env.termination_manager.get_term("goal_reached") & ~done
        else:  # same 0.6 m test, but the robot is not reset: it stays standing at the goal
            rel = robot.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
            reached = (torch.linalg.norm(rel - (MAZE_CELLS - 1) * CELL_SIZE, dim=1) < 0.6) & ~done
        crashed = env.termination_manager.get_term("body_contact") & ~done
        if reached.any():
            finish_time[reached] = (k + 1) * dt
            cells_at_finish[reached] = pilot.cells_walked[reached] + 1
            done |= reached
            for i in torch.nonzero(reached).squeeze(1).tolist():
                print(f"[MAZE] robot {i}: GOAL at {(k + 1) * dt:.1f} s after {cells_at_finish[i]} cells", flush=True)
        if crashed[0]:
            last_crash_t = (k + 1) * dt
        if crashed.any():
            crashes += crashed.long()
            ids = torch.nonzero(crashed).squeeze(1)
            pilot.reset(ids)
            print(f"[MAZE] t={(k + 1) * dt:.1f}s crash/fall: robots {ids.tolist()} (back to start)", flush=True)
        if done.all():
            if viewer is not None and not REC_DIR:
                continue  # keep the window open on the finished maze until it is closed
            if not REC_DIR:
                break
            hold_until = hold_until or k + int(REC_HOLD / dt)
            if k >= hold_until:
                break
        if k % int(10.0 / dt) == 0:
            c0 = pilot.cell_of(pos_xy)[0].tolist()
            print(f"[MAZE] t={k * dt:5.1f}s robot0 cell={c0} cells walked={pilot.cells_walked[0].item()}", flush=True)

    ok = int(done.sum())
    times = finish_time[done]
    print("\n[MAZE] map seen by robot 0 ('?' = never sensed):\n" + pilot.map_ascii(0), flush=True)
    print(
        f"[MAZE] RESULT strategy={STRATEGY} sensor={SENSOR}: {ok}/{n} reached the goal"
        + (f", time mean {times.mean():.1f} s (min {times.min():.1f}, max {times.max():.1f})" if ok else "")
        + (f", cells mean {cells_at_finish[done].float().mean():.1f}" if ok else "")
        + f", crashes/falls {int(crashes.sum())}  (wall clock {time.time() - wall_t0:.0f} s)",
        flush=True,
    )


play_rsl_rl.run_playback = autopilot_playback

if __name__ == "__main__":
    play_rsl_rl.run(sys.argv[1:])
