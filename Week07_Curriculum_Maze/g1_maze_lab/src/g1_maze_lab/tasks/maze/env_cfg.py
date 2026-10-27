# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 maze environments: a pretrained G1 walking policy + a maze terrain + a height scanner for the autopilot.

The walking policy is Isaac Lab's pretrained ``Isaac-Velocity-Rough-G1`` (trained with the terrain-level
curriculum on mesh terrain), so its observations are left untouched: it keeps its own small height scanner.
The maze scanner is an extra, larger height scanner that ``scripts/maze_autopilot.py`` reads to find the
walls and turn them into velocity commands.

``G1-Maze-*-FlatPolicy`` tasks are the same mazes for the Week 3 flat policy (no height scan in its
observations). It walks fine on the infinite plane but falls on the maze's mesh floor (Exercise 7.3).
"""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg

from ...terrains import HazardMazeTerrainCfg, HazardTerrainGeneratorCfg, HazardTerrainImporterCfg, MazeTerrainCfg
from . import mdp

# The fixed challenge maze (5 x 5 cells of 2 m). Start S = bottom-left, goal G = top-right.
# Shortest route: 10 cells. Left-hand wall following: 18 cells. Right-hand: 30 cells.
CHALLENGE_LAYOUT = """
+-+-+-+-+-+
|       |G|
+ + +-+ + +
| | | |   |
+ + + +-+ +
| | |   | |
+ + + + + +
| | | | | |
+-+ + +-+ +
|S  |     |
+-+-+-+-+-+
"""

MAZE_CELLS = 5
CELL_SIZE = 2.0


def maze_terrain_cfg(layout: str | None, num_rows: int, num_cols: int) -> terrain_gen.TerrainGeneratorCfg:
    """Terrain generator with one maze per sub-terrain (``num_rows x num_cols`` mazes)."""
    side = MAZE_CELLS * CELL_SIZE + 1.0
    return terrain_gen.TerrainGeneratorCfg(
        size=(side, side),
        border_width=2.0,
        border_height=0.0,
        num_rows=num_rows,
        num_cols=num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        seed=7,
        sub_terrains={
            "maze": MazeTerrainCfg(
                proportion=1.0, num_cells=MAZE_CELLS, cell_size=CELL_SIZE, wall_height=1.0, layout=layout
            )
        },
    )


def setup_maze(cfg, layout: str | None = CHALLENGE_LAYOUT, num_rows: int = 1, num_cols: int = 1):
    """Turn a G1 velocity env config into a maze env config (shared by the rough- and flat-policy tasks)."""
    # scene: maze terrain instead of the infinite plane
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = maze_terrain_cfg(layout, num_rows, num_cols)
    cfg.scene.terrain.max_init_terrain_level = None
    # no terrain curriculum: it would move a robot to another maze every time it resets
    cfg.curriculum.terrain_levels = None

    # the walking policy keeps its own small height scanner (1.6 m x 1.0 m, from G1 Rough).
    # maze scanner for the autopilot: 3.2 m x 3.2 m grid around the robot, 10 cm spacing (33 x 33 = 1089 rays)
    # the rays start 20 m above the torso and point down; a wall shows up as a 1 m tall bump
    cfg.scene.maze_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[3.2, 3.2]),
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
        update_period=cfg.decimation * cfg.sim.dt,
        debug_vis=False,
    )
    # optional 2D lidar at chest height: 72 horizontal rays, one every 5 degrees (Exercise 7.4)
    cfg.scene.wall_lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1, vertical_fov_range=(0.0, 0.0), horizontal_fov_range=(-180.0, 180.0), horizontal_res=5.0
        ),
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
        max_distance=6.0,
        update_period=cfg.decimation * cfg.sim.dt,
        debug_vis=False,
    )

    # commands: the autopilot writes v_x / v_y / w_z directly
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    cfg.commands.base_velocity.debug_vis = False

    # events: start at the centre of cell (0, 0), facing +x, no pushes
    cfg.events.reset_base.params = {
        "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
        "velocity_range": {},
    }
    cfg.events.push_robot = None
    cfg.events.base_external_force_torque = None

    # terminations: a fall ends the run; reaching the goal cell ends it too
    cfg.episode_length_s = 180.0
    cfg.terminations.goal_reached = DoneTerm(
        func=mdp.maze_goal_reached,
        params={"num_cells": MAZE_CELLS, "cell_size": CELL_SIZE, "threshold": 0.6},
    )
    # a wall touching the arms or the pelvis also counts as a crash
    cfg.terminations.body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["torso_link", "pelvis", ".*_elbow_.*", ".*_palm_link"]
            ),
            "threshold": 1.0,
        },
    )
    cfg.terminations.base_contact = None


def play_mode_keep_maze(cfg, parent_play_mode):
    """Run the shared play mode but keep the maze grid (it would turn the terrain into 5 x 5 tiles)."""
    gen = cfg.scene.terrain.terrain_generator
    rows, cols, curriculum = gen.num_rows, gen.num_cols, gen.curriculum
    parent_play_mode()
    gen.num_rows, gen.num_cols, gen.curriculum = rows, cols, curriculum


@configclass
class G1MazeEnvCfg(G1RoughEnvCfg):
    """G1 in a maze. Every environment gets its own maze; the robot starts in cell (0, 0) facing +x."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        setup_maze(self)

    def play_mode(self):
        play_mode_keep_maze(self, super().play_mode)


@configclass
class G1MazeFlatPolicyEnvCfg(G1FlatEnvCfg):
    """Same maze for the Week 3 flat policy (Exercise 7.3)."""

    def __post_init__(self):
        super().__post_init__()
        # the flat config shrinks the contact budget for the analytic plane; the maze is a mesh
        self.sim.physics.default.solver_cfg.njmax = 300
        self.sim.physics.default.solver_cfg.nconmax = 300
        setup_maze(self)

    def play_mode(self):
        play_mode_keep_maze(self, super().play_mode)


@configclass
class G1MazeRandomEnvCfg(G1MazeEnvCfg):
    """16 robots in a 4 x 4 grid of random mazes. Rows go from open (row 0) to perfect mazes (last row)."""

    def __post_init__(self):
        super().__post_init__()
        setup_maze(self, None, num_rows=4, num_cols=4)
        self.scene.num_envs = 16
        self.scene.terrain.terrain_generator.curriculum = True  # difficulty grows with the row index


# No walls at all: only the maze floor (a mesh). Used to test the walking policies on that floor.
OPEN_LAYOUT = "\n".join("+ + + + + +" if k % 2 == 0 else "+         +" for k in range(11))


@configclass
class G1MazeOpenEnvCfg(G1MazeEnvCfg):
    """Maze floor without walls (rough policy)."""

    def __post_init__(self):
        super().__post_init__()
        setup_maze(self, OPEN_LAYOUT)


@configclass
class G1MazeOpenFlatPolicyEnvCfg(G1MazeFlatPolicyEnvCfg):
    """Maze floor without walls (flat policy)."""

    def __post_init__(self):
        super().__post_init__()
        setup_maze(self, OPEN_LAYOUT)


# Challenge levels: the same 5 x 5 maze, fewer shortcuts at every level. A new seed = new mazes.
LEVEL_OPEN_RATIOS = (0.3, 0.2, 0.1, 0.0)  # share of inner walls knocked down (0.0 = perfect maze: one route)


@configclass
class G1MazeLevelsEnvCfg(G1MazeEnvCfg):
    """4 robots, one per level, left (easy) to right (perfect maze). Change the mazes with
    ``env.scene.terrain.terrain_generator.seed=<n>``."""

    def __post_init__(self):
        super().__post_init__()
        setup_maze(self, None, num_rows=1, num_cols=len(LEVEL_OPEN_RATIOS))
        gen = self.scene.terrain.terrain_generator
        template = gen.sub_terrains["maze"]
        gen.sub_terrains = {
            f"level{k + 1}": template.replace(proportion=1.0, open_ratio_range=(r, r))
            for k, r in enumerate(LEVEL_OPEN_RATIOS)
        }
        gen.curriculum = True  # column k = level k+1 (sub-terrains are laid out by proportion, in order)
        gen.seed = 1681  # example mazes for the slides: shortest route 8 / 12 / 16 / 20 cells
        self.scene.num_envs = len(LEVEL_OPEN_RATIOS)


# ---------------------------------------------------------------- Challenge 7.5: hazard maze
HARD_CELLS = 7  # 7 x 7 cells of 2 m, perfect maze (one route), gaps / ice / rough cells


def setup_hazard_maze(cfg, num_rows: int = 1, num_cols: int = 1, **hazards):
    """Turn a maze env config into the hazard maze of the challenge (see ``terrains/hazard_maze.py``)."""
    setup_maze(cfg, None, num_rows, num_cols)
    old = cfg.scene.terrain
    gen = old.terrain_generator
    side = HARD_CELLS * CELL_SIZE + 1.0
    cfg.scene.terrain = HazardTerrainImporterCfg(
        prim_path=old.prim_path,
        terrain_type="generator",
        terrain_generator=HazardTerrainGeneratorCfg(
            size=(side, side),
            border_width=gen.border_width,
            border_height=gen.border_height,
            num_rows=num_rows,
            num_cols=num_cols,
            horizontal_scale=gen.horizontal_scale,
            vertical_scale=gen.vertical_scale,
            slope_threshold=gen.slope_threshold,
            use_cache=False,
            seed=2026,
            sub_terrains={
                "hazard_maze": HazardMazeTerrainCfg(
                    proportion=1.0, num_cells=HARD_CELLS, cell_size=CELL_SIZE, wall_height=1.0, **hazards
                )
            },
        ),
        max_init_terrain_level=None,
        collision_group=old.collision_group,
        physics_material=old.physics_material,
        visual_material=old.visual_material,
        debug_vis=False,
    )
    cfg.terminations.goal_reached.params["num_cells"] = HARD_CELLS
    cfg.episode_length_s = 300.0


@configclass
class G1MazeHardEnvCfg(G1MazeEnvCfg):
    """Challenge 7.5: one robot, one random 7 x 7 hazard maze. New maze: ``env.scene.terrain.terrain_generator.seed=<n>``."""

    def __post_init__(self):
        super().__post_init__()
        setup_hazard_maze(self)
        self.scene.num_envs = 1
