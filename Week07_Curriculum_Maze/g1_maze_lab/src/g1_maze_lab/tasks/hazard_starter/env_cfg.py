# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Challenge 7.5 starter task ``G1-Hazard-Starter``: G1 Rough on ground that looks a bit like the hazard maze.

It only uses Isaac Lab's built-in sub-terrains, so it is a starting point, not the answer:
    gaps    -> a trench around the spawn platform, wider on harder terrain levels
    slopes  -> pyramid slopes (ramps), steeper on harder levels
    boxes   -> loose-box ground (rough)
    rough   -> random bumps
    friction-> the robot's feet get a random friction from 0.2 (ice) to 1.0
The observation and action spaces are G1 Rough's (do not change them: the upload page checks the network).

Train (fine-tune from G1 Rough, ~30 min for 1500 iterations on an RTX 3060 Laptop):
    uv run isaaclab train --rl_library rsl_rl --task G1-Hazard-Starter --num_envs 1024 \
        --checkpoint pretrained/g1_rough/model_pretrained.pt agent.max_iterations=1500
Try it in the maze:
    uv run python scripts/challenge.py --policy logs/rsl_rl/g1_hazard_starter/<run>/model_XXXX.pt --seed 7 --viz newton

Ideas (TODO): shapes closer to the maze (the ramp is 0.75 m up, 0.5 m flat, 0.75 m down; trenches cross the whole
corridor; spike beds), a reward for lifting the feet over a trench, your own curriculum term (Lab 7.1).
"""

import torch

import isaaclab.terrains as terrain_gen
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1RoughPPORunnerCfg
from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg

def physics_blowup(env: ManagerBasedRLEnv, max_speed: float = 20.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Robots whose state is NaN / inf or flying off (> max_speed [m/s]): reset them before the observation is computed.

    On steep slopes and trench edges the solver now and then blows up one robot; without this term its NaN
    observation stops training ("observation group 'policy' ... contains NaN values").
    """
    asset: Articulation = env.scene[asset_cfg.name]
    root = torch.cat((asset.data.root_pos_w.torch, asset.data.root_lin_vel_w.torch), dim=1)
    joints = torch.cat((asset.data.joint_pos.torch, asset.data.joint_vel.torch), dim=1)
    bad = ~torch.isfinite(root).all(dim=1) | ~torch.isfinite(joints).all(dim=1)
    return bad | (torch.linalg.norm(torch.nan_to_num(asset.data.root_lin_vel_w.torch, nan=0.0), dim=1) > max_speed)


STARTER_TERRAIN = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=10,  # terrain levels: the curriculum (terrain_levels_vel) moves robots to harder rows
    num_cols=12,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # difficulty 0 -> 1 goes from the first value of each range to the second
        "gaps": terrain_gen.MeshGapTerrainCfg(proportion=0.3, gap_width_range=(0.05, 0.30), platform_width=3.0),
        "slopes": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.05, 0.45), platform_width=2.0, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.25, grid_width=0.45, grid_height_range=(0.02, 0.10), platform_width=2.0
        ),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.08), noise_step=0.02, border_width=0.25
        ),
    },
)


@configclass
class G1HazardStarterEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1024
        self.scene.terrain.terrain_generator = STARTER_TERRAIN
        self.scene.terrain.max_init_terrain_level = 3
        # ice: random foot friction per robot (the maze's ice is 0.2, normal floor 1.0)
        self.events.physics_material.params["static_friction_range"] = (0.2, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.2, 1.0)
        # the challenge pilots send v_x 0.15-0.6 m/s: train around those speeds
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.8)
        # keep training alive when the physics blows up for one robot (see physics_blowup)
        self.terminations.physics_blowup = DoneTerm(func=physics_blowup)


@configclass
class G1HazardStarterPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_hazard_starter"
