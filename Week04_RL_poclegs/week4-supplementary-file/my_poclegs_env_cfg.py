# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Poclegs velocity-tracking environments - Week 4 skeleton.

Copy this file to src/poclegs_lab/tasks/locomotion/config/poclegs/env_cfg.py, then fill the TODOs
with the classes in terms/ (same flow as the original course: terms/observation.py, terms/rewards.py).
"""

import math

import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

from poclegs_lab.assets.poclegs import POCLEGS_CFG, POCLEGS_URDF_USD

from ... import mdp

##
# Terrain (original course: gentle slopes, low stairs, waves, noise)
##

POCLEGS_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.3),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.0, 0.1),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.0, 0.1),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "wave_terrain": terrain_gen.HfWaveTerrainCfg(
            proportion=0.2, amplitude_range=(0.0, 0.2), num_waves=4, border_width=0.25
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.0, 0.06), noise_step=0.02, border_width=0.25
        ),
    },
)


##
# MDP
##


@configclass
class PoclegsObservationsCfg:
    """Observation specifications for the MDP."""

    # TODO (Step 3 · Observations): ลบคลาสนี้ทั้งคลาส แล้วแปะ terms/observations.py แทน


@configclass
class PoclegsRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    # TODO (Step 3 · Rewards): ลบคลาสนี้ทั้งคลาส แล้วแปะ terms/rewards.py แทน


##
# Environments
##


@configclass
class PoclegsRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Poclegs on the original-course rough terrain with a terrain-level curriculum."""

    observations: PoclegsObservationsCfg = PoclegsObservationsCfg()
    rewards: PoclegsRewardsCfg = PoclegsRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # scene
        self.scene.robot = POCLEGS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_generator = POCLEGS_ROUGH_TERRAINS_CFG
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.height_scanner = None
        # commands
        self.commands.base_velocity.marker_pos_offset = (0.0, 0.0, 0.5)


@configclass
class PoclegsFlatEnvCfg(PoclegsRoughEnvCfg):
    """Poclegs on flat ground (fast to train on a laptop GPU)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None


# TODO (Exercise 4.3 / Week 6): แปะ terms/tuned.py ต่อท้ายไฟล์นี้
