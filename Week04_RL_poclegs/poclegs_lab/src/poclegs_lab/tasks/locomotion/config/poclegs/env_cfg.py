# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Poclegs velocity-tracking environments (original-course env, ported to Isaac Lab 3 + Newton)."""

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

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        hip_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.03, n_max=0.03),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HR"])},
        )
        kfe_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HAA", ".*HFE", ".*KFE"])},
        )
        ffe_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.08, n_max=0.08),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*FFE"])},
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class PoclegsRewardsCfg(RewardsCfg):
    """Reward terms for the MDP (original-course values)."""

    dof_acc_l2 = None
    # extra terms, off (weight 0) to match the original course - turn them on with Hydra overrides
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=0.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_range,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ffe"),
            "command_name": "base_velocity",
            "threshold_min": 0.2,
            "threshold_max": 0.5,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ffe"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ffe"),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*hfe", ".*haa"]), "threshold": 1.0},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HR", ".*HAA"])},
    )
    joint_deviation_knee = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*KFE"])},
    )


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


##
# Tuned recipe (Week 4 "make it walk"): the Unitree H1 reward set from Isaac Lab 3, adapted to poclegs
##


@configclass
class PoclegsTunedRewardsCfg(RewardsCfg):
    """H1-style rewards: reward single-stance steps instead of penalizing short ones."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = None
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ffe"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ffe"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ffe"),
        },
    )
    undesired_contacts = None
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*FFE")}
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HR", ".*HAA"])},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    dof_torques_l2 = None
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)


@configclass
class PoclegsTunedFlatEnvCfg(PoclegsFlatEnvCfg):
    """Poclegs that learns to walk in ~20 min on a laptop GPU (see the Week 4 exercises)."""

    rewards: PoclegsTunedRewardsCfg = PoclegsTunedRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # poclegs' legs are spread along x and swing about x: its "forward" is the base +-y axis
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # the USD caps HAA/KFE at 1.57 rad/s (90 deg/s): use the actuator speed limits instead
        for name, limit in {"hr": 23.0, "haa": 15.0, "kfe": 20.0, "ffe": 23.0}.items():
            self.scene.robot.actuators[name].joint_velocity_limit = limit


@configclass
class PoclegsUrdfFlatEnvCfg(PoclegsTunedFlatEnvCfg):
    """Poclegs rebuilt from URDF (Week 6), trained with the tuned recipe. Swap the robot with Hydra overrides, e.g.

    ``env.scene.robot.spawn.usd_path=/path/robot.usda "env.scene.robot.init_state.pos=[0,0,0.539]"``
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.usd_path = POCLEGS_URDF_USD
        # the URDF importer nests links (base/ll_hr/ll_haa/...), so match bodies at any depth
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*"
        # URDF has no collision groups: overlapping hip hulls would touch the base at spawn
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False
