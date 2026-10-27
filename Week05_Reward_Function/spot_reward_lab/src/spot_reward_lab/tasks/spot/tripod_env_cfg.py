# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lab 5.3: Spot walks on three legs, the front-right leg is "broken" (held up).

Same as Isaac Lab's Spot flat task, but the trot gait term is replaced by
BrokenLegGaitReward + broken_leg_elevation_reward (ported from the Isaac Lab 2 version).
Nothing in the robot is disabled: the policy still controls all 12 joints, only the reward changes.
"""

from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity.config.spot.flat_env_cfg import SpotFlatEnvCfg, SpotRewardsCfg

from . import mdp as spot_mdp


@configclass
class SpotTripodRewardsCfg(SpotRewardsCfg):
    # the trot term (diagonal pairs) does not make sense with one leg up
    gait = None

    tripod_gait = RewardTermCfg(
        func=spot_mdp.BrokenLegGaitReward,
        weight=10.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "working_feet_names": ("fl_foot", "hl_foot", "hr_foot"),
            "broken_foot_name": "fr_foot",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )
    broken_leg_elevation = RewardTermCfg(
        func=spot_mdp.broken_leg_elevation_reward,
        weight=5.0,
        params={
            "target_height": 0.15,
            "std": 0.05,
            "asset_cfg": SceneEntityCfg("robot", body_names="fr_foot"),
        },
    )


@configclass
class SpotTripodEnvCfg(SpotFlatEnvCfg):
    rewards: SpotTripodRewardsCfg = SpotTripodRewardsCfg()
