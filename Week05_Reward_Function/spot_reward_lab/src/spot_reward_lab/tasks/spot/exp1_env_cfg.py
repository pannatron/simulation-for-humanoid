# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lab 5.2 Experiment 1: lift the feet higher and change which feet move together."""

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.velocity.config.spot.flat_env_cfg import SpotFlatEnvCfg


@configclass
class SpotExp1EnvCfg(SpotFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # swing feet should clear 0.28 m instead of 0.10 m
        clearance = self.rewards.foot_clearance
        clearance.weight = 4.0  # 0.5 -> 4.0
        clearance.params["std"] = 0.02  # 0.05 -> 0.02 (stricter)
        clearance.params["tanh_mult"] = 12.0  # 2.0 -> 12.0 (counts even slow-moving feet as swinging)
        clearance.params["target_height"] = 0.28  # 0.10 -> 0.28
        # front pair and rear pair in sync (bound) instead of diagonal pairs (trot)
        gait = self.rewards.gait
        gait.weight = 25.0  # 10 -> 25
        gait.params["std"] = 0.02  # 0.1 -> 0.02
        gait.params["max_err"] = 0.04  # 0.2 -> 0.04
        gait.params["velocity_threshold"] = 0.0  # 0.5 -> 0.0
        gait.params["synced_feet_pair_names"] = (("fl_foot", "fr_foot"), ("hl_foot", "hr_foot"))
