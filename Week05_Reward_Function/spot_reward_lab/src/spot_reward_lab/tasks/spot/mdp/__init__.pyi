# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "BrokenLegGaitReward",
    "air_time_reward_any_feet",
    "broken_foot_contact",
    "broken_leg_elevation_reward",
]

# Forward Isaac Lab's Spot reward terms (GaitReward, foot_clearance_reward, ...), then add ours.
from isaaclab_tasks.contrib.velocity.config.spot.mdp import *  # noqa: F401, F403

from .rewards import BrokenLegGaitReward, air_time_reward_any_feet, broken_foot_contact, broken_leg_elevation_reward
