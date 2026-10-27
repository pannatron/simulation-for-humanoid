# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "AdaptivePushCurriculum",
    "CommandRangeCurriculum",
    "RewardWeightRampCurriculum",
]

# Forward the velocity-task MDP terms lazily (terrain_levels_vel, rewards, ...), then add ours.
from isaaclab_tasks.core.velocity.mdp import *  # noqa: F401, F403

from .curriculums import AdaptivePushCurriculum, CommandRangeCurriculum, RewardWeightRampCurriculum
