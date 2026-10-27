# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "feet_air_time_range",
]

# Forward the velocity-task MDP terms lazily, then add project-specific terms below.
from isaaclab_tasks.core.velocity.mdp import *  # noqa: F401, F403

from .rewards import feet_air_time_range
