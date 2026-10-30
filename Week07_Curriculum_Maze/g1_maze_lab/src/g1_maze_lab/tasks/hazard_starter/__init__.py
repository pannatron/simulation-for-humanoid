# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Challenge 7.5 starter: fine-tune G1 Rough on hazard-like ground (copy this folder and make it your own)."""

import gymnasium as gym

gym.register(
    id="G1-Hazard-Starter",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1HazardStarterEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{__name__}.env_cfg:G1HazardStarterPPORunnerCfg",
    },
)
