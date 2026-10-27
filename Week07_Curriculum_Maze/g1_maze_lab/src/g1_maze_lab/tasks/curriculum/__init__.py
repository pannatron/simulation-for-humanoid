# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 curriculum environments (G1 Rough + one or more extra curriculum terms)."""

import gymnasium as gym

from . import agents

for _name in ("Base", "Reward", "Push", "Velocity", "Full", "ActionRate"):
    gym.register(
        id=f"G1-Curriculum-{_name}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:G1Curriculum{_name}EnvCfg",
            "default_agent": "rsl_rl",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1CurriculumPPORunnerCfg",
        },
    )
