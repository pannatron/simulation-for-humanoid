# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Week 5: customize Spot's reward function (Isaac Lab 3 + Newton)."""

import gymnasium as gym

SPOT_AGENT = "isaaclab_tasks.contrib.velocity.config.spot.agents.rsl_rl_ppo_cfg:SpotFlatPPORunnerCfg"

##
# Register Gym environments.
##

gym.register(
    id="Spot-Reward-Exp1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.exp1_env_cfg:SpotExp1EnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": SPOT_AGENT,
    },
)

gym.register(
    id="Spot-Reward-Tripod",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tripod_env_cfg:SpotTripodEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": SPOT_AGENT,
    },
)

# exercise tasks (Week 5, Exercise 5.2-5.4)
for _id, _cfg in (
    ("Spot-Reward-Pace", "SpotPaceEnvCfg"),
    ("Spot-Reward-Tripod-Fix", "SpotTripodFixEnvCfg"),
    ("Spot-Limp", "SpotLimpEnvCfg"),
    ("Spot-Reward-Tripod-Limp", "SpotTripodLimpEnvCfg"),
):
    gym.register(
        id=_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.exercise_env_cfg:{_cfg}",
            "default_agent": "rsl_rl",
            "rsl_rl_cfg_entry_point": SPOT_AGENT,
        },
    )
