# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 maze challenge environments."""

import gymnasium as gym

from . import agents


def _register(task_id: str, cfg_name: str, agent_cfg: str):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{cfg_name}",
            "default_agent": "rsl_rl",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{agent_cfg}",
        },
    )


# pretrained G1 Rough walking policy (pretrained/g1_rough/model_pretrained.pt)
_register("G1-Maze-Challenge", "G1MazeEnvCfg", "G1MazePPORunnerCfg")
_register("G1-Maze-Random", "G1MazeRandomEnvCfg", "G1MazePPORunnerCfg")
_register("G1-Maze-Open", "G1MazeOpenEnvCfg", "G1MazePPORunnerCfg")
_register("G1-Maze-Levels", "G1MazeLevelsEnvCfg", "G1MazePPORunnerCfg")
# Week 3 G1 Flat walking policy (pretrained/g1_flat/model_pretrained.pt)
_register("G1-Maze-Challenge-FlatPolicy", "G1MazeFlatPolicyEnvCfg", "G1MazeFlatPolicyPPORunnerCfg")
_register("G1-Maze-Open-FlatPolicy", "G1MazeOpenFlatPolicyEnvCfg", "G1MazeFlatPolicyPPORunnerCfg")
# Challenge 7.5: random hazard mazes 7 x 7 / 9 x 9 (practice levels), scripts/challenge.py
_register("G1-Maze-Hard", "G1MazeHardEnvCfg", "G1MazePPORunnerCfg")
_register("G1-Maze-Hard9", "G1MazeHard9EnvCfg", "G1MazePPORunnerCfg")
