# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg, G1RoughPPORunnerCfg


@configclass
class G1MazePPORunnerCfg(G1RoughPPORunnerCfg):
    """Same network as Isaac Lab's G1 Rough, so its checkpoint loads into the maze task."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_maze"


@configclass
class G1MazeFlatPolicyPPORunnerCfg(G1FlatPPORunnerCfg):
    """Same network as Week 3's G1 Flat."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_maze_flat"
