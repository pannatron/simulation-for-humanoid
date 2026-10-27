# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1RoughPPORunnerCfg


@configclass
class G1CurriculumPPORunnerCfg(G1RoughPPORunnerCfg):
    """G1 Rough network (so the pretrained checkpoint loads), shorter runs for fine-tuning."""

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.save_interval = 50
        self.experiment_name = "g1_curriculum"
