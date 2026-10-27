# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 Rough + extra curriculum terms (Week 7).

Each config starts from Isaac Lab's ``G1RoughEnvCfg`` (which already has the ``terrain_levels``
curriculum) and adds one more curriculum term. ``Full`` adds all three.
Training on a 6 GB laptop GPU: fine-tune the pretrained G1 Rough checkpoint for a few hundred iterations
(see the slides), training from scratch needs ~5000 iterations x 4096 robots.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg

from . import mdp

STEPS_PER_IT = 24  # num_steps_per_env of the PPO runner: 1 iteration = 24 env steps


def reward_ramp_term() -> CurrTerm:
    # feet_air_time 0.25 -> 1.0 between iteration 50 and 150: longer steps once the robot already walks
    return CurrTerm(
        func=mdp.RewardWeightRampCurriculum,
        params={
            "term_name": "feet_air_time",
            "start_weight": 0.25,
            "end_weight": 1.0,
            "start_step": 50 * STEPS_PER_IT,
            "end_step": 150 * STEPS_PER_IT,
        },
    )


def push_term() -> CurrTerm:
    # push velocity 0 -> up to 1.5 m/s, +0.1 every 20 iterations while < 5 % of the episodes end in a fall
    return CurrTerm(
        func=mdp.AdaptivePushCurriculum,
        params={
            "event_name": "push_robot",
            "fall_term": "base_contact",
            "start_vel": 0.0,
            "max_vel": 1.5,
            "step": 0.1,
            "raise_below": 0.05,
            "lower_above": 0.2,
            "update_every": 20 * STEPS_PER_IT,
        },
    )


def velocity_term() -> CurrTerm:
    # forward-speed command max 1.0 -> up to 2.0 m/s: +0.1 at most every 20 iterations, if tracking >= 80 %
    return CurrTerm(
        func=mdp.CommandRangeCurriculum,
        params={
            "command_name": "base_velocity",
            "reward_term": "track_lin_vel_xy_exp",
            "start_max": 1.0,
            "final_max": 2.0,
            "step": 0.1,
            "threshold": 0.8,
            "update_every": 20 * STEPS_PER_IT,
        },
    )


@configclass
class G1CurriculumBaseEnvCfg(G1RoughEnvCfg):
    """G1 Rough sized for a laptop GPU: 1024 robots, 6 x 10 terrain tiles."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1024
        self.scene.terrain.terrain_generator.num_rows = 6
        self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.terrain.max_init_terrain_level = 3


@configclass
class G1CurriculumRewardEnvCfg(G1CurriculumBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.curriculum.feet_air_time_ramp = reward_ramp_term()


@configclass
class G1CurriculumPushEnvCfg(G1CurriculumBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.curriculum.push_strength = push_term()


@configclass
class G1CurriculumVelocityEnvCfg(G1CurriculumBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.curriculum.command_vel_x = velocity_term()


@configclass
class G1CurriculumFullEnvCfg(G1CurriculumBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.curriculum.feet_air_time_ramp = reward_ramp_term()
        self.curriculum.push_strength = push_term()
        self.curriculum.command_vel_x = velocity_term()


# ---------------------------------------------------------------- Exercise 7.2 (answer)
def action_rate_schedule(env, env_ids, old_value, start_step: int, end_step: int, start: float, end: float):
    """modify_fn for the built-in ``modify_term_cfg``: ramp a value between two training steps."""
    frac = min(max((env.common_step_counter - start_step) / max(end_step - start_step, 1), 0.0), 1.0)
    return start + frac * (end - start)


@configclass
class G1CurriculumActionRateEnvCfg(G1CurriculumBaseEnvCfg):
    """Exercise 7.2: no new class, only ``modify_term_cfg`` + a small modify_fn."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.action_rate_weight = CurrTerm(
            func=mdp.modify_term_cfg,
            params={
                "address": "rewards.action_rate_l2.weight",
                "modify_fn": action_rate_schedule,
                "modify_params": {
                    "start_step": 0,
                    "end_step": 80 * STEPS_PER_IT,
                    "start": -0.005,
                    "end": -0.02,
                },
            },
        )
