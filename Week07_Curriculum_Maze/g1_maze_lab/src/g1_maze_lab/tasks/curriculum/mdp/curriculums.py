# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Three curriculum terms for G1 (Week 7).

Isaac Lab ships ``terrain_levels_vel`` (terrain difficulty) plus generic helpers in
``isaaclab.envs.mdp`` (``modify_reward_weight``, ``modify_term_cfg``, ``DifficultyScheduler``).
The terms below change *other* parts of the task, based on how well the robot is doing:

=============================  ==================================================================
term                           what it changes
=============================  ==================================================================
RewardWeightRampCurriculum     a reward weight, linearly between two training steps
AdaptivePushCurriculum         push strength, up when few robots fall, down when many fall
CommandRangeCurriculum         the max forward speed command, up when tracking is good
=============================  ==================================================================

Every term is called by the curriculum manager **when environments reset** (``env_ids`` = the robots
that just finished an episode), before their episode rewards are cleared. The value it returns is logged
to TensorBoard as ``Curriculum/<term name>``.

Time is measured with ``env.common_step_counter`` = environment steps since the start of training
(one PPO iteration = ``num_steps_per_env`` = 24 steps).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class RewardWeightRampCurriculum(ManagerTermBase):
    """Move a reward weight from ``start_weight`` to ``end_weight`` between two training steps.

    Isaac Lab's built-in ``modify_reward_weight`` switches the weight in one jump; a ramp avoids the sudden
    change in the reward scale that the critic has to re-learn.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._term_cfg = env.reward_manager.get_term_cfg(cfg.params["term_name"])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        start_weight: float,
        end_weight: float,
        start_step: int,
        end_step: int,
    ) -> float:
        frac = (env.common_step_counter - start_step) / max(end_step - start_step, 1)
        frac = min(max(frac, 0.0), 1.0)
        self._term_cfg.weight = start_weight + frac * (end_weight - start_weight)
        env.reward_manager.set_term_cfg(term_name, self._term_cfg)
        return self._term_cfg.weight


class AdaptivePushCurriculum(ManagerTermBase):
    """Push harder while the robots cope, back off when too many fall.

    Every ``update_every`` steps it looks at the episodes that ended since the last update:
    fall rate < ``raise_below`` -> push velocity + ``step``; fall rate > ``lower_above`` -> - ``step``.
    The push event term must use ``push_by_setting_velocity`` (``velocity_range`` param).
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._push_cfg = env.event_manager.get_term_cfg(cfg.params.get("event_name", "push_robot"))
        self.level = cfg.params.get("start_vel", 0.0)
        self._ended = 0
        self._fell = 0
        self._last_update = 0
        self._apply(cfg.params.get("event_name", "push_robot"), env)

    def _apply(self, event_name: str, env: ManagerBasedRLEnv):
        v = self.level
        self._push_cfg.params["velocity_range"] = {"x": (-v, v), "y": (-v, v)}
        env.event_manager.set_term_cfg(event_name, self._push_cfg)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        event_name: str = "push_robot",
        fall_term: str = "base_contact",
        start_vel: float = 0.0,
        max_vel: float = 1.5,
        step: float = 0.1,
        raise_below: float = 0.05,
        lower_above: float = 0.2,
        update_every: int = 24 * 20,
    ) -> float:
        if env.common_step_counter == 0:  # the very first reset at start-up is not an episode end
            return self.level
        fell = env.termination_manager.get_term(fall_term)[env_ids]
        self._ended += len(env_ids)
        self._fell += int(fell.sum())
        if env.common_step_counter - self._last_update >= update_every and self._ended > 0:
            fall_rate = self._fell / self._ended
            if fall_rate < raise_below:
                self.level = min(self.level + step, max_vel)
            elif fall_rate > lower_above:
                self.level = max(self.level - step, 0.0)
            self._apply(event_name, env)
            self._ended, self._fell, self._last_update = 0, 0, env.common_step_counter
        return self.level


class CommandRangeCurriculum(ManagerTermBase):
    """Raise the max forward-speed command when the robots track the current range well.

    Same rule as legged_gym's command curriculum: if the mean episode reward of ``reward_term`` is above
    ``threshold`` x its maximum possible value, widen ``lin_vel_x`` by ``step`` (up to ``final_max``).

    The term is called at *every* reset -- several times per step once episodes end at different times --
    so the decision is made at most once every ``update_every`` steps, from all episodes that ended since.
    Without this the range went from 1.0 to 2.0 m/s in 4 iterations (see the slides).
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._ranges = env.command_manager.get_term(cfg.params.get("command_name", "base_velocity")).cfg.ranges
        self._ranges.lin_vel_x = (self._ranges.lin_vel_x[0], cfg.params.get("start_max", 0.5))
        self._sum, self._count, self._last_update = 0.0, 0, 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str = "base_velocity",
        reward_term: str = "track_lin_vel_xy_exp",
        start_max: float = 0.5,
        final_max: float = 1.5,
        step: float = 0.1,
        threshold: float = 0.8,
        update_every: int = 24 * 20,
    ) -> float:
        if env.common_step_counter == 0 or len(env_ids) == 0:
            return self._ranges.lin_vel_x[1]
        # episode sum / episode length = average reward per second; its maximum is the weight itself
        sums = env.reward_manager._episode_sums[reward_term][env_ids]
        self._sum += float(sums.sum()) / env.max_episode_length_s
        self._count += len(env_ids)
        if env.common_step_counter - self._last_update >= update_every:
            weight = env.reward_manager.get_term_cfg(reward_term).weight
            if self._sum / self._count > threshold * weight:
                lo, hi = self._ranges.lin_vel_x
                self._ranges.lin_vel_x = (lo, min(hi + step, final_max))
            self._sum, self._count, self._last_update = 0.0, 0, env.common_step_counter
        return self._ranges.lin_vel_x[1]
