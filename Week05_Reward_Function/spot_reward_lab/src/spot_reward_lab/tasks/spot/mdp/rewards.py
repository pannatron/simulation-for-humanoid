# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Week 5 reward terms for Spot walking on three legs (front-right leg "broken").

Ported from the Isaac Lab 2 supplementary ``reward.py``. Isaac Lab 3 changes:
  * sensor/asset data are warp arrays -> read them with ``.torch``
  * contact-sensor bodies are looked up with ``find_sensors`` (was ``find_bodies``)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def _gait_enabled(env: ManagerBasedRLEnv, asset: Articulation, velocity_threshold: float) -> torch.Tensor:
    """Only enforce a gait when the robot is told to move (or is already moving)."""
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b.torch[:, :2], dim=1)
    return torch.logical_or(cmd > 0.0, body_vel > velocity_threshold)


class BrokenLegGaitReward(ManagerTermBase):
    """Walk on the three working legs and keep the broken foot off the ground.

    reward = sync(3 working feet) * exp(-contact_time(broken foot) / std)
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        working_feet_names = cfg.params["working_feet_names"]
        if len(working_feet_names) != 3:
            raise ValueError("This reward expects exactly 3 working feet for a broken leg gait.")
        # keep the order given in the config (preserve_order) so pairs are predictable
        self.working_feet = self.contact_sensor.find_sensors(working_feet_names, preserve_order=True)[0]
        self.broken_foot = self.contact_sensor.find_sensors([cfg.params["broken_foot_name"]])[0][0]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        working_feet_names,
        broken_foot_name,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        reward = self._sync_working_feet() * self._broken_leg_penalty()
        return torch.where(_gait_enabled(env, self.asset, self.velocity_threshold), reward, 0.0)

    def _sync_working_feet(self) -> torch.Tensor:
        """Reward the three working feet for having the same air/contact times (all pairs)."""
        air_time = self.contact_sensor.data.current_air_time.torch
        contact_time = self.contact_sensor.data.current_contact_time.torch
        total_error = torch.zeros(self.num_envs, device=self.device)
        for a, b in ((0, 1), (1, 2), (0, 2)):
            foot_a, foot_b = self.working_feet[a], self.working_feet[b]
            total_error += torch.clip(torch.square(air_time[:, foot_a] - air_time[:, foot_b]), max=self.max_err**2)
            total_error += torch.clip(
                torch.square(contact_time[:, foot_a] - contact_time[:, foot_b]), max=self.max_err**2
            )
        return torch.exp(-total_error / self.std)

    def _broken_leg_penalty(self) -> torch.Tensor:
        """1.0 while the broken foot is in the air, decays the longer it stays on the ground."""
        contact_time = self.contact_sensor.data.current_contact_time.torch
        return torch.exp(-contact_time[:, self.broken_foot] / self.std)


def broken_leg_elevation_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float
) -> torch.Tensor:
    """Reward the broken foot for staying at ``target_height`` (world z, metres)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    return torch.exp(-torch.sum(torch.square(foot_z - target_height), dim=1) / std)


def air_time_reward_any_feet(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Same as Spot's ``air_time_reward`` but works for any number of feet.

    The Isaac Lab version uses ``.expand(-1, 4)``, so it only accepts exactly 4 feet.
    """
    # TODO (Exercise 5.3): start from Spot's air_time_reward in
    # isaaclab_tasks/contrib/velocity/config/spot/mdp/rewards.py and remove the hard-coded 4 feet.
    raise NotImplementedError("Exercise 5.3: write air_time_reward_any_feet")


def broken_foot_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """1.0 when the broken foot touches the ground (use with a negative weight)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, :]
    return (torch.linalg.norm(forces, dim=-1) > threshold).any(dim=1).float()
