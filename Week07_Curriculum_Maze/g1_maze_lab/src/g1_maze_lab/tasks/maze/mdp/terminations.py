# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the maze tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def maze_goal_offset(num_cells: int, cell_size: float) -> tuple[float, float]:
    """Goal position relative to the start cell: the goal is cell (n-1, n-1), the start cell (0, 0)."""
    d = (num_cells - 1) * cell_size
    return d, d


def maze_goal_reached(
    env: ManagerBasedRLEnv,
    num_cells: int,
    cell_size: float,
    threshold: float = 0.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True when the robot is within ``threshold`` metres of the centre of the goal cell."""
    robot = env.scene[asset_cfg.name]
    gx, gy = maze_goal_offset(num_cells, cell_size)
    rel = robot.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
    goal = torch.tensor((gx, gy), device=env.device)
    return torch.linalg.norm(rel - goal, dim=1) < threshold
