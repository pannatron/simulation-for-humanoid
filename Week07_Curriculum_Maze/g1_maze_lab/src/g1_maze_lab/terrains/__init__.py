# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Maze terrain."""

from .maze import DIRS, GENERATED, MazeTerrainCfg, generate_maze, maze_stats, maze_to_ascii, parse_layout
from .hazard_maze import (
    HAZARDS,
    HazardMazeTerrainCfg,
    HazardTerrainGeneratorCfg,
    HazardTerrainImporterCfg,
)
