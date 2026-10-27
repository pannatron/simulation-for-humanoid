# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Maze generator + autopilot planner tests (no simulator)."""

import numpy as np
import pytest
import torch

from g1_maze_lab.autopilot import MazeAutopilot
from g1_maze_lab.terrains.maze import DIRS, generate_maze, maze_to_ascii, parse_layout

CHALLENGE = """
+-+-+-+-+-+
|       |G|
+ + +-+ + +
| | | |   |
+ + + +-+ +
| | |   | |
+ + + + + +
| | | | | |
+-+ + +-+ +
|S  |     |
+-+-+-+-+-+
"""


def _consistent(w):
    n = w.shape[0]
    for i in range(n):
        for j in range(n):
            for d, (di, dj) in enumerate(DIRS):
                a, b = i + di, j + dj
                if 0 <= a < n and 0 <= b < n:
                    assert w[i, j, d] == w[a, b, (d + 2) % 4]
                else:
                    assert w[i, j, d]


def _solve(walls, strategy):
    """Teleport a 'robot' from cell centre to cell centre; return cells walked (None = gave up)."""
    n = walls.shape[0]
    pilot = MazeAutopilot(1, n, 2.0, "cpu", strategy)
    pos = torch.zeros(1, 2)
    for _ in range(400):
        cell = pilot.cell_of(pos)
        if tuple(cell[0].tolist()) == (n - 1, n - 1):
            return pilot.cells_walked.item() + 1  # the last move into the goal is not a decision
        pilot.step(pos, torch.zeros(1), torch.tensor(walls[cell[0, 0], cell[0, 1]]).unsqueeze(0), 0.02)
        pos = pilot.centre_of(pilot.target)
    return None


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(20))
def test_generated_mazes_are_consistent_and_roundtrip(seed):
    w = generate_maze(5, np.random.default_rng(seed), 0.2 * (seed % 2))
    _consistent(w)
    assert (parse_layout(maze_to_ascii(w)) == w).all()


@pytest.mark.unit
@pytest.mark.parametrize("strategy,cells", [("left", 18), ("right", 30), ("flood", 10)])
def test_challenge_route_lengths(strategy, cells):
    w = parse_layout(CHALLENGE)
    _consistent(w)
    assert _solve(w, strategy) == cells


@pytest.mark.unit
@pytest.mark.parametrize("strategy", ["left", "right", "flood"])
def test_every_strategy_solves_random_mazes(strategy):
    for seed in range(50):
        assert _solve(generate_maze(5, np.random.default_rng(seed), 0.2 * (seed % 2)), strategy) is not None
