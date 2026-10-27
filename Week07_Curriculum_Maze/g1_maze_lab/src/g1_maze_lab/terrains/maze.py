# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Maze sub-terrain for the Week 7 humanoid challenge.

The walls are part of the terrain mesh (``/World/ground``), so the robot collides with them and the
height scanner (a ray caster that only hits the ground mesh) sees them as tall bumps.

Grid convention (used everywhere in this project):
    * the maze has ``n x n`` square cells of ``cell_size`` metres
    * cell ``(i, j)``: ``i`` counts along +x, ``j`` along +y
    * the robot starts in cell ``(0, 0)`` facing +x, the goal is cell ``(n-1, n-1)``
    * the returned terrain origin is the centre of the start cell, so for every maze
      ``goal - start = ((n-1) * cell_size, (n-1) * cell_size)``
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

# direction index -> (di, dj): 0 = +x (east), 1 = +y (north), 2 = -x (west), 3 = -y (south)
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def generate_maze(n: int, rng: np.random.Generator, open_ratio: float = 0.0) -> np.ndarray:
    """Return a boolean wall array ``walls[i, j, d]`` (True = wall on side ``d`` of cell ``(i, j)``).

    A perfect maze is carved with a randomized depth-first search (one path between any two cells),
    then ``open_ratio`` of the remaining inner walls are removed to create loops / open areas.
    """
    walls = np.ones((n, n, 4), dtype=bool)
    visited = np.zeros((n, n), dtype=bool)
    stack = [(0, 0)]
    visited[0, 0] = True
    while stack:
        i, j = stack[-1]
        options = [
            d for d, (di, dj) in enumerate(DIRS) if 0 <= i + di < n and 0 <= j + dj < n and not visited[i + di, j + dj]
        ]
        if not options:
            stack.pop()
            continue
        d = options[rng.integers(len(options))]
        di, dj = DIRS[d]
        walls[i, j, d] = walls[i + di, j + dj, (d + 2) % 4] = False
        visited[i + di, j + dj] = True
        stack.append((i + di, j + dj))
    # knock down some inner walls (east / north sides only, so each wall is counted once)
    if open_ratio > 0.0:
        inner = [(i, j, d) for i in range(n) for j in range(n) for d in (0, 1) if walls[i, j, d]]
        inner = [(i, j, d) for i, j, d in inner if i + DIRS[d][0] < n and j + DIRS[d][1] < n]
        for k in rng.permutation(len(inner))[: int(round(open_ratio * len(inner)))]:
            i, j, d = inner[k]
            di, dj = DIRS[d]
            walls[i, j, d] = walls[i + di, j + dj, (d + 2) % 4] = False
    return walls


def parse_layout(layout: str) -> np.ndarray:
    """Turn an ASCII drawing into a wall array (same format as :func:`generate_maze`).

    The drawing has ``2n+1`` lines of ``2n+1`` characters; the top line is the north edge (+y).
    Cells sit at odd (row, column) positions; ``|`` and ``-`` between two cells are walls
    (``+`` corners and anything else are ignored). Example (3 x 3)::

        +-+-+-+
        |S    |
        + +-+ +
        | |   |
        + + +-+
        |    G|
        +-+-+-+

    ``S`` / ``G`` are just labels: the start is always the bottom-left cell, the goal the top-right.
    """
    rows = [r for r in layout.strip("\n").splitlines() if r.strip()]
    n = (len(rows) - 1) // 2
    rows = [r.ljust(2 * n + 1) for r in rows]
    walls = np.zeros((n, n, 4), dtype=bool)
    for i in range(n):
        for j in range(n):
            r, c = 2 * (n - 1 - j) + 1, 2 * i + 1  # text row 1 is the top (+y) row of cells
            walls[i, j, 0] = rows[r][c + 1] == "|"
            walls[i, j, 1] = rows[r - 1][c] == "-"
            walls[i, j, 2] = rows[r][c - 1] == "|"
            walls[i, j, 3] = rows[r + 1][c] == "-"
    return walls


def maze_to_ascii(walls: np.ndarray) -> str:
    """Draw a wall array as text (inverse of :func:`parse_layout`)."""
    n = walls.shape[0]
    out = []
    for j in range(n - 1, -1, -1):
        out.append("+" + "".join("-+" if walls[i, j, 1] else " +" for i in range(n)))
        line = "|" if walls[0, j, 2] else " "
        for i in range(n):
            label = "S" if (i, j) == (0, 0) else "G" if (i, j) == (n - 1, n - 1) else " "
            line += label + ("|" if walls[i, j, 0] else " ")
        out.append(line)
    out.append("+" + "".join("-+" if walls[i, 0, 3] else " +" for i in range(n)))
    return "\n".join(out)


GENERATED: list[dict] = []
"""Every random maze built in this process, in generation order (for printing the layouts / their stats)."""


def maze_stats(walls: np.ndarray) -> dict:
    """Shortest route start -> goal (cells moved), dead ends (cells with 3 walls) and openings, of a wall array."""
    from collections import deque

    n = walls.shape[0]
    dist = {(0, 0): 0}
    queue = deque([(0, 0)])
    while queue:
        i, j = queue.popleft()
        for d, (di, dj) in enumerate(DIRS):
            if not walls[i, j, d] and (i + di, j + dj) not in dist:
                dist[(i + di, j + dj)] = dist[(i, j)] + 1
                queue.append((i + di, j + dj))
    dead_ends = int((walls.sum(axis=2) == 3).sum())
    return {"shortest": dist.get((n - 1, n - 1)), "dead_ends": dead_ends}


def maze_terrain(difficulty: float, cfg: MazeTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate the maze meshes (ground slab + wall boxes) and the start-cell origin."""
    n, c = cfg.num_cells, cfg.cell_size
    if cfg.layout is not None:
        walls = parse_layout(cfg.layout)
        if walls.shape[0] != n:
            raise ValueError(f"layout is {walls.shape[0]}x{walls.shape[0]} cells but num_cells={n}")
    else:
        # harder = fewer extra openings (difficulty 1.0 -> perfect maze, a single route to the goal)
        open_ratio = cfg.open_ratio_range[1] + difficulty * (cfg.open_ratio_range[0] - cfg.open_ratio_range[1])
        seed = getattr(cfg, "seed", None)
        seed = None if seed is None else seed + int(difficulty * 1e6)
        walls = generate_maze(n, np.random.default_rng(seed), open_ratio)
        GENERATED.append({"open_ratio": open_ratio, "walls": walls})

    # the maze is centred on the sub-terrain
    x0 = 0.5 * (cfg.size[0] - n * c)
    y0 = 0.5 * (cfg.size[1] - n * c)
    t, h = cfg.wall_thickness, cfg.wall_height

    meshes = [
        trimesh.creation.box(
            (cfg.size[0], cfg.size[1], 1.0),
            trimesh.transformations.translation_matrix((0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5)),
        )
    ]

    def add_wall(cx: float, cy: float, sx: float, sy: float):
        tf = trimesh.transformations.translation_matrix((cx, cy, 0.5 * h))
        meshes.append(trimesh.creation.box((sx, sy, h), tf))

    for i in range(n):
        for j in range(n):
            # west and south walls of every cell + east / north walls on the outer border
            if walls[i, j, 2]:
                add_wall(x0 + i * c, y0 + (j + 0.5) * c, t, c + t)
            if walls[i, j, 3]:
                add_wall(x0 + (i + 0.5) * c, y0 + j * c, c + t, t)
            if i == n - 1 and walls[i, j, 0]:
                add_wall(x0 + n * c, y0 + (j + 0.5) * c, t, c + t)
            if j == n - 1 and walls[i, j, 1]:
                add_wall(x0 + (i + 0.5) * c, y0 + n * c, c + t, t)

    origin = np.array([x0 + 0.5 * c, y0 + 0.5 * c, 0.0])
    return meshes, origin


@configclass
class MazeTerrainCfg(SubTerrainBaseCfg):
    """A square maze of ``num_cells x num_cells`` cells with walls the robot cannot step over."""

    function = maze_terrain

    num_cells: int = 5
    """Cells per side. The start is cell (0, 0), the goal cell (n-1, n-1)."""

    cell_size: float = 2.0
    """Cell width [m] (corridor width is ``cell_size - wall_thickness``)."""

    wall_thickness: float = 0.15
    wall_height: float = 1.0

    open_ratio_range: tuple[float, float] = (0.0, 0.3)
    """Fraction of inner walls removed at difficulty 1.0 (first value) and 0.0 (second value)."""

    layout: str | None = None
    """Optional fixed ASCII layout (see :func:`parse_layout`). If set, the random maze is not used."""
