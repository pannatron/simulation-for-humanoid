# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hazard maze generator tests (Challenge 7.5, no simulator)."""

import numpy as np

from g1_maze_lab.terrains import HAZARDS, HazardMazeTerrainCfg
from g1_maze_lab.terrains.hazard_maze import _PENDING_ICE, _PENDING_SPIKES, hazard_maze_terrain
from g1_maze_lab.terrains.maze import maze_stats


def _build(seed):
    cfg = HazardMazeTerrainCfg(size=(15.0, 15.0))
    cfg.seed = seed
    meshes, origin = hazard_maze_terrain(0.5, cfg)
    return cfg, meshes, origin, HAZARDS[-1], list(_PENDING_ICE)


def test_hazards_placed_and_reachable():
    for seed in range(20):
        cfg, meshes, origin, hz, ice = _build(seed)
        n = cfg.num_cells
        assert len(hz["ice"]) == cfg.short_ice + cfg.long_ice + cfg.extra_ice == len(ice)
        assert len(hz["rough"]) >= cfg.long_rough + cfg.extra_rough
        short_only = set(hz["routes"]["short"]) - set(hz["routes"]["long"])
        hard = {g[:2] for g in hz["gap"] + hz["ramp"] + hz["spike"]} | {tuple(x) for x in hz["ice"]} | {(1, 0), (0, 1)}
        assert all(tuple(c) in {tuple(x) for x in hz["rough"]} for c in short_only - hard), "plain cell on the short route"
        assert len(hz["gap"]) == cfg.short_gaps and len(hz["spike"]) == cfg.short_spikes
        assert len(hz["ramp"]) == cfg.short_ramps + cfg.long_ramps
        cells = [g[:2] for g in hz["gap"] + hz["ramp"] + hz["spike"]] + hz["ice"] + hz["rough"]
        assert len(set(cells)) == len(cells), "two hazards in one cell"
        for banned in ((0, 0), (1, 0), (0, 1), (n - 1, n - 1)):
            assert banned not in cells
        np.testing.assert_allclose(origin[:2], 0.5 * (15.0 - n * cfg.cell_size) + 0.5 * cfg.cell_size)


def test_short_route_hard_long_route_easy():
    from g1_maze_lab.terrains.hazard_maze import route_turns

    for seed in range(20):
        cfg, _, _, hz, _ = _build(seed)
        short, long = hz["routes"]["short"], hz["routes"]["long"]
        assert len(long) - len(short) >= cfg.min_route_gap
        assert route_turns(long) >= route_turns(short) + cfg.min_extra_turns
        short_only = set(short) - set(long)
        long_only = set(long) - set(short)
        for i, j, _, w in hz["gap"]:
            assert (i, j) in short_only and cfg.short_gap_width_range[0] <= w <= cfg.short_gap_width_range[1]
        for i, j, *_ in hz["spike"]:
            assert (i, j) in short_only
        for i, j, _, h in hz["ramp"]:
            rng = cfg.short_ramp_height_range if (i, j) in short_only else cfg.long_ramp_height_range
            assert (i, j) in short_only | long_only and rng[0] <= h <= rng[1]
        assert all(tuple(x) in long_only for x in hz["rough"][: cfg.long_rough])


def test_spread_mode_without_route_split():
    cfg = HazardMazeTerrainCfg(size=(15.0, 15.0), route_split=False)
    cfg.seed = 3
    hazard_maze_terrain(0.5, cfg)
    hz = HAZARDS[-1]
    assert len(hz["ice"]) == cfg.num_ice and len(hz["rough"]) == cfg.num_rough
    assert len(hz["gap"]) <= cfg.num_gaps and len(hz["ramp"]) <= cfg.num_ramps


def test_gaps_only_in_straight_corridors():
    from g1_maze_lab.terrains import GENERATED

    for seed in range(20):
        _, _, _, hz, _ = _build(seed)
        walls = GENERATED[-1]["walls"]
        assert maze_stats(walls)["shortest"] is not None
        for i, j, axis, *_ in hz["gap"] + hz["ramp"] + hz["spike"]:
            w = walls[i, j]
            if axis == 0:
                assert not w[0] and not w[2] and w[1] and w[3]
            else:
                assert not w[1] and not w[3] and w[0] and w[2]


def test_same_seed_same_maze():
    from g1_maze_lab.terrains import GENERATED

    _build(7)
    a = GENERATED[-1]["walls"].copy(), dict(HAZARDS[-1])
    _build(7)
    b = GENERATED[-1]["walls"], HAZARDS[-1]
    assert (a[0] == b[0]).all() and a[1] == b[1]


def test_spike_bed_leaves_a_free_lane():
    for seed in range(20):
        cfg, _, _, hz, _ = _build(seed)
        c, hw = cfg.cell_size, 0.5 * (cfg.cell_size - cfg.wall_thickness)
        if hz["spike"]:
            assert _PENDING_SPIKES, "spike cones missing"
        for i, j, axis, x0, y0, x1, y1 in hz["spike"]:
            lo, hi = (y0 - j * c, y1 - j * c) if axis == 0 else (x0 - i * c, x1 - i * c)
            along = (x1 - x0) if axis == 0 else (y1 - y0)
            assert abs(along - cfg.spike_length) < 1e-6
            # one end at a wall, the other just past the middle: the free lane on the other side
            assert abs(abs(lo if abs(lo) > abs(hi) else hi) - hw) < 1e-6
            free = hw - min(abs(lo), abs(hi))
            assert hw - cfg.spike_reach_range[1] - 1e-6 <= free <= hw - cfg.spike_reach_range[0] + 1e-6
        near = [g[:2] for g in hz["spike"] + hz["gap"] + hz["ramp"]]
        for i, j, *_ in hz["spike"] + hz["ramp"]:
            assert sum(abs(i - a) + abs(j - b) == 1 for a, b in near) == 0, "ramp / spike bed next to another hazard"


def test_start_fork_gives_two_routes():
    from g1_maze_lab.terrains import GENERATED

    for seed in range(20):
        _build(seed)
        w = GENERATED[-1]["walls"]
        assert not w[0, 0, 0] and not w[0, 0, 1], "start cell must open east and north"
        assert not w[1, 0, 2] and not w[0, 1, 3]
        assert maze_stats(w)["shortest"] is not None


def test_first_hard_hazard_near_the_fork():
    for seed in range(20):
        cfg, _, _, hz, _ = _build(seed)
        short = hz["routes"]["short"]
        hard = {(x[0], x[1]): k for k in ("gap", "ramp", "spike") for x in hz[k] if (x[0], x[1]) in short}
        first = min(short.index(c) for c in hard)
        assert first < cfg.peek_cells and hard[short[first]] in ("ramp", "spike")
