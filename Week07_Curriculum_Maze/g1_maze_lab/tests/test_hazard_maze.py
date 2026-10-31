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
            # one end at a wall, the other near the middle: the free lane on the other side
            wall, inner = (lo, hi) if abs(lo) > abs(hi) else (hi, lo)
            assert abs(abs(wall) - hw) < 1e-6
            reach = -inner * np.sign(wall)  # how far the bed reaches past the middle
            assert cfg.spike_reach_range[0] - 1e-6 <= reach <= cfg.spike_reach_range[1] + 1e-6
            assert hw - reach >= 0.92, "free lane too narrow for a walking G1"
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


def _build_level(seed, **kw):
    from g1_maze_lab.terrains import GENERATED

    n = kw.get("num_cells", 7)
    cfg = HazardMazeTerrainCfg(size=(n * 2.0 + 1.0,) * 2, **kw)
    cfg.seed = seed
    _, origin = hazard_maze_terrain(0.5, cfg)
    return cfg, origin, HAZARDS[-1], GENERATED[-1]


def test_level9_slalom_and_long_route_spikes():
    kw = dict(num_cells=9, short_gaps=2, short_slaloms=1, long_ramps=3, long_spikes=2, long_rough=4, long_ice=3)
    for seed in range(10):
        cfg, _, hz, _ = _build_level(seed, **kw)
        short_only = set(hz["routes"]["short"]) - set(hz["routes"]["long"])
        long_only = set(hz["routes"]["long"]) - set(hz["routes"]["short"])
        assert len(hz["slalom"]) == 1 and hz["slalom"][0] in short_only
        cells = [tuple(b[:2]) for b in hz["spike"]]
        assert cells.count(hz["slalom"][0]) == 2, "a slalom cell has two beds"
        assert sum(c in long_only for c in cells) == cfg.long_spikes
        assert len(hz["gap"]) == 2
        # slalom: the beds are on opposite sides of the corridor, with a gap along it
        i, j, a, *r1 = next(b for b in hz["spike"] if tuple(b[:2]) == hz["slalom"][0])
        r2 = [b[3:] for b in hz["spike"] if tuple(b[:2]) == hz["slalom"][0]][1]
        across = (1, 3) if a == 0 else (0, 2)
        along = (0, 2) if a == 0 else (1, 3)
        mid = (j if a == 0 else i) * cfg.cell_size
        s1 = sum(r1[k] for k in across) / 2 - mid
        s2 = sum(r2[k] for k in across) / 2 - mid
        assert s1 * s2 < 0
        assert abs(min(r1[along[1]], r2[along[1]]) - max(r1[along[0]], r2[along[0]])) >= cfg.slalom_gap - 1e-6


def test_random_ends():
    for seed in range(15):
        cfg, origin, hz, gen = _build_level(seed, num_cells=10, random_ends=True, long_spikes=2)
        s, g = hz["start"], hz["goal"]
        assert abs(s[0] - g[0]) + abs(s[1] - g[1]) >= 10
        assert gen["start"] == s and gen["goal"] == g
        assert len(hz["routes"]["long"]) - len(hz["routes"]["short"]) >= cfg.min_route_gap
        cells = [tuple(b[:2]) for b in hz["gap"] + hz["ramp"] + hz["spike"]] + hz["ice"] + hz["rough"]
        assert s not in cells and g not in cells
        # the robot spawns in the start cell
        x0 = 0.5 * (cfg.size[0] - 10 * cfg.cell_size)
        np.testing.assert_allclose(origin[:2], [x0 + (s[0] + 0.5) * cfg.cell_size, x0 + (s[1] + 0.5) * cfg.cell_size])
        assert len({hz["start"] for _ in [0]}) == 1
    starts = {_build_level(k, num_cells=10, random_ends=True)[2]["start"] for k in range(15)}
    assert len(starts) > 3, "start cell does not change with the seed"


def test_spike_hit_rule():
    import importlib.util
    import os
    import sys
    import types

    import torch

    # spike_hit only needs torch: load it from scripts/challenge.py without starting the simulator
    src = open(os.path.join(os.path.dirname(__file__), "..", "scripts", "challenge.py")).read()
    code = src[src.index("def spike_hit"): src.index("# ---", src.index("def spike_hit"))]
    ns = {"torch": torch}
    exec(code, ns)
    beds = [(0.0, 0.0, 1.0, 0.5)]
    assert ns["spike_hit"](torch.tensor([[0.5, 0.2, 0.1]]), beds, 0.37)
    assert not ns["spike_hit"](torch.tensor([[0.5, 0.2, 0.6]]), beds, 0.37), "above the tips"
    assert not ns["spike_hit"](torch.tensor([[0.5, 0.6, 0.1]]), beds, 0.37), "beside the bed"
