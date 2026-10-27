# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hazard maze for the Week 7 challenge: a bigger random maze whose floor is not always kind.

Three kinds of hazard are placed in random cells (never in the start / goal cell or right next to the start):

* **gap**   a trench across a straight corridor (``gap_width_range`` wide, ``pit_depth`` deep).
            The robot has to step over it; a foot in the trench usually means a fall.
* **ice**   a slippery tile (friction ``ice_friction``). It is a separate mesh ``/World/ground/ice`` 1 cm above
            the floor with its own physics material, drawn light blue. The height scanners see it only as a 1 cm step.
* **rough** loose boxes 2-8 cm high on the floor (the G1 Rough policy was trained on worse).
* **ramp**  a hump across a straight corridor: up a slope, a short flat top, down again (``ramp_height_range``
            high over ``ramp_length``, 13-22 deg with the defaults).
* **spike** a bed of sharp, slippery cones (25-35 cm, friction ``spike_friction``) on one side of a straight corridor, from the wall to just past the
            middle. No extra rule: the cones are just hard to walk on (feet slip off the tips, toes catch), so walking
            past on the free side is faster and safer. A separate red mesh ``/World/ground/spikes``; the height
            scanners see the cones.

The maze is a perfect maze plus a fork at the start (``start_fork``): both sides of the start cell are open, which
makes exactly one loop, so there are two routes to the goal. A hazard on the shared part of the routes cannot be
walked around; elsewhere the robot can take the other way.
Grid convention, start / goal and origin are the same as :mod:`g1_maze_lab.terrains.maze`.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg, TerrainImporter, TerrainImporterCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.terrains.utils import create_prim_from_mesh
from isaaclab.utils import configclass

from .maze import DIRS, GENERATED, MazeTerrainCfg, generate_maze, parse_layout

_PENDING_ICE: list[tuple[float, float, float, float]] = []
"""Ice tiles of the sub-terrain generated last: (cx, cy, sx, sy) relative to its start-cell centre."""

_PENDING_SPIKES: list[trimesh.Trimesh] = []
"""Spike cones of the sub-terrain generated last, relative to its start-cell centre."""

HAZARDS: list[dict] = []
"""Hazards of every hazard maze built in this process (generation order):
{"gap": [(i, j, axis, width)], "ice": [(i, j)], "rough": [(i, j)], "ramp": [(i, j, axis, height)],
 "spike": [(i, j, axis, x0, y0, x1, y1)]} -- a spike bed's rectangle is relative to the start-cell centre."""


def _straight_cells(walls: np.ndarray) -> list[tuple[int, int, int]]:
    """Cells that are a straight corridor: open on two opposite sides, closed on the other two.
    Returns (i, j, axis) with axis 0 = corridor along x, 1 = along y."""
    out = []
    n = walls.shape[0]
    for i in range(n):
        for j in range(n):
            w = walls[i, j]
            if not w[0] and not w[2] and w[1] and w[3]:
                out.append((i, j, 0))
            elif not w[1] and not w[3] and w[0] and w[2]:
                out.append((i, j, 1))
    return out


def route_pair(walls: np.ndarray) -> tuple[list, list]:
    """The two routes of a maze with a start fork: (shorter, longer) lists of cells after the start cell.

    Distances are taken without passing through the start cell, so each route is the shortest way to the goal
    after leaving the start to the east or to the north.
    """
    n = walls.shape[0]
    goal = (n - 1, n - 1)
    dist = {goal: 0}
    queue = deque([goal])
    while queue:
        i, j = queue.popleft()
        for d, (di, dj) in enumerate(DIRS):
            nb = (i + di, j + dj)
            if not walls[i, j, d] and nb != (0, 0) and nb not in dist:
                dist[nb] = dist[(i, j)] + 1
                queue.append(nb)
    paths = []
    for first in ((1, 0), (0, 1)):
        path = [first]
        while dist[path[-1]] > 0:
            i, j = path[-1]
            path.append(next((i + di, j + dj) for d, (di, dj) in enumerate(DIRS)
                             if not walls[i, j, d] and dist.get((i + di, j + dj)) == dist[(i, j)] - 1))
        paths.append(path)
    return tuple(sorted(paths, key=len))


def route_turns(path: list) -> int:
    """Number of turns walking from the start cell along ``path``."""
    p = [(0, 0)] + list(path)
    return sum((b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]) for a, b, c in zip(p, p[1:], p[2:]))


def _slots(cells: list, straight: set, banned: set) -> list:
    """Straight cells along ``cells`` (in route order), never two next to each other."""
    out = []
    for x in cells:
        if x in straight and x not in banned and all(abs(x[0] - a) + abs(x[1] - b) > 1 for a, b in out):
            out.append(x)
    return out


def _routes_ok(walls: np.ndarray, routes: tuple[list, list], cfg: HazardMazeTerrainCfg) -> bool:
    """Short route clearly shorter, long route clearly twistier, and room for the hazards on each."""
    short, long = routes
    n = walls.shape[0]
    straight = {s[:2] for s in _straight_cells(walls)}
    banned = {(0, 0), (1, 0), (0, 1), (n - 1, n - 1)}
    short_slots = _slots([x for x in short if x not in long], straight, banned)
    n_short = len(short_slots)
    n_long = len(_slots([x for x in long if x not in short], straight, banned))
    # a hard hazard within the first `peek_cells` cells after the fork: a robot that walks in a little sees it
    near = short_slots and short.index(short_slots[0]) < cfg.peek_cells
    return (
        near and
        len(long) - len(short) >= cfg.min_route_gap
        and route_turns(long) >= route_turns(short) + cfg.min_extra_turns
        and n_short >= cfg.short_gaps + cfg.short_ramps + cfg.short_spikes
        and n_long >= cfg.long_ramps + 1
    )


def hazard_maze_terrain(difficulty: float, cfg: HazardMazeTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Maze walls + a floor with gaps / rough boxes, and the ice tiles (handed to :class:`HazardTerrainGenerator`)."""
    n, c = cfg.num_cells, cfg.cell_size
    seed = getattr(cfg, "seed", None)
    rng = np.random.default_rng(None if seed is None else seed + int(difficulty * 1e6))
    routes = None
    if cfg.layout is not None:
        walls = parse_layout(cfg.layout)
    else:
        for _ in range(cfg.max_tries):
            walls = generate_maze(n, rng, cfg.open_ratio_range[0])
            if not cfg.start_fork:
                break
            # open both sides of the start cell: one loop through the start = two routes to the goal
            for d in (0, 1):
                di, dj = DIRS[d]
                walls[0, 0, d] = walls[di, dj, (d + 2) % 4] = False
            routes = route_pair(walls)
            if not cfg.route_split or _routes_ok(walls, routes, cfg):
                break
    GENERATED.append({"open_ratio": cfg.open_ratio_range[0], "walls": walls})

    # pick the hazard cells: keep the start, its neighbours and the goal clean.
    # Ramps, spike beds and trenches are never next to each other: two ramps in a row make a V the robot cannot
    # climb out of, and the robot needs a cell to change lanes / square up to a trench.
    banned = {(0, 0), (1, 0), (0, 1), (n - 1, n - 1)}
    straight = {s[:2]: s for s in _straight_cells(walls) if s[:2] not in banned}
    placed = []

    def pick(cells, k):
        cand = [straight[x] for x in cells if x in straight]
        out = []
        for idx in rng.permutation(len(cand)):
            if len(out) == k:
                break
            i, j, _ = cand[idx]
            if all(abs(i - a) + abs(j - b) > 1 for a, b, _ in placed):
                out.append(cand[idx])
                placed.append(cand[idx])
        return out

    def free(cells, k):
        used = {p[:2] for p in placed} | banned | set(ice) | set(rough)
        cand = [x for x in cells if x not in used]
        return [cand[i] for i in rng.permutation(len(cand))[:k]]

    ice, rough = [], []
    if cfg.route_split and routes is not None:
        short, long = routes
        short_only = [x for x in short if x not in long]
        long_only = [x for x in long if x not in short]
        off_route = [(i, j) for i in range(n) for j in range(n) if (i, j) not in short and (i, j) not in long]
        # short route: hard (wide trench, steep ramps, spike beds); long route: easy (low ramps, boxes, ice)
        slots = _slots(short_only, set(straight), banned)
        kinds = ["gap"] * cfg.short_gaps + ["ramp"] * cfg.short_ramps + ["spike"] * cfg.short_spikes
        # the first slot (near the fork, see _routes_ok) always gets a hazard the scanner sees well: ramp or spikes
        chosen = [0] + sorted(1 + rng.permutation(len(slots) - 1)[: len(kinds) - 1])
        kinds = [kinds[i] for i in rng.permutation(len(kinds))]
        first = next(i for i, k in enumerate(kinds) if k in ("ramp", "spike"))
        kinds.insert(0, kinds.pop(first))
        kinds = kinds[: len(chosen)]
        hard = {k: [straight[slots[i]] for i, kk in zip(chosen, kinds) if kk == k] for k in ("gap", "ramp", "spike")}
        placed += hard["gap"] + hard["ramp"] + hard["spike"]
        gaps, ramps_s, spikes = hard["gap"], hard["ramp"], hard["spike"]
        ramps_l = pick(long_only, cfg.long_ramps)
        ice += free(short_only, cfg.short_ice)
        ice += free(long_only, cfg.long_ice)
        rough += free(long_only, cfg.long_rough)
        ice += free(off_route, cfg.extra_ice)
        rough += free(off_route, cfg.extra_rough)
        if cfg.short_rough_fill:  # every cell of the short route that is still plain floor gets loose boxes
            taken = {p[:2] for p in placed} | banned | set(ice) | set(rough)
            rough += [x for x in short_only if x not in taken]
        ramps = ramps_s + ramps_l
        gap_widths = rng.uniform(*cfg.short_gap_width_range, size=len(gaps))
        ramp_heights = np.concatenate(
            (rng.uniform(*cfg.short_ramp_height_range, size=len(ramps_s)),
             rng.uniform(*cfg.long_ramp_height_range, size=len(ramps_l)))
        )
    else:
        cells = list(straight)
        gaps = pick(cells, cfg.num_gaps)
        ramps = pick(cells, cfg.num_ramps)
        spikes = pick(cells, cfg.num_spikes)
        all_cells = [(i, j) for i in range(n) for j in range(n)]
        ice += free(all_cells, cfg.num_ice)
        rough += free(all_cells, cfg.num_rough)
        gap_widths = rng.uniform(*cfg.gap_width_range, size=len(gaps))
        ramp_heights = rng.uniform(*cfg.ramp_height_range, size=len(ramps))
    # spike bed: across the corridor from the wall on one side to `reach` past the middle, along it +-half
    n_cells_spk = len(spikes)
    spike_side = rng.choice((-1.0, 1.0), size=n_cells_spk)
    spike_reach = rng.uniform(*cfg.spike_reach_range, size=n_cells_spk)
    hw, half = 0.5 * (cfg.cell_size - cfg.wall_thickness), 0.5 * cfg.spike_length
    spike_rects = []
    for (i, j, a), sd, r in zip(spikes, spike_side, spike_reach):
        lo, hi = sorted((sd * hw, -sd * r))
        if a == 0:  # corridor along x: the bed spans y in [lo, hi]
            spike_rects.append((i * c - half, j * c + lo, i * c + half, j * c + hi))
        else:
            spike_rects.append((i * c + lo, j * c - half, i * c + hi, j * c + half))
    HAZARDS.append(
        {
            "gap": [(int(i), int(j), int(a), float(w)) for (i, j, a), w in zip(gaps, gap_widths)],
            "ice": [tuple(map(int, x)) for x in ice],
            "rough": [tuple(map(int, x)) for x in rough],
            "ramp": [(int(i), int(j), int(a), float(hh)) for (i, j, a), hh in zip(ramps, ramp_heights)],
            "spike": [(int(i), int(j), int(a), *map(float, r)) for (i, j, a), r in zip(spikes, spike_rects)],
            "routes": None if routes is None else {k: [tuple(map(int, x)) for x in r] for k, r in zip(("short", "long"), routes)},
        }
    )

    x0 = 0.5 * (cfg.size[0] - n * c)
    y0 = 0.5 * (cfg.size[1] - n * c)
    t, h, depth = cfg.wall_thickness, cfg.wall_height, cfg.pit_depth
    meshes = []

    def box(cx, cy, cz, sx, sy, sz):
        meshes.append(trimesh.creation.box((sx, sy, sz), trimesh.transformations.translation_matrix((cx, cy, cz))))

    # bottom of the pits, then the floor (top at z = 0) as one slab per cell + the margin around the maze
    box(0.5 * cfg.size[0], 0.5 * cfg.size[1], -depth - 0.5, cfg.size[0], cfg.size[1], 1.0)
    fz, ft = -0.5 * depth, depth
    box(0.5 * cfg.size[0], 0.5 * y0, fz, cfg.size[0], y0, ft)  # south margin
    box(0.5 * cfg.size[0], cfg.size[1] - 0.5 * y0, fz, cfg.size[0], y0, ft)  # north margin
    box(0.5 * x0, 0.5 * cfg.size[1], fz, x0, n * c, ft)  # west margin
    box(cfg.size[0] - 0.5 * x0, 0.5 * cfg.size[1], fz, x0, n * c, ft)  # east margin
    gap_of = {(i, j): (a, w) for (i, j, a), w in zip(gaps, gap_widths)}
    for i in range(n):
        for j in range(n):
            cx, cy = x0 + (i + 0.5) * c, y0 + (j + 0.5) * c
            if (i, j) not in gap_of:
                box(cx, cy, fz, c, c, ft)
                continue
            a, w = gap_of[(i, j)]
            part = 0.5 * (c - w)  # floor on both sides of the trench
            off = 0.5 * (c - part)
            if a == 0:  # corridor along x: trench across it (along y)
                box(cx - off, cy, fz, part, c, ft)
                box(cx + off, cy, fz, part, c, ft)
            else:
                box(cx, cy - off, fz, c, part, ft)
                box(cx, cy + off, fz, c, part, ft)

    # rough cells: loose boxes inside the corridor
    inner = c - t - 0.3
    for i, j in rough:
        cx, cy = x0 + (i + 0.5) * c, y0 + (j + 0.5) * c
        for _ in range(cfg.rough_boxes_per_cell):
            sx, sy = rng.uniform(0.15, 0.4, size=2)
            bh = rng.uniform(*cfg.rough_height_range)
            px, py = rng.uniform(-0.5 * inner, 0.5 * inner, size=2)
            box(cx + px, cy + py, 0.5 * bh, sx, sy, bh)

    # ramps: a trapezoid prism across the corridor (up `ramp_length`, flat top, down `ramp_length`)
    for (i, j, a), rh in zip(ramps, ramp_heights):
        cx, cy = x0 + (i + 0.5) * c, y0 + (j + 0.5) * c
        L = cfg.ramp_length
        prof = [(-0.5 * c, -0.01), (0.5 * c, -0.01), (0.5 * c - L, rh), (-0.5 * c + L, rh)]  # (along, z)
        pts = []
        for u, z in prof:
            for v in (-0.5 * c, 0.5 * c):
                pts.append((cx + u, cy + v, z) if a == 0 else (cx + v, cy + u, z))
        meshes.append(trimesh.convex.convex_hull(np.array(pts)))

    # spike beds: a grid of cones (separate red mesh, see HazardTerrainImporter), relative to the start-cell centre
    _PENDING_SPIKES.clear()
    pitch, rad = cfg.spike_pitch, 0.45 * cfg.spike_pitch
    for x_lo, y_lo, x_hi, y_hi in spike_rects:
        for px in np.arange(x_lo + rad, x_hi - rad + 1e-6, pitch):
            for py in np.arange(y_lo + rad, y_hi - rad + 1e-6, pitch):
                ch = rng.uniform(*cfg.spike_height_range)
                cone = trimesh.creation.cone(radius=rad, height=ch, sections=8)
                cone.apply_translation((px, py, 0.0))
                _PENDING_SPIKES.append(cone)

    # walls (down to the pit bottom so the trenches have side walls)
    wz, wh = 0.5 * (h - depth), h + depth
    for i in range(n):
        for j in range(n):
            if walls[i, j, 2]:
                box(x0 + i * c, y0 + (j + 0.5) * c, wz, t, c + t, wh)
            if walls[i, j, 3]:
                box(x0 + (i + 0.5) * c, y0 + j * c, wz, c + t, t, wh)
            if i == n - 1 and walls[i, j, 0]:
                box(x0 + n * c, y0 + (j + 0.5) * c, wz, t, c + t, wh)
            if j == n - 1 and walls[i, j, 1]:
                box(x0 + (i + 0.5) * c, y0 + n * c, wz, c + t, t, wh)

    # ice tiles, relative to the start-cell centre (the generator moves them into place)
    _PENDING_ICE.clear()
    for i, j in ice:
        _PENDING_ICE.append((i * c, j * c, c - t, c - t))

    origin = np.array([x0 + 0.5 * c, y0 + 0.5 * c, 0.0])
    return meshes, origin


class HazardTerrainGenerator(TerrainGenerator):
    """Terrain generator that also collects the ice tiles of every hazard maze (in world coordinates)."""

    last: HazardTerrainGenerator | None = None

    def __init__(self, cfg, device: str = "cpu"):
        self._ice_by_tile: dict[tuple[int, int], list] = {}
        self._spikes_by_tile: dict[tuple[int, int], list] = {}
        _PENDING_ICE.clear()
        _PENDING_SPIKES.clear()
        super().__init__(cfg, device)
        # terrain_origins are final (centred) here: ice = start-cell centre of its tile + relative offset
        self.ice_boxes = []
        for (row, col), tiles in self._ice_by_tile.items():
            ox, oy = self.terrain_origins[row, col, :2]
            self.ice_boxes += [(ox + cx, oy + cy, sx, sy) for cx, cy, sx, sy in tiles]
        self.spike_meshes = []
        for (row, col), cones in self._spikes_by_tile.items():
            shift = (*self.terrain_origins[row, col, :2], 0.0)
            self.spike_meshes += [m.copy().apply_translation(shift) for m in cones]
        HazardTerrainGenerator.last = self

    def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
        self._ice_by_tile[(row, col)] = list(_PENDING_ICE)
        self._spikes_by_tile[(row, col)] = list(_PENDING_SPIKES)
        _PENDING_ICE.clear()
        _PENDING_SPIKES.clear()
        super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)


class HazardTerrainImporter(TerrainImporter):
    """Terrain importer that adds the ice tiles as ``/World/ground/ice`` with a low-friction material
    and the spike beds as ``/World/ground/spikes`` (red, normal friction).

    MuJoCo (Newton) normally takes the *larger* friction of the two touching shapes, so the ice mesh gets a
    higher contact priority (``mjc:priority``): its friction is used whatever the feet have.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        gen = HazardTerrainGenerator.last
        if gen is None:
            return
        from pxr import Sdf

        if gen.spike_meshes:
            mu = cfg.spike_friction
            create_prim_from_mesh(
                "/World/ground/spikes",
                trimesh.util.concatenate(gen.spike_meshes),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.12, 0.1), roughness=0.4),
                physics_material=RigidBodyMaterialBaseCfg(static_friction=mu, dynamic_friction=mu),
            )
            prim = sim_utils.get_current_stage().GetPrimAtPath("/World/ground/spikes/mesh")
            prim.CreateAttribute("mjc:priority", Sdf.ValueTypeNames.Int).Set(1)
        if not gen.ice_boxes:
            return
        ice = trimesh.util.concatenate(
            [
                trimesh.creation.box((sx, sy, 0.02), trimesh.transformations.translation_matrix((cx, cy, 0.0)))
                for cx, cy, sx, sy in gen.ice_boxes
            ]
        )  # top 1 cm above the floor
        mu = cfg.ice_friction
        create_prim_from_mesh(
            "/World/ground/ice",
            ice,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 1.0), roughness=0.1),
            physics_material=RigidBodyMaterialBaseCfg(static_friction=mu, dynamic_friction=mu),
        )
        prim = sim_utils.get_current_stage().GetPrimAtPath("/World/ground/ice/mesh")
        prim.CreateAttribute("mjc:priority", Sdf.ValueTypeNames.Int).Set(1)


@configclass
class HazardMazeTerrainCfg(MazeTerrainCfg):
    """A perfect ``num_cells x num_cells`` maze with gaps, ice and rough cells (see the module docstring)."""

    function = hazard_maze_terrain

    num_cells: int = 7
    open_ratio_range: tuple[float, float] = (0.0, 0.0)
    """Share of inner walls knocked down (first value is used). 0 = perfect maze, one route."""
    start_fork: bool = True
    """Open both sides of the start cell: two routes to the goal (ignored for a fixed ``layout``)."""
    route_split: bool = True
    """With ``start_fork``: make one route short and hard, the other long, twisty and easy (the ``short_*`` and
    ``long_*`` settings below). Off: the ``num_*`` hazards are spread over the whole maze."""
    min_route_gap: int = 6
    """The long route is at least this many cells longer than the short one."""
    min_extra_turns: int = 2
    """... and has at least this many more turns."""
    max_tries: int = 2000
    peek_cells: int = 3
    """The first hard hazard of the short route is in one of its first ``peek_cells`` cells (the first is never one:
    it is next to the start), so walking a cell or two into the short route shows it."""
    """Mazes generated (from the same seed) until one fits the two conditions above."""

    short_gaps: int = 1
    short_gap_width_range: tuple[float, float] = (0.15, 0.22)
    short_ramps: int = 2
    short_ramp_height_range: tuple[float, float] = (0.24, 0.3)
    """Steeper than G1 Rough climbs (it gets stuck from ~0.24 m)."""
    short_spikes: int = 2
    short_ice: int = 1
    short_rough_fill: bool = True
    """Fill the short route's plain cells with loose boxes (rough floor all the way)."""
    long_ramps: int = 2
    long_ramp_height_range: tuple[float, float] = (0.1, 0.18)
    long_rough: int = 3
    long_ice: int = 2
    extra_ice: int = 2
    """Ice / rough cells off both routes (dead ends)."""
    extra_rough: int = 1

    num_gaps: int = 3
    gap_width_range: tuple[float, float] = (0.15, 0.3)
    """Trench width [m]."""
    pit_depth: float = 1.0

    num_ice: int = 5
    num_rough: int = 4
    rough_boxes_per_cell: int = 10
    rough_height_range: tuple[float, float] = (0.02, 0.08)

    num_ramps: int = 3
    ramp_height_range: tuple[float, float] = (0.18, 0.3)
    """Height of the flat top [m]."""
    ramp_length: float = 0.75
    """Length of each slope along the corridor [m] (0.3 m over 0.75 m = 22 deg, the steepest slope G1 Rough trained on)."""

    num_spikes: int = 3
    spike_reach_range: tuple[float, float] = (0.05, 0.2)
    """How far the bed reaches past the middle of the corridor [m]; the free lane is 0.7-0.9 m wide."""
    spike_length: float = 1.4
    """Length of the bed along the corridor [m]."""
    spike_pitch: float = 0.15
    spike_height_range: tuple[float, float] = (0.25, 0.35)


@configclass
class HazardTerrainGeneratorCfg(TerrainGeneratorCfg):
    class_type: type = HazardTerrainGenerator


@configclass
class HazardTerrainImporterCfg(TerrainImporterCfg):
    class_type: type = HazardTerrainImporter

    ice_friction: float = 0.2
    """Friction of the ice tiles (the normal floor has 1.0)."""
    spike_friction: float = 0.3
    """Friction of the spike cones (feet slide off the tips)."""
