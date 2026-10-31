# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Maze autopilot: height scan -> walls -> next cell -> velocity command (for every env at once).

Three layers, each one a function you can change on its own:

1. ``walls_from_height_scan``  (perception)  which sides of the current cell are blocked?
2. ``choose_direction``        (planning)    which neighbour cell do we go to next?
3. ``velocity_command``        (control)     v_x, v_y, w_z that drive the robot to that cell

The pretrained G1 walking policy then turns the velocity command into joint targets, exactly like the
keyboard in Week 3. Directions: 0 = +x (east), 1 = +y (north), 2 = -x (west), 3 = -y (south).
"""

from __future__ import annotations

import math

import torch

DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))


class MazeAutopilot:
    """Cell-by-cell maze solver for ``num_envs`` robots.

    Args:
        num_envs: number of robots.
        num_cells: maze cells per side.
        cell_size: cell width [m].
        strategy: ``"left"`` / ``"right"`` (wall following) or ``"flood"`` (flood fill on the walls seen so far).
        speed: forward speed command [m/s].
        turn_early: pick the next cell once the robot is this close to the centre of the cell it is walking to [m].
            0.35 = walk to the centre, then turn there. 0.5 = start the turn half a metre early and walk it as an arc.
        min_speed: forward speed during a sharp turn [m/s] (0.15 = nearly on the spot).
        max_turn: largest yaw rate command [rad/s].
        start, goal: start / goal cell (default ``(0, 0)`` / ``(n-1, n-1)``). Positions are always relative to the
            centre of cell ``(0, 0)``.
    """

    def __init__(
        self,
        num_envs: int,
        num_cells: int,
        cell_size: float,
        device: str,
        strategy: str = "left",
        speed: float = 0.6,
        turn_early: float = 0.35,
        min_speed: float = 0.15,
        max_turn: float = 0.8,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] | None = None,
    ):
        self.n, self.c, self.device = num_cells, cell_size, device
        self.start = tuple(int(v) for v in start)
        self.goal = (num_cells - 1, num_cells - 1) if goal is None else tuple(int(v) for v in goal)
        self.strategy, self.speed = strategy, speed
        self.turn_early, self.min_speed = turn_early, min_speed
        self.num_envs = num_envs
        self._dirs = torch.tensor(DIRS, device=device)
        # wall map: -1 unknown, 0 open, 1 wall (outer border is known from the start)
        self.walls = torch.full((num_envs, num_cells, num_cells, 4), -1, dtype=torch.int8, device=device)
        self.walls[:, -1, :, 0] = 1
        self.walls[:, :, -1, 1] = 1
        self.walls[:, 0, :, 2] = 1
        self.walls[:, :, 0, 3] = 1
        start_t = torch.tensor(self.start, dtype=torch.long, device=device)
        self.target = start_t.repeat(num_envs, 1)  # cell we are walking to
        self.origin = start_t.repeat(num_envs, 1)  # cell the last decision was made in
        self.heading = torch.zeros(num_envs, dtype=torch.long, device=device)  # last direction travelled
        self.need_decision = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.cells_walked = torch.zeros(num_envs, dtype=torch.long, device=device)
        # command shaping (see velocity_command)
        self.cmd = torch.zeros(num_envs, 3, device=device)
        self.max_turn, self.max_acc, self.max_ang_acc = max_turn, 0.8, 2.0
        self.debug = False

    # ------------------------------------------------------------------ helpers
    def cell_of(self, pos_xy: torch.Tensor) -> torch.Tensor:
        """Maze cell index of a position given relative to the centre of cell (0, 0)."""
        return torch.round(pos_xy / self.c).long().clamp(0, self.n - 1)

    def centre_of(self, cell: torch.Tensor) -> torch.Tensor:
        return cell.float() * self.c

    # ------------------------------------------------------------------ 1. perception
    def walls_from_height_scan(
        self, hits_xy: torch.Tensor, hits_z: torch.Tensor, cell: torch.Tensor, ground_z: float = 0.0
    ) -> torch.Tensor:
        """Return ``(num_envs, 4)`` bools: is there a wall on side d of ``cell``?

        ``hits_xy`` are the ray hit points (relative to the centre of cell (0, 0)) and ``hits_z`` their heights.
        A side is blocked when at least ``min_hits`` tall points (> 0.5 m) fall inside a thin box
        that sits on that side of the cell (0.3 m deep, 1.2 m long).
        """
        centre = self.centre_of(cell).unsqueeze(1)  # (N, 1, 2)
        rel = hits_xy - centre  # (N, R, 2)
        tall = hits_z > ground_z + 0.5
        out = torch.zeros(hits_xy.shape[0], 4, dtype=torch.bool, device=self.device)
        half = 0.5 * self.c
        for d, (dx, dy) in enumerate(DIRS):
            along = rel[..., 0] * dx + rel[..., 1] * dy  # distance towards side d
            across = rel[..., 0] * dy - rel[..., 1] * dx
            in_box = (torch.abs(along - half) < 0.15) & (torch.abs(across) < 0.6)
            out[:, d] = (in_box & tall).sum(dim=1) >= 3
        return out

    def walls_from_lidar(self, distances: torch.Tensor, angles_w: torch.Tensor, offset_xy: torch.Tensor):
        """Same output as :meth:`walls_from_height_scan`, from a horizontal 2D lidar (Exercise 7.4).

        ``distances``: (N, R) range of every ray, ``angles_w``: (N, R) world yaw of every ray,
        ``offset_xy``: (N, 2) robot position minus the current cell centre.
        """
        out = torch.zeros(distances.shape[0], 4, dtype=torch.bool, device=self.device)
        for d in range(4):
            yaw = d * math.pi / 2
            diff = torch.atan2(torch.sin(angles_w - yaw), torch.cos(angles_w - yaw))
            in_cone = torch.abs(diff) < math.radians(10)
            dist = torch.where(in_cone, distances, torch.full_like(distances, 1e6)).min(dim=1).values
            # distance from the robot to side d of the cell, corrected for the robot's offset from the centre
            to_side = 0.5 * self.c - (offset_xy[:, 0] * DIRS[d][0] + offset_xy[:, 1] * DIRS[d][1])
            out[:, d] = dist < to_side + 0.3
        return out

    # ------------------------------------------------------------------ 2. planning
    def flood_fill(self, walls: torch.Tensor) -> torch.Tensor:
        """Steps from every cell to the goal, treating unknown walls as open. (N, n, n, 4) -> (N, n, n)."""
        n = self.n
        big = n * n + 1
        dist = torch.full(walls.shape[:3], big, dtype=torch.long, device=self.device)
        dist[:, self.goal[0], self.goal[1]] = 0
        blocked = walls == 1
        for _ in range(n * n):
            new = dist.clone()
            # neighbour in direction d is reachable if side d is not blocked
            new[:, :-1, :] = torch.minimum(new[:, :-1, :], torch.where(blocked[:, :-1, :, 0], big, dist[:, 1:, :] + 1))
            new[:, 1:, :] = torch.minimum(new[:, 1:, :], torch.where(blocked[:, 1:, :, 2], big, dist[:, :-1, :] + 1))
            new[:, :, :-1] = torch.minimum(new[:, :, :-1], torch.where(blocked[:, :, :-1, 1], big, dist[:, :, 1:] + 1))
            new[:, :, 1:] = torch.minimum(new[:, :, 1:], torch.where(blocked[:, :, 1:, 3], big, dist[:, :, :-1] + 1))
            if torch.equal(new, dist):
                break
            dist = new
        return dist

    def choose_direction(
        self, cell: torch.Tensor, open_sides: torch.Tensor, heading: torch.Tensor, walls: torch.Tensor
    ) -> torch.Tensor:
        """Pick the next direction for each robot.

        Args:
            cell: (B, 2) current cell, open_sides: (B, 4) which sides are open,
            heading: (B,) direction the robot came in with, walls: (B, n, n, 4) wall map seen so far.
        """
        h = heading
        if self.strategy in ("left", "right"):
            # wall following: try left, straight, right, back (mirror for the right hand)
            turns = (1, 0, 3, 2) if self.strategy == "left" else (3, 0, 1, 2)
            choice = (h + 2) % 4
            for t in reversed(turns):  # the first open side in the preference list wins
                d = (h + t) % 4
                choice = torch.where(open_sides.gather(1, d.unsqueeze(1)).squeeze(1), d, choice)
            return choice
        if self.strategy == "flood":
            dist = self.flood_fill(walls)
            idx = torch.arange(len(h), device=self.device)
            best = torch.full_like(h, -1)
            best_cost = torch.full(h.shape, 10**6, dtype=torch.long, device=self.device)
            # straight first, so ties keep the robot walking straight (fewer turns)
            for t in (0, 1, 3, 2):
                d = (h + t) % 4
                nb = (cell + self._dirs[d]).clamp(0, self.n - 1)
                cost = dist[idx, nb[:, 0], nb[:, 1]]
                ok = open_sides[idx, d] & (cost < best_cost)
                best = torch.where(ok, d, best)
                best_cost = torch.where(ok, cost, best_cost)
            return torch.where(best >= 0, best, (h + 2) % 4)
        raise ValueError(f"unknown strategy {self.strategy!r}")

    # ------------------------------------------------------------------ 3. control
    def velocity_command(self, pos_xy: torch.Tensor, yaw: torch.Tensor, dt: float) -> torch.Tensor:
        """(v_x, v_y, w_z) in the robot frame that drives each robot to the centre of its target cell.

        Speed follows cos^2 of the heading error (slow down to ``min_speed`` for sharp turns instead of spinning
        on the spot) and both v_x and w_z are rate limited. With the G1 Rough policy the rate limit turned out
        not to be needed (MAZE_ACC=1e9,1e9: same 0 falls, ~2 s faster); it is a safety margin for other policies.
        """
        to_target = self.centre_of(self.target) - pos_xy
        yaw_err = torch.atan2(to_target[:, 1], to_target[:, 0]) - yaw
        yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err))
        # full speed when facing the target, down to min_speed when it is 90 deg or more to the side
        v_des = self.min_speed + (self.speed - self.min_speed) * torch.clamp(torch.cos(yaw_err), min=0.0) ** 2
        w_des = torch.clamp(1.5 * yaw_err, -self.max_turn, self.max_turn)
        # rate limits [m/s^2], [rad/s^2]
        self.cmd[:, 0] += torch.clamp(v_des - self.cmd[:, 0], -self.max_acc * dt, self.max_acc * dt)
        self.cmd[:, 2] += torch.clamp(w_des - self.cmd[:, 2], -self.max_ang_acc * dt, self.max_ang_acc * dt)
        return self.cmd.clone()

    # ------------------------------------------------------------------ main step
    def step(self, pos_xy: torch.Tensor, yaw: torch.Tensor, walls_now: torch.Tensor, dt: float) -> torch.Tensor:
        """Update the plan and return the velocity command. ``walls_now`` = sensed walls of the current cell."""
        cell = self.cell_of(pos_xy)
        near_centre = torch.linalg.norm(pos_xy - self.centre_of(cell), dim=1) < self.turn_early
        arrived = near_centre & (cell == self.target).all(dim=1)
        # a turn picked early sees the far wall of the cell only at the edge of the scan: if the side we are
        # heading out of turns out to be a wall while we are still in that cell, decide again
        still_there = (cell == self.origin).all(dim=1) & ~self.need_decision
        surprise = still_there & walls_now.gather(1, self.heading.unsqueeze(1)).squeeze(1)
        surprise &= self.walls[torch.arange(len(cell), device=self.device), cell[:, 0], cell[:, 1], self.heading] == 0
        decide = self.need_decision | arrived | surprise
        if decide.any():
            idx = torch.nonzero(decide).squeeze(1)
            # remember what we saw (and the same wall from the neighbour's side)
            for d, (di, dj) in enumerate(DIRS):
                ci, cj = cell[idx, 0], cell[idx, 1]
                val = walls_now[idx, d].to(torch.int8)
                self.walls[idx, ci, cj, d] = val
                ni, nj = ci + di, cj + dj
                inside = (ni >= 0) & (ni < self.n) & (nj >= 0) & (nj < self.n)
                k = idx[inside]
                self.walls[k, ni[inside], nj[inside], (d + 2) % 4] = val[inside]
            open_sides = self.walls[idx, cell[idx, 0], cell[idx, 1]] == 0
            if self.debug and decide[0]:
                print(
                    f"[PILOT] robot0 cell={cell[0].tolist()} sensed={walls_now[0].int().tolist()} "
                    f"open={open_sides[0].int().tolist()}",
                    flush=True,
                )
            d = self.choose_direction(cell[idx], open_sides, self.heading[idx], self.walls[idx])
            moved = ~self.need_decision[idx] & ~surprise[idx]
            self.cells_walked[idx] += moved.long()
            self.heading[idx] = d
            self.target[idx] = (cell[idx] + self._dirs[d]).clamp(0, self.n - 1)
            self.origin[idx] = cell[idx]
            self.need_decision[idx] = False
        return self.velocity_command(pos_xy, yaw, dt)

    def reset(self, env_ids: torch.Tensor):
        """Forget everything for the given robots (e.g. after a fall)."""
        self.walls[env_ids] = -1
        self.walls[env_ids, -1, :, 0] = 1
        self.walls[env_ids, :, -1, 1] = 1
        self.walls[env_ids, 0, :, 2] = 1
        self.walls[env_ids, :, 0, 3] = 1
        start = torch.tensor(self.start, dtype=torch.long, device=self.device)
        self.target[env_ids] = start
        self.origin[env_ids] = start
        self.heading[env_ids] = 0
        self.need_decision[env_ids] = True
        self.cells_walked[env_ids] = 0
        self.cmd[env_ids] = 0.0

    def map_ascii(self, env_id: int = 0) -> str:
        """Draw the walls robot ``env_id`` has seen: '-'/'|' wall, '?' never looked at."""
        w = self.walls[env_id].cpu()
        n = self.n

        def ch(v, wall):
            return wall if v == 1 else ("?" if v == -1 else " ")

        rows = []
        for j in range(n - 1, -1, -1):
            rows.append("+" + "".join(ch(w[i, j, 1], "-") + "+" for i in range(n)))
            line = ch(w[0, j, 2], "|")
            for i in range(n):
                label = "S" if (i, j) == self.start else "G" if (i, j) == self.goal else " "
                line += label + ch(w[i, j, 0], "|")
            rows.append(line)
        rows.append("+" + "".join(ch(w[i, 0, 3], "-") + "+" for i in range(n)))
        return "\n".join(rows)
