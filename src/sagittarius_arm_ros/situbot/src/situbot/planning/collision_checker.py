#!/usr/bin/env python3
"""Collision checker for tabletop object placement (2D footprint + height)."""

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class ObjectFootprint:
    """Footprint of an object on the table.

    Convention (matches objects.yaml and placement_optimizer):
      width  = catalog 'w' = left-right extent = Y-axis
      depth  = catalog 'd' = front-back extent = X-axis
      height = catalog 'h' = vertical extent   = Z-axis
    """
    name: str
    cx: float
    cy: float
    width: float
    depth: float
    height: float = 0.0
    min_clearance: float = 0.02


class CollisionChecker:
    """Checks and resolves placement collisions on a tabletop.

    Uses axis-aligned bounding boxes with configurable clearance.
    Supports both 2D footprint overlap and 3D transit path checks.
    """

    def __init__(self, workspace_bounds: dict, min_clearance: float = 0.02):
        self.bounds = workspace_bounds
        self.min_clearance = min_clearance

    def check_collision(self, obj: ObjectFootprint,
                        placed: List[ObjectFootprint]) -> bool:
        for other in placed:
            if self._overlaps(obj, other):
                return True
        return False

    def check_in_bounds(self, obj: ObjectFootprint) -> bool:
        b = self.bounds
        hx = obj.depth / 2
        hy = obj.width / 2
        return (b["x_min"] <= obj.cx - hx and obj.cx + hx <= b["x_max"] and
                b["y_min"] <= obj.cy - hy and obj.cy + hy <= b["y_max"])

    def check_transit_collision(self, from_xy: Tuple[float, float],
                                to_xy: Tuple[float, float],
                                transit_height: float,
                                carried_width: float,
                                obstacles: List[ObjectFootprint],
                                clearance: float = 0.03) -> List[str]:
        colliding = []
        for obs in obstacles:
            if obs.height < (transit_height - clearance):
                continue
            if self._swept_path_intersects(from_xy, to_xy, carried_width, obs):
                colliding.append(obs.name)
        return colliding

    def compute_safe_transit_height(self, obstacles: List[ObjectFootprint],
                                     clearance: float = 0.05) -> float:
        if not obstacles:
            return clearance
        max_h = max(obs.height for obs in obstacles)
        return max_h + clearance

    def find_nearest_free(self, obj: ObjectFootprint,
                          placed: List[ObjectFootprint],
                          max_shift: float = 0.10,
                          step: float = 0.01) -> Optional[Tuple[float, float]]:
        import numpy as np

        best_pos = None
        best_dist = float("inf")

        for dx in np.arange(-max_shift, max_shift + step, step):
            for dy in np.arange(-max_shift, max_shift + step, step):
                candidate = ObjectFootprint(
                    name=obj.name,
                    cx=obj.cx + dx,
                    cy=obj.cy + dy,
                    width=obj.width,
                    depth=obj.depth,
                    height=obj.height,
                    min_clearance=obj.min_clearance,
                )
                if self.check_in_bounds(candidate) and not self.check_collision(candidate, placed):
                    dist = (dx ** 2 + dy ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (candidate.cx, candidate.cy)

        return best_pos

    def _overlaps(self, a: ObjectFootprint, b: ObjectFootprint) -> bool:
        gap = max(a.min_clearance, b.min_clearance)
        hx_a = a.depth / 2 + gap / 2
        hy_a = a.width / 2 + gap / 2
        hx_b = b.depth / 2 + gap / 2
        hy_b = b.width / 2 + gap / 2

        return (abs(a.cx - b.cx) < hx_a + hx_b and
                abs(a.cy - b.cy) < hy_a + hy_b)

    @staticmethod
    def _swept_path_intersects(from_xy: Tuple[float, float],
                                to_xy: Tuple[float, float],
                                carried_width: float,
                                obs: ObjectFootprint) -> bool:
        import math

        ax, ay = from_xy
        bx, by = to_xy
        dx = bx - ax
        dy = by - ay
        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq < 1e-10:
            dist = math.sqrt((obs.cx - ax) ** 2 + (obs.cy - ay) ** 2)
        else:
            t = max(0.0, min(1.0,
                ((obs.cx - ax) * dx + (obs.cy - ay) * dy) / seg_len_sq))
            proj_x = ax + t * dx
            proj_y = ay + t * dy
            dist = math.sqrt((obs.cx - proj_x) ** 2 + (obs.cy - proj_y) ** 2)

        obs_radius = max(obs.width, obs.depth) / 2
        threshold = carried_width / 2 + obs_radius + obs.min_clearance

        return dist < threshold
