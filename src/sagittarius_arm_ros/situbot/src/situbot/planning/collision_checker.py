#!/usr/bin/env python3
"""Simple 2D collision checker for tabletop object placement."""

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class ObjectFootprint:
    """2D footprint of an object on the table.

    Convention (matches objects.yaml and placement_optimizer):
      width  = catalog 'w' = left-right extent = Y-axis
      depth  = catalog 'd' = front-back extent = X-axis
    """
    name: str
    cx: float  # center x (front-back axis)
    cy: float  # center y (left-right axis)
    width: float  # extent along Y-axis (left-right, from catalog 'w')
    depth: float  # extent along X-axis (front-back, from catalog 'd')
    min_clearance: float = 0.02  # minimum gap to other objects


class CollisionChecker:
    """Checks and resolves placement collisions on a tabletop.

    Uses axis-aligned bounding boxes with configurable clearance.
    """

    def __init__(self, workspace_bounds: dict, min_clearance: float = 0.02):
        """
        Args:
            workspace_bounds: Dict with x_min, x_max, y_min, y_max.
            min_clearance: Minimum gap between objects (meters).
        """
        self.bounds = workspace_bounds
        self.min_clearance = min_clearance

    def check_collision(self, obj: ObjectFootprint,
                        placed: List[ObjectFootprint]) -> bool:
        """Check if obj collides with any already-placed object.

        Returns:
            True if collision detected, False if placement is safe.
        """
        for other in placed:
            if self._overlaps(obj, other):
                return True
        return False

    def check_in_bounds(self, obj: ObjectFootprint) -> bool:
        """Check if object footprint is within workspace bounds.

        Returns:
            True if within bounds.
        """
        b = self.bounds
        hx = obj.depth / 2   # half X-extent (from catalog 'd')
        hy = obj.width / 2   # half Y-extent (from catalog 'w')
        return (b["x_min"] <= obj.cx - hx and obj.cx + hx <= b["x_max"] and
                b["y_min"] <= obj.cy - hy and obj.cy + hy <= b["y_max"])

    def find_nearest_free(self, obj: ObjectFootprint,
                          placed: List[ObjectFootprint],
                          max_shift: float = 0.10,
                          step: float = 0.01) -> Optional[Tuple[float, float]]:
        """Find nearest collision-free position by spiraling outward.

        Args:
            obj: Object to place.
            placed: Already-placed objects.
            max_shift: Maximum search radius.
            step: Search grid step size.

        Returns:
            (x, y) of nearest free position, or None if no space found.
        """
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
                    min_clearance=obj.min_clearance,
                )
                if self.check_in_bounds(candidate) and not self.check_collision(candidate, placed):
                    dist = (dx ** 2 + dy ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (candidate.cx, candidate.cy)

        return best_pos

    def _overlaps(self, a: ObjectFootprint, b: ObjectFootprint) -> bool:
        """Check if two footprints overlap (with clearance).

        Uses AABB overlap: objects overlap iff they overlap in BOTH axes.
        depth = X-extent (catalog 'd'), width = Y-extent (catalog 'w').
        """
        gap = max(a.min_clearance, b.min_clearance)
        # half-extents per axis (depth→X, width→Y)
        hx_a = a.depth / 2 + gap / 2
        hy_a = a.width / 2 + gap / 2
        hx_b = b.depth / 2 + gap / 2
        hy_b = b.width / 2 + gap / 2

        return (abs(a.cx - b.cx) < hx_a + hx_b and
                abs(a.cy - b.cy) < hy_a + hy_b)
