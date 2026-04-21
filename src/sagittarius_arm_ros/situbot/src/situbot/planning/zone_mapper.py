#!/usr/bin/env python3
"""Qualitative zone-based placement mapper.

Inspired by V-CAGE (arXiv:2604.09036) §III-A2: instead of asking the LLM
for exact (x, y) coordinates (which LLMs are notoriously bad at), we let
the LLM output qualitative zone names and convert to coordinates
programmatically.

Zone layout (from the person's perspective, sitting at x_min side):

    y_max (left)                              y_min (right)
    ┌──────────┬──────────┬──────────┐  x_max (far)
    │ back-    │ back-    │ back-    │
    │ left     │ center   │ right   │
    ├──────────┼──────────┼──────────┤
    │ mid-     │ center   │ mid-    │
    │ left     │          │ right   │
    ├──────────┼──────────┼──────────┤  x_min (near/person)
    │ front-   │ front-   │ front-  │
    │ left     │ center   │ right   │
    └──────────┴──────────┴──────────┘

Each zone has a nominal center. Objects assigned to the same zone are
spread within it using sub-grid offsets to avoid stacking.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Zone definitions ──────────────────────────────────────────────────

ZONE_NAMES = [
    "front-left", "front-center", "front-right",
    "mid-left", "center", "mid-right",
    "back-left", "back-center", "back-right",
]

# Semantic aliases the LLM might produce → canonical zone name
ZONE_ALIASES: Dict[str, str] = {
    # exact matches
    "front-left": "front-left",
    "front-center": "front-center",
    "front-right": "front-right",
    "mid-left": "mid-left",
    "center": "center",
    "mid-right": "mid-right",
    "back-left": "back-left",
    "back-center": "back-center",
    "back-right": "back-right",
    # common LLM variations
    "front left": "front-left",
    "front center": "front-center",
    "front right": "front-right",
    "middle left": "mid-left",
    "middle": "center",
    "middle right": "mid-right",
    "middle center": "center",
    "middle-left": "mid-left",
    "middle-center": "center",
    "middle-right": "mid-right",
    "rear left": "back-left",
    "rear center": "back-center",
    "rear right": "back-right",
    "rear-left": "back-left",
    "rear-center": "back-center",
    "rear-right": "back-right",
    "far left": "back-left",
    "far center": "back-center",
    "far right": "back-right",
    "far-left": "back-left",
    "far-center": "back-center",
    "far-right": "back-right",
    "far corner": "back-right",       # default "remove" zone
    "far-corner": "back-right",
    "back corner": "back-right",
    "back-corner": "back-right",
}

# Role → preferred zones (used as fallback if LLM outputs invalid zone)
ROLE_DEFAULT_ZONES: Dict[str, List[str]] = {
    "prominent": ["front-center", "center"],
    "accessible": ["front-left", "front-right", "mid-left", "mid-right"],
    "peripheral": ["back-left", "back-center", "back-right"],
    "remove": ["back-right", "back-left"],
}


@dataclass
class ZoneSpec:
    """A 2D rectangular zone on the table."""
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def cx(self) -> float:
        return (self.x_min + self.x_max) / 2

    @property
    def cy(self) -> float:
        return (self.y_min + self.y_max) / 2


class ZoneMapper:
    """Converts qualitative zone names to (x, y) coordinates.

    Divides the workspace into a 3×3 grid and assigns coordinates
    within the appropriate cell, spreading multiple objects in the
    same zone to avoid overlap.
    """

    def __init__(self, workspace_bounds: Dict[str, float],
                 inner_padding: float = 0.02):
        """
        Args:
            workspace_bounds: Dict with x_min, x_max, y_min, y_max, z_surface.
            inner_padding: Padding inside each zone boundary (meters).
        """
        self.bounds = workspace_bounds
        self.padding = inner_padding
        self.zones = self._build_zones()

        # Track how many objects are placed in each zone for spreading
        self._zone_counts: Dict[str, int] = {z: 0 for z in ZONE_NAMES}

    def reset(self):
        """Reset placement counters (call before each new arrangement)."""
        self._zone_counts = {z: 0 for z in ZONE_NAMES}

    def _build_zones(self) -> Dict[str, ZoneSpec]:
        """Build the 3×3 zone grid from workspace bounds."""
        b = self.bounds
        # x-axis: front (near person) = x_min, back (far) = x_max
        x_thirds = [
            b["x_min"],
            b["x_min"] + (b["x_max"] - b["x_min"]) / 3,
            b["x_min"] + 2 * (b["x_max"] - b["x_min"]) / 3,
            b["x_max"],
        ]
        # y-axis: right = y_min, left = y_max
        y_thirds = [
            b["y_min"],
            b["y_min"] + (b["y_max"] - b["y_min"]) / 3,
            b["y_min"] + 2 * (b["y_max"] - b["y_min"]) / 3,
            b["y_max"],
        ]

        # Grid layout:  row 0 = front (x_min), row 2 = back (x_max)
        #               col 0 = right (y_min), col 2 = left (y_max)
        grid = [
            ["front-right",  "front-center", "front-left"],
            ["mid-right",    "center",       "mid-left"],
            ["back-right",   "back-center",  "back-left"],
        ]

        zones = {}
        for row in range(3):
            for col in range(3):
                name = grid[row][col]
                zones[name] = ZoneSpec(
                    name=name,
                    x_min=x_thirds[row],
                    x_max=x_thirds[row + 1],
                    y_min=y_thirds[col],
                    y_max=y_thirds[col + 1],
                )
        return zones

    def resolve_zone(self, zone_str: str,
                     role: str = "accessible") -> str:
        """Resolve a potentially fuzzy zone string to a canonical zone name.

        Args:
            zone_str: Zone name from LLM output (may be fuzzy).
            role: Object role, used for fallback if zone is unrecognised.

        Returns:
            Canonical zone name from ZONE_NAMES.
        """
        normalised = zone_str.strip().lower().replace("_", "-")

        # Direct match
        if normalised in ZONE_ALIASES:
            return ZONE_ALIASES[normalised]

        # Partial match: check if any alias is contained in the string
        for alias, canonical in ZONE_ALIASES.items():
            if alias in normalised:
                return canonical

        # Fallback to role-based default
        defaults = ROLE_DEFAULT_ZONES.get(role, ["center"])
        chosen = defaults[self._zone_counts.get(defaults[0], 0) % len(defaults)]
        logger.warning(
            f"Unrecognised zone '{zone_str}' for role '{role}', "
            f"falling back to '{chosen}'"
        )
        return chosen

    def zone_to_coordinates(
        self, zone_name: str,
        object_width: float = 0.10,
        object_depth: float = 0.10,
    ) -> Tuple[float, float]:
        """Convert a zone name to (x, y) coordinates.

        Spreads multiple objects within the same zone using a sub-grid
        pattern to avoid stacking.

        Args:
            zone_name: Canonical zone name.
            object_width: Object width in meters (for clearance).
            object_depth: Object depth in meters (for clearance).

        Returns:
            (x, y) in workspace coordinates.
        """
        zone = self.zones[zone_name]
        count = self._zone_counts[zone_name]
        self._zone_counts[zone_name] = count + 1

        # Usable area within zone (after padding)
        p = self.padding
        x_lo = zone.x_min + p + object_depth / 2
        x_hi = zone.x_max - p - object_depth / 2
        y_lo = zone.y_min + p + object_width / 2
        y_hi = zone.y_max - p - object_width / 2

        # Clamp if zone is too small
        x_lo = min(x_lo, zone.cx)
        x_hi = max(x_hi, zone.cx)
        y_lo = min(y_lo, zone.cy)
        y_hi = max(y_hi, zone.cy)

        # Sub-grid spreading pattern for multiple objects in same zone
        # Pattern: center, then offset positions
        offsets = [
            (0.0, 0.0),        # 1st object: zone center
            (-0.3, 0.3),       # 2nd: front-left of zone
            (0.3, -0.3),       # 3rd: back-right of zone
            (-0.3, -0.3),      # 4th: front-right
            (0.3, 0.3),        # 5th: back-left
        ]
        ox, oy = offsets[count % len(offsets)]

        # Map offset (in [-0.5, 0.5] normalised zone coords) to real coords
        x = zone.cx + ox * (x_hi - x_lo)
        y = zone.cy + oy * (y_hi - y_lo)

        # Final clamp to workspace bounds
        b = self.bounds
        hw, hd = object_width / 2, object_depth / 2
        x = max(b["x_min"] + hd, min(b["x_max"] - hd, x))
        y = max(b["y_min"] + hw, min(b["y_max"] - hw, y))

        return (x, y)

    def map_placements(
        self,
        zone_assignments: List[Dict],
        object_catalog: Dict,
        z_surface: float,
    ) -> List[Dict]:
        """Convert a list of zone assignments to placements with coordinates.

        Args:
            zone_assignments: List of dicts from LLM, each with keys:
                name, zone, role, reason.
            object_catalog: Dict of object_name → object info (with dimensions).
            z_surface: Table surface height.

        Returns:
            List of placement dicts with name, x, y, z, reason, role.
        """
        self.reset()
        placements = []

        for assignment in zone_assignments:
            obj_name = assignment.get("name", "unknown")
            zone_raw = assignment.get("zone", "center")
            role = assignment.get("role", "accessible")
            reason = assignment.get("reason", "")

            # Resolve zone
            zone_name = self.resolve_zone(zone_raw, role)

            # Get object dimensions
            obj_info = object_catalog.get(obj_name, {})
            dims = obj_info.get("dimensions", {"w": 0.10, "d": 0.10})
            obj_w = dims.get("w", 0.10)
            obj_d = dims.get("d", 0.10)

            # Convert to coordinates
            x, y = self.zone_to_coordinates(zone_name, obj_w, obj_d)

            placements.append({
                "name": obj_name,
                "x": x,
                "y": y,
                "z": z_surface,
                "reason": reason,
                "role": role,
                "zone": zone_name,
            })

        return placements
