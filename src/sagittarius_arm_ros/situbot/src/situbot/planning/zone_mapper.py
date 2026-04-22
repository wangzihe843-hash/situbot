#!/usr/bin/env python3
"""Qualitative zone-based placement mapper.

Inspired by V-CAGE (arXiv:2604.09036) SIII-A2: instead of asking the LLM
for exact (x, y) coordinates, we let the LLM output qualitative zone names
and convert to coordinates programmatically.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ZONE_NAMES = [
    "front-left", "front-center", "front-right",
    "mid-left", "center", "mid-right",
    "back-left", "back-center", "back-right",
]

ZONE_ALIASES: Dict[str, str] = {
    "front-left": "front-left",
    "front-center": "front-center",
    "front-right": "front-right",
    "mid-left": "mid-left",
    "center": "center",
    "mid-right": "mid-right",
    "back-left": "back-left",
    "back-center": "back-center",
    "back-right": "back-right",
    "front left": "front-left",
    "front center": "front-center",
    "front right": "front-right",
    "middle left": "mid-left",
    "middle": "center",
    "middle right": "mid-right",
    "middle-center": "center",
    "rear left": "back-left",
    "rear center": "back-center",
    "rear right": "back-right",
    "far left": "back-left",
    "far center": "back-center",
    "far right": "back-right",
    "far corner": "back-right",
    "back corner": "back-right",
}

ROLE_DEFAULT_ZONES: Dict[str, List[str]] = {
    "prominent": ["front-center", "center"],
    "accessible": ["front-left", "front-right", "mid-left", "mid-right"],
    "peripheral": ["back-left", "back-center", "back-right"],
    "remove": ["back-right", "back-left"],
}


@dataclass
class ZoneSpec:
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
    """Converts qualitative zone names to (x, y) coordinates."""

    def __init__(self, workspace_bounds: Dict[str, float],
                 inner_padding: float = 0.02):
        self.bounds = workspace_bounds
        self.padding = inner_padding
        self.zones = self._build_zones()
        self._zone_counts: Dict[str, int] = {z: 0 for z in ZONE_NAMES}
        self._fallback_count: int = 0

    def reset(self):
        self._zone_counts = {z: 0 for z in ZONE_NAMES}
        self._fallback_count = 0

    def _build_zones(self) -> Dict[str, ZoneSpec]:
        b = self.bounds
        x_thirds = [
            b["x_min"],
            b["x_min"] + (b["x_max"] - b["x_min"]) / 3,
            b["x_min"] + 2 * (b["x_max"] - b["x_min"]) / 3,
            b["x_max"],
        ]
        y_thirds = [
            b["y_min"],
            b["y_min"] + (b["y_max"] - b["y_min"]) / 3,
            b["y_min"] + 2 * (b["y_max"] - b["y_min"]) / 3,
            b["y_max"],
        ]

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
        normalised = zone_str.strip().lower().replace("_", "-")

        if normalised in ZONE_ALIASES:
            return ZONE_ALIASES[normalised]

        for alias, canonical in ZONE_ALIASES.items():
            if alias in normalised:
                return canonical

        defaults = ROLE_DEFAULT_ZONES.get(role, ["center"])
        chosen = defaults[self._fallback_count % len(defaults)]
        self._fallback_count += 1
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
        zone = self.zones[zone_name]
        count = self._zone_counts[zone_name]
        self._zone_counts[zone_name] = count + 1

        p = self.padding
        x_lo = zone.x_min + p + object_depth / 2
        x_hi = zone.x_max - p - object_depth / 2
        y_lo = zone.y_min + p + object_width / 2
        y_hi = zone.y_max - p - object_width / 2

        x_lo = min(x_lo, zone.cx)
        x_hi = max(x_hi, zone.cx)
        y_lo = min(y_lo, zone.cy)
        y_hi = max(y_hi, zone.cy)

        offsets = [
            (0.0, 0.0),
            (-0.3, 0.3),
            (0.3, -0.3),
            (-0.3, -0.3),
            (0.3, 0.3),
        ]
        ox, oy = offsets[count % len(offsets)]

        x = zone.cx + ox * (x_hi - x_lo)
        y = zone.cy + oy * (y_hi - y_lo)

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
        self.reset()
        placements = []

        for assignment in zone_assignments:
            obj_name = assignment.get("name", "unknown")
            zone_raw = assignment.get("zone", "center")
            role = assignment.get("role", "accessible")
            reason = assignment.get("reason", "")

            zone_name = self.resolve_zone(zone_raw, role)

            obj_info = object_catalog.get(obj_name, {})
            dims = obj_info.get("dimensions", {"w": 0.10, "d": 0.10})
            obj_w = dims.get("w", 0.10)
            obj_d = dims.get("d", 0.10)

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
