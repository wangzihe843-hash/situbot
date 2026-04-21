#!/usr/bin/env python3
"""Pick-and-place sequence planner with collision avoidance."""

import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from .collision_checker import CollisionChecker, ObjectFootprint

logger = logging.getLogger(__name__)


@dataclass
class PickPlaceAction:
    """A single pick-and-place action."""
    sequence_order: int
    action_type: str  # "pick" or "place"
    object_name: str
    instance_id: str
    x: float
    y: float
    z: float
    reason: str = ""
    zone: str = ""  # qualitative zone (v2, for logging/debugging)


class SequencePlanner:
    """Plans collision-free pick-and-place sequences.

    Strategy:
    1. Sort objects by priority (prominent first, remove last)
    2. For each object, pick from current position, place at target
    3. Resolve collisions by nudging target positions
    4. Generate alternating pick/place action sequence

    Objects are moved one at a time: pick → place → pick → place → ...
    The planner uses a greedy nearest-first approach within each priority tier.
    """

    ROLE_PRIORITY = {"prominent": 0, "accessible": 1, "peripheral": 2, "remove": 3}

    def __init__(self, workspace_bounds: dict, object_catalog: dict,
                 min_clearance: float = 0.02):
        """
        Args:
            workspace_bounds: Dict with x_min, x_max, y_min, y_max, z_surface.
            object_catalog: Dict of object_name → object info (with dimensions).
            min_clearance: Minimum gap between placed objects.
        """
        self.bounds = workspace_bounds
        self.catalog = object_catalog
        self.collision_checker = CollisionChecker(workspace_bounds, min_clearance)

    def plan(self, current_positions: Dict[str, Tuple[float, float, float]],
             target_placements: list) -> List[PickPlaceAction]:
        """Plan a sequence of pick-and-place actions.

        Args:
            current_positions: Dict of object_name → (x, y, z) current positions.
            target_placements: List of Placement objects (from SituationReasoner).

        Returns:
            Ordered list of PickPlaceAction.
        """
        def placement_key(placement):
            return getattr(placement, "grounded_instance_id", "") or placement.name

        # Filter to graspable objects only
        graspable_targets = []
        for p in target_placements:
            obj_info = self.catalog.get(p.name, {})
            if not obj_info.get("graspable", True):
                logger.info(f"Skipping non-graspable object: {p.name}")
                continue
            key = placement_key(p)
            if key not in current_positions and p.name not in current_positions:
                logger.warning(f"Object {p.name} ({key}) not in current positions, skipping")
                continue
            graspable_targets.append(p)

        # Sort by role priority, then by distance from current to target (nearest first)
        def sort_key(p):
            priority = self.ROLE_PRIORITY.get(getattr(p, "role", ""), 2)
            cur = current_positions.get(placement_key(p), current_positions.get(p.name, (0, 0, 0)))
            dist = ((cur[0] - p.x) ** 2 + (cur[1] - p.y) ** 2) ** 0.5
            return (priority, dist)

        sorted_targets = sorted(graspable_targets, key=sort_key)

        # Generate pick-place sequence with collision resolution
        actions = []
        placed_footprints = []
        seq = 0

        for placement in sorted_targets:
            obj_info = self.catalog.get(placement.name, {})
            dims = obj_info.get("dimensions", {"w": 0.10, "d": 0.10})

            # Check collision at target position
            footprint = ObjectFootprint(
                name=placement.name,
                cx=placement.x,
                cy=placement.y,
                width=dims.get("w", 0.10),
                depth=dims.get("d", 0.10),
            )

            if self.collision_checker.check_collision(footprint, placed_footprints):
                # Try to find nearest free position
                free_pos = self.collision_checker.find_nearest_free(
                    footprint, placed_footprints
                )
                if free_pos:
                    logger.info(f"Nudged {placement.name} from ({placement.x:.3f}, {placement.y:.3f}) "
                                f"to ({free_pos[0]:.3f}, {free_pos[1]:.3f})")
                    footprint.cx, footprint.cy = free_pos
                else:
                    logger.warning(f"Could not find collision-free position for {placement.name}")
                    continue

            key = placement_key(placement)
            cur = current_positions.get(key, current_positions[placement.name])

            # Pick action
            actions.append(PickPlaceAction(
                sequence_order=seq,
                action_type="pick",
                object_name=placement.name,
                instance_id=key,
                x=cur[0], y=cur[1], z=cur[2],
                reason=f"Pick {placement.name} from current position",
            ))
            seq += 1

            # Place action
            actions.append(PickPlaceAction(
                sequence_order=seq,
                action_type="place",
                object_name=placement.name,
                instance_id=key,
                x=footprint.cx, y=footprint.cy,
                z=self.bounds["z_surface"],
                reason=placement.reason,
            ))
            seq += 1

            placed_footprints.append(footprint)

        logger.info(f"Planned {len(actions)} actions for {len(sorted_targets)} objects")
        return actions
