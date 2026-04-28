#!/usr/bin/env python3
"""Pick-and-place sequence planner with collision avoidance.

Extended with height-aware transit collision checks.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from .collision_checker import CollisionChecker, ObjectFootprint

logger = logging.getLogger(__name__)


@dataclass
class PickPlaceAction:
    sequence_order: int
    action_type: str
    object_name: str
    instance_id: str
    x: float
    y: float
    z: float
    reason: str = ""
    zone: str = ""


class SequencePlanner:
    """Plans collision-free pick-and-place sequences."""

    ROLE_PRIORITY = {"prominent": 0, "accessible": 1, "peripheral": 2, "remove": 3}

    def __init__(self, workspace_bounds: dict, object_catalog: dict,
                 min_clearance: float = 0.02,
                 lift_height: float = 0.08):
        self.bounds = workspace_bounds
        self.catalog = object_catalog
        self.lift_height = lift_height
        self.collision_checker = CollisionChecker(workspace_bounds, min_clearance)

    def plan(self, current_positions: Dict[str, Tuple[float, float, float]],
             target_placements: list) -> List[PickPlaceAction]:
        def placement_key(placement):
            return getattr(placement, "grounded_instance_id", "") or placement.name

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

        all_current_footprints = self._build_current_footprints(
            graspable_targets, current_positions, placement_key
        )

        sorted_targets = self._sort_with_height_priority(
            graspable_targets, current_positions, placement_key,
            all_current_footprints
        )

        actions = []
        placed_footprints = []
        removed_names = set()
        seq = 0

        for placement in sorted_targets:
            obj_info = self.catalog.get(placement.name, {})
            dims = obj_info.get("dimensions", {"w": 0.10, "d": 0.10, "h": 0.05})
            obj_h = dims.get("h", 0.05)
            key = placement_key(placement)

            footprint = ObjectFootprint(
                name=key,
                cx=placement.x,
                cy=placement.y,
                width=dims.get("w", 0.10),
                depth=dims.get("d", 0.10),
                height=obj_h,
            )

            if self.collision_checker.check_collision(footprint, placed_footprints):
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

            cur = current_positions.get(key)
            if cur is None:
                cur = current_positions.get(placement.name)
            if cur is None:
                logger.warning(f"Object {placement.name} ({key}) missing current position at plan time, skipping")
                continue

            remaining_obstacles = [
                fp for fp in all_current_footprints
                if fp.name != key and fp.name not in removed_names
            ]
            transit_obstacles = remaining_obstacles + placed_footprints

            transit_collisions = self.collision_checker.check_transit_collision(
                from_xy=(cur[0], cur[1]),
                to_xy=(footprint.cx, footprint.cy),
                transit_height=self.lift_height,
                carried_width=dims.get("w", 0.10),
                obstacles=transit_obstacles,
            )
            if transit_collisions:
                safe_h = self.collision_checker.compute_safe_transit_height(
                    transit_obstacles
                )
                logger.warning(
                    f"Transit collision for {placement.name}: arm at "
                    f"{self.lift_height:.3f}m may hit [{', '.join(transit_collisions)}]. "
                    f"MoveIt planning scene should handle avoidance. "
                    f"Safe transit height: {safe_h:.3f}m"
                )

            actions.append(PickPlaceAction(
                sequence_order=seq,
                action_type="pick",
                object_name=placement.name,
                instance_id=key,
                x=cur[0], y=cur[1], z=cur[2],
                reason=f"Pick {placement.name} from current position",
            ))
            seq += 1
            removed_names.add(key)

            actions.append(PickPlaceAction(
                sequence_order=seq,
                action_type="place",
                object_name=placement.name,
                instance_id=key,
                x=footprint.cx, y=footprint.cy,
                z=self.bounds["z_surface"],
                reason=getattr(placement, "reason", ""),
            ))
            seq += 1

            placed_footprints.append(footprint)

        logger.info(f"Planned {len(actions)} actions for {len(sorted_targets)} objects")
        return actions

    def _build_current_footprints(
        self,
        targets: list,
        current_positions: dict,
        key_fn,
    ) -> List[ObjectFootprint]:
        footprints = []
        for p in targets:
            key = key_fn(p)
            cur = current_positions.get(key, current_positions.get(p.name))
            if cur is None:
                continue
            obj_info = self.catalog.get(p.name, {})
            dims = obj_info.get("dimensions", {"w": 0.10, "d": 0.10, "h": 0.05})
            footprints.append(ObjectFootprint(
                name=key,
                cx=cur[0],
                cy=cur[1],
                width=dims.get("w", 0.10),
                depth=dims.get("d", 0.10),
                height=dims.get("h", 0.05),
            ))
        return footprints

    def _sort_with_height_priority(
        self,
        targets: list,
        current_positions: dict,
        key_fn,
        current_footprints: List[ObjectFootprint],
    ) -> list:
        blocker_counts: Dict[str, int] = {}
        for p in targets:
            key = key_fn(p)
            cur = current_positions.get(key, current_positions.get(p.name, (0, 0, 0)))
            others = [fp for fp in current_footprints if fp.name != key]
            hits = self.collision_checker.check_transit_collision(
                from_xy=(cur[0], cur[1]),
                to_xy=(p.x, p.y),
                transit_height=self.lift_height,
                carried_width=0.10,
                obstacles=others,
            )
            for hit_name in hits:
                blocker_counts[hit_name] = blocker_counts.get(hit_name, 0) + 1

        def sort_key(p):
            priority = self.ROLE_PRIORITY.get(getattr(p, "role", ""), 2)
            key = key_fn(p)
            blocker_bonus = -blocker_counts.get(key, 0)
            cur = current_positions.get(key, current_positions.get(p.name, (0, 0, 0)))
            dist = ((cur[0] - p.x) ** 2 + (cur[1] - p.y) ** 2) ** 0.5
            return (priority, blocker_bonus, dist)

        return sorted(targets, key=sort_key)
