#!/usr/bin/env python3
"""Scene description and visual grounding helpers for SituBot perception.

Member C owns the bridge between image detections and the LLM/planner stack.
This module keeps that bridge explicit: detections become instance-level
objects, a compact scene description, spatial relations, and grounding records
for LLM-generated placements.
"""

from dataclasses import dataclass, field
from math import hypot
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_WORKSPACE = {
    "x_min": 0.15,
    "x_max": 0.75,
    "y_min": -0.40,
    "y_max": 0.40,
    "z_min": 0.00,
    "z_max": 0.60,
    "z_surface": 0.00,
}


@dataclass
class SceneObject:
    """Instance-level object used by scene description and grounding."""

    instance_id: str
    name: str
    x: float
    y: float
    z: float
    confidence: float = 0.0
    width: float = 0.0
    depth: float = 0.0
    height: float = 0.0
    zone: str = "unknown"


@dataclass
class SceneSummary:
    """Human- and LLM-readable summary of the current table scene."""

    description: str
    relations: List[str] = field(default_factory=list)
    objects: List[SceneObject] = field(default_factory=list)


@dataclass
class GroundingInfo:
    """Result of grounding one LLM placement to one detected instance."""

    placement_name: str
    instance_id: str = ""
    grounded: bool = False
    note: str = ""


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from a dataclass/object/dict without caring which."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _set(obj: Any, name: str, value: Any) -> None:
    """Set a field when the object supports dynamic attributes."""
    try:
        setattr(obj, name, value)
    except Exception:
        pass


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower()).strip("_") or "object"


def assign_instance_ids(detections: Iterable[Any]) -> List[Any]:
    """Assign deterministic per-frame instance IDs such as mug_01, mug_02.

    IDs are intentionally simple and local to one perception snapshot. They are
    still enough to prevent downstream code from collapsing multiple objects
    with the same class name into a single dictionary entry.
    """
    counts: Dict[str, int] = {}
    out = []
    for det in detections:
        name = str(_get(det, "name", "object"))
        base = _safe_name(name)
        counts[base] = counts.get(base, 0) + 1
        instance_id = str(_get(det, "instance_id", "") or f"{base}_{counts[base]:02d}")
        _set(det, "instance_id", instance_id)
        out.append(det)
    return out


def zone_for_position(x: float, y: float, workspace: Optional[Dict[str, float]] = None) -> str:
    """Return a 3x3 table zone name for a robot-frame point.

    Convention follows the existing prompt/zone mapper:
    x_min is front/near the person, x_max is back/far, y_min is the person's
    right, and y_max is the person's left.
    """
    b = {**DEFAULT_WORKSPACE, **(workspace or {})}
    x_span = max(1e-9, b["x_max"] - b["x_min"])
    y_span = max(1e-9, b["y_max"] - b["y_min"])
    x_ratio = (x - b["x_min"]) / x_span
    y_ratio = (y - b["y_min"]) / y_span

    if x_ratio < 1.0 / 3.0:
        row = "front"
    elif x_ratio < 2.0 / 3.0:
        row = "mid"
    else:
        row = "back"

    if y_ratio < 1.0 / 3.0:
        col = "right"
    elif y_ratio < 2.0 / 3.0:
        col = "center"
    else:
        col = "left"

    return "center" if row == "mid" and col == "center" else f"{row}-{col}"


def _scene_objects(detections: Iterable[Any],
                   workspace: Optional[Dict[str, float]]) -> List[SceneObject]:
    objects = []
    for det in detections:
        x = float(_get(det, "x", 0.0))
        y = float(_get(det, "y", 0.0))
        zone = zone_for_position(x, y, workspace)
        _set(det, "zone", zone)
        objects.append(SceneObject(
            instance_id=str(_get(det, "instance_id", "")),
            name=str(_get(det, "name", "object")),
            x=x,
            y=y,
            z=float(_get(det, "z", 0.0)),
            confidence=float(_get(det, "confidence", 0.0)),
            width=float(_get(det, "width", 0.0)),
            depth=float(_get(det, "depth", 0.0)),
            height=float(_get(det, "height", 0.0)),
            zone=zone,
        ))
    return objects


def build_scene_description(
    detections: Iterable[Any],
    workspace: Optional[Dict[str, float]] = None,
    near_distance: float = 0.12,
    relation_axis_threshold: float = 0.08,
    max_relations: int = 24,
) -> SceneSummary:
    """Build a compact scene description and relation list from detections."""
    objects = _scene_objects(detections, workspace)
    if not objects:
        return SceneSummary(
            description="No objects detected on the tabletop.",
            relations=[],
            objects=[],
        )

    object_bits = [
        f"{obj.instance_id} ({obj.name}) at {obj.zone} "
        f"[x={obj.x:.3f}, y={obj.y:.3f}, z={obj.z:.3f}, conf={obj.confidence:.2f}]"
        for obj in objects
    ]

    relations: List[str] = []
    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            if len(relations) >= max_relations:
                break
            dx = a.x - b.x
            dy = a.y - b.y
            distance = hypot(dx, dy)
            if distance <= near_distance:
                relations.append(f"{a.instance_id} is near {b.instance_id}")
            if abs(dx) >= relation_axis_threshold:
                front, back = (a, b) if a.x < b.x else (b, a)
                relations.append(f"{front.instance_id} is in front of {back.instance_id}")
            if abs(dy) >= relation_axis_threshold:
                left, right = (a, b) if a.y > b.y else (b, a)
                relations.append(f"{left.instance_id} is left of {right.instance_id}")
            if len(relations) >= max_relations:
                break

    description = (
        f"Detected {len(objects)} tabletop object(s). "
        + "; ".join(object_bits)
    )
    if relations:
        description += ". Key spatial relations: " + "; ".join(relations[:8])

    return SceneSummary(description=description, relations=relations, objects=objects)


def ground_placements_to_scene(
    placements: Iterable[Any],
    detections: Iterable[Any],
    distance_tolerance: float = 0.20,
) -> List[GroundingInfo]:
    """Ground LLM placement decisions to detected object instances.

    The LLM usually outputs catalog names (e.g. "mug") rather than instance
    IDs. We match by name, choosing the nearest detected instance to the target
    pose when there are duplicates.
    """
    detected = _scene_objects(detections, DEFAULT_WORKSPACE)
    by_name: Dict[str, List[SceneObject]] = {}
    by_id: Dict[str, SceneObject] = {}
    for obj in detected:
        by_name.setdefault(obj.name, []).append(obj)
        by_id[obj.instance_id] = obj

    result: List[GroundingInfo] = []
    used_ids = set()

    for placement in placements:
        name = str(_get(placement, "name", ""))
        explicit_id = str(_get(placement, "grounded_instance_id", "") or "")

        if explicit_id and explicit_id in by_id:
            used_ids.add(explicit_id)
            result.append(GroundingInfo(
                placement_name=name,
                instance_id=explicit_id,
                grounded=True,
                note="Matched explicit detected instance ID.",
            ))
            continue

        candidates = [obj for obj in by_name.get(name, []) if obj.instance_id not in used_ids]
        if not candidates:
            result.append(GroundingInfo(
                placement_name=name,
                grounded=False,
                note=f"No detected instance for object name '{name}'.",
            ))
            continue

        target_x = float(_get(placement, "x", candidates[0].x))
        target_y = float(_get(placement, "y", candidates[0].y))
        chosen = min(candidates, key=lambda obj: hypot(obj.x - target_x, obj.y - target_y))
        used_ids.add(chosen.instance_id)
        distance = hypot(chosen.x - target_x, chosen.y - target_y)

        if len(candidates) == 1:
            note = "Matched unique detected instance by object name."
        else:
            note = "Multiple detected instances share this name; chose nearest to target pose."
        if distance > distance_tolerance:
            note += f" Target is {distance:.2f}m from current detection; verify calibration."

        result.append(GroundingInfo(
            placement_name=name,
            instance_id=chosen.instance_id,
            grounded=True,
            note=note,
        ))

    return result
