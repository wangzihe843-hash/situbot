#!/usr/bin/env python3
"""Three-stage situation reasoning pipeline.

Chain: Situation → Need Inference → Object Relevance → Spatial Arrangement
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .llm_client import DashScopeClient
from . import prompts

logger = logging.getLogger(__name__)


@dataclass
class Placement:
    """A single object placement decision."""
    name: str
    x: float
    y: float
    z: float
    reason: str
    role: str = ""  # prominent/accessible/peripheral/remove


@dataclass
class ArrangementResult:
    """Complete arrangement result from the reasoning pipeline."""
    situation: str
    placements: List[Placement]
    needs: Dict[str, Any]
    roles: Dict[str, Any]
    layout_description: str = ""
    reasoning_trace: str = ""  # full chain-of-thought for debugging


class SituationReasoner:
    """Orchestrates the 3-stage reasoning chain for situation-conditioned arrangement.

    Stage 1: Need Inference — what does the person need?
    Stage 2: Object Relevance — what role does each object play?
    Stage 3: Spatial Arrangement — exact (x, y) coordinates for each object.
    """

    def __init__(self, llm_client: DashScopeClient,
                 workspace_bounds: Dict[str, float],
                 object_catalog: List[Dict[str, Any]]):
        """
        Args:
            llm_client: Configured DashScope client.
            workspace_bounds: Dict with keys x_min, x_max, y_min, y_max, z_surface.
            object_catalog: List of object dicts from objects.yaml.
        """
        self.llm = llm_client
        self.bounds = workspace_bounds
        self.catalog = {obj["name"]: obj for obj in object_catalog}

    def reason(self, situation: str,
               detected_objects: List[str]) -> ArrangementResult:
        """Run the full 3-stage reasoning pipeline.

        Args:
            situation: Natural language situation description.
            detected_objects: Names of objects detected on the table.

        Returns:
            ArrangementResult with placements and full reasoning trace.
        """
        trace_parts = []

        # Stage 1: Need Inference
        logger.info("Stage 1: Inferring needs for situation...")
        needs = self._infer_needs(situation)
        trace_parts.append(f"=== NEEDS ===\n{json.dumps(needs, indent=2, ensure_ascii=False)}")

        # Stage 2: Object Relevance
        logger.info("Stage 2: Determining object relevance...")
        available = [self.catalog[n] for n in detected_objects if n in self.catalog]
        roles = self._determine_relevance(situation, needs, available)
        trace_parts.append(f"=== ROLES ===\n{json.dumps(roles, indent=2, ensure_ascii=False)}")

        # Stage 3: Spatial Arrangement
        logger.info("Stage 3: Computing spatial arrangement...")
        arrangement = self._compute_arrangement(situation, roles, available)
        trace_parts.append(f"=== ARRANGEMENT ===\n{json.dumps(arrangement, indent=2, ensure_ascii=False)}")

        # Build result with robust parsing
        placements = []
        role_map = {r["name"]: r.get("role", "") for r in roles.get("object_roles", [])}
        for p in arrangement.get("placements", []):
            try:
                # Force float conversion — handle LLM outputs like "0.25", "0.25m"
                x_val = float(str(p.get("x", 0.0)).replace('m', '').strip())
                y_val = float(str(p.get("y", 0.0)).replace('m', '').strip())
                z_val = float(str(p.get("z", self.bounds["z_surface"])).replace('m', '').strip())

                placements.append(Placement(
                    name=p.get("name", "unknown_object"),
                    x=x_val,
                    y=y_val,
                    z=z_val,
                    reason=p.get("reason", "No reason provided"),
                    role=role_map.get(p.get("name"), "accessible"),
                ))
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to parse coordinates for object {p.get('name')}: {e}")
                continue  # skip unparseable objects, keep pipeline alive

        return ArrangementResult(
            situation=situation,
            placements=placements,
            needs=needs,
            roles=roles,
            layout_description=arrangement.get("layout_description", ""),
            reasoning_trace="\n\n".join(trace_parts),
        )

    def _infer_needs(self, situation: str) -> Dict[str, Any]:
        """Stage 1: Infer human needs from situation description."""
        messages = [
            {"role": "system", "content": prompts.NEED_INFERENCE_SYSTEM},
            {"role": "user", "content": prompts.NEED_INFERENCE_USER.format(
                situation=situation
            )},
        ]
        return self.llm.chat_json(messages)

    def _determine_relevance(self, situation: str,
                             needs: Dict[str, Any],
                             objects: List[Dict]) -> Dict[str, Any]:
        """Stage 2: Determine each object's role given the situation needs."""
        messages = [
            {"role": "system", "content": prompts.OBJECT_RELEVANCE_SYSTEM},
            {"role": "user", "content": prompts.OBJECT_RELEVANCE_USER.format(
                situation=situation,
                needs_json=json.dumps(needs, indent=2, ensure_ascii=False),
                objects_list=prompts.format_objects_list(objects),
            )},
        ]
        return self.llm.chat_json(messages)

    def _compute_arrangement(self, situation: str,
                             roles: Dict[str, Any],
                             objects: List[Dict]) -> Dict[str, Any]:
        """Stage 3: Compute exact spatial coordinates for each object."""
        b = self.bounds
        x_mid = (b["x_min"] + b["x_max"]) / 2
        y_q1 = b["y_min"] + (b["y_max"] - b["y_min"]) * 0.25
        y_q3 = b["y_min"] + (b["y_max"] - b["y_min"]) * 0.75

        # Format object dimensions
        dim_lines = []
        for obj in objects:
            d = obj.get("dimensions", {})
            dim_lines.append(f"- {obj['name']}: {d.get('w', 0.1)}m wide × {d.get('d', 0.1)}m deep")
        object_dims = "\n".join(dim_lines)

        messages = [
            {"role": "system", "content": prompts.SPATIAL_ARRANGEMENT_SYSTEM},
            {"role": "user", "content": prompts.SPATIAL_ARRANGEMENT_USER.format(
                situation=situation,
                roles_json=json.dumps(roles, indent=2, ensure_ascii=False),
                x_min=b["x_min"], x_max=b["x_max"],
                y_min=b["y_min"], y_max=b["y_max"],
                z_surface=b["z_surface"],
                x_mid=x_mid, y_q1=y_q1, y_q3=y_q3,
                object_dims=object_dims,
            )},
        ]
        return self.llm.chat_json(messages)
