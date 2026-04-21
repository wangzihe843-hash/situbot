#!/usr/bin/env python3
"""Three-stage situation reasoning pipeline with zone-based placement
and rejection sampling.

Chain: Situation → Need Inference → Object Relevance → Zone Assignment
       → Zone→Coord Mapping → L-BFGS-B Optimisation → (optional) Roundtrip
       rejection sampling.

Changes from v1 (inspired by V-CAGE, arXiv:2604.09036):
  - Stage 3 outputs qualitative zones instead of exact coordinates (§III-A2)
  - ZoneMapper converts zones to coordinates programmatically
  - PlacementOptimiser refines positions via L-BFGS-B (§III-A4)
  - Rejection sampling generates N candidates, keeps the best (§III-C)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

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
    role: str = ""       # prominent/accessible/peripheral/remove
    zone: str = ""       # qualitative zone name (new in v2)


@dataclass
class ArrangementResult:
    """Complete arrangement result from the reasoning pipeline."""
    situation: str
    placements: List[Placement]
    needs: Dict[str, Any]
    roles: Dict[str, Any]
    layout_description: str = ""
    reasoning_trace: str = ""  # full chain-of-thought for debugging
    rejection_sampling_info: Optional[Dict] = None  # stats from sampling


class SituationReasoner:
    """Orchestrates the 3-stage reasoning chain for situation-conditioned
    arrangement.

    Stage 1: Need Inference — what does the person need?
    Stage 2: Object Relevance — what role does each object play?
    Stage 3: Zone Assignment — which zone on the table for each object?
             (converted to coordinates by ZoneMapper + PlacementOptimiser)
    """

    def __init__(self, llm_client: DashScopeClient,
                 workspace_bounds: Dict[str, float],
                 object_catalog: List[Dict[str, Any]],
                 use_zone_placement: bool = True):
        """
        Args:
            llm_client: Configured DashScope client.
            workspace_bounds: Dict with keys x_min, x_max, y_min, y_max, z_surface.
            object_catalog: List of object dicts from objects.yaml.
            use_zone_placement: If True (default), use zone-based Stage 3.
                If False, use legacy exact-coordinate Stage 3 (for ablation).
        """
        self.llm = llm_client
        self.bounds = workspace_bounds
        self.catalog = {obj["name"]: obj for obj in object_catalog}
        self.use_zone_placement = use_zone_placement

        # Lazy-initialised (only when zone placement is used)
        self._zone_mapper = None
        self._placement_optimiser = None

    @property
    def zone_mapper(self):
        if self._zone_mapper is None:
            from situbot.planning.zone_mapper import ZoneMapper
            self._zone_mapper = ZoneMapper(self.bounds)
        return self._zone_mapper

    @property
    def placement_optimiser(self):
        if self._placement_optimiser is None:
            from situbot.planning.placement_optimizer import (
                PlacementOptimiser, SCIPY_AVAILABLE,
            )
            if SCIPY_AVAILABLE:
                self._placement_optimiser = PlacementOptimiser(self.bounds)
            else:
                self._placement_optimiser = None
        return self._placement_optimiser

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

        # Stage 3: Spatial Arrangement (zone-based or legacy)
        if self.use_zone_placement:
            logger.info("Stage 3: Computing zone-based arrangement...")
            placements = self._compute_zone_arrangement(situation, roles, available)
        else:
            logger.info("Stage 3: Computing legacy coordinate arrangement...")
            placements = self._compute_legacy_arrangement(situation, roles, available)

        trace_parts.append(
            f"=== PLACEMENTS ===\n"
            + json.dumps([vars(p) for p in placements], indent=2, ensure_ascii=False)
        )

        return ArrangementResult(
            situation=situation,
            placements=placements,
            needs=needs,
            roles=roles,
            layout_description="",  # filled by caller if needed
            reasoning_trace="\n\n".join(trace_parts),
        )

    def reason_with_rejection_sampling(
        self,
        situation: str,
        detected_objects: List[str],
        evaluator_fn: Callable[["ArrangementResult"], float],
        n_candidates: int = 3,
    ) -> ArrangementResult:
        """Generate N candidate arrangements, evaluate each, keep the best.

        Inspired by V-CAGE §III-C: rejection sampling for quality assurance.
        The evaluator_fn can be a Roundtrip Test or any scoring function.

        Args:
            situation: Natural language situation description.
            detected_objects: Names of objects detected on the table.
            evaluator_fn: Callable that takes an ArrangementResult and
                returns a quality score (higher is better). Could be
                roundtrip confidence, human rating, or composite metric.
            n_candidates: Number of candidate arrangements to generate.

        Returns:
            Best ArrangementResult by evaluator score.
        """
        if n_candidates <= 1:
            return self.reason(situation, detected_objects)

        logger.info(
            f"Rejection sampling: generating {n_candidates} candidates..."
        )

        # Stage 1 & 2 are run once and shared across candidates to reduce cost.
        # Note: with temperature > 0, results aren't strictly deterministic,
        # but the variation is small and re-running would just add cost.
        # The diversity comes from Stage 3 (spatial arrangement).
        needs = self._infer_needs(situation)
        available = [self.catalog[n] for n in detected_objects if n in self.catalog]
        roles = self._determine_relevance(situation, needs, available)

        candidates = []
        scores = []

        for i in range(n_candidates):
            logger.info(f"  Candidate {i+1}/{n_candidates}...")
            try:
                if self.use_zone_placement:
                    placements = self._compute_zone_arrangement(
                        situation, roles, available
                    )
                else:
                    placements = self._compute_legacy_arrangement(
                        situation, roles, available
                    )

                result = ArrangementResult(
                    situation=situation,
                    placements=placements,
                    needs=needs,
                    roles=roles,
                )

                score = evaluator_fn(result)
                candidates.append(result)
                scores.append(score)
                logger.info(f"    Score: {score:.3f}")

            except Exception as e:
                logger.error(f"    Candidate {i+1} failed: {e}")
                scores.append(-1.0)
                candidates.append(None)

        # Select best
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best = candidates[best_idx]

        if best is None:
            logger.error("All candidates failed, running single attempt...")
            return self.reason(situation, detected_objects)

        best.rejection_sampling_info = {
            "n_candidates": n_candidates,
            "scores": scores,
            "best_index": best_idx,
            "best_score": scores[best_idx],
        }

        logger.info(
            f"Rejection sampling: best candidate {best_idx+1} "
            f"(score={scores[best_idx]:.3f})"
        )
        return best

    # ── Stage implementations ───────────────────────────────────────

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

    def _compute_zone_arrangement(
        self, situation: str,
        roles: Dict[str, Any],
        objects: List[Dict],
    ) -> List[Placement]:
        """Stage 3 (zone-based): LLM outputs zones, we convert to coords."""

        messages = [
            {"role": "system", "content": prompts.SPATIAL_ARRANGEMENT_SYSTEM},
            {"role": "user", "content": prompts.SPATIAL_ARRANGEMENT_USER.format(
                situation=situation,
                roles_json=json.dumps(roles, indent=2, ensure_ascii=False),
            )},
        ]
        arrangement = self.llm.chat_json(messages)
        zone_assignments = arrangement.get("zone_assignments", [])

        # Convert zones to initial coordinates via ZoneMapper
        raw_placements = self.zone_mapper.map_placements(
            zone_assignments,
            self.catalog,
            self.bounds["z_surface"],
        )

        # Optimise positions via L-BFGS-B if available
        if self.placement_optimiser is not None and raw_placements:
            from situbot.planning.placement_optimizer import PlacementTarget
            targets = []
            for p in raw_placements:
                obj_info = self.catalog.get(p["name"], {})
                dims = obj_info.get("dimensions", {"w": 0.10, "d": 0.10})
                targets.append(PlacementTarget(
                    name=p["name"],
                    x=p["x"],
                    y=p["y"],
                    width=dims.get("w", 0.10),
                    depth=dims.get("d", 0.10),
                    graspable=obj_info.get("graspable", True),
                ))
            optimised = self.placement_optimiser.optimise(targets)

            # Merge optimised positions back
            opt_map = {o.name: o for o in optimised}
            for p in raw_placements:
                if p["name"] in opt_map:
                    p["x"] = opt_map[p["name"]].x
                    p["y"] = opt_map[p["name"]].y

        # Build Placement objects
        placements = []
        for p in raw_placements:
            placements.append(Placement(
                name=p["name"],
                x=p["x"],
                y=p["y"],
                z=p["z"],
                reason=p.get("reason", ""),
                role=p.get("role", "accessible"),
                zone=p.get("zone", ""),
            ))
        return placements

    def _compute_legacy_arrangement(
        self, situation: str,
        roles: Dict[str, Any],
        objects: List[Dict],
    ) -> List[Placement]:
        """Stage 3 (legacy): LLM outputs exact coordinates.

        Kept for ablation comparison against zone-based approach.
        """
        b = self.bounds
        x_mid = (b["x_min"] + b["x_max"]) / 2
        y_q1 = b["y_min"] + (b["y_max"] - b["y_min"]) * 0.25
        y_q3 = b["y_min"] + (b["y_max"] - b["y_min"]) * 0.75

        dim_lines = []
        for obj in objects:
            d = obj.get("dimensions", {})
            dim_lines.append(f"- {obj['name']}: {d.get('w', 0.1)}m wide × {d.get('d', 0.1)}m deep")
        object_dims = "\n".join(dim_lines)

        messages = [
            {"role": "system", "content": prompts.SPATIAL_ARRANGEMENT_SYSTEM_LEGACY},
            {"role": "user", "content": prompts.SPATIAL_ARRANGEMENT_USER_LEGACY.format(
                situation=situation,
                roles_json=json.dumps(roles, indent=2, ensure_ascii=False),
                x_min=b["x_min"], x_max=b["x_max"],
                y_min=b["y_min"], y_max=b["y_max"],
                z_surface=b["z_surface"],
                x_mid=x_mid, y_q1=y_q1, y_q3=y_q3,
                object_dims=object_dims,
            )},
        ]
        arrangement = self.llm.chat_json(messages)

        role_map = {r["name"]: r.get("role", "")
                    for r in roles.get("object_roles", [])}
        placements = []
        for p in arrangement.get("placements", []):
            try:
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
                logger.error(f"Failed to parse coordinates for {p.get('name')}: {e}")
                continue

        return placements
