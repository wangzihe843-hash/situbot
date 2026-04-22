#!/usr/bin/env python3
"""Collision-free placement optimisation using L-BFGS-B.

Inspired by V-CAGE (arXiv:2604.09036) SIII-A4: formulates placement
refinement as a constrained optimisation problem.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.optimize import minimize as scipy_minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available - L-BFGS-B optimiser disabled, "
                   "falling back to greedy nudge. Install: pip install scipy")


@dataclass
class PlacementTarget:
    name: str
    x: float
    y: float
    width: float
    depth: float
    height: float = 0.0
    graspable: bool = True


@dataclass
class OptimisedPlacement:
    name: str
    x: float
    y: float
    displaced: float


class PlacementOptimiser:
    """L-BFGS-B based collision-free placement refinement."""

    def __init__(
        self,
        workspace_bounds: Dict[str, float],
        min_clearance: float = 0.02,
        lambda_coll: float = 100.0,
        lambda_disp: float = 1.0,
        lambda_bnd: float = 50.0,
        lambda_transit: float = 10.0,
        num_restarts: int = 5,
        max_iter: int = 200,
        lift_height: float = 0.08,
    ):
        self.bounds = workspace_bounds
        self.min_clearance = min_clearance
        self.lambda_coll = lambda_coll
        self.lambda_disp = lambda_disp
        self.lambda_bnd = lambda_bnd
        self.lambda_transit = lambda_transit
        self.num_restarts = num_restarts
        self.max_iter = max_iter
        self.lift_height = lift_height

    def optimise(self, targets: List[PlacementTarget]) -> List[OptimisedPlacement]:
        if not targets:
            return []

        if not SCIPY_AVAILABLE:
            logger.warning("scipy unavailable, returning targets unchanged")
            return [
                OptimisedPlacement(name=t.name, x=t.x, y=t.y, displaced=0.0)
                for t in targets
            ]

        n = len(targets)
        x0 = np.array([[t.x, t.y] for t in targets]).flatten()
        target_pos = x0.copy()

        widths = np.array([t.width for t in targets])
        depths = np.array([t.depth for t in targets])
        heights = np.array([t.height for t in targets])

        areas = widths * depths
        disp_weights = areas / (areas.mean() + 1e-8)

        b = self.bounds
        var_bounds = []
        for t in targets:
            var_bounds.append((b["x_min"] + t.depth / 2, b["x_max"] - t.depth / 2))
            var_bounds.append((b["y_min"] + t.width / 2, b["y_max"] - t.width / 2))

        def cost(x_flat):
            positions = x_flat.reshape(n, 2)
            total = 0.0

            for i in range(n):
                for j in range(i + 1, n):
                    dx = abs(positions[i, 0] - positions[j, 0])
                    dy = abs(positions[i, 1] - positions[j, 1])
                    req_x = (depths[i] + depths[j]) / 2 + self.min_clearance
                    req_y = (widths[i] + widths[j]) / 2 + self.min_clearance
                    overlap_x = max(0.0, req_x - dx)
                    overlap_y = max(0.0, req_y - dy)
                    if overlap_x > 0 and overlap_y > 0:
                        total += self.lambda_coll * (overlap_x ** 2 + overlap_y ** 2)

            for i in range(n):
                dx = positions[i, 0] - target_pos[2 * i]
                dy = positions[i, 1] - target_pos[2 * i + 1]
                total += self.lambda_disp * disp_weights[i] * (dx ** 2 + dy ** 2)

            for i in range(n):
                x_i, y_i = positions[i]
                hw, hd = widths[i] / 2, depths[i] / 2
                total += self.lambda_bnd * (
                    max(0, b["x_min"] + hd - x_i) ** 2 +
                    max(0, x_i + hd - b["x_max"]) ** 2 +
                    max(0, b["y_min"] + hw - y_i) ** 2 +
                    max(0, y_i + hw - b["y_max"]) ** 2
                )

            if self.lambda_transit > 0:
                for i in range(n):
                    if heights[i] <= self.lift_height:
                        continue
                    for j in range(n):
                        if i == j:
                            continue
                        dx = abs(positions[i, 0] - positions[j, 0])
                        dy = abs(positions[i, 1] - positions[j, 1])
                        proximity = max(0.0, 0.15 - min(dx, dy))
                        if proximity > 0:
                            excess_h = heights[i] - self.lift_height
                            total += self.lambda_transit * proximity * excess_h

            return total

        best_result = None
        best_cost = float("inf")

        for restart in range(self.num_restarts):
            if restart == 0:
                init = x0.copy()
            else:
                noise = np.random.uniform(-0.03, 0.03, size=x0.shape)
                init = x0 + noise
                for i in range(len(init)):
                    lo, hi = var_bounds[i]
                    init[i] = np.clip(init[i], lo, hi)

            result = scipy_minimize(
                cost,
                init,
                method="L-BFGS-B",
                bounds=var_bounds,
                options={"maxiter": self.max_iter, "ftol": 1e-10},
            )

            if result.fun < best_cost:
                best_cost = result.fun
                best_result = result

        final_pos = best_result.x.reshape(n, 2)
        placements = []
        for i, t in enumerate(targets):
            displaced = float(np.sqrt(
                (final_pos[i, 0] - t.x) ** 2 + (final_pos[i, 1] - t.y) ** 2
            ))
            placements.append(OptimisedPlacement(
                name=t.name,
                x=float(final_pos[i, 0]),
                y=float(final_pos[i, 1]),
                displaced=displaced,
            ))
            if displaced > 0.01:
                logger.info(
                    f"Optimiser nudged {t.name}: "
                    f"({t.x:.3f},{t.y:.3f}) -> ({final_pos[i,0]:.3f},{final_pos[i,1]:.3f}) "
                    f"[{displaced*100:.1f}cm]"
                )

        logger.info(
            f"Placement optimisation complete: cost={best_cost:.4f}, "
            f"{sum(1 for p in placements if p.displaced > 0.01)}/{n} objects nudged"
        )
        return placements
