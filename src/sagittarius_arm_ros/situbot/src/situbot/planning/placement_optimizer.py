#!/usr/bin/env python3
"""Collision-free placement optimisation using L-BFGS-B.

Inspired by V-CAGE (arXiv:2604.09036) §III-A4: formulates placement
refinement as a constrained optimisation problem that balances three
objectives:
  1. Collision penalty  — penalise pairwise AABB overlap + enforce min gap
  2. Displacement cost  — penalise deviation from initial target positions
  3. Boundary penalty   — penalise positions outside workspace bounds

Uses scipy.optimize.minimize with L-BFGS-B and multiple restarts.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try importing scipy; mark as unavailable if missing so the module
# degrades gracefully to the greedy fallback in CollisionChecker.
try:
    from scipy.optimize import minimize as scipy_minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available — L-BFGS-B optimiser disabled, "
                   "falling back to greedy nudge. Install: pip install scipy")


@dataclass
class PlacementTarget:
    """An object to be placed, with initial target and physical dims."""
    name: str
    x: float          # initial target x from zone mapper
    y: float          # initial target y from zone mapper
    width: float      # extent along y-axis
    depth: float      # extent along x-axis
    graspable: bool = True


@dataclass
class OptimisedPlacement:
    """Result of the optimisation for one object."""
    name: str
    x: float
    y: float
    displaced: float   # distance from original target


class PlacementOptimiser:
    """L-BFGS-B based collision-free placement refinement.

    Follows V-CAGE Eq. 1:
        min_x  J(x) = λ_c · J_coll(x) + λ_d · J_disp(x) + λ_b · J_bnd(x)

    Where:
        J_coll: sum of pairwise overlap penalties (smooth hinge on gap)
        J_disp: weighted sum of squared displacements from targets
        J_bnd:  penalty for positions outside workspace
    """

    def __init__(
        self,
        workspace_bounds: Dict[str, float],
        min_clearance: float = 0.02,
        lambda_coll: float = 100.0,
        lambda_disp: float = 1.0,
        lambda_bnd: float = 50.0,
        num_restarts: int = 5,
        max_iter: int = 200,
    ):
        """
        Args:
            workspace_bounds: Dict with x_min, x_max, y_min, y_max.
            min_clearance: Minimum gap between objects (meters).
            lambda_coll: Weight for collision penalty.
            lambda_disp: Weight for displacement penalty.
            lambda_bnd: Weight for boundary penalty.
            num_restarts: Number of random restarts for L-BFGS-B.
            max_iter: Max iterations per optimisation run.
        """
        self.bounds = workspace_bounds
        self.min_clearance = min_clearance
        self.lambda_coll = lambda_coll
        self.lambda_disp = lambda_disp
        self.lambda_bnd = lambda_bnd
        self.num_restarts = num_restarts
        self.max_iter = max_iter

    def optimise(
        self,
        targets: List[PlacementTarget],
    ) -> List[OptimisedPlacement]:
        """Optimise placements to be collision-free while staying near targets.

        Args:
            targets: List of PlacementTarget with initial positions.

        Returns:
            List of OptimisedPlacement with refined positions.
        """
        if not targets:
            return []

        if not SCIPY_AVAILABLE:
            logger.warning("scipy unavailable, returning targets unchanged")
            return [
                OptimisedPlacement(name=t.name, x=t.x, y=t.y, displaced=0.0)
                for t in targets
            ]

        n = len(targets)
        # Initial guess: flatten target positions [x0, y0, x1, y1, ...]
        x0 = np.array([[t.x, t.y] for t in targets]).flatten()
        target_pos = x0.copy()

        # Object dimensions for collision computation
        widths = np.array([t.width for t in targets])
        depths = np.array([t.depth for t in targets])

        # Area-based displacement weights (larger objects penalised more,
        # following V-CAGE: "larger objects are penalized more to preserve
        # the layout backbone")
        areas = widths * depths
        disp_weights = areas / (areas.mean() + 1e-8)

        # Variable bounds for L-BFGS-B
        b = self.bounds
        var_bounds = []
        for t in targets:
            var_bounds.append((b["x_min"] + t.depth / 2, b["x_max"] - t.depth / 2))
            var_bounds.append((b["y_min"] + t.width / 2, b["y_max"] - t.width / 2))

        def cost(x_flat):
            positions = x_flat.reshape(n, 2)
            total = 0.0

            # J_coll: pairwise collision penalty
            for i in range(n):
                for j in range(i + 1, n):
                    dx = abs(positions[i, 0] - positions[j, 0])
                    dy = abs(positions[i, 1] - positions[j, 1])
                    # Required separation
                    req_x = (depths[i] + depths[j]) / 2 + self.min_clearance
                    req_y = (widths[i] + widths[j]) / 2 + self.min_clearance
                    # Overlap in each axis (positive = overlapping)
                    overlap_x = max(0.0, req_x - dx)
                    overlap_y = max(0.0, req_y - dy)
                    # Only penalise if overlapping in BOTH axes (AABB overlap)
                    if overlap_x > 0 and overlap_y > 0:
                        total += self.lambda_coll * (overlap_x ** 2 + overlap_y ** 2)

            # J_disp: displacement from targets (area-weighted)
            for i in range(n):
                dx = positions[i, 0] - target_pos[2 * i]
                dy = positions[i, 1] - target_pos[2 * i + 1]
                total += self.lambda_disp * disp_weights[i] * (dx ** 2 + dy ** 2)

            # J_bnd: boundary penalty (should be handled by var_bounds,
            # but add soft penalty for robustness)
            for i in range(n):
                x_i, y_i = positions[i]
                hw, hd = widths[i] / 2, depths[i] / 2
                total += self.lambda_bnd * (
                    max(0, b["x_min"] + hd - x_i) ** 2 +
                    max(0, x_i + hd - b["x_max"]) ** 2 +
                    max(0, b["y_min"] + hw - y_i) ** 2 +
                    max(0, y_i + hw - b["y_max"]) ** 2
                )

            return total

        # Multi-restart optimisation
        best_result = None
        best_cost = float("inf")

        for restart in range(self.num_restarts):
            if restart == 0:
                init = x0.copy()
            else:
                # Add small random perturbation
                noise = np.random.uniform(-0.03, 0.03, size=x0.shape)
                init = x0 + noise
                # Clamp to bounds
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

        # Extract optimised positions
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
                    f"({t.x:.3f},{t.y:.3f}) → ({final_pos[i,0]:.3f},{final_pos[i,1]:.3f}) "
                    f"[{displaced*100:.1f}cm]"
                )

        logger.info(
            f"Placement optimisation complete: cost={best_cost:.4f}, "
            f"{sum(1 for p in placements if p.displaced > 0.01)}/{n} objects nudged"
        )
        return placements
