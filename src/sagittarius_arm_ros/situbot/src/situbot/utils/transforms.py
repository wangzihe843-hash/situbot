#!/usr/bin/env python3
"""Coordinate transforms between pixel, world, and robot frames."""

import numpy as np
import yaml
from typing import Dict, Optional, Tuple


# Default camera intrinsics (placeholder — calibrate for actual camera)
# Assumes a top-down camera looking at the table
DEFAULT_CAMERA_MATRIX = np.array([
    [600.0, 0.0, 320.0],   # fx, 0, cx
    [0.0, 600.0, 240.0],   # 0, fy, cy
    [0.0, 0.0, 1.0],
])

# Default camera-to-robot extrinsics (placeholder)
# Assumes camera is mounted directly above the table center
DEFAULT_CAMERA_TO_ROBOT = np.eye(4)


def load_linear_regression_config(config_path: str) -> Optional[Dict[str, float]]:
    """Load pixel-to-robot linear regression parameters from a YAML file."""
    if not config_path:
        return None

    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    params = data.get("LinearRegression", {})
    required = ("k1", "b1", "k2", "b2")
    if not all(key in params for key in required):
        raise ValueError(
            f"LinearRegression block missing one of {required} in {config_path}"
        )

    return {key: float(params[key]) for key in required}


def pixel_to_world_linear_regression(px: float, py: float, depth: float,
                                     linear_regression: Dict[str, float]) -> Tuple[float, float, float]:
    """Map pixel coordinates to robot coordinates using calibrated regression."""
    x = linear_regression["k1"] * py + linear_regression["b1"]
    y = linear_regression["k2"] * px + linear_regression["b2"]
    return (x, y, depth)


def pixel_to_world(px: float, py: float, depth: float,
                   image_shape: Tuple[int, ...],
                   camera_matrix: Optional[np.ndarray] = None,
                   workspace_bounds: Optional[dict] = None,
                   linear_regression: Optional[Dict[str, float]] = None) -> Tuple[float, float, float]:
    """Convert pixel coordinates to world (robot base) frame.

    For simplicity, uses a linear mapping from pixel space to workspace bounds
    when camera_matrix is not provided. Replace with proper projection when
    camera calibration is available.

    Args:
        px, py: Pixel coordinates (center of detected object).
        depth: Estimated depth in meters.
        image_shape: (H, W, C) of the image.
        camera_matrix: 3×3 camera intrinsic matrix.
        workspace_bounds: Dict with x_min, x_max, y_min, y_max.
        linear_regression: Dict with k1, b1, k2, b2 from calibration.

    Returns:
        (x, y, z) in robot base frame (meters).
    """
    H, W = image_shape[0], image_shape[1]

    if linear_regression is not None:
        return pixel_to_world_linear_regression(px, py, depth, linear_regression)

    if camera_matrix is not None:
        # Proper pinhole projection inverse
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        # Camera frame coordinates
        cam_x = (px - cx) * depth / fx
        cam_y = (py - cy) * depth / fy
        cam_z = depth

        # TODO: Apply camera-to-robot transform (extrinsics)
        # For now, assume camera aligned with robot frame
        robot_x = cam_z   # depth → forward
        robot_y = -cam_x  # right → robot y
        robot_z = -cam_y  # down → robot z (not used, table surface)

        return (robot_x, robot_y, depth)

    # Fallback: linear mapping from pixel to workspace bounds
    if workspace_bounds is None:
        workspace_bounds = {
            "x_min": 0.15, "x_max": 0.75,
            "y_min": -0.40, "y_max": 0.40,
        }

    b = workspace_bounds
    # py maps to x (top of image = far from person = x_max)
    x = b["x_min"] + (py / H) * (b["x_max"] - b["x_min"])
    # px maps to y (left of image = y_max, right = y_min)
    y = b["y_max"] - (px / W) * (b["y_max"] - b["y_min"])

    return (x, y, depth)


def world_to_robot(x: float, y: float, z: float,
                   transform_matrix: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
    """Transform world coordinates to robot base frame.

    Args:
        x, y, z: World coordinates.
        transform_matrix: 4×4 homogeneous transform. Identity if None.

    Returns:
        (x, y, z) in robot base frame.
    """
    if transform_matrix is None:
        return (x, y, z)

    point = np.array([x, y, z, 1.0])
    result = transform_matrix @ point
    return (float(result[0]), float(result[1]), float(result[2]))


def world_to_pixel(x: float, y: float, z: float,
                   image_shape: Tuple[int, ...],
                   workspace_bounds: Optional[dict] = None,
                   linear_regression: Optional[Dict[str, float]] = None) -> Tuple[int, int]:
    """Inverse of pixel_to_world (linear mapping version).

    Args:
        x, y, z: World coordinates.
        image_shape: (H, W, C).
        workspace_bounds: Dict with x_min, x_max, y_min, y_max.
        linear_regression: Dict with k1, b1, k2, b2 from calibration.

    Returns:
        (px, py) pixel coordinates.
    """
    H, W = image_shape[0], image_shape[1]

    if linear_regression is not None:
        k1 = linear_regression["k1"]
        b1 = linear_regression["b1"]
        k2 = linear_regression["k2"]
        b2 = linear_regression["b2"]
        if abs(k1) < 1e-12 or abs(k2) < 1e-12:
            raise ValueError("Linear regression slopes must be non-zero")
        py = int(round((x - b1) / k1))
        px = int(round((y - b2) / k2))
        return (px, py)

    if workspace_bounds is None:
        workspace_bounds = {
            "x_min": 0.15, "x_max": 0.75,
            "y_min": -0.40, "y_max": 0.40,
        }

    b = workspace_bounds
    py = int((x - b["x_min"]) / (b["x_max"] - b["x_min"]) * H)
    px = int((b["y_max"] - y) / (b["y_max"] - b["y_min"]) * W)
    return (px, py)
