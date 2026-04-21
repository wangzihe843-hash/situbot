#!/usr/bin/env python3
"""HSV-based color detector — lightweight fallback when YOLO-World is unavailable.

Extracted from github.com/umamtiti/sagittarius_openclaw_bridge (command_bridge.py).
Use when:
  - YOLO-World weights haven't downloaded yet
  - Running on a machine without GPU
  - You need faster detection (HSV is ~100x faster than YOLO)
  - Camera calibration uses vision_config.yaml with HSV bounds

This is NOT a replacement for ObjectDetector (YOLO-World). It only detects
pre-defined colors, not arbitrary objects. Use it as a fallback or for
quick hardware debugging.
"""

import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class HSVDetection:
    """Result from HSV color detection."""
    color: str
    pixel_x: float
    pixel_y: float
    world_x: float
    world_y: float
    area: float


class HSVColorDetector:
    """HSV color segmentation detector with pixel-to-world mapping.

    Loads color bounds and linear regression calibration from the same
    vision_config.yaml used by sagittarius_object_color_detector.
    """

    DEFAULT_COLORS = ("red", "green", "blue")

    def __init__(self, vision_config_path: str, min_area: float = 2500.0):
        self.min_area = min_area
        self.config = self._load_config(vision_config_path)
        self.k1, self.b1, self.k2, self.b2 = self._load_calibration()

    def _load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            content = yaml.safe_load(f) or {}
        if "LinearRegression" not in content:
            raise ValueError(f"vision_config missing LinearRegression: {path}")
        return content

    def _load_calibration(self):
        reg = self.config.get("LinearRegression", {})
        k1, b1 = float(reg.get("k1", 0)), float(reg.get("b1", 0))
        k2, b2 = float(reg.get("k2", 0)), float(reg.get("b2", 0))
        if abs(k1) < 1e-9 and abs(k2) < 1e-9:
            raise ValueError("LinearRegression not calibrated (k1≈0, k2≈0)")
        return k1, b1, k2, b2

    def _get_hsv_bounds(self, color: str):
        if color not in self.config:
            raise KeyError(f"Color '{color}' not in vision_config")
        c = self.config[color]
        lower = np.array([int(float(c["hmin"]) / 2), int(float(c["smin"])), int(float(c["vmin"]))], dtype=np.uint8)
        upper = np.array([int(float(c["hmax"]) / 2), int(float(c["smax"])), int(float(c["vmax"]))], dtype=np.uint8)
        return lower, upper

    def _find_largest(self, image, lower, upper):
        """Find largest color blob. Returns (area, cx, cy)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if lower[0] > upper[0]:  # hue wrap (e.g. red)
            m1 = cv2.inRange(hsv, np.array([0, lower[1], lower[2]]), upper)
            m2 = cv2.inRange(hsv, lower, np.array([180, upper[1], upper[2]]))
            mask = cv2.add(m1, m2)
        else:
            mask = cv2.inRange(hsv, lower, upper)

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        best_area, bx, by = 0.0, 0.0, 0.0
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(int)
            cx = sum(p[0] for p in box) / 4.0
            cy = sum(p[1] for p in box) / 4.0
            w = np.linalg.norm(box[0] - box[1])
            h = np.linalg.norm(box[0] - box[3])
            area = float(w * h)
            if area > best_area:
                best_area, bx, by = area, float(cx), float(cy)
        return best_area, bx, by

    def pixel_to_world(self, px: float, py: float) -> Tuple[float, float]:
        """Pixel → world using linear regression calibration."""
        return self.k1 * py + self.b1, self.k2 * px + self.b2

    def detect_all(self, image, colors=None) -> List[HSVDetection]:
        """Detect all specified colors in one frame."""
        colors = colors or self.DEFAULT_COLORS
        results = []
        for color in colors:
            lower, upper = self._get_hsv_bounds(color)
            area, px, py = self._find_largest(image, lower, upper)
            if area >= self.min_area:
                wx, wy = self.pixel_to_world(px, py)
                results.append(HSVDetection(color, px, py, wx, wy, area))
        results.sort(key=lambda d: d.area, reverse=True)
        return results

    def wait_stable(self, image_fn, colors=None, stable_n=5,
                    tolerance_px=10.0, timeout=60.0) -> Optional[HSVDetection]:
        """Wait for a stable detection across multiple frames.

        Args:
            image_fn: callable returning BGR image or None
            stable_n: consecutive stable frames required
            tolerance_px: max pixel drift between frames
            timeout: seconds (0 = wait forever)
        """
        colors = colors or self.DEFAULT_COLORS
        deadline = None if timeout <= 0 else time.time() + timeout
        counts = {c: 0 for c in colors}
        last = {c: None for c in colors}

        while deadline is None or time.time() < deadline:
            img = image_fn()
            if img is None:
                continue
            for color in colors:
                lower, upper = self._get_hsv_bounds(color)
                area, px, py = self._find_largest(img, lower, upper)
                if area < self.min_area:
                    counts[color] = 0
                    last[color] = None
                    continue
                prev = last[color]
                if prev and abs(px - prev[0]) <= tolerance_px and abs(py - prev[1]) <= tolerance_px:
                    counts[color] += 1
                else:
                    counts[color] = 1
                last[color] = (px, py)
                if counts[color] >= stable_n:
                    wx, wy = self.pixel_to_world(px, py)
                    return HSVDetection(color, px, py, wx, wy, area)
        return None
