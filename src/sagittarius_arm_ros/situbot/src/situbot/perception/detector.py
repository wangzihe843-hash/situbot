#!/usr/bin/env python3
"""Open-vocabulary object detection for SituBot.

The detector is intentionally strict about model files. On RB8 and other
offline robot deployments, a hidden first-run download can make perception look
like it has frozen. Pass local weights through ROS params or environment
variables instead.
"""

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DetectedObject:
    """A single detected object with position and dimensions."""
    name: str
    x: float
    y: float
    z: float
    confidence: float
    width: float = 0.0
    depth: float = 0.0
    height: float = 0.0
    bbox_pixels: Tuple[int, int, int, int] = (0, 0, 0, 0)
    instance_id: str = ""
    pixel_x: float = 0.0
    pixel_y: float = 0.0
    zone: str = ""


class ObjectDetector:
    """Open-vocabulary detector for catalog objects.

    Supported backends:
    - yolo_world: ultralytics YOLO-World with a local .pt file.
    - grounding_dino: GroundingDINO with local config and checkpoint files.
    """

    def __init__(self, model_name: str = "yolo_world",
                 confidence_threshold: float = 0.3,
                 nms_threshold: float = 0.5,
                 object_names: Optional[List[str]] = None,
                 object_catalog: Optional[List[Dict]] = None,
                 workspace_bounds: Optional[Dict[str, float]] = None,
                 coordinate_mapping_mode: str = "workspace_linear",
                 linear_regression: Optional[Dict[str, float]] = None,
                 min_bbox_area: float = 400.0,
                 max_detections_per_class: int = 3,
                 model_weights: str = "",
                 allow_model_download: bool = False,
                 grounding_dino_config: str = "",
                 grounding_dino_weights: str = "",
                 grounding_dino_text_threshold: float = 0.25,
                 grounding_dino_device: str = "cpu"):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.object_catalog = {
            obj["name"]: obj for obj in (object_catalog or []) if "name" in obj
        }
        catalog_names = list(self.object_catalog.keys())
        self.object_names = object_names or catalog_names
        self.known_names = set(self.object_names)
        self.workspace_bounds = workspace_bounds
        self.coordinate_mapping_mode = coordinate_mapping_mode
        self.linear_regression = linear_regression
        self.min_bbox_area = float(min_bbox_area)
        self.max_detections_per_class = max(1, int(max_detections_per_class))

        self.model_weights = model_weights
        self.allow_model_download = self._as_bool(allow_model_download)
        self.grounding_dino_config = grounding_dino_config
        self.grounding_dino_weights = grounding_dino_weights
        self.grounding_dino_text_threshold = float(grounding_dino_text_threshold)
        self.grounding_dino_device = grounding_dino_device
        self._grounding_transform = None
        self._grounding_prompt = self._build_grounding_prompt(self.object_names)

        self.model = None
        self._model_load_failed = False
        self._model_fail_time = 0.0
        self._model_retry_interval = 60.0

    def load_model(self):
        """Load the selected detection model."""
        if self.model_name == "yolo_world":
            from ultralytics import YOLOWorld
            weights = self._resolve_yolo_world_weights()
            self.model = YOLOWorld(weights)
            self.model.set_classes(self.object_names)
        elif self.model_name == "grounding_dino":
            self._load_grounding_dino()
        else:
            raise ValueError(f"Unknown detection model: {self.model_name}")

    def detect(self, image: np.ndarray,
               depth_image: Optional[np.ndarray] = None) -> List[DetectedObject]:
        """Detect objects in a BGR camera image."""
        if self.model is None:
            if self._model_load_failed:
                import time
                if (time.time() - self._model_fail_time) < self._model_retry_interval:
                    return []
                self._model_load_failed = False
            try:
                self.load_model()
            except Exception:
                import time
                self._model_load_failed = True
                self._model_fail_time = time.time()
                raise

        raw_detections = self._run_detection(image)
        filtered = self._limit_per_class(self._apply_nms(raw_detections))

        results = []
        for det in filtered:
            width, depth, height = self._lookup_dimensions(det["name"])
            world_pos = self._estimate_position(
                det["bbox"], det["name"], depth_image, image.shape
            )
            results.append(DetectedObject(
                name=det["name"],
                x=world_pos[0],
                y=world_pos[1],
                z=world_pos[2],
                confidence=det["confidence"],
                width=width,
                depth=depth,
                height=height,
                bbox_pixels=tuple(det["bbox"]),
                pixel_x=(det["bbox"][0] + det["bbox"][2]) / 2.0,
                pixel_y=(det["bbox"][1] + det["bbox"][3]) / 2.0,
            ))
        return results

    def _run_detection(self, image: np.ndarray) -> List[dict]:
        if self.model_name == "yolo_world":
            return self._run_yolo_world(image)
        if self.model_name == "grounding_dino":
            return self._run_grounding_dino(image)
        return []

    def _run_yolo_world(self, image: np.ndarray) -> List[dict]:
        results = self.model.predict(
            image, conf=self.confidence_threshold, verbose=False
        )
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls.item() if hasattr(box.cls, "item") else box.cls)
            name = (
                self.object_names[cls_id]
                if cls_id < len(self.object_names)
                else f"class_{cls_id}"
            )
            if self.known_names and name not in self.known_names:
                continue

            bbox = self._clip_bbox(box.xyxy[0].tolist(), image.shape)
            if bbox is None or self._bbox_area(bbox) < self.min_bbox_area:
                continue

            detections.append({
                "name": name,
                "confidence": float(
                    box.conf.item() if hasattr(box.conf, "item") else box.conf
                ),
                "bbox": bbox,
            })
        return detections

    def _resolve_yolo_world_weights(self) -> str:
        explicit = self.model_weights or os.environ.get("YOLO_WORLD_WEIGHTS", "")
        if explicit:
            return self._require_file(explicit, "YOLO-World weights")

        local_candidates = [
            "yolov8s-worldv2.pt",
            os.path.join(os.getcwd(), "yolov8s-worldv2.pt"),
            os.path.join(os.getcwd(), "models", "yolov8s-worldv2.pt"),
            os.path.join(os.getcwd(), "config", "models", "yolov8s-worldv2.pt"),
        ]
        for candidate in local_candidates:
            expanded = os.path.abspath(os.path.expanduser(os.path.expandvars(candidate)))
            if os.path.isfile(expanded):
                return expanded

        if self.allow_model_download:
            return "yolov8s-worldv2.pt"

        raise FileNotFoundError(
            "YOLO-World weights were not found. Put yolov8s-worldv2.pt on RB8 "
            "and pass model_weights:=/path/to/yolov8s-worldv2.pt, set "
            "YOLO_WORLD_WEIGHTS, or use allow_model_download:=true only when "
            "network access is available."
        )

    def _load_grounding_dino(self):
        config_path = self._require_file(
            self.grounding_dino_config, "GroundingDINO config"
        )
        weights_path = self._require_file(
            self.grounding_dino_weights, "GroundingDINO weights"
        )
        try:
            from groundingdino.datasets import transforms as T
            from groundingdino.util.inference import load_model
        except ImportError as exc:
            raise ImportError(
                "GroundingDINO is not installed. Install it on RB8 or use "
                "detection_model:=yolo_world with local YOLO-World weights."
            ) from exc

        self._grounding_transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.model = load_model(
            config_path,
            weights_path,
            device=self.grounding_dino_device,
        )

    def _run_grounding_dino(self, image: np.ndarray) -> List[dict]:
        if self._grounding_transform is None:
            raise RuntimeError("GroundingDINO transform was not initialized")

        try:
            from PIL import Image
            from groundingdino.util.inference import predict
        except ImportError as exc:
            raise ImportError("GroundingDINO runtime dependencies are missing") from exc

        height, width = image.shape[:2]
        image_pil = Image.fromarray(image[:, :, ::-1])
        image_tensor, _ = self._grounding_transform(image_pil, None)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=self._grounding_prompt,
            box_threshold=self.confidence_threshold,
            text_threshold=self.grounding_dino_text_threshold,
            device=self.grounding_dino_device,
        )

        if hasattr(boxes, "detach"):
            boxes = boxes.detach().cpu().numpy()
        if hasattr(logits, "detach"):
            logits = logits.detach().cpu().numpy()

        detections = []
        scale = np.array([width, height, width, height], dtype=float)
        for box, score, phrase in zip(boxes, logits, phrases):
            name = self._match_grounding_phrase(str(phrase))
            if not name:
                continue
            cx, cy, box_w, box_h = box * scale
            bbox = self._clip_bbox(
                [
                    cx - box_w / 2.0,
                    cy - box_h / 2.0,
                    cx + box_w / 2.0,
                    cy + box_h / 2.0,
                ],
                image.shape,
            )
            if bbox is None or self._bbox_area(bbox) < self.min_bbox_area:
                continue
            detections.append({
                "name": name,
                "confidence": float(score),
                "bbox": bbox,
            })
        return detections

    @staticmethod
    def _require_file(path: str, description: str) -> str:
        if not path:
            raise FileNotFoundError(f"{description} path is empty")
        expanded = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
        if not os.path.isfile(expanded):
            raise FileNotFoundError(f"{description} not found: {expanded}")
        return expanded

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    @staticmethod
    def _build_grounding_prompt(object_names: List[str]) -> str:
        return " . ".join(name.replace("_", " ") for name in object_names) + " ."

    def _match_grounding_phrase(self, phrase: str) -> Optional[str]:
        normalized = phrase.lower().replace("-", " ").replace("_", " ")
        for name in sorted(self.object_names, key=len, reverse=True):
            readable = name.lower().replace("_", " ")
            if readable in normalized or name.lower() in normalized:
                return name
        return None

    def _apply_nms(self, detections: List[dict]) -> List[dict]:
        """Apply non-maximum suppression to remove overlapping detections."""
        if not detections:
            return []

        detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        kept = []
        for det in detections:
            if all(self._iou(det["bbox"], k["bbox"]) < self.nms_threshold for k in kept):
                kept.append(det)
        return kept

    def _limit_per_class(self, detections: List[dict]) -> List[dict]:
        """Keep at most N top-scoring detections for each class."""
        counts = defaultdict(int)
        kept = []
        for det in detections:
            if counts[det["name"]] >= self.max_detections_per_class:
                continue
            kept.append(det)
            counts[det["name"]] += 1
        return kept

    def _lookup_dimensions(self, name: str) -> Tuple[float, float, float]:
        """Look up physical dimensions from the object catalog."""
        dims = self.object_catalog.get(name, {}).get("dimensions", {})
        return (
            float(dims.get("w", 0.0)),
            float(dims.get("d", 0.0)),
            float(dims.get("h", 0.0)),
        )

    @staticmethod
    def _bbox_area(bbox: List[int]) -> float:
        """Compute bbox area in pixels."""
        return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

    @staticmethod
    def _clip_bbox(bbox: list, image_shape: tuple) -> Optional[List[int]]:
        """Clamp a bbox to the image extent and discard invalid boxes."""
        height, width = image_shape[:2]
        x1 = int(max(0, min(round(bbox[0]), width - 1)))
        y1 = int(max(0, min(round(bbox[1]), height - 1)))
        x2 = int(max(0, min(round(bbox[2]), width - 1)))
        y2 = int(max(0, min(round(bbox[3]), height - 1)))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    @staticmethod
    def _iou(box1: list, box2: list) -> float:
        """Compute IoU between two bboxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def _estimate_position(self, bbox: list, name: str,
                           depth_image: Optional[np.ndarray],
                           image_shape: tuple) -> Tuple[float, float, float]:
        """Estimate robot-frame tabletop position from a 2D bounding box."""
        del name
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        if depth_image is not None:
            depth_region = depth_image[
                int(bbox[1]):int(bbox[3]),
                int(bbox[0]):int(bbox[2])
            ]
            valid = depth_region[depth_region > 0]
            z = float(np.median(valid)) if valid.size > 0 else 0.0
        else:
            z = 0.0

        use_regression = (
            self.coordinate_mapping_mode == "vision_config_linear"
            and self.linear_regression is not None
        )
        from situbot.utils.transforms import pixel_to_world
        x, y, _ = pixel_to_world(
            cx,
            cy,
            z,
            image_shape,
            workspace_bounds=self.workspace_bounds,
            linear_regression=self.linear_regression if use_regression else None,
        )
        return (x, y, z)
