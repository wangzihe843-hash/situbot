#!/usr/bin/env python3
"""ROS node: object detection from camera feed.

Subscribes to camera image, runs open-vocabulary detection,
publishes DetectedObjects message.
"""

import rospy
import yaml
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from situbot.msg import DetectedObject, DetectedObjects
from situbot.perception.detector import ObjectDetector
from situbot.perception.hsv_fallback import HSVColorDetector
from situbot.perception.scene_description import (
    assign_instance_ids,
    build_scene_description,
)
from situbot.utils.transforms import load_linear_regression_config


class PerceptionNode:
    """ROS wrapper for ObjectDetector."""

    def __init__(self):
        rospy.init_node("situbot_perception", anonymous=False)

        # Load params
        self.camera_topic = rospy.get_param("~camera_topic", "/usb_cam/image_raw")
        model_name = rospy.get_param("~detection_model", "yolo_world")
        confidence = rospy.get_param("~confidence_threshold", 0.3)
        nms = rospy.get_param("~nms_threshold", 0.5)
        min_bbox_area = rospy.get_param("~min_bbox_area", 400.0)
        max_per_class = rospy.get_param("~max_detections_per_class", 3)
        model_weights = rospy.get_param("~model_weights", "")
        allow_model_download = self._as_bool(
            rospy.get_param("~allow_model_download", False)
        )
        grounding_dino_config = rospy.get_param("~grounding_dino_config", "")
        grounding_dino_weights = rospy.get_param("~grounding_dino_weights", "")
        grounding_dino_text_threshold = rospy.get_param(
            "~grounding_dino_text_threshold", 0.25
        )
        grounding_dino_device = rospy.get_param("~grounding_dino_device", "cpu")
        coordinate_mapping_mode = rospy.get_param(
            "~coordinate_mapping_mode", "workspace_linear"
        )
        vision_config_file = rospy.get_param("~vision_config_file", "")
        self.debug_visualization = rospy.get_param("~debug_visualization", True)

        self.workspace_bounds = {
            "x_min": rospy.get_param("~workspace_bounds/x_min",
                                     rospy.get_param("/situbot/workspace/table/x_min", 0.15)),
            "x_max": rospy.get_param("~workspace_bounds/x_max",
                                     rospy.get_param("/situbot/workspace/table/x_max", 0.75)),
            "y_min": rospy.get_param("~workspace_bounds/y_min",
                                     rospy.get_param("/situbot/workspace/table/y_min", -0.40)),
            "y_max": rospy.get_param("~workspace_bounds/y_max",
                                     rospy.get_param("/situbot/workspace/table/y_max", 0.40)),
        }

        linear_regression = None
        if coordinate_mapping_mode == "vision_config_linear":
            try:
                linear_regression = load_linear_regression_config(vision_config_file)
            except Exception as exc:
                rospy.logwarn(
                    f"Failed to load linear regression config from '{vision_config_file}': {exc}. "
                    "Falling back to workspace_linear mapping."
                )
                coordinate_mapping_mode = "workspace_linear"

        # Load object catalog
        objects_file = rospy.get_param("~objects_file", "")
        if objects_file:
            with open(objects_file) as f:
                catalog = yaml.safe_load(f)
            object_catalog = catalog.get("objects", [])
            object_names = [obj["name"] for obj in object_catalog]
        else:
            object_names = [
                "textbook", "notebook", "mug", "water_bottle", "phone",
                "laptop", "snack_box", "tissue_box", "pen_holder",
                "desk_lamp", "photo_frame", "wine_glass", "tea_set",
                "candle", "highlighter_set",
            ]
            object_catalog = [{"name": name, "dimensions": {}} for name in object_names]

        # Initialize detector (with HSV fallback)
        self.hsv_detector = None
        self.use_hsv_fallback = False
        try:
            self.detector = ObjectDetector(
                model_name=model_name,
                confidence_threshold=confidence,
                nms_threshold=nms,
                object_names=object_names,
                object_catalog=object_catalog,
                workspace_bounds=self.workspace_bounds,
                coordinate_mapping_mode=coordinate_mapping_mode,
                linear_regression=linear_regression,
                min_bbox_area=min_bbox_area,
                max_detections_per_class=max_per_class,
                model_weights=model_weights,
                allow_model_download=allow_model_download,
                grounding_dino_config=grounding_dino_config,
                grounding_dino_weights=grounding_dino_weights,
                grounding_dino_text_threshold=grounding_dino_text_threshold,
                grounding_dino_device=grounding_dino_device,
            )
            # Eagerly load model to catch download/GPU failures early
            self.detector.load_model()
        except Exception as e:
            rospy.logwarn(f"{model_name} failed to load: {e}. Trying HSV fallback...")
            self.detector = None
            if vision_config_file:
                try:
                    self.hsv_detector = HSVColorDetector(vision_config_file)
                    self.use_hsv_fallback = True
                    rospy.logwarn(
                        "Using HSV color detection fallback (limited to predefined colors). "
                        "NOTE: HSV detects colors (red/green/blue), not object names. "
                        "The reasoning pipeline expects object names from objects.yaml. "
                        "HSV mode is suitable for hardware debugging and calibration only."
                    )
                except Exception as e2:
                    rospy.logerr(f"HSV fallback also failed: {e2}. Perception disabled.")
            else:
                rospy.logerr("No vision_config_file for HSV fallback. Perception disabled.")
        self.bridge = CvBridge()
        self._last_image_time = None

        # Publisher
        self.pub = rospy.Publisher(
            "~detected_objects", DetectedObjects, queue_size=1
        )
        self.debug_pub = None
        if self.debug_visualization:
            self.debug_pub = rospy.Publisher(
                "~debug_image", Image, queue_size=1
            )

        # Subscriber
        self.sub = rospy.Subscriber(
            self.camera_topic, Image, self.image_callback, queue_size=1
        )
        self._camera_watchdog = rospy.Timer(
            rospy.Duration(5.0), self._camera_watchdog_callback
        )

        rospy.loginfo(f"PerceptionNode ready. Listening on {self.camera_topic}")
        rospy.loginfo(
            f"Detection model: {model_name}"
            + (f", weights: {model_weights}" if model_weights else "")
            + (", model downloads allowed" if allow_model_download else "")
        )
        rospy.loginfo(
            f"Coordinate mapping mode: {coordinate_mapping_mode}"
            + (f" ({vision_config_file})" if linear_regression is not None else "")
        )

    def image_callback(self, msg: Image):
        """Process incoming camera image."""
        try:
            self._last_image_time = rospy.Time.now()
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            if self.use_hsv_fallback:
                detections = self._detect_hsv(cv_image)
            elif self.detector is not None:
                detections = self.detector.detect(cv_image)
            else:
                return  # perception disabled

            detections = assign_instance_ids(detections)
            scene = build_scene_description(detections, self.workspace_bounds)

            # Build message
            out = DetectedObjects()
            out.header = msg.header
            out.scene_description = scene.description
            out.spatial_relations = scene.relations
            for det in detections:
                obj_msg = DetectedObject()
                obj_msg.instance_id = getattr(det, "instance_id", "")
                obj_msg.name = det.name
                obj_msg.x = det.x
                obj_msg.y = det.y
                obj_msg.z = det.z
                obj_msg.confidence = det.confidence
                obj_msg.width = det.width
                obj_msg.depth = det.depth
                obj_msg.height = det.height
                obj_msg.bbox_x1 = int(getattr(det, "bbox_pixels", (0, 0, 0, 0))[0])
                obj_msg.bbox_y1 = int(getattr(det, "bbox_pixels", (0, 0, 0, 0))[1])
                obj_msg.bbox_x2 = int(getattr(det, "bbox_pixels", (0, 0, 0, 0))[2])
                obj_msg.bbox_y2 = int(getattr(det, "bbox_pixels", (0, 0, 0, 0))[3])
                obj_msg.pixel_x = float(getattr(det, "pixel_x", 0.0))
                obj_msg.pixel_y = float(getattr(det, "pixel_y", 0.0))
                obj_msg.zone = getattr(det, "zone", "")
                out.objects.append(obj_msg)

            self.pub.publish(out)
            if self.debug_pub is not None:
                debug_image = self._draw_detections(cv_image, detections)
                debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
                debug_msg.header = msg.header
                self.debug_pub.publish(debug_msg)

        except Exception as e:
            rospy.logerr(f"Perception error: {e}")

    def _camera_watchdog_callback(self, event):
        """Warn when the configured image topic is not producing frames."""
        del event
        if self._last_image_time is None:
            rospy.logwarn_throttle(
                30.0,
                f"No camera frames received yet on {self.camera_topic}. "
                "For the bundled USB camera launch this should be /usb_cam/image_raw.",
            )
            return

        age = (rospy.Time.now() - self._last_image_time).to_sec()
        if age > 5.0:
            rospy.logwarn_throttle(
                30.0,
                f"Last camera frame on {self.camera_topic} was {age:.1f}s ago.",
            )

    @staticmethod
    def _draw_detections(image, detections):
        """Render bbox and pose estimates for debugging."""
        canvas = image.copy()
        for det in detections:
            name = det.name if hasattr(det, 'name') else str(det)
            instance_id = getattr(det, "instance_id", "")
            conf = det.confidence if hasattr(det, 'confidence') else 0.0
            x_w = det.x if hasattr(det, 'x') else 0.0
            y_w = det.y if hasattr(det, 'y') else 0.0
            z_w = det.z if hasattr(det, 'z') else 0.0

            label_name = f"{instance_id}:{name}" if instance_id else name
            label = f"{label_name} {conf:.2f}"
            pose_line = f"({x_w:.3f}, {y_w:.3f}, {z_w:.3f})"

            has_bbox = (hasattr(det, 'bbox_pixels')
                        and det.bbox_pixels != (0, 0, 0, 0))

            if has_bbox:
                x1, y1, x2, y2 = det.bbox_pixels
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 220, 60), 2)
                text_x = x1
                lines = [label, pose_line]
                w_val = getattr(det, 'width', 0)
                d_val = getattr(det, 'depth', 0)
                h_val = getattr(det, 'height', 0)
                if w_val > 0 and d_val > 0 and h_val > 0:
                    lines.append(f"{w_val:.2f}x{d_val:.2f}x{h_val:.2f}m")
                for idx, text in enumerate(lines):
                    y_text = max(18, y1 - 8 - 18 * (len(lines) - idx - 1))
                    cv2.putText(
                        canvas, text, (text_x, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
                    )
            else:
                # HSV fallback: draw crosshair at pixel coords if available
                px = int(getattr(det, 'pixel_x', 0) if hasattr(det, 'pixel_x')
                         else 0)
                py = int(getattr(det, 'pixel_y', 0) if hasattr(det, 'pixel_y')
                         else 0)
                if px > 0 or py > 0:
                    cv2.drawMarker(canvas, (px, py), (60, 220, 60),
                                   cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(canvas, label, (px + 12, py - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                                cv2.LINE_AA)
                    cv2.putText(canvas, pose_line, (px + 12, py + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                                cv2.LINE_AA)
        return canvas

    def _detect_hsv(self, cv_image):
        """Run HSV fallback detection, returning DetectedObject-like results."""
        from situbot.perception.detector import DetectedObject as DetObj
        hsv_results = self.hsv_detector.detect_all(cv_image)
        detections = []
        for h in hsv_results:
            det = DetObj(
                name=h.color,
                x=h.world_x,
                y=h.world_y,
                z=0.0,
                confidence=min(1.0, h.area / 10000.0),
            )
            # Stash pixel coords for debug visualization
            det.pixel_x = h.pixel_x
            det.pixel_y = h.pixel_y
            det.bbox_pixels = (0, 0, 0, 0)
            detections.append(det)
        return detections

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)


if __name__ == "__main__":
    try:
        node = PerceptionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
