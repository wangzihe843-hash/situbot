#!/usr/bin/env python3
"""MoveIt executor for sagittarius arm pick-and-place operations.

Adapted from sagittarius_perception/sgr_ctrl.py (MoveItSGRTool class).
Extended with PlanningScene obstacle management for height-aware
collision avoidance during arm transit.
"""

import sys
import math
import copy
import logging
import numpy as np
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)


class MoveItExecutor:
    """Wraps moveit_commander for pick-and-place with the sagittarius arm.

    Based on the proven MoveItSGRTool from sgr_ctrl.py.
    Uses 'sagittarius_arm' and 'sagittarius_gripper' planning groups,
    with the -0.07m end-effector offset and dynamic pitch/yaw calculation.

    Provides high-level pick/place/go_home methods that handle:
    - End-effector offset compensation (-0.07m gripper-to-wrist)
    - Dynamic approach angle calculation from target position
    - Approach trajectory (move above object)
    - Grasp/release via MoveIt gripper planning group
    - Error recovery (return to home on failure)
    """

    GRIPPER_OPEN = [0.0, 0.0]
    GRIPPER_CLOSE = [-0.021, -0.021]
    ARM_HALF_LENGTH = 0.532 / 2
    EE_GRASP_OFFSET = -0.07

    def __init__(self, planning_group: str = "sagittarius_arm",
                 gripper_group: str = "sagittarius_gripper",
                 planning_time: float = 5.0,
                 max_velocity_scaling: float = 0.5,
                 max_acceleration_scaling: float = 0.5,
                 approach_height: float = 0.04,
                 lift_height: float = 0.08,
                 position_tolerance: float = 0.001,
                 orientation_tolerance: float = 0.001,
                 scene_obstacle_padding: float = 0.01):
        self.planning_group = planning_group
        self.gripper_group_name = gripper_group
        self.planning_time = planning_time
        self.max_velocity_scaling = max_velocity_scaling
        self.max_acceleration_scaling = max_acceleration_scaling
        self.approach_height = approach_height
        self.lift_height = lift_height
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.scene_obstacle_padding = scene_obstacle_padding

        self.arm_group = None
        self.gripper_group = None
        self.scene = None
        self.reference_frame = None
        self._scene_objects: Dict[str, dict] = {}

    def initialize(self):
        import rospy
        import moveit_commander

        moveit_commander.roscpp_initialize(sys.argv)

        self.arm_group = moveit_commander.MoveGroupCommander(
            self.planning_group,
            robot_description="/sgr532/robot_description",
            ns="/sgr532"
        )
        self.arm_group.allow_replanning(False)
        self.arm_group.set_goal_position_tolerance(self.position_tolerance)
        self.arm_group.set_goal_orientation_tolerance(self.orientation_tolerance)
        self.arm_group.set_max_acceleration_scaling_factor(self.max_acceleration_scaling)
        self.arm_group.set_max_velocity_scaling_factor(self.max_velocity_scaling)

        self.reference_frame = "sgr532/base_link"
        self.arm_group.set_pose_reference_frame(self.reference_frame)

        self.gripper_group = moveit_commander.MoveGroupCommander(
            self.gripper_group_name,
            robot_description="/sgr532/robot_description",
            ns="/sgr532"
        )
        self.gripper_group.set_pose_reference_frame(self.reference_frame)
        self.gripper_group.set_goal_joint_tolerance(0.001)

        self.scene = moveit_commander.PlanningSceneInterface(ns="/sgr532")
        rospy.sleep(0.5)

        ee_link = self.arm_group.get_end_effector_link()
        logger.info(f"MoveIt executor initialized: group={self.planning_group}, "
                     f"ee_link={ee_link}, ref_frame={self.reference_frame}")

    def add_scene_obstacle(self, name: str, x: float, y: float,
                           z_surface: float, w: float, d: float, h: float):
        if self.scene is None:
            logger.warning("Planning scene not initialized, skipping add_scene_obstacle")
            return
        if h < 0.005:
            return

        import rospy
        from geometry_msgs.msg import PoseStamped

        pad = self.scene_obstacle_padding
        box_h = h + pad

        pose = PoseStamped()
        pose.header.frame_id = self.reference_frame
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z_surface + box_h / 2.0
        pose.pose.orientation.w = 1.0

        self.scene.add_box(name, pose, size=(d + pad, w + pad, box_h))
        self._scene_objects[name] = {
            "x": x, "y": y, "z_surface": z_surface,
            "w": w, "d": d, "h": h,
        }
        logger.debug(f"Scene obstacle added: {name} at ({x:.3f}, {y:.3f}), "
                     f"size=({d:.3f}, {w:.3f}, {h:.3f})")

    def remove_scene_obstacle(self, name: str):
        if self.scene is None:
            return
        if name in self._scene_objects:
            self.scene.remove_world_object(name)
            del self._scene_objects[name]
            logger.debug(f"Scene obstacle removed: {name}")

    def update_scene_obstacle(self, name: str, new_x: float, new_y: float,
                              z_surface: float):
        if name not in self._scene_objects:
            return
        info = self._scene_objects[name]
        self.remove_scene_obstacle(name)
        self.add_scene_obstacle(
            name, new_x, new_y, z_surface, info["w"], info["d"], info["h"]
        )

    def populate_scene_from_detections(self, objects: list,
                                       object_catalog: dict,
                                       z_surface: float):
        for old_name in list(self._scene_objects.keys()):
            self.remove_scene_obstacle(old_name)

        for obj in objects:
            name = obj.name if hasattr(obj, "name") else obj.get("name", "")
            x = obj.x if hasattr(obj, "x") else obj.get("x", 0)
            y = obj.y if hasattr(obj, "y") else obj.get("y", 0)

            cat_dims = object_catalog.get(name, {}).get("dimensions", {})
            if hasattr(obj, "width"):
                w = getattr(obj, "width", 0)
                d = getattr(obj, "depth", 0)
                h = getattr(obj, "height", 0)
            elif isinstance(obj, dict):
                w = obj.get("width", 0)
                d = obj.get("depth", 0)
                h = obj.get("height", 0)
            else:
                w = d = h = 0
            w = w or cat_dims.get("w", 0.10)
            d = d or cat_dims.get("d", 0.10)
            h = h or cat_dims.get("h", 0.05)

            obj_id = getattr(obj, "instance_id", "") or name
            self.add_scene_obstacle(obj_id, x, y, z_surface, w, d, h)

        import rospy
        rospy.sleep(0.2)
        logger.info(f"Planning scene populated: {len(self._scene_objects)} obstacles")

    def clear_scene(self):
        if self.scene is None:
            return
        for name in list(self._scene_objects.keys()):
            self.scene.remove_world_object(name)
        self._scene_objects.clear()
        logger.info("Planning scene cleared")

    def go_home(self) -> bool:
        success = self.go_named_pose("home")
        logger.info(f"go_home: {'success' if success else 'failed'}")
        return success

    def go_named_pose(self, pose_name: str) -> bool:
        import rospy
        self.arm_group.set_named_target(pose_name)
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        rospy.sleep(1)
        logger.info(f"go_named_pose('{pose_name}'): {'success' if success else 'failed'}")
        return success

    def go_sleep(self) -> bool:
        import rospy
        self.go_home()
        self._gripper_open()
        self.arm_group.set_named_target('sleep')
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        rospy.sleep(1)
        return success

    def pick(self, x: float, y: float, z: float,
             object_name: str = "") -> bool:
        logger.info(f"Picking {object_name} at ({x:.3f}, {y:.3f}, {z:.3f})")

        roll, pitch, yaw = self._ee_xyz_get_rpy(x, y, z)
        ox, oy, oz, oroll, opitch, oyaw = self._ee_target_offset(
            x, y, z, roll, pitch, yaw
        )

        if not self._is_plan_success(ox, oy, oz, oroll, opitch, oyaw):
            logger.error(f"No valid plan to reach {object_name}")
            return False
        if not self._is_plan_success(ox, oy, oz + self.approach_height, oroll, opitch, oyaw):
            logger.error(f"No valid plan for approach height for {object_name}")
            return False

        if not self._gripper_open():
            return False

        if not self._move_to_pose_euler(ox, oy, oz + self.approach_height,
                                         oroll, opitch, oyaw, wait_time=1.0):
            logger.error(f"Failed to reach approach position for {object_name}")
            return False

        if not self._move_to_pose_euler(ox, oy, oz, oroll, opitch, oyaw, wait_time=0.2):
            logger.error(f"Failed to descend to grasp {object_name}")
            self._move_to_pose_euler(ox, oy, oz + self.approach_height,
                                      oroll, opitch, oyaw)
            return False

        if not self._gripper_close():
            return False

        if not self._move_to_pose_euler(ox, oy, oz + self.lift_height,
                                         oroll, opitch, oyaw, wait_time=0.2):
            logger.warning(f"Lift failed for {object_name}")

        logger.info(f"Successfully picked {object_name}")
        return True

    def place(self, x: float, y: float, z: float,
              object_name: str = "") -> bool:
        logger.info(f"Placing {object_name} at ({x:.3f}, {y:.3f}, {z:.3f})")

        roll, pitch, yaw = self._ee_xyz_get_rpy(x, y, z)
        ox, oy, oz, oroll, opitch, oyaw = self._ee_target_offset(
            x, y, z, roll, pitch, yaw
        )

        if not self._move_to_pose_euler(ox, oy, oz + self.approach_height,
                                         oroll, opitch, oyaw, wait_time=1.0):
            logger.error(f"Failed to reach approach position for placing {object_name}")
            return False

        if not self._move_to_pose_euler(ox, oy, oz, oroll, opitch, oyaw, wait_time=0.2):
            logger.error(f"Failed to descend for placing {object_name}")
            return False

        if not self._gripper_open():
            return False

        self._move_to_pose_euler(ox, oy, oz + self.lift_height,
                                 oroll, opitch, oyaw, wait_time=0.5)

        logger.info(f"Successfully placed {object_name}")
        return True

    def _ee_xyz_get_rpy(self, x: float, y: float, z: float) -> Tuple:
        import rospy
        current_pose = self.arm_group.get_current_pose()
        cx, cy, cz = current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z

        dx = x - cx
        dy = y - cy
        dz = z - cz

        yaw = math.atan2(dy, dx)
        dist_xy = math.sqrt(dx*dx + dy*dy)
        pitch = math.atan2(-dz, dist_xy)
        roll = 0.0

        return roll, pitch, yaw

    def _ee_target_offset(self, x: float, y: float, z: float,
                         roll: float, pitch: float, yaw: float):
        ox = x + self.EE_GRASP_OFFSET * math.cos(yaw) * math.cos(pitch)
        oy = y + self.EE_GRASP_OFFSET * math.sin(yaw) * math.cos(pitch)
        oz = z + self.EE_GRASP_OFFSET * math.sin(pitch)
        return ox, oy, oz, roll, pitch, yaw

    def _is_plan_success(self, x: float, y: float, z: float,
                        roll: float, pitch: float, yaw: float) -> bool:
        """Check if a valid motion plan exists (does NOT move the arm)."""
        pose = self._make_pose(x, y, z, roll, pitch, yaw)
        self.arm_group.set_pose_target(pose)
        plan = self.arm_group.plan()
        self.arm_group.clear_pose_targets()
        # MoveIt plan() returns (success, plan, ...) tuple in newer API
        if isinstance(plan, tuple):
            return plan[0]  # bool success flag
        return len(plan.joint_trajectory.points) != 0

    def _move_to_pose_euler(self, x: float, y: float, z: float,
                            roll: float, pitch: float, yaw: float,
                            wait_time: float = 1.0) -> bool:
        import rospy
        pose = self._make_pose(x, y, z, roll, pitch, yaw)
        self.arm_group.set_pose_target(pose)
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        rospy.sleep(wait_time)
        return success

    def _make_pose(self, x: float, y: float, z: float,
                   roll: float, pitch: float, yaw: float):
        from geometry_msgs.msg import Pose
        from tf.transformations import quaternion_from_euler
        q = quaternion_from_euler(roll, pitch, yaw)
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        return pose

    def _gripper_open(self) -> bool:
        import rospy
        self.gripper_group.set_joint_value_target(self.GRIPPER_OPEN)
        success = self.gripper_group.go(wait=True)
        self.gripper_group.stop()
        self.gripper_group.clear_pose_targets()
        rospy.sleep(0.5)
        return success

    def _gripper_close(self) -> bool:
        import rospy
        self.gripper_group.set_joint_value_target(self.GRIPPER_CLOSE)
        success = self.gripper_group.go(wait=True)
        self.gripper_group.stop()
        self.gripper_group.clear_pose_targets()
        rospy.sleep(0.5)
        return success
