#!/usr/bin/env python3
"""MoveIt executor for sagittarius arm pick-and-place operations.

Adapted from sagittarius_perception/sgr_ctrl.py (MoveItSGRTool class).
"""

import sys
import math
import copy
import logging
import numpy as np
from typing import Optional, Tuple, List

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

    # Gripper joint values (from sgr_ctrl.py MoveItSGRTool)
    GRIPPER_OPEN = [0.0, 0.0]
    GRIPPER_CLOSE = [-0.021, -0.021]

    # Arm geometry for pitch calculation (from sgr_ctrl.py)
    ARM_HALF_LENGTH = 0.532 / 2  # half total arm length for triangulation

    # End-effector grasp offset (from sgr_ctrl.py ee_target_offset)
    EE_GRASP_OFFSET = -0.07  # meters, wrist-to-grasp-point along local x

    def __init__(self, planning_group: str = "sagittarius_arm",
                 gripper_group: str = "sagittarius_gripper",
                 planning_time: float = 5.0,
                 max_velocity_scaling: float = 0.5,
                 max_acceleration_scaling: float = 0.5,
                 approach_height: float = 0.04,
                 lift_height: float = 0.08,
                 position_tolerance: float = 0.001,
                 orientation_tolerance: float = 0.001):
        """
        Args:
            planning_group: MoveIt arm planning group (sagittarius_arm).
            gripper_group: MoveIt gripper planning group (sagittarius_gripper).
            planning_time: Maximum planning time per attempt.
            max_velocity_scaling: Velocity scaling factor [0, 1].
            max_acceleration_scaling: Acceleration scaling factor [0, 1].
            approach_height: Height above object for approach (meters). Default 0.04 from sgr_ctrl.
            lift_height: Height to lift after grasp (meters). Default 0.08 from sgr_ctrl.
            position_tolerance: Position goal tolerance (meters).
            orientation_tolerance: Orientation goal tolerance (radians).
        """
        self.planning_group = planning_group
        self.gripper_group_name = gripper_group
        self.planning_time = planning_time
        self.max_velocity_scaling = max_velocity_scaling
        self.max_acceleration_scaling = max_acceleration_scaling
        self.approach_height = approach_height
        self.lift_height = lift_height
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance

        self.arm_group = None
        self.gripper_group = None
        self.reference_frame = None

    def initialize(self):
        """Initialize MoveIt commander and planning groups.

        Must be called after rospy.init_node().
        """
        import rospy
        import moveit_commander

        moveit_commander.roscpp_initialize(sys.argv)

        # Arm group (from sgr_ctrl.py)
        self.arm_group = moveit_commander.MoveGroupCommander(self.planning_group)
        self.arm_group.allow_replanning(False)
        self.arm_group.set_goal_position_tolerance(self.position_tolerance)
        self.arm_group.set_goal_orientation_tolerance(self.orientation_tolerance)
        self.arm_group.set_max_acceleration_scaling_factor(self.max_acceleration_scaling)
        self.arm_group.set_max_velocity_scaling_factor(self.max_velocity_scaling)

        # Reference frame (from sgr_ctrl.py)
        self.reference_frame = rospy.get_namespace()[1:] + 'base_link'
        self.arm_group.set_pose_reference_frame(self.reference_frame)

        # Gripper group (from sgr_ctrl.py)
        self.gripper_group = moveit_commander.MoveGroupCommander(self.gripper_group_name)
        self.gripper_group.set_pose_reference_frame(self.reference_frame)
        self.gripper_group.set_goal_joint_tolerance(0.001)

        # Log end effector link
        ee_link = self.arm_group.get_end_effector_link()
        logger.info(f"MoveIt executor initialized: group={self.planning_group}, "
                     f"ee_link={ee_link}, ref_frame={self.reference_frame}")

    def go_home(self) -> bool:
        """Move to home position using SRDF named target.

        Returns:
            True if motion succeeded.
        """
        import rospy
        self.arm_group.set_named_target('home')
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        rospy.sleep(1)
        logger.info(f"go_home: {'success' if success else 'failed'}")
        return success

    def go_sleep(self) -> bool:
        """Move to sleep (compact) position using SRDF named target.

        Returns:
            True if motion succeeded.
        """
        import rospy
        self.go_home()
        self._gripper_open()  # release any held object BEFORE compacting
        self.arm_group.set_named_target('sleep')
        success = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        rospy.sleep(1)
        return success

    def pick(self, x: float, y: float, z: float,
             object_name: str = "") -> bool:
        """Execute a pick operation at the given world coordinates.

        Sequence (from sgr_ctrl.py SGRCtrlActionServer):
          1. Open gripper
          2. Compute approach angles (roll, pitch, yaw) from target position
          3. Apply end-effector offset (-0.07m)
          4. Move to approach height (z + 0.04)
          5. Descend to object (z)
          6. Close gripper
          7. Lift (z + 0.08)

        Args:
            x, y, z: World coordinates of the object center.
            object_name: Name for logging.

        Returns:
            True if pick succeeded.
        """
        logger.info(f"Picking {object_name} at ({x:.3f}, {y:.3f}, {z:.3f})")

        # Compute approach orientation from position (from sgr_ctrl.py)
        roll, pitch, yaw = self._ee_xyz_get_rpy(x, y, z)

        # Apply end-effector offset (from sgr_ctrl.py)
        ox, oy, oz, oroll, opitch, oyaw = self._ee_target_offset(
            x, y, z, roll, pitch, yaw
        )

        # Validate plan feasibility (from sgr_ctrl.py)
        if not self._is_plan_success(ox, oy, oz, oroll, opitch, oyaw):
            logger.error(f"No valid plan to reach {object_name}")
            return False
        if not self._is_plan_success(ox, oy, oz + self.approach_height, oroll, opitch, oyaw):
            logger.error(f"No valid plan for approach height for {object_name}")
            return False

        # Step 1: Open gripper
        if not self._gripper_open():
            return False

        # Step 2: Move above the object
        if not self._move_to_pose_euler(ox, oy, oz + self.approach_height,
                                         oroll, opitch, oyaw, wait_time=1.0):
            logger.error(f"Failed to reach approach position for {object_name}")
            return False

        # Step 3: Descend to object
        if not self._move_to_pose_euler(ox, oy, oz, oroll, opitch, oyaw, wait_time=0.2):
            logger.error(f"Failed to descend to grasp {object_name}")
            self._move_to_pose_euler(ox, oy, oz + self.approach_height,
                                      oroll, opitch, oyaw)
            return False

        # Step 4: Close gripper
        if not self._gripper_close():
            return False

        # Step 5: Lift
        if not self._move_to_pose_euler(ox, oy, oz + self.lift_height,
                                         oroll, opitch, oyaw, wait_time=0.2):
            logger.warning(f"Lift failed for {object_name}")

        logger.info(f"Successfully picked {object_name}")
        return True

    def place(self, x: float, y: float, z: float,
              object_name: str = "") -> bool:
        """Execute a place operation at the given world coordinates.

        Sequence:
          1. Compute approach angles and apply EE offset
          2. Move to approach height (z + 0.04)
          3. Descend to place position (z)
          4. Open gripper (release)
          5. Lift (z + 0.08)

        Args:
            x, y, z: Target world coordinates.
            object_name: Name for logging.

        Returns:
            True if place succeeded.
        """
        logger.info(f"Placing {object_name} at ({x:.3f}, {y:.3f}, {z:.3f})")

        # Compute orientation and offset
        roll, pitch, yaw = self._ee_xyz_get_rpy(x, y, z)
        ox, oy, oz, oroll, opitch, oyaw = self._ee_target_offset(
            x, y, z, roll, pitch, yaw
        )

        # Step 1: Move above the target
        if not self._move_to_pose_euler(ox, oy, oz + self.approach_height,
                                         oroll, opitch, oyaw, wait_time=1.0):
            logger.error(f"Failed to reach approach position for placing {object_name}")
            return False

        # Step 2: Descend to place position
        if not self._move_to_pose_euler(ox, oy, oz, oroll, opitch, oyaw, wait_time=0.2):
            logger.error(f"Failed to descend for placing {object_name}")
            return False

        # Step 3: Open gripper (release)
        if not self._gripper_open():
            return False

        # Step 4: Lift away
        self._move_to_pose_euler(ox, oy, oz + self.lift_height,
                                  oroll, opitch, oyaw, wait_time=0.2)

        logger.info(f"Successfully placed {object_name}")
        return True

    # =========================================================================
    # Internal methods — adapted from sgr_ctrl.py MoveItSGRTool
    # =========================================================================

    def _move_to_pose_euler(self, x: float, y: float, z: float,
                             roll: float = 0, pitch: float = 0, yaw: float = 0,
                             wait_time: float = 0.2) -> bool:
        """Move end effector to target pose specified as Euler angles.

        Adapted from MoveItSGRTool.to_pose_eular().

        Returns:
            True if motion succeeded.
        """
        import rospy

        self.arm_group.set_pose_target([x, y, z, roll, pitch, yaw])
        plan = self.arm_group.plan()

        # Python 3 returns a tuple; index [1] is RobotTrajectory
        if isinstance(plan, tuple):
            plan = plan[1]

        if len(plan.joint_trajectory.points) == 0:
            logger.warning(f"No plan found for pose ({x:.3f}, {y:.3f}, {z:.3f})")
            self.arm_group.clear_pose_targets()
            return False

        self.arm_group.execute(plan, wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        rospy.sleep(wait_time)
        return True

    def _is_plan_success(self, x: float, y: float, z: float,
                          roll: float = 0, pitch: float = 0, yaw: float = 0) -> bool:
        """Check if a valid motion plan exists for the target pose.

        Adapted from MoveItSGRTool.isPlanSuccess().
        """
        self.arm_group.set_pose_target([x, y, z, roll, pitch, yaw])
        plan = self.arm_group.plan()

        if isinstance(plan, tuple):
            plan = plan[1]

        self.arm_group.clear_pose_targets()
        return len(plan.joint_trajectory.points) != 0

    def _gripper_open(self) -> bool:
        """Open the gripper.

        From MoveItSGRTool.gripper_open():
          gripper.set_joint_value_target([0.0, 0.0])
        """
        import rospy
        self.gripper_group.set_joint_value_target(self.GRIPPER_OPEN)
        ret = self.gripper_group.go(wait=True)
        if ret:
            rospy.sleep(2)
        logger.debug(f"Gripper open: {'success' if ret else 'failed'}")
        return ret

    def _gripper_close(self) -> bool:
        """Close the gripper.

        From MoveItSGRTool.gripper_catch():
          gripper.set_joint_value_target([-0.021, -0.021])
        """
        import rospy
        self.gripper_group.set_joint_value_target(self.GRIPPER_CLOSE)
        ret = self.gripper_group.go(wait=True)
        if ret:
            rospy.sleep(2)
        logger.debug(f"Gripper close: {'success' if ret else 'failed'}")
        return ret

    def _ee_xyz_get_rpy(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Compute end-effector roll, pitch, yaw from target position.

        From MoveItSGRTool.ee_xyz_get_rpy().
        Uses arm geometry triangulation to estimate a reasonable pitch angle,
        and arctangent for yaw.
        """
        yaw = math.atan2(y, x)
        roll = 0.0

        a = self.ARM_HALF_LENGTH
        b = self.ARM_HALF_LENGTH
        c = math.sqrt(x * x + y * y + z * z)

        if c < 1e-6:
            # Target at origin — no meaningful orientation
            return (0.0, 0.0, 0.0)

        if a + b <= c:
            # Triangle inequality fails — target at or beyond reach
            pitch = 0.0
        else:
            cos_arg = (a * a - b * b - c * c) / (-2 * b * c)
            cos_arg = max(-1.0, min(1.0, cos_arg))  # clamp for float imprecision
            sin_arg = max(-1.0, min(1.0, z / c))
            pitch = math.acos(cos_arg) - math.asin(sin_arg)
            pitch = max(0.0, min(1.57, pitch))

        return (roll, pitch, yaw)

    def _ee_target_offset(self, px: float, py: float, pz: float,
                           roll: float = 0, pitch: float = 0, yaw: float = 0,
                           ee_type: str = 'grasp') -> Tuple[float, float, float, float, float, float]:
        """Apply end-effector offset to target pose.

        From MoveItSGRTool.ee_target_offset().
        The gripper grasp point is -0.07m along the local x-axis from the wrist.

        Returns:
            (x, y, z, roll, pitch, yaw) with offset applied.
        """
        # TODO: Requires tf.transformations — install via: sudo apt install ros-noetic-tf
        # If tf is unavailable, install transforms3d: pip install transforms3d
        # See: https://github.com/matthew-brett/transforms3d
        import tf.transformations as transformations

        M = transformations.compose_matrix(
            angles=[roll, pitch, yaw], translate=[px, py, pz]
        )
        if ee_type == 'grasp':
            M1 = np.dot(M, transformations.translation_matrix([self.EE_GRASP_OFFSET, 0, 0]))
        else:
            M1 = M

        scale, shear, angles, translate, perspective = transformations.decompose_matrix(M1)
        return (translate[0], translate[1], translate[2],
                angles[0], angles[1], angles[2])
