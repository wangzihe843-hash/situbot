#!/usr/bin/env python3
"""ROS node: MoveIt arm executor.

Subscribes to PlannedAction messages and executes pick/place operations
via MoveIt commander on the sagittarius arm.

Uses sagittarius_arm and sagittarius_gripper planning groups,
with end-effector offset and dynamic approach angles from sgr_ctrl.py.

Extended: subscribes to DetectedObjects to populate the MoveIt planning
scene with collision boxes, so the arm avoids tall objects during transit.
"""

import threading

import rospy
import yaml

from situbot.msg import PlannedAction, DetectedObjects
from situbot.execution.moveit_executor import MoveItExecutor


class ExecutorNode:
    """ROS wrapper for MoveItExecutor.

    Collects PlannedAction messages, buffers them, and executes
    in sequence_order when a complete batch is received.

    Also subscribes to perception output to keep the MoveIt planning
    scene up-to-date with obstacle boxes for all detected objects.
    """

    def __init__(self):
        rospy.init_node("situbot_executor", anonymous=False)

        def _get_param_with_global_fallback(private_key, global_key, default):
            if rospy.has_param(private_key):
                return rospy.get_param(private_key)
            if rospy.has_param(global_key):
                return rospy.get_param(global_key)
            return default

        # Load params (matched to sagittarius hardware)
        planning_group = _get_param_with_global_fallback(
            "~moveit/planning_group", "moveit/planning_group", "sagittarius_arm"
        )
        gripper_group = _get_param_with_global_fallback(
            "~moveit/gripper_group", "moveit/gripper_group", "sagittarius_gripper"
        )
        planning_time = _get_param_with_global_fallback(
            "~moveit/planning_time", "moveit/planning_time", 5.0
        )
        max_vel = _get_param_with_global_fallback(
            "~moveit/max_velocity_scaling", "moveit/max_velocity_scaling", 0.5
        )
        max_acc = _get_param_with_global_fallback(
            "~moveit/max_acceleration_scaling", "moveit/max_acceleration_scaling", 0.5
        )
        approach_h = rospy.get_param("~gripper/approach_height", 0.04)
        lift_h = rospy.get_param("~gripper/lift_height", 0.08)
        self.start_pose = rospy.get_param("~moveit/start_pose", "home")
        self.recovery_pose = rospy.get_param("~moveit/recovery_pose", self.start_pose)

        self.executor = MoveItExecutor(
            planning_group=planning_group,
            gripper_group=gripper_group,
            planning_time=planning_time,
            max_velocity_scaling=max_vel,
            max_acceleration_scaling=max_acc,
            approach_height=approach_h,
            lift_height=lift_h,
        )
        self.executor.initialize()
        rospy.loginfo(
            f"Executor poses: start_pose='{self.start_pose}', recovery_pose='{self.recovery_pose}'"
        )

        # Load object catalog for dimension lookups
        objects_file = rospy.get_param("~objects_file", "")
        self.object_catalog = {}
        if objects_file:
            try:
                with open(objects_file) as f:
                    data = yaml.safe_load(f)
                self.object_catalog = {
                    obj["name"]: obj for obj in data.get("objects", [])
                }
                rospy.loginfo(f"Loaded {len(self.object_catalog)} objects from catalog")
            except Exception as e:
                rospy.logwarn(f"Failed to load object catalog: {e}")

        # Workspace z_surface for obstacle placement
        self.z_surface = rospy.get_param("~workspace/table/z_surface", 0.00)

        # Action buffer (collect all actions, execute after quiet period)
        self.action_buffer = []
        self.buffer_timer = None
        self.buffer_timeout = rospy.get_param("~buffer_timeout", 2.0)
        self._first_action_time = None
        self._max_buffer_wait = rospy.get_param("~max_buffer_wait", 5.0)
        self._executing = threading.Lock()
        self._buffer_lock = threading.Lock()

        # Subscribers
        self.sub = rospy.Subscriber(
            "/situbot_planner/planned_actions",
            PlannedAction, self.action_callback, queue_size=50,
        )
        self.sub_objects = rospy.Subscriber(
            "/situbot_perception/detected_objects",
            DetectedObjects, self.objects_callback, queue_size=1,
        )

        rospy.loginfo("ExecutorNode ready. Waiting for planned actions...")

    def objects_callback(self, msg: DetectedObjects):
        """Update MoveIt planning scene from latest perception snapshot."""
        if not self._executing.acquire(blocking=False):
            return
        try:
            self.executor.populate_scene_from_detections(
                msg.objects, self.object_catalog, self.z_surface
            )
        finally:
            self._executing.release()

    def action_callback(self, msg: PlannedAction):
        """Buffer incoming actions and schedule execution."""
        should_execute_now = False
        with self._buffer_lock:
            self.action_buffer.append(msg)
            now = rospy.Time.now()

            if self._first_action_time is None:
                self._first_action_time = now

            if self.buffer_timer:
                self.buffer_timer.shutdown()

            elapsed = (now - self._first_action_time).to_sec()
            if elapsed >= self._max_buffer_wait:
                should_execute_now = True
            else:
                self.buffer_timer = rospy.Timer(
                    rospy.Duration(self.buffer_timeout),
                    self.execute_buffered,
                    oneshot=True,
                )
        if should_execute_now:
            self.execute_buffered()

    def execute_buffered(self, event=None):
        """Execute all buffered actions in sequence order."""
        if not self._executing.acquire(blocking=False):
            # Scene updates may briefly hold this lock; retry so buffered actions
            # are not dropped if timer fires during an objects callback.
            rospy.Timer(
                rospy.Duration(0.2),
                self.execute_buffered,
                oneshot=True,
            )
            return
        try:
            self._execute_buffered_inner()
        finally:
            self._executing.release()

    def _execute_buffered_inner(self):
        with self._buffer_lock:
            if not self.action_buffer:
                return
            actions = sorted(self.action_buffer, key=lambda a: a.sequence_order)
            self.action_buffer = []
            self._first_action_time = None

        rospy.loginfo(f"Executing {len(actions)} actions...")

        start_pose = getattr(self, "start_pose", "home")
        recovery_pose = getattr(self, "recovery_pose", start_pose)

        rospy.loginfo(f"Safety: releasing gripper and moving to start pose '{start_pose}'...")
        if not self.executor._gripper_open():
            # On real hardware this can be false even when jaws are already open.
            # Do not abort solely on this signal; verify via start-pose move result.
            rospy.logwarn(
                "Safety step warning: gripper open command returned false; "
                "continuing to start-pose check"
            )
        if not self._go_pose(start_pose):
            self._log_planning_diagnostics(f"safety_move_to_{start_pose}")
            rospy.logerr(
                f"Safety step failed: unable to reach start pose '{start_pose}', "
                "aborting current action batch"
            )
            rospy.loginfo(f"Execution complete: 0 succeeded, {len(actions)} failed")
            return

        success_count = 0
        fail_count = 0

        for action in actions:
            pos = action.pose.position
            name = action.object_name
            obj_id = action.instance_id or name
            label = f"{name} ({action.instance_id})" if action.instance_id else name

            if action.action_type == "pick":
                self.executor.remove_scene_obstacle(obj_id)
                rospy.sleep(0.1)
                success = self.executor.pick(pos.x, pos.y, pos.z, label)
            elif action.action_type == "place":
                success = self.executor.place(pos.x, pos.y, pos.z, label)
                if success:
                    cat = self.object_catalog.get(name, {})
                    dims = cat.get("dimensions", {})
                    w = dims.get("w", 0.10)
                    d = dims.get("d", 0.10)
                    h = dims.get("h", 0.05)
                    self.executor.add_scene_obstacle(
                        obj_id, pos.x, pos.y, self.z_surface, w, d, h
                    )
            else:
                rospy.logwarn(f"Unknown action type: {action.action_type}")
                continue

            if success:
                success_count += 1
            else:
                fail_count += 1
                rospy.logwarn(f"Action failed: {action.action_type} {label}, "
                              f"returning to '{recovery_pose}' and continuing")
                if not self._go_pose(recovery_pose):
                    self._log_planning_diagnostics(
                        f"recovery_after_{action.action_type}_{obj_id}"
                    )

        self._go_pose(recovery_pose)
        rospy.loginfo(f"Execution complete: {success_count} succeeded, {fail_count} failed")

    def _go_pose(self, pose_name: str):
        """Move to a named pose, with compatibility fallback for test doubles."""
        if hasattr(self.executor, "go_named_pose"):
            return self.executor.go_named_pose(pose_name)
        if pose_name == "home" and hasattr(self.executor, "go_home"):
            return self.executor.go_home()
        rospy.logwarn(f"Executor missing pose API for '{pose_name}', skipping move")
        return False

    def _log_planning_diagnostics(self, context: str):
        """Emit compact diagnostics to help triage runtime planning/collision failures."""
        scene_count = len(getattr(self.executor, "_scene_objects", {}) or {})
        rospy.logwarn(
            f"[diagnostics] context={context}, scene_obstacles={scene_count}, "
            f"start_pose='{getattr(self, 'start_pose', 'home')}', "
            f"recovery_pose='{getattr(self, 'recovery_pose', getattr(self, 'start_pose', 'home'))}'"
        )


if __name__ == "__main__":
    try:
        node = ExecutorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
