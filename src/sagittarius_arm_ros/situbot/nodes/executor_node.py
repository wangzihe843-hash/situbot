#!/usr/bin/env python3
"""ROS node: MoveIt arm executor.

Subscribes to PlannedAction messages and executes pick/place operations
via MoveIt commander on the sagittarius arm.

Uses sagittarius_arm and sagittarius_gripper planning groups,
with end-effector offset and dynamic approach angles from sgr_ctrl.py.
"""

import threading

import rospy
from collections import defaultdict

from situbot.msg import PlannedAction
from situbot.execution.moveit_executor import MoveItExecutor


class ExecutorNode:
    """ROS wrapper for MoveItExecutor.

    Collects PlannedAction messages, buffers them, and executes
    in sequence_order when a complete batch is received.
    """

    def __init__(self):
        rospy.init_node("situbot_executor", anonymous=False)

        # Load params (matched to sagittarius hardware)
        planning_group = rospy.get_param("~moveit/planning_group", "sagittarius_arm")
        gripper_group = rospy.get_param("~moveit/gripper_group", "sagittarius_gripper")
        planning_time = rospy.get_param("~moveit/planning_time", 5.0)
        max_vel = rospy.get_param("~moveit/max_velocity_scaling", 0.5)
        max_acc = rospy.get_param("~moveit/max_acceleration_scaling", 0.5)
        approach_h = rospy.get_param("~gripper/approach_height", 0.04)
        lift_h = rospy.get_param("~gripper/lift_height", 0.08)

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

        # Action buffer (collect all actions, execute after quiet period)
        self.action_buffer = []
        self.buffer_timer = None
        self.buffer_timeout = rospy.get_param("~buffer_timeout", 2.0)
        self._first_action_time = None
        self._max_buffer_wait = rospy.get_param("~max_buffer_wait", 5.0)
        self._executing = threading.Lock()

        # Subscriber
        self.sub = rospy.Subscriber(
            "/situbot_planner/planned_actions",
            PlannedAction, self.action_callback, queue_size=50,
        )

        rospy.loginfo("ExecutorNode ready. Waiting for planned actions...")

    def action_callback(self, msg: PlannedAction):
        """Buffer incoming actions and schedule execution."""
        self.action_buffer.append(msg)
        now = rospy.Time.now()

        if self._first_action_time is None:
            self._first_action_time = now

        # Cancel any pending timer
        if self.buffer_timer:
            self.buffer_timer.shutdown()

        # If we've been buffering too long, execute immediately
        elapsed = (now - self._first_action_time).to_sec()
        if elapsed >= self._max_buffer_wait:
            self.execute_buffered()
        else:
            # Otherwise reset the quiet-period timer
            self.buffer_timer = rospy.Timer(
                rospy.Duration(self.buffer_timeout),
                self.execute_buffered,
                oneshot=True,
            )

    def execute_buffered(self, event=None):
        """Execute all buffered actions in sequence order."""
        if not self._executing.acquire(blocking=False):
            return  # another thread is already executing
        try:
            self._execute_buffered_inner()
        finally:
            self._executing.release()

    def _execute_buffered_inner(self):
        if not self.action_buffer:
            return

        actions = sorted(self.action_buffer, key=lambda a: a.sequence_order)
        self.action_buffer = []
        self._first_action_time = None  # reset for next batch

        rospy.loginfo(f"Executing {len(actions)} actions...")

        # Safety: open gripper before homing in case we're holding something
        # from a previous failed batch
        rospy.loginfo("Safety: releasing gripper and moving to home...")
        self.executor._gripper_open()
        self.executor.go_home()

        success_count = 0
        fail_count = 0

        for action in actions:
            pos = action.pose.position
            name = action.object_name
            label = f"{name} ({action.instance_id})" if action.instance_id else name

            if action.action_type == "pick":
                success = self.executor.pick(pos.x, pos.y, pos.z, label)
            elif action.action_type == "place":
                success = self.executor.place(pos.x, pos.y, pos.z, label)
            else:
                rospy.logwarn(f"Unknown action type: {action.action_type}")
                continue

            if success:
                success_count += 1
            else:
                fail_count += 1
                rospy.logwarn(f"Action failed: {action.action_type} {label}, "
                              "returning to home and continuing")
                self.executor.go_home()

        # Return home after all actions
        self.executor.go_home()

        rospy.loginfo(f"Execution complete: {success_count} succeeded, {fail_count} failed")


if __name__ == "__main__":
    try:
        node = ExecutorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
