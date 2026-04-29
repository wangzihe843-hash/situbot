#!/usr/bin/env python3
"""ROS node: pick-and-place sequence planner.

Subscribes to ArrangementPlan, computes collision-free pick-place sequence,
publishes PlannedAction messages in order.

Includes workspace clamping to handle upstream reasoning modules that
may output coordinates outside the robot's reachable workspace.
"""

import rospy
import yaml

from geometry_msgs.msg import Pose, Point, Quaternion
from situbot.msg import ArrangementPlan, PlannedAction, DetectedObjects
from situbot.reasoning.situation_reasoner import Placement
from situbot.planning.sequence_planner import SequencePlanner


class PlannerNode:
    """ROS wrapper for SequencePlanner."""

    def __init__(self):
        rospy.init_node("situbot_planner", anonymous=False)

        self.workspace = {
            "x_min": rospy.get_param("~workspace/table/x_min", 0.15),
            "x_max": rospy.get_param("~workspace/table/x_max", 0.75),
            "y_min": rospy.get_param("~workspace/table/y_min", -0.40),
            "y_max": rospy.get_param("~workspace/table/y_max", 0.40),
            "z_min": rospy.get_param("~workspace/table/z_min", 0.00),
            "z_max": rospy.get_param("~workspace/table/z_max", 0.60),
            "z_surface": rospy.get_param("~workspace/table/z_surface", 0.00),
        }

        objects_file = rospy.get_param("~objects_file", "")
        catalog = {}
        if objects_file:
            with open(objects_file) as f:
                data = yaml.safe_load(f)
            catalog = {obj["name"]: obj for obj in data.get("objects", [])}

        lift_height = rospy.get_param("~gripper/lift_height", 0.08)

        self.planner = SequencePlanner(
            workspace_bounds=self.workspace,
            object_catalog=catalog,
            lift_height=lift_height,
        )

        self.current_positions = {}
        self.current_names = {}

        self.sub_plan = rospy.Subscriber(
            "/situbot_reasoning/arrangement_plan",
            ArrangementPlan, self.plan_callback, queue_size=1,
        )
        self.sub_objects = rospy.Subscriber(
            "/situbot_perception/detected_objects",
            DetectedObjects, self.objects_callback, queue_size=1,
        )

        self.pub = rospy.Publisher(
            "~planned_actions", PlannedAction, queue_size=10,
        )

        rospy.loginfo("PlannerNode ready.")

    def objects_callback(self, msg: DetectedObjects):
        """Update current object positions from perception (full snapshot)."""
        self.current_positions = {}
        self.current_names = {}
        for obj in msg.objects:
            key = obj.instance_id or obj.name
            self.current_positions[key] = (obj.x, obj.y, obj.z)
            self.current_names[key] = obj.name
            if obj.name not in self.current_positions:
                self.current_positions[obj.name] = (obj.x, obj.y, obj.z)
                self.current_names[obj.name] = obj.name

    def _clamp_to_workspace(self, placements):
        """Clamp target coordinates into workspace bounds with inward margin.

        Upstream reasoning modules may output coordinates in a different
        frame (e.g. normalised [0,1] or [-1,1]).  This method detects
        out-of-bounds placements and remaps them:

        1. If ANY placement is outside bounds, check whether all coords
           look like a normalised space (all in [0,1] or [-1,1]) and
           remap the entire batch linearly into the workspace.
        2. Otherwise, clamp individual outliers to the nearest edge
           with a small inward margin so objects don't sit on the boundary.
        """
        b = self.workspace
        margin = 0.03  # keep objects 3cm inside edges

        x_lo, x_hi = b["x_min"] + margin, b["x_max"] - margin
        y_lo, y_hi = b["y_min"] + margin, b["y_max"] - margin

        xs = [p.x for p in placements]
        ys = [p.y for p in placements]

        any_oob = any(
            p.x < b["x_min"] or p.x > b["x_max"] or
            p.y < b["y_min"] or p.y > b["y_max"]
            for p in placements
        )

        if not any_oob:
            return placements

        clamped_count = 0

        # Detect normalised coordinate space: all values in [-1, 1]
        all_normalised = (
            all(-1.0 <= v <= 1.0 for v in xs) and
            all(-1.0 <= v <= 1.0 for v in ys)
        )

        if all_normalised and len(placements) > 1:
            # Linear remap from the actual value range into workspace
            src_x_min, src_x_max = min(xs), max(xs)
            src_y_min, src_y_max = min(ys), max(ys)

            def remap(val, src_lo, src_hi, dst_lo, dst_hi):
                if src_hi - src_lo < 1e-8:
                    return (dst_lo + dst_hi) / 2
                return dst_lo + (val - src_lo) / (src_hi - src_lo) * (dst_hi - dst_lo)

            for p in placements:
                old_x, old_y = p.x, p.y
                p.x = remap(p.x, src_x_min, src_x_max, x_lo, x_hi)
                p.y = remap(p.y, src_y_min, src_y_max, y_lo, y_hi)
                p.z = b["z_surface"]
                clamped_count += 1
                rospy.loginfo(
                    f"Remapped {p.name}: ({old_x:.3f}, {old_y:.3f}) -> "
                    f"({p.x:.3f}, {p.y:.3f})"
                )

            rospy.logwarn(
                f"Detected normalised coordinate space, remapped all "
                f"{clamped_count} placements into workspace"
            )
        else:
            # Hard clamp: individual outliers pulled to nearest valid point
            for p in placements:
                old_x, old_y = p.x, p.y
                p.x = max(x_lo, min(x_hi, p.x))
                p.y = max(y_lo, min(y_hi, p.y))
                p.z = b["z_surface"]
                if old_x != p.x or old_y != p.y:
                    clamped_count += 1
                    rospy.logwarn(
                        f"Clamped {p.name}: ({old_x:.3f}, {old_y:.3f}) -> "
                        f"({p.x:.3f}, {p.y:.3f})"
                    )

            if clamped_count:
                rospy.logwarn(
                    f"Clamped {clamped_count}/{len(placements)} out-of-bounds "
                    f"placements into workspace [{b['x_min']:.2f}..{b['x_max']:.2f}] x "
                    f"[{b['y_min']:.2f}..{b['y_max']:.2f}]"
                )

        return placements

    def plan_callback(self, msg: ArrangementPlan):
        """Receive an arrangement plan and compute pick-place sequence."""
        rospy.loginfo(f"Received plan for: {msg.situation[:60]}...")

        target_placements = []
        for p in msg.placements:
            target_placements.append(Placement(
                name=p.name,
                x=p.target_pose.position.x,
                y=p.target_pose.position.y,
                z=p.target_pose.position.z,
                reason=p.reason,
                role=p.role,
            ))
            target_placements[-1].grounded_instance_id = p.grounded_instance_id

        # Clamp/remap any out-of-bounds targets before planning
        target_placements = self._clamp_to_workspace(target_placements)

        if not self.current_positions:
            rospy.logwarn("No current positions from perception, using spread defaults")
            n = len(target_placements)
            cx = (self.workspace["x_min"] + self.workspace["x_max"]) / 2
            y_lo = self.workspace["y_min"] + 0.03
            y_hi = self.workspace["y_max"] - 0.03
            for i, p in enumerate(target_placements):
                key = getattr(p, "grounded_instance_id", "") or p.name
                if key not in self.current_positions:
                    frac = i / max(n - 1, 1)
                    y_pos = y_lo + frac * (y_hi - y_lo)
                    self.current_positions[key] = (cx, y_pos, self.workspace["z_surface"])
                    self.current_names[key] = p.name

        actions = self.planner.plan(self.current_positions, target_placements)

        for action in actions:
            action_msg = PlannedAction()
            action_msg.action_type = action.action_type
            action_msg.object_name = action.object_name
            action_msg.instance_id = action.instance_id
            action_msg.pose = Pose(
                position=Point(x=action.x, y=action.y, z=action.z),
                orientation=Quaternion(x=0, y=0.707, z=0, w=0.707),
            )
            action_msg.sequence_order = action.sequence_order
            self.pub.publish(action_msg)

        rospy.loginfo(f"Published {len(actions)} planned actions")


if __name__ == "__main__":
    try:
        node = PlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass