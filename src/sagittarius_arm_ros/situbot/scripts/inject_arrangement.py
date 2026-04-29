#!/usr/bin/env python3
"""Publish an ArrangementPlan from arrangement_result.json.

Bypasses the reasoning node — feeds Member A/B's output directly into
the planner and executor nodes for end-to-end hardware testing.

Usage (inside the Docker container, with ROS running):

  # Terminal: make sure planner + executor are up (situbot_full.launch)

  # Then in a separate terminal:
  rosrun situbot inject_arrangement.py \
    --input /path/to/arrangement_result.json \
    --scenario 0

  # Or with roslaunch (if you prefer):
  # Just run this script directly — it inits its own node.

What happens:
  1. Reads the JSON file
  2. Publishes an ArrangementPlan message to /situbot_reasoning/arrangement_plan
  3. Your planner_node picks it up, clamps coordinates, plans sequence
  4. Your executor_node executes the pick-place actions on the real arm
"""

import argparse
import json
import sys
import os

import rospy
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Header
from situbot.msg import ArrangementPlan, ObjectPlacement


def load_scenario(input_path, scenario_idx):
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    if scenario_idx >= len(data):
        rospy.logerr(f"Scenario index {scenario_idx} out of range (have {len(data)})")
        sys.exit(1)

    return data[scenario_idx]


def build_arrangement_plan(scenario):
    """Convert JSON scenario to ArrangementPlan ROS message."""
    plan = ArrangementPlan()
    plan.header = Header()
    plan.header.stamp = rospy.Time.now()
    plan.situation = scenario["situation"]
    plan.scene_description = scenario.get("layout_summary", "")
    plan.reasoning_trace = "injected from arrangement_result.json"
    plan.grounding_warnings = []

    for item in scenario["object_arrangements"]:
        coord = item["coordinate"]

        placement = ObjectPlacement()
        placement.name = item["name"]
        placement.grounded_instance_id = item["name"]  # use name as instance id
        placement.grounded = True
        placement.target_pose = Pose(
            position=Point(
                x=float(coord["x"]),
                y=float(coord["y"]),
                z=float(coord.get("z", 0.0)),
            ),
            orientation=Quaternion(x=0, y=0.707, z=0, w=0.707),  # top-down
        )
        placement.reason = item.get("reasoning", "")
        placement.role = item.get("role", "accessible")
        placement.grounding_note = "injected"

        plan.placements.append(placement)

    return plan


def main():
    parser = argparse.ArgumentParser(
        description="Inject arrangement_result.json into ROS planner pipeline"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to arrangement_result.json")
    parser.add_argument("--scenario", type=int, default=0,
                        help="Scenario index in the JSON array (default: 0)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds to wait for subscribers before publishing (default: 2)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Publish N times with 1s gap (default: 1)")

    # rospy.init_node eats some args, so parse known only
    args, _ = parser.parse_known_args()

    rospy.init_node("inject_arrangement", anonymous=True)

    # Load and build message
    scenario = load_scenario(args.input, args.scenario)
    plan = build_arrangement_plan(scenario)

    rospy.loginfo(f"Loaded scenario: {scenario['scenario_id']}")
    rospy.loginfo(f"Situation: {scenario['situation']}")
    rospy.loginfo(f"Objects: {len(plan.placements)}")
    for p in plan.placements:
        pos = p.target_pose.position
        rospy.loginfo(
            f"  {p.name:20s} ({pos.x:6.2f}, {pos.y:5.2f})  [{p.role}]"
        )

    # Publish on the same topic the reasoning node uses (latched)
    pub = rospy.Publisher(
        "/situbot_reasoning/arrangement_plan",
        ArrangementPlan,
        queue_size=1,
        latch=True,
    )

    # Wait for planner to subscribe
    rospy.loginfo(f"Waiting {args.delay}s for subscribers...")
    rospy.sleep(args.delay)

    n_subs = pub.get_num_connections()
    if n_subs == 0:
        rospy.logwarn(
            "No subscribers on /situbot_reasoning/arrangement_plan! "
            "Is planner_node running? (roslaunch situbot situbot_full.launch)"
        )
    else:
        rospy.loginfo(f"{n_subs} subscriber(s) connected")

    for i in range(args.repeat):
        plan.header.stamp = rospy.Time.now()
        pub.publish(plan)
        rospy.loginfo(f"Published ArrangementPlan ({i+1}/{args.repeat})")
        if i < args.repeat - 1:
            rospy.sleep(1.0)

    # Keep alive briefly so the latched message is received
    rospy.loginfo("Done. Keeping node alive for 5s (latched message)...")
    rospy.sleep(5.0)
    rospy.loginfo("Exiting.")


if __name__ == "__main__":
    main()