#!/usr/bin/env python3
"""ROS node: LLM-based situation reasoning.

Provides a service that takes a situation description + detected objects,
runs the 3-stage reasoning chain, and returns an ArrangementPlan.
"""

import json
import rospy
import yaml

from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point, Quaternion
from situbot.msg import DetectedObjects, ArrangementPlan, ObjectPlacement
from situbot.srv import GetArrangement, GetArrangementResponse
from situbot.reasoning.llm_client import DashScopeClient
from situbot.reasoning.situation_reasoner import SituationReasoner
from situbot.perception.scene_description import ground_placements_to_scene


class ReasoningNode:
    """ROS wrapper for SituationReasoner."""

    def __init__(self):
        rospy.init_node("situbot_reasoning", anonymous=False)

        endpoint = rospy.get_param("~llm/endpoint",
                                   rospy.get_param("/situbot/llm/endpoint",
                                   "https://dashscope.aliyuncs.com/compatible-mode/v1"))
        api_key = rospy.get_param("~llm/api_key",
                                  rospy.get_param("/situbot/llm/api_key", ""))
        model = rospy.get_param("~llm/model",
                                rospy.get_param("/situbot/llm/model", "qwen-plus"))
        temperature = rospy.get_param("~llm/temperature",
                                      rospy.get_param("/situbot/llm/temperature", 0.3))
        max_tokens = rospy.get_param("~llm/max_tokens",
                                     rospy.get_param("/situbot/llm/max_tokens", 2048))

        if not api_key or api_key == "sk-REPLACE_WITH_YOUR_KEY":
            rospy.logwarn("No valid DashScope API key configured!")

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
        if objects_file:
            with open(objects_file) as f:
                catalog_data = yaml.safe_load(f)
            self.object_catalog = catalog_data.get("objects", [])
        else:
            self.object_catalog = []
            rospy.logwarn("No objects file configured")

        llm_client = DashScopeClient(
            endpoint=endpoint, api_key=api_key, model=model,
            temperature=temperature, max_tokens=max_tokens,
        )
        self.reasoner = SituationReasoner(
            llm_client=llm_client,
            workspace_bounds=self.workspace,
            object_catalog=self.object_catalog,
            use_zone_placement=rospy.get_param(
                "~vcage_enhancements/use_zone_placement", True
            ),
        )

        self.latest_objects = []
        self.latest_detected_objects = []
        self.latest_scene_description = ""

        self.sub = rospy.Subscriber(
            "/situbot_perception/detected_objects",
            DetectedObjects, self.objects_callback, queue_size=1,
        )

        self.service = rospy.Service(
            "~get_arrangement", GetArrangement, self.handle_get_arrangement
        )

        self.pub = rospy.Publisher(
            "~arrangement_plan", ArrangementPlan, queue_size=1, latch=True
        )

        rospy.loginfo("ReasoningNode ready.")

    def objects_callback(self, msg: DetectedObjects):
        """Update latest detected objects."""
        self.latest_detected_objects = list(msg.objects)
        self.latest_scene_description = msg.scene_description
        seen = set()
        self.latest_objects = []
        for obj in msg.objects:
            if obj.name in seen:
                continue
            seen.add(obj.name)
            self.latest_objects.append(obj.name)

    def handle_get_arrangement(self, req):
        """Service handler: generate arrangement for a situation."""
        resp = GetArrangementResponse()

        if not self.latest_objects:
            rospy.logwarn("No detected objects available, using full catalog")
            object_names = [obj["name"] for obj in self.object_catalog]
        else:
            object_names = self.latest_objects

        try:
            result = self.reasoner.reason(req.situation, object_names)

            plan = ArrangementPlan()
            plan.header.stamp = rospy.Time.now()
            plan.situation = result.situation
            plan.scene_description = self.latest_scene_description
            plan.reasoning_trace = result.reasoning_trace

            grounding_infos = ground_placements_to_scene(
                result.placements,
                self.latest_detected_objects,
            )

            for p, grounding in zip(result.placements, grounding_infos):
                placement_msg = ObjectPlacement()
                placement_msg.name = p.name
                placement_msg.grounded_instance_id = grounding.instance_id
                placement_msg.grounded = grounding.grounded
                placement_msg.target_pose = Pose(
                    position=Point(x=p.x, y=p.y, z=p.z),
                    orientation=Quaternion(x=0, y=0.707, z=0, w=0.707),
                )
                placement_msg.reason = p.reason
                placement_msg.role = getattr(p, "role", "")
                placement_msg.grounding_note = grounding.note
                plan.placements.append(placement_msg)
                if (not grounding.grounded) or ("verify calibration" in grounding.note):
                    plan.grounding_warnings.append(grounding.note)

            resp.plan = plan
            resp.success = True
            self.pub.publish(plan)
            rospy.loginfo(f"Generated arrangement with {len(plan.placements)} placements")

        except Exception as e:
            resp.success = False
            resp.error = str(e)
            rospy.logerr(f"Reasoning failed: {e}")

        return resp


if __name__ == "__main__":
    try:
        node = ReasoningNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
