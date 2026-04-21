#!/usr/bin/env python3
"""ROS node: roundtrip evaluation after rearrangement.

Captures the scene state and runs the roundtrip test to evaluate
whether the arrangement communicates the intended situation.
"""

import rospy
import yaml

from situbot.srv import EvaluateScene, EvaluateSceneResponse
from situbot.msg import DetectedObjects
from situbot.reasoning.llm_client import DashScopeClient
from situbot.evaluation.roundtrip import RoundtripEvaluator


class EvaluatorNode:
    """ROS wrapper for RoundtripEvaluator."""

    def __init__(self):
        rospy.init_node("situbot_evaluator", anonymous=False)

        # Load evaluator LLM config (check private ns first, then shared /situbot/ ns)
        endpoint = rospy.get_param("~evaluator_llm/endpoint",
                                   rospy.get_param("/situbot/evaluator_llm/endpoint",
                                   "https://dashscope.aliyuncs.com/compatible-mode/v1"))
        api_key = rospy.get_param("~evaluator_llm/api_key",
                                  rospy.get_param("/situbot/evaluator_llm/api_key", ""))
        model = rospy.get_param("~evaluator_llm/model",
                                rospy.get_param("/situbot/evaluator_llm/model", "qwen-max"))
        temperature = rospy.get_param("~evaluator_llm/temperature",
                                      rospy.get_param("/situbot/evaluator_llm/temperature", 0.0))
        num_candidates = rospy.get_param("~evaluation/num_candidates", 5)

        # Load SituBench scenarios (supports both YAML and JSON)
        bench_file = rospy.get_param("~situbench_file", "")
        scenarios = []
        if bench_file:
            with open(bench_file) as f:
                if bench_file.endswith(".json"):
                    import json
                    data = json.load(f)
                    # JSON format: list of scenario dicts directly
                    scenarios = data if isinstance(data, list) else data.get("scenarios", [])
                else:
                    data = yaml.safe_load(f)
                    scenarios = data.get("scenarios", [])

        # Initialize evaluator
        eval_llm = DashScopeClient(
            endpoint=endpoint, api_key=api_key, model=model,
            temperature=temperature,
        )
        self.evaluator = RoundtripEvaluator(
            evaluator_llm=eval_llm,
            all_scenarios=scenarios,
            num_candidates=num_candidates,
        )

        # Latest detected objects for building arrangement description
        self.latest_objects = []
        self.sub = rospy.Subscriber(
            "/situbot_perception/detected_objects",
            DetectedObjects, self.objects_callback, queue_size=1,
        )

        # Service
        self.service = rospy.Service(
            "~evaluate_scene", EvaluateScene, self.handle_evaluate
        )

        rospy.loginfo("EvaluatorNode ready.")

    def objects_callback(self, msg: DetectedObjects):
        """Update latest scene state."""
        self.latest_objects = [
            {"name": obj.name, "x": obj.x, "y": obj.y, "z": obj.z}
            for obj in msg.objects
        ]

    def handle_evaluate(self, req):
        """Service handler: run roundtrip evaluation."""
        resp = EvaluateSceneResponse()

        if not self.latest_objects:
            rospy.logwarn("No scene data available for evaluation")
            resp.predicted_situation = ""
            resp.confidence = 0.0
            resp.reasoning = "No scene data available"
            return resp

        # Use explicit ground_truth field if provided; fall back to [0] for
        # backwards compatibility with older service callers.
        if not req.candidate_situations:
            rospy.logerr("No candidate situations provided")
            resp.reasoning = "No candidates provided"
            return resp

        ground_truth = req.ground_truth_situation or req.candidate_situations[0]

        result = self.evaluator.evaluate(
            ground_truth_situation=ground_truth,
            placements=self.latest_objects,
        )

        resp.predicted_situation = result.get("predicted", "")
        resp.confidence = result.get("confidence", 0.0)
        resp.reasoning = result.get("reasoning", "")

        rospy.loginfo(f"Evaluation: predicted='{resp.predicted_situation}', "
                      f"correct={result.get('correct')}, conf={resp.confidence:.2f}")

        return resp


if __name__ == "__main__":
    try:
        node = EvaluatorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
