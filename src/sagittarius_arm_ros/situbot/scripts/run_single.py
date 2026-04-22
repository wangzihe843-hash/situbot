#!/usr/bin/env python3
"""Run a single situation through the SituBot reasoning pipeline (no ROS required).

Usage:
    python run_single.py --situation "A student preparing for exams" --api-key sk-xxx
    python run_single.py --scenario-id F01 --api-key sk-xxx
"""

import argparse
import json
import os
import sys
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from situbot.reasoning.llm_client import DashScopeClient
from situbot.reasoning.situation_reasoner import SituationReasoner
from situbot.utils.visualization import plot_arrangement


def main():
    parser = argparse.ArgumentParser(description="Run SituBot reasoning for one situation")
    parser.add_argument("--situation", type=str, help="Situation description text")
    parser.add_argument("--scenario-id", type=str, help="SituBench scenario ID (e.g., F01, C05, E10)")
    parser.add_argument("--api-key", type=str, default=os.environ.get("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--model", type=str, default="qwen-plus")
    parser.add_argument("--endpoint", type=str,
                        default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--config-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "config"))
    parser.add_argument("--output", type=str, help="Save result JSON to file")
    parser.add_argument("--plot", action="store_true", help="Show arrangement plot")
    parser.add_argument("--save-plot", type=str, help="Save plot to file")
    parser.add_argument("--use-legacy-coords", action="store_true",
                        help="Use legacy exact-coordinate placement instead of zone-based (for ablation)")
    parser.add_argument("--rejection-samples", type=int, default=1,
                        help="Number of rejection sampling candidates (1=disabled). "
                             "Requires --eval-model and evaluator LLM.")
    parser.add_argument("--eval-model", type=str, default="qwen-max",
                        help="Evaluator model for rejection sampling")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: No API key. Set DASHSCOPE_API_KEY or use --api-key")
        sys.exit(1)

    config_dir = args.config_dir
    with open(os.path.join(config_dir, "objects.yaml")) as f:
        objects_data = yaml.safe_load(f)
    object_catalog = objects_data["objects"]

    with open(os.path.join(config_dir, "situbench.yaml")) as f:
        bench_data = yaml.safe_load(f)
    scenarios = bench_data["scenarios"]

    situation = args.situation
    scenario = None
    if args.scenario_id:
        for s in scenarios:
            if s["id"] == args.scenario_id:
                scenario = s
                situation = s["situation"]
                break
        if not scenario:
            print(f"ERROR: Scenario {args.scenario_id} not found in SituBench")
            sys.exit(1)

    if not situation:
        print("ERROR: Provide --situation or --scenario-id")
        sys.exit(1)

    if scenario and "objects" in scenario:
        available_objects = scenario["objects"]
    else:
        available_objects = [obj["name"] for obj in object_catalog]

    workspace = {
        "x_min": 0.15, "x_max": 0.75,
        "y_min": -0.40, "y_max": 0.40,
        "z_min": 0.00, "z_max": 0.60,
        "z_surface": 0.00,
    }

    llm = DashScopeClient(
        endpoint=args.endpoint, api_key=args.api_key,
        model=args.model, temperature=0.3,
    )
    reasoner = SituationReasoner(
        llm_client=llm, workspace_bounds=workspace,
        object_catalog=object_catalog,
        use_zone_placement=not args.use_legacy_coords,
    )

    print(f"\n{'='*60}")
    print(f"Situation: {situation}")
    print(f"Objects: {', '.join(available_objects)}")
    print(f"{'='*60}\n")

    if args.rejection_samples > 1:
        from situbot.evaluation.roundtrip import RoundtripEvaluator
        eval_llm = DashScopeClient(
            endpoint=args.endpoint, api_key=args.api_key,
            model=args.eval_model, temperature=0.0,
        )
        evaluator = RoundtripEvaluator(
            evaluator_llm=eval_llm,
            all_scenarios=scenarios,
        )

        def _eval_fn(arrangement):
            placements = [
                {"name": p.name, "x": p.x, "y": p.y, "z": p.z}
                for p in arrangement.placements
            ]
            r = evaluator.evaluate(situation, placements)
            return r.get("confidence", 0.0)

        result = reasoner.reason_with_rejection_sampling(
            situation, available_objects, _eval_fn, args.rejection_samples,
        )
    else:
        result = reasoner.reason(situation, available_objects)

    print(f"\n{'='*60}")
    print(f"Layout: {result.layout_description}")
    print(f"\nPlacements ({len(result.placements)}):")
    for p in result.placements:
        zone_str = f" [{p.zone}]" if hasattr(p, 'zone') and p.zone else ""
        print(f"  [{p.role:11s}] {p.name:18s} -> ({p.x:.3f}, {p.y:.3f}){zone_str}  {p.reason}")
    print(f"{'='*60}\n")

    if args.output:
        output_data = {
            "situation": situation,
            "scenario_id": args.scenario_id,
            "layout_description": result.layout_description,
            "placements": [
                {"name": p.name, "x": p.x, "y": p.y, "z": p.z,
                 "role": p.role, "zone": getattr(p, 'zone', ''),
                 "reason": p.reason}
                for p in result.placements
            ],
            "reasoning_trace": result.reasoning_trace,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Saved result to {args.output}")

    if args.plot or args.save_plot:
        catalog_dict = {obj["name"]: obj for obj in object_catalog}
        placements_for_plot = [
            {"name": p.name, "x": p.x, "y": p.y, "role": p.role}
            for p in result.placements
        ]
        plot_arrangement(
            placements_for_plot, workspace,
            title=situation,
            object_catalog=catalog_dict,
            save_path=args.save_plot,
            show=args.plot,
        )


if __name__ == "__main__":
    main()
