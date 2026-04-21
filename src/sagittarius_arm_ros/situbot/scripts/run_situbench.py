#!/usr/bin/env python3
"""Run full SituBench evaluation (no ROS required).

Iterates all 30 scenarios, generates arrangements, runs roundtrip evaluation,
and reports aggregate metrics.

Usage:
    python run_situbench.py --api-key sk-xxx
    python run_situbench.py --api-key sk-xxx --level functional  # one level only
    python run_situbench.py --api-key sk-xxx --output results/
"""

import argparse
import json
import os
import sys
import time
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from situbot.reasoning.llm_client import DashScopeClient
from situbot.reasoning.situation_reasoner import SituationReasoner
from situbot.evaluation.roundtrip import RoundtripEvaluator
from situbot.evaluation.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Run SituBench evaluation")
    parser.add_argument("--api-key", type=str, default=os.environ.get("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--eval-api-key", type=str, default="",
                        help="Separate API key for evaluator LLM (default: same as --api-key)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible distractor selection")
    parser.add_argument("--model", type=str, default="qwen-plus")
    parser.add_argument("--eval-model", type=str, default="qwen-max")
    parser.add_argument("--endpoint", type=str,
                        default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--config-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "config"))
    parser.add_argument("--level", type=str, choices=["functional", "cultural", "emotional"],
                        help="Run only one difficulty level")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between API calls (rate limiting)")
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: No API key. Set DASHSCOPE_API_KEY or use --api-key")
        sys.exit(1)

    # Load configs
    with open(os.path.join(args.config_dir, "objects.yaml")) as f:
        object_catalog = yaml.safe_load(f)["objects"]
    with open(os.path.join(args.config_dir, "situbench.yaml")) as f:
        scenarios = yaml.safe_load(f)["scenarios"]

    # Filter by level if specified
    if args.level:
        scenarios = [s for s in scenarios if s["level"] == args.level]

    print(f"Running SituBench: {len(scenarios)} scenarios")
    print(f"Reasoning model: {args.model} | Evaluator model: {args.eval_model}")

    # Setup
    workspace = {
        "x_min": 0.15, "x_max": 0.75,
        "y_min": -0.40, "y_max": 0.40,
        "z_min": 0.00, "z_max": 0.60,
        "z_surface": 0.00,
    }

    reasoning_llm = DashScopeClient(
        endpoint=args.endpoint, api_key=args.api_key,
        model=args.model, temperature=0.3,
    )
    eval_llm = DashScopeClient(
        endpoint=args.endpoint, api_key=args.eval_api_key or args.api_key,
        model=args.eval_model, temperature=0.0,
    )
    reasoner = SituationReasoner(
        llm_client=reasoning_llm, workspace_bounds=workspace,
        object_catalog=object_catalog,
    )
    evaluator = RoundtripEvaluator(
        evaluator_llm=eval_llm,
        all_scenarios=scenarios,
        num_candidates=args.num_candidates,
        seed=args.seed,
    )

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Run all scenarios
    all_results = []
    for i, scenario in enumerate(scenarios):
        sid = scenario["id"]
        situation = scenario["situation"]
        objects = scenario.get("objects", [obj["name"] for obj in object_catalog])

        print(f"\n[{i+1}/{len(scenarios)}] {sid}: {situation[:50]}...")

        try:
            # Stage 1-3: Generate arrangement
            result = reasoner.reason(situation, objects)
            placements = [
                {"name": p.name, "x": p.x, "y": p.y, "z": p.z}
                for p in result.placements
            ]
            print(f"  → {len(placements)} placements, layout: {result.layout_description[:60]}")

            time.sleep(args.delay)  # rate limiting

            # Stage 4: Roundtrip evaluation
            eval_result = evaluator.evaluate(situation, placements)
            eval_result["scenario_id"] = sid
            eval_result["level"] = scenario["level"]
            eval_result["layout_description"] = result.layout_description

            status = "✓" if eval_result["correct"] else "✗"
            print(f"  → Roundtrip: {status} (conf={eval_result['confidence']:.2f})")

            all_results.append(eval_result)

            # Save per-scenario result
            with open(os.path.join(args.output, f"{sid}.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "scenario": scenario,
                    "placements": placements,
                    "reasoning_trace": result.reasoning_trace,
                    "evaluation": eval_result,
                }, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"  → ERROR: {e}")
            all_results.append({
                "scenario_id": sid, "level": scenario["level"],
                "correct": False, "ground_truth": situation,
                "error": str(e),
            })

        time.sleep(args.delay)

    # Compute and print metrics
    metrics = compute_metrics(all_results)
    print(f"\n{'='*60}")
    print(f"RESULTS: {metrics['overall']['correct']}/{metrics['overall']['total']} "
          f"({metrics['overall']['accuracy']:.1%} roundtrip accuracy)")
    for level, lm in metrics.get("by_level", {}).items():
        print(f"  {level:12s}: {lm['correct']}/{lm['total']} ({lm['accuracy']:.1%})")
    print(f"{'='*60}")

    # Save aggregate results
    with open(os.path.join(args.output, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()
