#!/usr/bin/env python3
"""Offline test for Member D's planner pipeline.

Reads arrangement_result.json (from Member A/B's reasoning output),
runs it through the full planning pipeline:
  coordinate clamping → collision checking → sequence planning

No API key, no ROS, no hardware required.

Usage:
    python3 test_planner_offline.py
    python3 test_planner_offline.py --input /path/to/arrangement_result.json
    python3 test_planner_offline.py --scenario 0     # pick scenario by index
    python3 test_planner_offline.py --plot            # show arrangement plot
"""

import argparse
import json
import os
import sys

# Add src to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, "..", "src"))

from situbot.reasoning.situation_reasoner import Placement
from situbot.planning.sequence_planner import SequencePlanner
from situbot.planning.collision_checker import CollisionChecker

# Default workspace bounds (must match situbot_params.yaml)
WORKSPACE = {
    "x_min": 0.15, "x_max": 0.75,
    "y_min": -0.40, "y_max": 0.40,
    "z_min": 0.00, "z_max": 0.60,
    "z_surface": 0.00,
}


def clamp_to_workspace(placements, workspace):
    """Clamp/remap out-of-bounds coordinates into workspace.
    
    Same logic as planner_node.py _clamp_to_workspace().
    """
    b = workspace
    margin = 0.03
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
        print("  All coordinates in bounds — no clamping needed.")
        return placements

    all_normalised = (
        all(-1.0 <= v <= 1.0 for v in xs) and
        all(-1.0 <= v <= 1.0 for v in ys)
    )

    if all_normalised and len(placements) > 1:
        src_x_min, src_x_max = min(xs), max(xs)
        src_y_min, src_y_max = min(ys), max(ys)

        def remap(val, src_lo, src_hi, dst_lo, dst_hi):
            if src_hi - src_lo < 1e-8:
                return (dst_lo + dst_hi) / 2
            return dst_lo + (val - src_lo) / (src_hi - src_lo) * (dst_hi - dst_lo)

        print(f"  Detected normalised coordinates, remapping into workspace:")
        print(f"    x: [{src_x_min:.2f}, {src_x_max:.2f}] -> [{x_lo:.2f}, {x_hi:.2f}]")
        print(f"    y: [{src_y_min:.2f}, {src_y_max:.2f}] -> [{y_lo:.2f}, {y_hi:.2f}]")

        for p in placements:
            old_x, old_y = p.x, p.y
            p.x = remap(p.x, src_x_min, src_x_max, x_lo, x_hi)
            p.y = remap(p.y, src_y_min, src_y_max, y_lo, y_hi)
            p.z = b["z_surface"]
            print(f"    {p.name:20s} ({old_x:6.2f}, {old_y:5.2f}) -> ({p.x:.3f}, {p.y:.3f})")
    else:
        print(f"  Clamping outliers to workspace bounds:")
        for p in placements:
            old_x, old_y = p.x, p.y
            p.x = max(x_lo, min(x_hi, p.x))
            p.y = max(y_lo, min(y_hi, p.y))
            p.z = b["z_surface"]
            if old_x != p.x or old_y != p.y:
                print(f"    {p.name:20s} ({old_x:6.2f}, {old_y:5.2f}) -> ({p.x:.3f}, {p.y:.3f})")

    return placements


def simulate_current_positions(placements, workspace):
    """Simulate 'current' object positions (as if perception placed them).
    
    Spreads objects evenly across the workspace centre line.
    """
    n = len(placements)
    cx = (workspace["x_min"] + workspace["x_max"]) / 2
    y_lo = workspace["y_min"] + 0.05
    y_hi = workspace["y_max"] - 0.05
    positions = {}
    for i, p in enumerate(placements):
        frac = i / max(n - 1, 1)
        y_pos = y_lo + frac * (y_hi - y_lo)
        positions[p.name] = (cx, y_pos, workspace["z_surface"])
    return positions


def main():
    parser = argparse.ArgumentParser(description="Offline planner test for Member D")
    parser.add_argument("--input", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "..", "..", "..", "..", "..",
                                             os.path.expanduser("~"),
                                             ".openclaw", "workspace",
                                             "arrangement_result.json"),
                        help="Path to arrangement_result.json")
    parser.add_argument("--objects", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "config", "objects.yaml"),
                        help="Path to objects.yaml")
    parser.add_argument("--scenario", type=int, default=0,
                        help="Scenario index in the JSON array (default: 0)")
    parser.add_argument("--plot", action="store_true",
                        help="Show matplotlib plot of arrangement")
    args = parser.parse_args()

    # --- Load input ---
    input_path = args.input
    if not os.path.exists(input_path):
        # Try workspace path directly
        input_path = os.path.join(os.path.expanduser("~"),
                                   ".openclaw", "workspace",
                                   "arrangement_result.json")
    if not os.path.exists(input_path):
        print(f"ERROR: Cannot find arrangement_result.json at {args.input}")
        print("  Use --input /path/to/arrangement_result.json")
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    if args.scenario >= len(data):
        print(f"ERROR: Scenario index {args.scenario} out of range (have {len(data)} scenarios)")
        sys.exit(1)

    scenario = data[args.scenario]
    print(f"{'=' * 70}")
    print(f"Scenario: {scenario['scenario_id']}")
    print(f"Situation: {scenario['situation']}")
    print(f"Objects: {len(scenario['object_arrangements'])}")
    print(f"{'=' * 70}")

    # --- Load object catalog ---
    objects_path = args.objects
    catalog = {}
    if os.path.exists(objects_path):
        import yaml
        with open(objects_path) as f:
            obj_data = yaml.safe_load(f)
        catalog = {obj["name"]: obj for obj in obj_data.get("objects", [])}
        print(f"\nLoaded {len(catalog)} objects from catalog")
    else:
        print(f"\nWARNING: objects.yaml not found at {objects_path}, using defaults")

    # --- Build placements ---
    print(f"\n--- Step 1: Parse input coordinates ---")
    placements = []
    for item in scenario["object_arrangements"]:
        coord = item["coordinate"]
        p = Placement(
            name=item["name"],
            x=coord["x"],
            y=coord["y"],
            z=coord.get("z", WORKSPACE["z_surface"]),
            reason=item.get("reasoning", ""),
            role=item.get("role", "accessible"),
        )
        in_bounds = (WORKSPACE["x_min"] <= p.x <= WORKSPACE["x_max"] and
                     WORKSPACE["y_min"] <= p.y <= WORKSPACE["y_max"])
        tag = "OK" if in_bounds else "OOB"
        print(f"  {p.name:20s} ({p.x:6.2f}, {p.y:5.2f})  {tag:3s}  [{p.role}]")
        placements.append(p)

    # --- Clamp ---
    print(f"\n--- Step 2: Coordinate clamping ---")
    placements = clamp_to_workspace(placements, WORKSPACE)

    # --- Simulate current positions ---
    print(f"\n--- Step 3: Simulate current positions (as if from perception) ---")
    current_positions = simulate_current_positions(placements, WORKSPACE)
    for name, (x, y, z) in current_positions.items():
        print(f"  {name:20s} currently at ({x:.3f}, {y:.3f}, {z:.3f})")

    # --- Plan sequence ---
    print(f"\n--- Step 4: Sequence planning ---")
    planner = SequencePlanner(
        workspace_bounds=WORKSPACE,
        object_catalog=catalog,
        lift_height=0.08,
    )
    actions = planner.plan(current_positions, placements)

    if not actions:
        print("  WARNING: No actions planned!")
    else:
        print(f"\n  Planned {len(actions)} actions:")
        print(f"  {'#':>3s}  {'Type':6s}  {'Object':20s}  {'X':>7s}  {'Y':>7s}  {'Z':>7s}")
        print(f"  {'---':>3s}  {'------':6s}  {'--------------------':20s}  {'-------':>7s}  {'-------':>7s}  {'-------':>7s}")
        for a in actions:
            print(f"  {a.sequence_order:3d}  {a.action_type:6s}  {a.object_name:20s}  {a.x:7.3f}  {a.y:7.3f}  {a.z:7.3f}")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    pick_count = sum(1 for a in actions if a.action_type == "pick")
    place_count = sum(1 for a in actions if a.action_type == "place")
    print(f"Summary: {pick_count} picks + {place_count} places = {len(actions)} total actions")
    unique_objects = set(a.object_name for a in actions)
    skipped = set(p.name for p in placements) - unique_objects
    if skipped:
        print(f"Skipped (non-graspable or missing): {', '.join(skipped)}")
    print(f"{'=' * 70}")

    # --- Save output ---
    output_path = os.path.join(os.path.dirname(input_path) if os.path.dirname(input_path) else ".",
                                "planner_test_output.json")
    output = {
        "scenario_id": scenario["scenario_id"],
        "situation": scenario["situation"],
        "actions": [
            {
                "sequence_order": a.sequence_order,
                "action_type": a.action_type,
                "object_name": a.object_name,
                "instance_id": a.instance_id,
                "x": a.x, "y": a.y, "z": a.z,
            }
            for a in actions
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved action sequence to: {output_path}")

    # --- Plot ---
    if args.plot:
        try:
            _plot(placements, actions, current_positions, catalog, scenario)
        except ImportError:
            print("\nWARNING: matplotlib not installed, skipping plot")

    return 0


def _plot(placements, actions, current_positions, catalog, scenario):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, title, positions in [
        (ax1, "Current (before)", current_positions),
        (ax2, "Target (after)", {p.name: (p.x, p.y) for p in placements}),
    ]:
        b = WORKSPACE
        ax.set_xlim(b["x_min"] - 0.05, b["x_max"] + 0.05)
        ax.set_ylim(b["y_min"] - 0.05, b["y_max"] + 0.05)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("X (front → back)")
        ax.set_ylabel("Y (right → left)")

        # Draw workspace
        ws = patches.Rectangle(
            (b["x_min"], b["y_min"]),
            b["x_max"] - b["x_min"], b["y_max"] - b["y_min"],
            linewidth=2, edgecolor="black", facecolor="lightyellow",
        )
        ax.add_patch(ws)

        # Draw objects
        role_colors = {
            "prominent": "#e74c3c",
            "accessible": "#3498db",
            "peripheral": "#95a5a6",
            "remove": "#7f8c8d",
        }
        for p in placements:
            pos = positions.get(p.name)
            if pos is None:
                continue
            x, y = pos[0] if len(pos) == 3 else pos[0], pos[1]
            info = catalog.get(p.name, {})
            dims = info.get("dimensions", {"w": 0.08, "d": 0.08})
            w, d = dims.get("w", 0.08), dims.get("d", 0.08)
            color = role_colors.get(p.role, "#3498db")
            rect = patches.Rectangle(
                (x - d / 2, y - w / 2), d, w,
                linewidth=1, edgecolor=color, facecolor=color, alpha=0.4,
            )
            ax.add_patch(rect)
            ax.annotate(p.name, (x, y), fontsize=7, ha="center", va="center")

    fig.suptitle(f"{scenario['scenario_id']}: {scenario['situation']}", fontsize=11)
    plt.tight_layout()
    plt.savefig("planner_test_plot.png", dpi=150)
    print(f"Saved plot to: planner_test_plot.png")
    plt.show()


if __name__ == "__main__":
    sys.exit(main())