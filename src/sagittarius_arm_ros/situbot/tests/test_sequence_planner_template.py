"""Pytest templates for situbot planning logic.

These tests are written as templates you can adapt to your real object schema.
"""

from types import SimpleNamespace
import importlib.util
from pathlib import Path
import sys
import types

_PLANNING_DIR = Path(__file__).resolve().parents[1] / "src" / "situbot" / "planning"

if "situbot" not in sys.modules:
    situbot_pkg = types.ModuleType("situbot")
    situbot_pkg.__path__ = [str(_PLANNING_DIR.parent)]
    sys.modules["situbot"] = situbot_pkg

if "situbot.planning" not in sys.modules:
    planning_pkg = types.ModuleType("situbot.planning")
    planning_pkg.__path__ = [str(_PLANNING_DIR)]
    sys.modules["situbot.planning"] = planning_pkg

_CC_SPEC = importlib.util.spec_from_file_location(
    "situbot.planning.collision_checker", _PLANNING_DIR / "collision_checker.py"
)
_CC_MOD = importlib.util.module_from_spec(_CC_SPEC)
assert _CC_SPEC and _CC_SPEC.loader
_CC_SPEC.loader.exec_module(_CC_MOD)
sys.modules["situbot.planning.collision_checker"] = _CC_MOD

_SP_SPEC = importlib.util.spec_from_file_location(
    "situbot.planning.sequence_planner", _PLANNING_DIR / "sequence_planner.py"
)
_SP_MOD = importlib.util.module_from_spec(_SP_SPEC)
assert _SP_SPEC and _SP_SPEC.loader
_SP_SPEC.loader.exec_module(_SP_MOD)
SequencePlanner = _SP_MOD.SequencePlanner


def _placement(name, x, y, z=0.0, grounded_instance_id="", role="accessible", reason=""):
    p = SimpleNamespace(
        name=name,
        x=x,
        y=y,
        z=z,
        role=role,
        reason=reason,
        grounded_instance_id=grounded_instance_id,
    )
    return p


def _planner():
    workspace = {
        "x_min": 0.15,
        "x_max": 0.75,
        "y_min": -0.40,
        "y_max": 0.40,
        "z_min": 0.0,
        "z_max": 0.60,
        "z_surface": 0.0,
    }
    catalog = {
        "cup": {"name": "cup", "dimensions": {"w": 0.06, "d": 0.06, "h": 0.10}, "graspable": True},
        "box": {"name": "box", "dimensions": {"w": 0.08, "d": 0.08, "h": 0.05}, "graspable": True},
    }
    return SequencePlanner(workspace_bounds=workspace, object_catalog=catalog, lift_height=0.08)


def test_plan_uses_grounded_instance_id_without_keyerror():
    planner = _planner()
    placements = [_placement("cup", 0.35, 0.10, grounded_instance_id="cup#1")]
    current_positions = {"cup#1": (0.25, -0.10, 0.0)}

    actions = planner.plan(current_positions=current_positions, target_placements=placements)

    assert len(actions) == 2
    assert actions[0].action_type == "pick"
    assert actions[0].instance_id == "cup#1"
    assert actions[1].action_type == "place"
    assert actions[1].instance_id == "cup#1"


def test_plan_keeps_two_same_name_instances_separate():
    planner = _planner()
    placements = [
        _placement("cup", 0.30, 0.05, grounded_instance_id="cup#1"),
        _placement("cup", 0.32, -0.05, grounded_instance_id="cup#2"),
    ]
    current_positions = {
        "cup#1": (0.22, 0.05, 0.0),
        "cup#2": (0.22, -0.05, 0.0),
    }

    actions = planner.plan(current_positions=current_positions, target_placements=placements)
    pick_ids = [a.instance_id for a in actions if a.action_type == "pick"]

    assert "cup#1" in pick_ids
    assert "cup#2" in pick_ids
    assert len(pick_ids) == 2
