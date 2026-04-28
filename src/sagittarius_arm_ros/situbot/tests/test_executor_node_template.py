"""Pytest templates for situbot executor buffering behavior."""

import types
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("rospy")

_EXECUTOR_NODE_PATH = (
    Path(__file__).resolve().parents[1] / "nodes" / "executor_node.py"
)
_SPEC = importlib.util.spec_from_file_location("executor_node", _EXECUTOR_NODE_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MOD)
ExecutorNode = _MOD.ExecutorNode


class _FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeExecutor:
    def __init__(self):
        self.calls = []

    def _gripper_open(self):
        self.calls.append(("gripper_open",))
        return True

    def go_home(self):
        self.calls.append(("go_home",))
        return True

    def remove_scene_obstacle(self, obj_id):
        self.calls.append(("remove_scene_obstacle", obj_id))

    def pick(self, x, y, z, label):
        self.calls.append(("pick", x, y, z, label))
        return True

    def place(self, x, y, z, label):
        self.calls.append(("place", x, y, z, label))
        return True

    def add_scene_obstacle(self, obj_id, x, y, z_surface, w, d, h):
        self.calls.append(("add_scene_obstacle", obj_id, x, y, z_surface, w, d, h))


def _action(seq, action_type, obj_name="cup", instance_id="cup#1", x=0.2, y=0.1, z=0.0):
    pos = types.SimpleNamespace(x=x, y=y, z=z)
    pose = types.SimpleNamespace(position=pos)
    return types.SimpleNamespace(
        sequence_order=seq,
        action_type=action_type,
        object_name=obj_name,
        instance_id=instance_id,
        pose=pose,
    )


@pytest.fixture
def executor_node_template():
    node = ExecutorNode.__new__(ExecutorNode)
    node._buffer_lock = _FakeLock()
    node.action_buffer = []
    node._first_action_time = None
    node.executor = _FakeExecutor()
    node.object_catalog = {"cup": {"dimensions": {"w": 0.06, "d": 0.06, "h": 0.10}}}
    node.z_surface = 0.0
    return node


def test_execute_buffered_inner_sorts_by_sequence_order(executor_node_template):
    node = executor_node_template
    node.action_buffer = [
        _action(2, "place"),
        _action(0, "pick"),
    ]

    node._execute_buffered_inner()

    op_names = [c[0] for c in node.executor.calls]
    assert op_names[0:2] == ["gripper_open", "go_home"]
    assert "pick" in op_names
    assert "place" in op_names


def test_execute_buffered_inner_pick_removes_scene_obstacle(executor_node_template):
    node = executor_node_template
    node.action_buffer = [_action(0, "pick", instance_id="cup#9")]

    node._execute_buffered_inner()

    assert ("remove_scene_obstacle", "cup#9") in node.executor.calls
