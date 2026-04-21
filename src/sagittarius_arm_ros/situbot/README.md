# SituBot — Situation-Conditioned Object Rearrangement

A ROS (Noetic) package that rearranges tabletop objects to reflect a described human situation, using LLM reasoning and a 6-DOF sagittarius arm.

## Architecture

Five ROS nodes form a linear pipeline:

```
Camera → Perception → Reasoning → Planner → Executor
                                      ↑
                                  Evaluator (offline)
```

| Node | Purpose | Key topic/service |
|---|---|---|
| `perception_node` | YOLO-World open-vocab detection (HSV fallback) | pub: `~detected_objects` |
| `reasoning_node` | 3-stage LLM chain: needs → roles → zones | srv: `~get_arrangement` |
| `planner_node` | Collision-aware pick-place sequencing | pub: `~planned_actions` |
| `executor_node` | MoveIt pick/place on sagittarius arm | sub: planned_actions |
| `evaluator_node` | Roundtrip Test (blind LLM guesses situation) | srv: `~evaluate_scene` |

## Reasoning Pipeline (V-CAGE–inspired)

Based on ideas from V-CAGE (arXiv:2604.09036):

1. **Need Inference** — LLM analyses situation → functional, emotional, cultural needs
2. **Object Relevance** — each object assigned a role: `prominent` / `accessible` / `peripheral` / `remove`
3. **Zone Assignment** — LLM outputs qualitative zone names (3×3 grid), converted to coordinates by `ZoneMapper` + L-BFGS-B optimiser (`PlacementOptimiser`)

Legacy exact-coordinate mode available for ablation (`--use-legacy-coords`).

## SituBench

30 scenarios × 3 difficulty levels in `config/situbench.yaml`:
- **Functional** (F01–F10): exam prep, video call, cooking, etc.
- **Cultural** (C01–C10): Chinese elders visiting, Ramadan, Japanese business partner, etc.
- **Emotional** (E01–E10): exam failure, job offer, homesickness, pet loss, etc.

Chinese-localised version with distractor objects in `config/situbench_zh.json`.

Evaluation metric: **Roundtrip Accuracy** — a blind evaluator LLM (different model) guesses the situation from the final arrangement among K candidates.

## Directory Layout

```
situbot/
├── config/
│   ├── objects.yaml          # 15-object catalog with dimensions
│   ├── situbench.yaml        # 30 scenarios (English)
│   ├── situbench_zh.json     # 30 scenarios (Chinese, with distractors)
│   └── situbot_params.yaml   # all tuneable parameters
├── launch/
│   ├── situbot_full.launch   # full 5-node pipeline
│   ├── situbot_sim.launch    # Gazebo + MoveIt + pipeline
│   ├── perception.launch     # perception only
│   └── evaluation.launch     # evaluator only
├── msg/                      # DetectedObject(s), ArrangementPlan, PlannedAction, ObjectPlacement
├── srv/                      # GetArrangement, EvaluateScene
├── nodes/                    # ROS node entry points (thin wrappers)
├── scripts/
│   ├── run_situbench.py      # full benchmark (no ROS required)
│   └── run_single.py         # single scenario test
└── src/situbot/
    ├── perception/
    │   ├── detector.py       # YOLO-World detector
    │   └── hsv_fallback.py   # HSV colour fallback
    ├── reasoning/
    │   ├── llm_client.py     # DashScope OpenAI-compatible client
    │   ├── situation_reasoner.py  # 3-stage pipeline + rejection sampling
    │   └── prompts.py        # all prompt templates
    ├── planning/
    │   ├── zone_mapper.py    # qualitative zone → (x,y)
    │   ├── placement_optimizer.py  # L-BFGS-B refinement (scipy)
    │   ├── collision_checker.py    # AABB overlap detection
    │   └── sequence_planner.py     # pick-place ordering
    ├── execution/
    │   └── moveit_executor.py      # MoveIt commander wrapper
    ├── evaluation/
    │   ├── roundtrip.py      # Roundtrip Test evaluator
    │   └── metrics.py        # aggregate metrics
    └── utils/
        ├── transforms.py     # pixel↔world coordinate transforms
        └── visualization.py  # matplotlib arrangement plots
```

## Quick Start

### Standalone (no ROS, no robot)

```bash
# Single scenario
python scripts/run_single.py --situation "A student preparing for exams" \
    --api-key $DASHSCOPE_API_KEY --plot

# Full benchmark
python scripts/run_situbench.py --api-key $DASHSCOPE_API_KEY --output results/
```

### ROS + Gazebo

```bash
# Terminal 1: simulation
roslaunch situbot situbot_sim.launch api_key:=sk-xxx

# Terminal 2: trigger a scenario
rosservice call /situbot_reasoning/get_arrangement \
    "situation: 'A university student preparing for a final exam'"
```

### ROS + Real Robot

```bash
roslaunch situbot situbot_full.launch api_key:=sk-xxx \
    coordinate_mapping_mode:=vision_config_linear \
    vision_config_file:=/path/to/vision_config.yaml
```

## Configuration

All parameters live in `config/situbot_params.yaml`. Key settings:

| Parameter | Default | Notes |
|---|---|---|
| `llm.model` | `qwen-plus` | Reasoning model (Qwen3.5-Plus) |
| `evaluator_llm.model` | `qwen-max` | Must differ from reasoning model |
| `perception.detection_model` | `yolo_world` | or `grounding_dino` with local config/checkpoint |
| `perception.model_weights` | empty | local YOLO-World `.pt` path; can also use `YOLO_WORLD_WEIGHTS` |
| `vcage_enhancements.use_zone_placement` | `true` | `false` for legacy ablation |
| `vcage_enhancements.placement_optimizer.enabled` | `true` | requires scipy |

Supports local models via Ollama (`http://localhost:11434/v1`) or vLLM — see comments in `situbot_params.yaml`.

## Hardware

- **Arm**: Sagittarius 6-DOF (`sagittarius_arm` + `sagittarius_gripper` MoveIt groups)
- **Camera**: USB camera on `/usb_cam/image_raw` by default; set `camera_topic` for RealSense/Gazebo
- **Coordinate mapping**: linear regression from `vision_config.yaml` (calibrated) or workspace-linear fallback

## Dependencies

- ROS Noetic, MoveIt, cv_bridge, tf2_ros
- Python: `openai` or `requests`, `pyyaml`, `numpy`, `opencv-python`
- Optional: `ultralytics` (YOLO-World), `scipy` (L-BFGS-B optimiser), `matplotlib` (plots)

## Team

AIR5021 Group 10 — CUHK(SZ)
