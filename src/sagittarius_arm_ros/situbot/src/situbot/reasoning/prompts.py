#!/usr/bin/env python3
"""Prompt templates for SituBot's situation reasoning pipeline.

Three-stage chain-of-thought:
  1. Need Inference — what does the person in this situation need?
  2. Object Relevance — which objects serve those needs, and how?
  3. Spatial Arrangement — where should each object go on the table?

Plus a separate Roundtrip Evaluation prompt.
"""

# ==============================================================================
# STAGE 1: NEED INFERENCE
# ==============================================================================

NEED_INFERENCE_SYSTEM = """You are a human behavior expert with deep understanding of emotions, cultural norms, and practical needs. Your task is to analyze a described human situation and infer what the person needs from their immediate physical environment.

CRITICAL RULE: If the situation involves grief, severe stress, or emotional vulnerability,
you MUST prioritize psychological safety and emotional comfort over functional efficiency
or typical "tidiness". If the situation involves specific cultural or religious contexts,
those norms override standard arrangement rules.

Think step by step:
1. What is the person's primary activity or state?
2. What are their emotional needs right now?
3. What are their practical/functional needs?
4. Are there any cultural or social considerations?
5. What environmental qualities would help? (calm, energetic, organized, cozy, etc.)

Respond in JSON format."""

NEED_INFERENCE_USER = """Situation: "{situation}"

Analyze this person's needs. Return a JSON object with:
{{
  "primary_activity": "what they are mainly doing or experiencing",
  "emotional_state": "their likely emotional state",
  "functional_needs": ["list of practical things they need from the environment"],
  "emotional_needs": ["list of emotional/psychological needs"],
  "cultural_considerations": ["any cultural norms that apply, or empty list"],
  "desired_atmosphere": "one-line description of ideal environment feel",
  "reasoning": "2-3 sentences explaining your analysis"
}}

Example for "A university student preparing for a final exam in 2 days":
{{
  "primary_activity": "intensive studying with time pressure",
  "emotional_state": "stressed but determined, needs focus",
  "functional_needs": ["easy access to study materials", "writing tools within reach", "hydration and light snacks for energy", "good lighting", "minimal distractions"],
  "emotional_needs": ["sense of control and organization", "reduced anxiety through preparedness", "small comforts to sustain long sessions"],
  "cultural_considerations": [],
  "desired_atmosphere": "focused, organized, and quietly supportive — a battle station for studying",
  "reasoning": "With exams in 2 days, the student needs maximum focus. Everything study-related should be prominent and accessible, while distractions (phone, non-essential items) should be minimized. Small comforts like a warm drink and snacks sustain marathon study sessions."
}}"""


# ==============================================================================
# STAGE 2: OBJECT RELEVANCE
# ==============================================================================

OBJECT_RELEVANCE_SYSTEM = """You are a thoughtful interior arrangement expert. Given a person's situation and needs, and a list of available objects on a table, you determine each object's role and importance.

Think step by step for each object:
1. Does this object directly serve the person's primary activity or emotional state?
2. Could it provide comfort, utility, or atmosphere?
3. Could it be a distraction or feel inappropriate in this situation?
4. Should it be grouped with other objects for functional or aesthetic reasons?

Then assign a role:
- "prominent": should be front-and-center, easily accessible, key to the situation
- "accessible": should be within reach but not dominant
- "peripheral": push to edges or corners, not important right now
- "remove": should ideally be removed from the scene (but since we can only rearrange on the table, push to far corner or hide behind larger objects). Use this for major distractions OR triggers of negative emotions (e.g., reminders of a recent loss).

Also specify WHY — the reasoning connects the object to the person's needs.

Respond in JSON format."""

OBJECT_RELEVANCE_USER = """Situation: "{situation}"

Person's needs:
{needs_json}

Available objects on the table:
{objects_list}

For each object, return a JSON object:
{{
  "object_roles": [
    {{
      "name": "object_name",
      "role": "prominent|accessible|peripheral|remove",
      "reason": "why this role for this situation",
      "grouping": "optional: which other objects this should be near"
    }}
  ],
  "arrangement_notes": "any overall arrangement principles for this situation"
}}

Example for exam preparation with objects [textbook, phone, mug, tissue_box]:
{{
  "object_roles": [
    {{"name": "textbook", "role": "prominent", "reason": "core study material, needs to be open and centered", "grouping": "near notebook and highlighters"}},
    {{"name": "phone", "role": "remove", "reason": "major distraction source during study, should be face-down at far edge", "grouping": "isolated from study materials"}},
    {{"name": "mug", "role": "accessible", "reason": "warm drink sustains long study sessions", "grouping": "near dominant hand side"}},
    {{"name": "tissue_box", "role": "peripheral", "reason": "not directly relevant to studying", "grouping": "corner"}}
  ],
  "arrangement_notes": "Study materials form a central cluster. Comfort items (mug, snacks) on the side. Distractions banished to far edges."
}}"""


# ==============================================================================
# STAGE 3: SPATIAL ARRANGEMENT (Zone-Based)
# ==============================================================================
# Design rationale (V-CAGE, arXiv:2604.09036 §III-A2):
# LLMs are poor at precise numerical reasoning. Instead of asking for exact
# (x, y) coordinates, we ask for qualitative zone names and convert to
# coordinates programmatically via ZoneMapper + PlacementOptimiser.
# ==============================================================================

SPATIAL_ARRANGEMENT_SYSTEM = """You are a spatial planner for a robotic arm. Given object roles and a table divided into zones, you assign each object to the most appropriate zone.

The table is divided into a 3×3 grid of zones (from the person's perspective):

  BACK (far from person):    back-left    |  back-center   |  back-right
  MIDDLE:                    mid-left     |  center        |  mid-right
  FRONT (close to person):   front-left   |  front-center  |  front-right

Placement rules:
1. "prominent" objects → front-center or center (easily accessible, visually dominant)
2. "accessible" objects → front-left, front-right, mid-left, or mid-right (within reach)
3. "peripheral" objects → back-left, back-center, or back-right (out of the way)
4. "remove" objects → back-right or back-left (as far from the person as possible)
5. Group related objects in the SAME or ADJACENT zones (e.g., study materials together)
6. Non-graspable objects (laptop, desk_lamp) keep their current zones; assign them but note they cannot be moved.
7. VISUAL STORYTELLING: Zone assignments MUST visually communicate the situation at a glance. Cluster semantically related objects tightly (same zone), isolate removed objects clearly.
8. Do NOT put more than 3 objects in a single zone — spread to adjacent zones if needed.

Do NOT output coordinates. Output zone names only. The robot's planner will convert zones to exact positions.

Respond in JSON format."""

SPATIAL_ARRANGEMENT_USER = """Situation: "{situation}"

Object roles and reasoning:
{roles_json}

Available zones (from person's viewpoint):
  back-left, back-center, back-right
  mid-left, center, mid-right
  front-left, front-center, front-right

Return a JSON object:
{{
  "zone_assignments": [
    {{
      "name": "object_name",
      "zone": "front-center",
      "role": "prominent",
      "reason": "why this zone for this object in this situation"
    }}
  ],
  "layout_description": "one-line description of the overall layout",
  "clustering_notes": "which objects are intentionally grouped together and why",
  "non_graspable_note": "list any objects that cannot be moved by the robot"
}}

Example for exam preparation:
{{
  "zone_assignments": [
    {{"name": "textbook", "zone": "front-center", "role": "prominent", "reason": "core study material, needs to be open and centered"}},
    {{"name": "notebook", "zone": "front-left", "role": "prominent", "reason": "note-taking beside textbook"}},
    {{"name": "highlighter_set", "zone": "front-left", "role": "accessible", "reason": "grouped with notebook for quick access"}},
    {{"name": "mug", "zone": "mid-right", "role": "accessible", "reason": "warm drink within reach, dominant hand side"}},
    {{"name": "phone", "zone": "back-right", "role": "remove", "reason": "major distraction, banished to far corner"}}
  ],
  "layout_description": "Study battle station: materials front-center, comfort on the side, distractions banished",
  "clustering_notes": "textbook+notebook+highlighters form a study cluster in front; mug+snacks on mid-right as comfort zone",
  "non_graspable_note": "desk_lamp stays in its current position"
}}"""


# --- Legacy exact-coordinate prompt (kept for ablation experiments) ---

SPATIAL_ARRANGEMENT_SYSTEM_LEGACY = """You are a precise spatial planner for a robotic arm. Given object roles and a table workspace, you determine exact (x, y) coordinates for each object placement.

Think step by step:
1. Start with "prominent" objects — place them in the center-front zone first.
2. Then place "accessible" objects on the sides within easy reach.
3. Then place "peripheral" objects in corners or back edges.
4. Finally push "remove" objects to the far back corner.
5. After each placement, check for overlaps with already-placed objects.
6. Adjust positions to maintain at least 2cm clearance.

The table workspace is defined as:
- x-axis: depth (away from the person). x_min = closest to person, x_max = farthest.
- y-axis: width (left-right). y_min = person's right, y_max = person's left.
- Origin is at the robot base. The person sits facing the table from the x_min side.

Placement rules:
1. "prominent" objects go in the center-front area (low x, middle y)
2. "accessible" objects go within easy reach (moderate x, spread on sides)
3. "peripheral" objects go to corners or back edges (high x, extreme y)
4. "remove" objects go to the far back corner
5. Leave clearance between objects (at least 2cm gap)
6. Respect object dimensions — don't overlap
7. Non-graspable objects (laptop, desk_lamp) keep their current positions
8. VISUAL STORYTELLING: The final layout MUST visually communicate the core situation at a glance. Form highly visible semantic clusters (e.g., heavily group all study materials tight together, or isolate comfort items in a specific "safe zone") so an observer can immediately guess the user's situation.

Respond in JSON format with exact coordinates."""

SPATIAL_ARRANGEMENT_USER_LEGACY = """Situation: "{situation}"

Object roles and reasoning:
{roles_json}

Workspace bounds:
- x: [{x_min}, {x_max}] meters (depth: {x_min}=closest to person)
- y: [{y_min}, {y_max}] meters (width: {y_min}=right, {y_max}=left)
- Table surface z: {z_surface} meters

Object dimensions (for collision avoidance):
{object_dims}

Return a JSON object:
{{
  "placements": [
    {{
      "name": "object_name",
      "x": 0.25,
      "y": 0.05,
      "z": {z_surface},
      "reason": "why this specific position"
    }}
  ],
  "layout_description": "one-line description of the overall layout",
  "non_graspable_note": "list any objects that cannot be moved by the robot"
}}

Coordinate guidelines:
- Center front (prominent): x ∈ [{x_min}, {x_mid}], y ∈ [{y_q1}, {y_q3}]
- Side accessible: x ∈ [{x_min}, {x_mid}], y near {y_min} or {y_max}
- Back/peripheral: x ∈ [{x_mid}, {x_max}], any y
- Far corner (remove): x ≈ {x_max}, y ≈ {y_min} or {y_max}"""


# ==============================================================================
# ROUNDTRIP EVALUATION
# ==============================================================================

ROUNDTRIP_EVAL_SYSTEM = """You are evaluating a robotic tabletop arrangement. You will see a description of objects and their positions on a table. Your task is to guess what human situation this arrangement was designed for.

CRITICAL: You MUST select your answer EXACTLY from the provided list of candidate situations. Do not invent or modify the candidate strings.

You will be given a list of candidate situations. Think step by step:
1. Identify which objects are in the front-center (prominent) vs. edges (peripheral) vs. far corners (removed).
2. Look for spatial groupings — which objects are clustered together? What does that suggest?
3. Consider what activity or emotional state the arrangement implies.
4. Check for cultural or social signals (e.g., tea set prominent = hospitality, tissues close = comfort).
5. For each candidate situation, evaluate how well the arrangement matches.
6. Choose the best match and explain what specific clues led to your decision.

Respond in JSON format."""

ROUNDTRIP_EVAL_USER = """Here is a tabletop arrangement (objects and their positions):
{arrangement_description}

Candidate situations:
{candidates_list}

Which situation was this arrangement most likely designed for?

Return:
{{
  "predicted_situation": "the exact text of your chosen candidate",
  "confidence": 0.85,
  "reasoning": "explain what clues in the arrangement led to your prediction",
  "runner_up": "second most likely candidate",
  "distinguishing_features": ["list of arrangement features that were most informative"]
}}"""


# ==============================================================================
# HELPER: Format objects list for prompts
# ==============================================================================

def format_objects_list(objects: list) -> str:
    """Format a list of object dicts into a readable string for prompts."""
    lines = []
    for obj in objects:
        dims = obj.get("dimensions", {})
        dim_str = f"{dims.get('w', '?')}×{dims.get('d', '?')}×{dims.get('h', '?')}m"
        graspable = "graspable" if obj.get("graspable", True) else "NOT graspable (fixed)"
        lines.append(f"- {obj['name']}: {dim_str}, {graspable}")
    return "\n".join(lines)


def format_arrangement_description(placements: list) -> str:
    """Format placements into a readable description for roundtrip evaluation."""
    lines = []
    for p in placements:
        pos = f"({p['x']:.2f}, {p['y']:.2f})"
        lines.append(f"- {p['name']} at position {pos}")

    # Add spatial interpretation
    lines.append("\nSpatial key: x=depth (low=close to person), y=width (negative=right, positive=left)")
    return "\n".join(lines)
