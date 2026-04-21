#!/usr/bin/env python3
"""Roundtrip evaluation: can an evaluator guess the situation from the arrangement?"""

import json
import logging
import random
import re
from typing import List, Dict, Optional

from situbot.reasoning.llm_client import DashScopeClient
from situbot.reasoning.prompts import (
    ROUNDTRIP_EVAL_SYSTEM, ROUNDTRIP_EVAL_USER,
    format_arrangement_description,
)

logger = logging.getLogger(__name__)


class RoundtripEvaluator:
    """Evaluates arrangements using the Roundtrip Test.

    Protocol:
    1. Robot arranges objects for situation S
    2. Arrangement description (or photo) shown to blind evaluator
    3. Evaluator picks from K candidates (including S)
    4. Roundtrip accuracy = fraction where evaluator picks S correctly
    """

    def __init__(self, evaluator_llm: DashScopeClient,
                 all_scenarios: List[Dict],
                 num_candidates: int = 5,
                 seed: Optional[int] = 42):
        """
        Args:
            evaluator_llm: LLM client for evaluation (should be DIFFERENT model
                          from the one that generated the arrangement).
            all_scenarios: List of all SituBench scenario dicts.
            num_candidates: Number of candidate situations (including ground truth).
        """
        self.llm = evaluator_llm
        self.all_scenarios = all_scenarios
        self.num_candidates = num_candidates
        self._rng = random.Random(seed)

    def evaluate(self, ground_truth_situation: str,
                 placements: List[Dict],
                 scene_photo_path: Optional[str] = None) -> Dict:
        """Run roundtrip evaluation for one arrangement.

        Args:
            ground_truth_situation: The actual situation the arrangement was made for.
            placements: List of dicts with name, x, y, z keys.
            scene_photo_path: Optional path to photo (for future VLM evaluation).

        Returns:
            Dict with: correct (bool), predicted, confidence, reasoning, candidates.
        """
        # Build candidate list: ground truth + (K-1) distractors
        candidates = self._select_candidates(ground_truth_situation)

        # Format arrangement for the evaluator
        arrangement_desc = format_arrangement_description(placements)
        candidates_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))

        # Query evaluator LLM
        messages = [
            {"role": "system", "content": ROUNDTRIP_EVAL_SYSTEM},
            {"role": "user", "content": ROUNDTRIP_EVAL_USER.format(
                arrangement_description=arrangement_desc,
                candidates_list=candidates_text,
            )},
        ]

        try:
            result = self.llm.chat_json(messages)
        except Exception as e:
            logger.error(f"Evaluation LLM call failed: {e}")
            return {
                "correct": False,
                "predicted": "",
                "confidence": 0.0,
                "reasoning": f"LLM error: {e}",
                "candidates": candidates,
                "ground_truth": ground_truth_situation,
            }

        predicted = result.get("predicted_situation", "")
        # Exact match first, then normalised comparison
        gt_norm = ground_truth_situation.strip().lower()
        pred_norm = predicted.strip().lower()
        correct = (gt_norm == pred_norm)
        if not correct:
            # Allow minor whitespace/punctuation differences but NOT substring
            def _normalize(s):
                return re.sub(r'[^a-z0-9 ]', '', s).strip()
            correct = (_normalize(gt_norm) == _normalize(pred_norm))

        return {
            "correct": correct,
            "predicted": predicted,
            "confidence": result.get("confidence", 0.0),
            "reasoning": result.get("reasoning", ""),
            "runner_up": result.get("runner_up", ""),
            "distinguishing_features": result.get("distinguishing_features", []),
            "candidates": candidates,
            "ground_truth": ground_truth_situation,
        }

    def _select_candidates(self, ground_truth: str) -> List[str]:
        """Select candidate situations for the roundtrip test.

        Includes the ground truth + (K-1) distractors from the same
        difficulty level when possible (harder discrimination).
        """
        # Find ground truth scenario
        gt_scenario = None
        for s in self.all_scenarios:
            if s["situation"] == ground_truth:
                gt_scenario = s
                break

        # Collect distractors, preferring same difficulty level
        distractors = [s["situation"] for s in self.all_scenarios
                       if s["situation"] != ground_truth]

        if gt_scenario:
            same_level = [s["situation"] for s in self.all_scenarios
                          if s.get("level") == gt_scenario.get("level")
                          and s["situation"] != ground_truth]
            diff_level = [s for s in distractors if s not in same_level]
            # Take as many same-level as possible
            n_needed = self.num_candidates - 1
            selected = self._rng.sample(same_level, min(len(same_level), n_needed))
            if len(selected) < n_needed:
                remaining = n_needed - len(selected)
                selected += self._rng.sample(diff_level, min(len(diff_level), remaining))
        else:
            selected = self._rng.sample(distractors, min(len(distractors), self.num_candidates - 1))

        # Insert ground truth at random position
        candidates = selected[:]
        insert_pos = self._rng.randint(0, len(candidates))
        candidates.insert(insert_pos, ground_truth)
        return candidates
