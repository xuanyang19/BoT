# -*- coding: utf-8 -*-
"""
Evaluation module for checking correctness of model responses across supported datasets.
Supports: medqa, winogrande

To add a new dataset evaluator:
  1. Implement an evaluate_<dataset>(row, response) function
  2. Register it in the DATASET_EVALUATORS dict at the bottom of this file
"""

import re
from typing import Dict, Any, Optional
import pandas as pd


# =========================
# Helper functions
# =========================

def _to_letter_any(x: Any) -> Optional[str]:
    """Convert answer to single letter (A, B, C, etc.)"""
    if x is None: return None
    s = str(x).strip().upper()
    return s[0] if s else None


# =========================
# Evaluation functions for each dataset
# =========================

def evaluate_medqa(row: pd.Series, response: Dict[str, Any]) -> Optional[bool]:
    """Evaluate MedQA dataset responses."""
    ans = response.get('answer', None)
    if ans is None:
        return None

    pred_letter = None
    # numeric index from model
    if isinstance(ans, (int, float)):
        if 0 <= int(ans) <= 25:
            pred_letter = chr(ord("A") + int(ans))
    elif isinstance(ans, str):
        s = ans.strip()
        if len(s) == 1 and s.isalpha():
            pred_letter = s.upper()
        else:
            # match prefix like "A." or "b)"
            m = re.match(r'([A-Za-z])[\.\)]', s)
            if m:
                pred_letter = m.group(1).upper()
            # fallback: exact option match
            opts = row.get("options", [])
            for idx, opt in enumerate(opts):
                if s.lower() == str(opt).lower():
                    pred_letter = chr(ord("A") + idx)
                    break

    gold_letter = str(row.get("medqa_gold_letter", "")).strip().upper()[:1]
    return pred_letter == gold_letter if gold_letter else None


def evaluate_winogrande(row: pd.Series, response: Dict[str, Any]) -> Optional[bool]:
    """
    Evaluate Winogrande dataset responses.
    Answer should be 1 or 2, or the actual text of the option.
    """
    ans = response.get('answer', None)
    if ans is None:
        return None

    gold_idx = row.get("winogrande_gold_index", None)
    if gold_idx is None or gold_idx == -1:
        return None

    # Try to extract answer index
    pred_idx = None

    # Case 1: Direct numeric answer (1 or 2)
    if isinstance(ans, (int, float)):
        pred_idx = int(ans) - 1  # Convert to 0-indexed
    elif isinstance(ans, str):
        ans_str = ans.strip()

        # Case 2: String "1" or "2"
        if ans_str in ["1", "2"]:
            pred_idx = int(ans_str) - 1

        # Case 3: String "A" or "B" (convert to 0/1)
        elif ans_str.upper() in ["A", "B"]:
            pred_idx = 0 if ans_str.upper() == "A" else 1

        # Case 4: Match against actual option text
        else:
            opts = row.get("options", [])
            if opts and len(opts) == 2:
                # Try exact match
                for i, opt in enumerate(opts):
                    if ans_str.lower() == str(opt).lower().strip():
                        pred_idx = i
                        break

                # Try partial match if no exact match
                if pred_idx is None:
                    for i, opt in enumerate(opts):
                        if ans_str.lower() in str(opt).lower() or str(opt).lower() in ans_str.lower():
                            pred_idx = i
                            break

    if pred_idx is None or pred_idx not in [0, 1]:
        return None

    return pred_idx == gold_idx


# =========================
# Dataset routing and main evaluation function
# =========================

DATASET_EVALUATORS = {
    "medqa": evaluate_medqa,
    "winogrande": evaluate_winogrande,
}


def evaluate_correctness(dataset: str, row: pd.Series, response: Dict[str, Any]) -> Optional[bool]:
    """
    Evaluate correctness of a model response based on the dataset type.

    Args:
        dataset: Name of the dataset
        row: Pandas Series containing the question data
        response: Dict containing model's response with 'answer' key

    Returns:
        True if correct, False if incorrect, None if cannot evaluate
    """
    evaluator = DATASET_EVALUATORS.get(dataset)
    if evaluator is None:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: {list(DATASET_EVALUATORS.keys())}")

    return evaluator(row, response)
