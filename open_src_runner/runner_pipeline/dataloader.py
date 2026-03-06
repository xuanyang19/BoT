import re
import json
import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from typing import Optional


# ===== MedQA =====
def df_from_medqa(split="test") -> pd.DataFrame:
    """
    Load MedQA from GBaker/MedQA-USMLE-4-options (test split by default).
    Returns columns: context, question, options(list), medqa_gold_letter(A-D)
    """
    ds = load_dataset("GBaker/MedQA-USMLE-4-options")
    if split not in ds:
        split = list(ds.keys())[0]

    rows = []
    for ex in ds[split]:
        q = ex.get("question", "")

        # Handle options - they are stored as a dictionary {"A": "...", "B": "...", "C": "...", "D": "..."}
        options_dict = ex.get("options", {})
        if isinstance(options_dict, dict):
            opts = [
                options_dict.get("A", ""),
                options_dict.get("B", ""),
                options_dict.get("C", ""),
                options_dict.get("D", "")
            ]
        else:
            opts = ["", "", "", ""]

        # Handle the answer
        gold_letter = ex.get("answer_idx")
        if isinstance(gold_letter, str):
            gold_letter = gold_letter.strip().upper()[:1] if gold_letter.strip() else None
        else:
            ans = ex.get("answer")
            if isinstance(ans, str):
                gold_letter = ans.strip().upper()[:1] if ans.strip() else None
            elif isinstance(ans, (int, float, np.integer)):
                gold_letter = chr(ord("A") + int(ans)) if 0 <= int(ans) <= 3 else None
            else:
                gold_letter = None

        if gold_letter not in ["A", "B", "C", "D"]:
            gold_letter = None

        rows.append({
            "context": "",
            "question": q,
            "options": opts,
            "medqa_gold_letter": gold_letter
        })

    return pd.DataFrame(rows)


# ===== Winogrande =====
def df_from_winogrande(split: str) -> pd.DataFrame:
    """
    Load Winogrande dataset.
    Returns columns: context, question, options, winogrande_gold_index
    """
    ds = load_dataset("winogrande", "winogrande_debiased")
    split = 'validation'

    rows = []
    for ex in ds[split]:
        sentence = ex.get("sentence", "")
        option1 = ex.get("option1", "")
        option2 = ex.get("option2", "")
        answer = ex.get("answer", "")

        # Split sentence at the blank "_"
        if "_" in sentence:
            query, end_of_target = sentence.split("_", 1)
            end_of_target = end_of_target.strip()
        else:
            query = sentence
            end_of_target = ""

        # Construct choices
        choices = [
            f"{option1} {end_of_target}",
            f"{option2} {end_of_target}"
        ]

        # Gold index
        gold_idx = int(answer) - 1 if answer != "" else -1

        rows.append({
            "context": "",
            "question": query,
            "options": choices,
            "winogrande_gold_index": gold_idx,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Custom Dataset Interface
# =============================================================================
#
# To add a new dataset, implement a loader function following this pattern:
#
#   def df_from_my_dataset(split: str) -> pd.DataFrame:
#       """
#       Load your dataset and return a DataFrame with these columns:
#         - context (str): Optional context/background for the question
#         - question (str): The question text
#         - options (list or None): Answer options for multiple-choice, None otherwise
#         - <dataset>_gold (str): Gold label column (e.g., my_dataset_gold)
#       """
#       rows = []
#       # ... load and process your data ...
#       return pd.DataFrame(rows)
#
# Then register it in load_dataset_router() below, and add:
#   - A prompt in actor_refinement.py (see ACTION_PROMPT_MEDQA as example)
#   - A query builder case in actor_refinement.py make_query()
#   - An evaluator in evaluate.py (see evaluate_medqa as example)
#   - Register the evaluator in evaluate.py DATASET_EVALUATORS dict
# =============================================================================


# ===== Router =====
def load_dataset_router(name: str, split: str) -> pd.DataFrame:
    """Load dataset by name."""
    if name == "medqa":
        return df_from_medqa(split)
    if name == "winogrande":
        return df_from_winogrande(split)

    raise ValueError(
        f"Unknown dataset: {name}. "
        f"Supported: medqa, winogrande. "
        f"See the Custom Dataset Interface section in this file to add new datasets."
    )
