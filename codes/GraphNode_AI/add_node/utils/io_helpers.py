"""English documentation."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def load_qa_pairs(path: Path) -> List[Dict[str, Any]]:
    """English documentation."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_qa_embeddings(path: Path) -> List[Dict[str, Any]]:
    """English documentation."""
    with path.open("rb") as f:
        data = pickle.load(f)
    return data


def load_cluster_results(path: Path) -> Dict[str, Any]:
    """English documentation."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_qa_embeddings_and_lengths(path: Path) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """English documentation."""
    with path.open("rb") as f:
        emb_records = pickle.load(f)

    qa_emb_dict = {}
    qa_length_dict = {}

    for rec in emb_records:
        qa_id = str(rec.get("qa_id"))
        qa_emb = rec.get("qa_embedding")
        if qa_emb is None:
            continue

        v = np.asarray(qa_emb, dtype=np.float32)
        if v.ndim == 2 and v.shape[0] == 1:
            v = v[0]

        qa_emb_dict[qa_id] = v
        qa_length = rec.get("qa_length", 1.0)
        qa_length_dict[qa_id] = max(float(qa_length), 1.0)

    return qa_emb_dict, qa_length_dict
