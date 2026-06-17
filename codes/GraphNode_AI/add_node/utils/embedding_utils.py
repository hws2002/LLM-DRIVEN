"""English documentation."""

from typing import Any, Dict, List

import numpy as np

from .logger import logger


def length_weighted_pool_embeddings(
    qa_ids: List[str],
    qa_emb_dict: Dict[str, np.ndarray],
    qa_length_dict: Dict[str, float],
) -> tuple[np.ndarray, float]:
    """English documentation."""
    valid_embs = []
    weights = []

    for qa_id in qa_ids:
        if qa_id in qa_emb_dict:
            valid_embs.append(qa_emb_dict[qa_id])
            weight = qa_length_dict.get(qa_id, 1.0)
            weights.append(weight)

    if not valid_embs:
        return None, 0.0

    weights = np.array(weights, dtype=np.float32)
    total_weight = np.sum(weights)

    if total_weight == 0:
        weights = np.ones(len(valid_embs), dtype=np.float32)
        total_weight = float(len(valid_embs))

    stacked = np.stack(valid_embs, axis=0)
    pooled = (weights[:, None] * stacked).sum(axis=0) / total_weight

    return pooled, float(total_weight)


def build_cluster_embeddings(
    cluster_json: Dict[str, Any],
    qa_emb_dict: Dict[str, np.ndarray],
    qa_length_dict: Dict[str, float],
    conv_id: str,
) -> Dict[int, Dict[str, Any]]:
    """English documentation."""
    cluster_embeddings = {}
    clusters = cluster_json.get("clusters", [])

    for cluster_info in clusters:
        cluster_id = int(cluster_info.get("cluster_id", -1))

        # English comment.
        if cluster_id < 0:
            continue

        qa_ids = [str(qid) for qid in cluster_info.get("qa_ids", [])]

        if not qa_ids:
            continue

        pooled_emb, weight_sum = length_weighted_pool_embeddings(
            qa_ids, qa_emb_dict, qa_length_dict
        )

        if pooled_emb is None:
            logger.warning(f"Cluster {cluster_id} has no valid embeddings, skipping")
            continue

        cluster_embeddings[cluster_id] = {
            "cluster_id": cluster_id,
            "conversation_id": conv_id,
            "size": len(qa_ids),
            "weight_sum": weight_sum,
            "embedding": pooled_emb.tolist(),
            "qa_ids": qa_ids,
        }

    return cluster_embeddings
