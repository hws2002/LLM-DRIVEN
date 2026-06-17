"""English documentation."""

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .logger import logger


def calculate_similarity_matrix(
    new_embeddings: np.ndarray,
    existing_embeddings: np.ndarray
) -> np.ndarray:
    """English documentation."""
    if new_embeddings.size == 0 or existing_embeddings.size == 0:
        return np.array([])

    similarity_matrix = cosine_similarity(new_embeddings, existing_embeddings)

    logger.info(f"Calculated similarity matrix: {similarity_matrix.shape}")

    return similarity_matrix


def create_edges_from_similarity(
    new_node_ids: List[int],
    existing_node_ids: List[int],
    similarity_matrix: np.ndarray,
    threshold: float = 0.5,
    top_k: int = 5,
    is_intra_cluster: bool = True
) -> List[Dict[str, Any]]:
    """English documentation."""
    if similarity_matrix.size == 0:
        logger.warning("Empty similarity matrix. No edges created.")
        return []

    edges = []

    for i, new_id in enumerate(new_node_ids):
        similarities = similarity_matrix[i]

        # English comment.
        valid_indices = np.where(similarities >= threshold)[0]

        if len(valid_indices) == 0:
            continue

        # English comment.
        sorted_indices = valid_indices[np.argsort(-similarities[valid_indices])]
        top_indices = sorted_indices[:top_k]

        for j in top_indices:
            existing_id = existing_node_ids[j]
            weight = float(similarities[j])

            # English comment.
            if new_id == existing_id:
                continue

            edge = {
                "source": new_id,
                "target": existing_id,
                "weight": weight,
                "type": "hard",
                "intraCluster": is_intra_cluster
            }
            edges.append(edge)

    logger.info(f"Created {len(edges)} edges (threshold={threshold}, top_k={top_k})")

    return edges
