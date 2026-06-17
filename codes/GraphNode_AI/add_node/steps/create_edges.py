"""English documentation."""

from typing import Any, Dict, List

import numpy as np

from ..utils import logger
from ..utils.similarity_utils import calculate_similarity_matrix


def create_hard_edges_for_new_nodes(
    new_nodes: List[Dict[str, Any]],
    existing_nodes: List[Dict[str, Any]],
    threshold: float = 0.5,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """English documentation."""
    logger.info(f"Creating edges: {len(new_nodes)} new nodes -> {len(existing_nodes)} existing nodes")

    if not new_nodes or not existing_nodes:
        logger.warning("No nodes to compare. Skipping edge creation.")
        return []

    # English comment.
    new_embeddings = []
    new_ids = []
    new_cluster_ids = []

    for node in new_nodes:
        embedding = node.get("embedding")
        if embedding and len(embedding) > 0:
            new_embeddings.append(embedding)
            new_ids.append(node.get("id"))
            new_cluster_ids.append(node.get("clusterId"))

    existing_embeddings = []
    existing_ids = []
    existing_cluster_ids = []

    for node in existing_nodes:
        embedding = node.get("embedding")
        if embedding and len(embedding) > 0:
            existing_embeddings.append(embedding)
            existing_ids.append(node.get("id"))
            existing_cluster_ids.append(node.get("clusterId"))

    if not new_embeddings or not existing_embeddings:
        logger.warning("No valid embeddings found. Skipping edge creation.")
        return []

    # English comment.
    new_emb_matrix = np.array(new_embeddings, dtype=np.float32)
    existing_emb_matrix = np.array(existing_embeddings, dtype=np.float32)

    # English comment.
    similarity_matrix = calculate_similarity_matrix(new_emb_matrix, existing_emb_matrix)

    # English comment.
    all_edges = []

    for i, new_id in enumerate(new_ids):
        similarities = similarity_matrix[i]
        new_cluster = new_cluster_ids[i]

        # English comment.
        valid_indices = np.where(similarities >= threshold)[0]

        if len(valid_indices) == 0:
            continue

        # English comment.
        sorted_indices = valid_indices[np.argsort(-similarities[valid_indices])]
        top_indices = sorted_indices[:top_k]

        for j in top_indices:
            existing_id = existing_ids[j]
            existing_cluster = existing_cluster_ids[j]
            weight = float(similarities[j])

            # English comment.
            if new_id == existing_id:
                continue

            # English comment.
            is_intra = (new_cluster == existing_cluster)

            edge = {
                "source": new_id,
                "target": existing_id,
                "weight": weight,
                "type": "hard",
                "intraCluster": is_intra
            }
            all_edges.append(edge)

    logger.info(f"Total edges created: {len(all_edges)}")

    return all_edges
