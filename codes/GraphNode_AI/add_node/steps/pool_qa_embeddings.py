"""English documentation."""

from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.io_helpers import load_cluster_results, load_qa_embeddings_and_lengths
from ..utils.embedding_utils import length_weighted_pool_embeddings, build_cluster_embeddings
from ..utils import logger

# English comment.
TMP_DIR = Path(__file__).parent.parent / "tmp"


def pool_embeddings(
    qa_emb_path: Path,
    conv_id: str,
    cluster_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """English documentation."""
    # English comment.
    if not qa_emb_path.exists():
        raise FileNotFoundError(f"QA embedding file not found: {qa_emb_path}")

    logger.info(f"Pooling embeddings for conversation {conv_id}")

    # English comment.
    cluster_embeddings = {}
    if cluster_path is not None and cluster_path.exists():
        cluster_json = load_cluster_results(cluster_path)
        n_clusters = cluster_json.get("n_clusters", 0)
        logger.info(f"Loaded {n_clusters} clusters")
    else:
        cluster_json = {"clusters": []}
        logger.info("No cluster file provided. Skipping cluster-level pooling.")

    # English comment.
    qa_emb_dict, qa_length_dict = load_qa_embeddings_and_lengths(qa_emb_path)
    logger.info(f"Loaded {len(qa_emb_dict)} QA embeddings")

    # English comment.
    if cluster_json.get("clusters"):
        cluster_embeddings = build_cluster_embeddings(
            cluster_json,
            qa_emb_dict,
            qa_length_dict,
            conv_id
        )

    logger.info(f"Pooled {len(cluster_embeddings)} cluster embeddings")

    # English comment.
    all_qa_ids = list(qa_emb_dict.keys())
    conv_embedding, total_weight = length_weighted_pool_embeddings(
        all_qa_ids, qa_emb_dict, qa_length_dict
    )

    if conv_embedding is not None:
        logger.info(f"Created conversation embedding from {len(all_qa_ids)} QAs, total_weight={total_weight:.1f}")
    else:
        logger.warning("Failed to create conversation embedding")

    return {
        "cluster_embeddings": cluster_embeddings,
        "conversation_embedding": conv_embedding.tolist() if conv_embedding is not None else None,
        "total_qa_count": len(all_qa_ids),
        "total_weight": total_weight
    }
