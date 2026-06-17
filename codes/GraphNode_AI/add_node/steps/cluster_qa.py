"""English documentation."""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from hdbscan import HDBSCAN

from ..utils.io_helpers import load_qa_embeddings
from ..utils.clustering_utils import merge_clusters_by_distance
from ..utils import logger

# English comment.
TMP_DIR = Path(__file__).parent.parent / "tmp"


def cluster_qa_single_conv(
    qa_emb_path: Path,
    conversation_id: str,
    min_cluster_size: int,
    min_samples: int | None,
    metric: str,
    output_path: Path,
    merge_distance_threshold: float | None,
) -> None:
    """English documentation."""
    qa_records = load_qa_embeddings(qa_emb_path)
    if not qa_records:
        logger.warning(f"No QA embeddings in {qa_emb_path}")
        return

    # English comment.
    qa_ids: List[str] = []
    X_list: List[np.ndarray] = []
    qa_emb_dict: Dict[str, np.ndarray] = {}

    for rec in qa_records:
        qa_id = rec.get("qa_id")
        qa_emb = rec.get("qa_embedding")
        if qa_emb is None:
            continue

        v = np.asarray(qa_emb, dtype=np.float32)
        if v.ndim == 2 and v.shape[0] == 1:
            v = v[0]

        qa_id_str = str(qa_id)
        qa_ids.append(qa_id_str)
        X_list.append(v)
        qa_emb_dict[qa_id_str] = v

    if not X_list:
        logger.warning(f"No valid embeddings for conversation_id={conversation_id}")
        return

    X = np.stack(X_list, axis=0)
    logger.info(f"Loaded {X.shape[0]} QA embeddings, dim={X.shape[1]}")

    # English comment.
    logger.info(f"Running HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})")
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))

    logger.info(f"Found {n_clusters} clusters, noise points: {n_noise}")

    # English comment.
    clusters: Dict[int, Dict[str, Any]] = {}
    qa_records_dict = {rec["qa_id"]: rec for rec in qa_records}

    for qa_id, label in zip(qa_ids, labels):
        label_int = int(label)
        rec = qa_records_dict.get(qa_id, {})
        qa_index_val = rec.get("qa_index")

        info = clusters.setdefault(
            label_int,
            {
                "cluster_id": label_int,
                "size": 0,
                "qa_ids": [],
                "conversation_id": conversation_id,
                "qa_indices": [],
            },
        )
        info["size"] += 1
        info["qa_ids"].append(qa_id)
        info["qa_indices"].append(qa_index_val)

    # Fallback: if HDBSCAN produced no positive clusters
    if n_clusters == 0:
        fallback_cluster = {
            "cluster_id": 0,
            "size": len(qa_ids),
            "qa_ids": qa_ids,
            "conversation_id": conversation_id,
            "qa_indices": [
                qa_records_dict.get(qid, {}).get("qa_index") for qid in qa_ids
            ],
        }
        clusters = {0: fallback_cluster}
        n_clusters = 1
        n_noise = 0
        logger.info("HDBSCAN produced no clusters; falling back to single cluster")

    merge_threshold = merge_distance_threshold if merge_distance_threshold is not None else 0.0
    merged_pairs = 0
    if merge_threshold > 0 and len(clusters) > 1:
        clusters, merged_pairs = merge_clusters_by_distance(
            clusters, qa_emb_dict, merge_threshold
        )
        if merged_pairs:
            logger.info(f"Merged {merged_pairs} cluster pairs (threshold: {merge_threshold})")

    n_clusters = sum(1 for cid in clusters.keys() if cid >= 0)

    # English comment.
    out = {
        "qa_embeddings_path": str(qa_emb_path),
        "conversation_id": conversation_id,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "metric": metric,
        "merge_distance_threshold": merge_distance_threshold,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "clusters": sorted(clusters.values(), key=lambda c: c["cluster_id"]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Cluster results saved: {output_path}")
