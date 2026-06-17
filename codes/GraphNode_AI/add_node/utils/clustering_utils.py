"""English documentation."""

from typing import Any, Dict, List, Tuple

import numpy as np


def cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine distance (1 - similarity) between two vectors."""
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 1.0
    sim = float(np.dot(vec_a, vec_b) / denom)
    return 1.0 - sim


def merge_clusters_by_distance(
    clusters: Dict[int, Dict[str, Any]],
    qa_emb_dict: Dict[str, np.ndarray],
    threshold: float,
) -> Tuple[Dict[int, Dict[str, Any]], int]:
    """Merge clusters whose centroids are closer than the given cosine-distance threshold."""
    if threshold <= 0:
        return clusters, 0

    positive_ids = sorted(cid for cid in clusters.keys() if cid >= 0)
    if len(positive_ids) <= 1:
        return clusters, 0

    centroids: Dict[int, np.ndarray] = {}
    for cid in positive_ids:
        info = clusters[cid]
        vectors = [
            qa_emb_dict[qa_id]
            for qa_id in info.get("qa_ids", [])
            if qa_id in qa_emb_dict
        ]
        if not vectors:
            continue

        stacked = np.stack(vectors, axis=0)
        centroid = stacked.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids[cid] = centroid

    centroid_ids = sorted(centroids.keys())
    if len(centroid_ids) <= 1:
        return clusters, 0

    parents: Dict[int, int] = {cid: cid for cid in centroid_ids}

    def find(x: int) -> int:
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if ra < rb:
            parents[rb] = ra
        else:
            parents[ra] = rb
        return True

    merge_count = 0
    for i, cid1 in enumerate(centroid_ids):
        vec1 = centroids[cid1]
        for cid2 in centroid_ids[i + 1:]:
            vec2 = centroids[cid2]
            dist = cosine_distance(vec1, vec2)
            if dist < threshold and union(cid1, cid2):
                merge_count += 1

    if merge_count == 0:
        return clusters, 0

    grouped: Dict[int, List[int]] = {}
    for cid in positive_ids:
        root = find(cid) if cid in parents else cid
        grouped.setdefault(root, []).append(cid)

    merged_clusters: Dict[int, Dict[str, Any]] = {}
    for root, member_ids in grouped.items():
        member_ids.sort()
        representative = min(member_ids)
        combined_pairs: List[Tuple[Any, str]] = []
        for mid in member_ids:
            info = clusters[mid]
            qa_ids = info.get("qa_ids", [])
            qa_indices = info.get("qa_indices", [])
            if qa_indices and len(qa_indices) == len(qa_ids):
                combined_pairs.extend(zip(qa_indices, qa_ids))
            else:
                combined_pairs.extend((None, qid) for qid in qa_ids)

        combined_pairs.sort(
            key=lambda pair: (
                float("inf") if pair[0] is None else pair[0],
                pair[1],
            )
        )
        new_qa_indices = [pair[0] for pair in combined_pairs]
        new_qa_ids = [pair[1] for pair in combined_pairs]

        merged_clusters[representative] = {
            "cluster_id": representative,
            "size": len(new_qa_ids),
            "qa_ids": new_qa_ids,
            "conversation_id": clusters[member_ids[0]].get("conversation_id"),
            "qa_indices": new_qa_indices,
        }

    if -1 in clusters:
        merged_clusters[-1] = clusters[-1]

    return merged_clusters, merge_count
