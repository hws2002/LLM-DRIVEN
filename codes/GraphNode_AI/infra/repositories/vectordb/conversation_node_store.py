"""ChromaDB store for add_node conversation node embeddings."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("GraphNodeDB")


class ConversationNodeStore:
    """English documentation."""

    COLLECTION = "add_node_conversation_nodes"

    def __init__(self, chroma_client: Any) -> None:
        # English comment.
        self._col = chroma_client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def store_node(
        self,
        *,
        orig_id: str,          # conversation_id (unique per user)
        user_id: str,
        cluster_id: str,
        cluster_name: str,
        embedding: List[float],
    ) -> None:
        """English documentation."""
        doc_id = f"{user_id}__{orig_id}"
        self._col.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[""],         # English comment.
            metadatas=[{
                "user_id": user_id,
                "orig_id": orig_id,
                "cluster_id": cluster_id,
                "cluster_name": cluster_name,
            }],
        )
        logger.debug("Stored conversation node: %s (cluster=%s)", doc_id, cluster_id)

    def get_nodes_by_cluster(
        self,
        *,
        query_embedding: List[float],
        user_id: str,
        cluster_id: str,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """English documentation."""
        count = self._col.count()
        if count == 0:
            return []

        n_results = min(top_k, count)
        where = {"$and": [
            {"user_id": {"$eq": user_id}},
            {"cluster_id": {"$eq": cluster_id}},
        ]}

        results = self._col.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["metadatas", "embeddings", "distances"],
        )

        out: List[Dict[str, Any]] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return out

        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            out.append({
                "id": doc_id,
                "orig_id": meta.get("orig_id"),
                "cluster_id": meta.get("cluster_id"),
                "cluster_name": meta.get("cluster_name"),
                "embedding": results["embeddings"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
        return out
