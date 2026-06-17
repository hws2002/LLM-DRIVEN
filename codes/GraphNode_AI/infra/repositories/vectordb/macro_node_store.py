"""ChromaDB store for macro graph node embeddings (macro_node collection)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import chromadb

logger = logging.getLogger(__name__)


class MacroNodeStore:
    """Stores and queries pre-computed conversation node embeddings for the macro pipeline."""

    DEFAULT_COLLECTION = "macro_node"

    def __init__(self, chroma_client: chromadb.ClientAPI, collection_name: str = DEFAULT_COLLECTION) -> None:
        self._client = chroma_client
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"MacroNodeStore initialized with collection '{collection_name}'")

    UPSERT_BATCH_SIZE = 250

    def add_embeddings(self, records) -> int:
        """Upsert pre-computed embeddings.

        Accepts EmbeddingRecord objects (.id, .embedding, .metadata) or
        plain dicts with keys 'id', 'embedding', 'metadata'.
        Returns number of records upserted.
        """
        if not records:
            return 0
        ids, embeddings, metadatas = [], [], []
        for r in records:
            if isinstance(r, dict):
                ids.append(r["id"])
                embeddings.append(r["embedding"])
                metadatas.append(self._sanitize(r.get("metadata", {})))
            else:
                ids.append(r.id)
                embeddings.append(r.embedding)
                metadatas.append(self._sanitize(r.metadata if r.metadata else {}))

        for i in range(0, len(ids), self.UPSERT_BATCH_SIZE):
            batch_ids = ids[i:i + self.UPSERT_BATCH_SIZE]
            batch_emb = embeddings[i:i + self.UPSERT_BATCH_SIZE]
            batch_meta = metadatas[i:i + self.UPSERT_BATCH_SIZE]
            self._collection.upsert(ids=batch_ids, embeddings=batch_emb, metadatas=batch_meta)
            logger.info(f"MacroNodeStore: upserted batch {i // self.UPSERT_BATCH_SIZE + 1} ({len(batch_ids)} records)")

        logger.info(f"MacroNodeStore: upserted {len(records)} records total")
        return len(records)

    def search(
        self,
        query_embedding: List[float],
        user_id: str = "",
        top_k: int = 10,
        cluster_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Similarity search with optional user_id / cluster_id filter."""
        clauses = []
        if user_id:
            clauses.append({"user_id": {"$eq": user_id}})
        if cluster_id:
            clauses.append({"cluster_id": {"$eq": cluster_id}})
        where = (
            clauses[0] if len(clauses) == 1
            else {"$and": clauses} if len(clauses) >= 2
            else None
        )
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["metadatas", "distances", "embeddings"],
        }
        if where:
            kwargs["where"] = where
        results = self._collection.query(**kwargs)
        out = []
        if results and results.get("ids") and results["ids"][0]:
            for i, rec_id in enumerate(results["ids"][0]):
                out.append({
                    "id": rec_id,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                    "embedding": results["embeddings"][0][i] if results.get("embeddings") else None,
                })
        return out

    @staticmethod
    def _sanitize(metadata: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in metadata.items():
            if v is None:
                out[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                out[k] = v
            else:
                out[k] = str(v)
        return out
