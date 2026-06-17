"""Vector store implementation using ChromaDB."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
import numpy as np

from .schema import EmbeddingRecord, SearchResult, VectorStoreConfig

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB wrapper for storing and querying conversation embeddings."""

    def __init__(self, config: VectorStoreConfig):
        """Initialize ChromaDB client and collection.

        Args:
            config: Vector store configuration

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If ChromaDB initialization fails
        """
        self.config = config
        self._client = None
        self._collection = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            # Create persist directory if it doesn't exist
            persist_path = Path(self.config.persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)

            # Initialize ChromaDB client with persistence
            self._client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Get or create collection (cosine distance for embedding similarity)
            self._collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                f"Initialized ChromaDB at {persist_path} "
                f"with collection '{self.config.collection_name}'"
            )

        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB: {e}") from e

    def add_embeddings(self, records: List[EmbeddingRecord]) -> int:
        """Batch add embeddings to the vector store.

        Args:
            records: List of embedding records to add

        Returns:
            Number of records successfully added

        Raises:
            ValueError: If records list is empty
            RuntimeError: If ChromaDB operation fails
        """
        if not records:
            raise ValueError("Cannot add empty list of records")

        try:
            ids = [record.id for record in records]
            embeddings = [record.embedding for record in records]
            metadatas = [self._sanitize_metadata(record.metadata) for record in records]

            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )

            logger.info(f"Added {len(records)} embeddings to vector store")
            return len(records)

        except Exception as e:
            raise RuntimeError(f"Failed to add embeddings: {e}") from e

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """Perform similarity search.

        Args:
            query_embedding: Query embedding vector
            top_k: Maximum number of results to return
            filters: Optional metadata filters (ChromaDB where clause)

        Returns:
            List of search results sorted by similarity score

        Raises:
            ValueError: If query_embedding is invalid
            RuntimeError: If search fails
        """
        if query_embedding is None or len(query_embedding) == 0:
            raise ValueError("query_embedding cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        try:
            # Normalize query embedding if needed
            if isinstance(query_embedding, np.ndarray):
                query_list = query_embedding.tolist()
            else:
                query_list = list(query_embedding)

            # Query ChromaDB
            results = self._collection.query(
                query_embeddings=[query_list],
                n_results=top_k,
                where=filters
            )

            # Parse results
            search_results = []
            if results["ids"] and results["ids"][0]:
                for i, result_id in enumerate(results["ids"][0]):
                    # ChromaDB returns distances, convert to similarity scores
                    # Using cosine distance: distance = 1 - similarity
                    # So similarity = 1 - distance
                    distance = results["distances"][0][i] if results.get("distances") else 0
                    score = max(0.0, min(1.0, 1.0 - distance))

                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}

                    search_results.append(
                        SearchResult(
                            id=result_id,
                            score=score,
                            metadata=metadata
                        )
                    )

            logger.debug(f"Search returned {len(search_results)} results")
            return search_results

        except Exception as e:
            raise RuntimeError(f"Search failed: {e}") from e

    def get_by_ids(self, ids: List[str]) -> List[EmbeddingRecord]:
        """Retrieve specific records by IDs.

        Args:
            ids: List of record IDs to retrieve

        Returns:
            List of embedding records (may be shorter than input if some IDs not found)

        Raises:
            ValueError: If ids list is empty
            RuntimeError: If retrieval fails
        """
        if not ids:
            raise ValueError("ids list cannot be empty")

        try:
            results = self._collection.get(ids=ids, include=["embeddings", "metadatas"])

            records = []
            for i, result_id in enumerate(results["ids"]):
                embedding = results["embeddings"][i] if results.get("embeddings") else []
                metadata = results["metadatas"][i] if results.get("metadatas") else {}

                records.append(
                    EmbeddingRecord(
                        id=result_id,
                        embedding=embedding,
                        metadata=metadata
                    )
                )

            logger.debug(f"Retrieved {len(records)} records by ID")
            return records

        except Exception as e:
            raise RuntimeError(f"Failed to retrieve records: {e}") from e

    def update_metadata(self, id: str, metadata: Dict) -> bool:
        """Update metadata for a specific record.

        Args:
            id: Record ID to update
            metadata: New metadata dictionary

        Returns:
            True if update successful, False otherwise

        Raises:
            RuntimeError: If update fails
        """
        try:
            sanitized_metadata = self._sanitize_metadata(metadata)
            self._collection.update(ids=[id], metadatas=[sanitized_metadata])
            logger.debug(f"Updated metadata for record {id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update metadata for {id}: {e}")
            return False

    def delete(self, ids: List[str]) -> int:
        """Delete records by IDs.

        Args:
            ids: List of record IDs to delete

        Returns:
            Number of records deleted

        Raises:
            ValueError: If ids list is empty
        """
        if not ids:
            raise ValueError("ids list cannot be empty")

        try:
            self._collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} records")
            return len(ids)

        except Exception as e:
            logger.error(f"Failed to delete records: {e}")
            return 0

    def count(self) -> int:
        """Get total number of records in the store.

        Returns:
            Total record count
        """
        try:
            return self._collection.count()
        except Exception as e:
            logger.error(f"Failed to get count: {e}")
            return 0

    def clear(self) -> None:
        """Clear all records from the collection.

        Warning: This operation cannot be undone.
        """
        try:
            # Delete collection and recreate it
            self._client.delete_collection(name=self.config.collection_name)
            self._collection = self._client.create_collection(
                name=self.config.collection_name,
                metadata={"embedding_dimension": self.config.embedding_dimension}
            )
            logger.warning(f"Cleared all records from collection '{self.config.collection_name}'")

        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise RuntimeError(f"Failed to clear collection: {e}") from e

    @staticmethod
    def _sanitize_metadata(metadata: Dict) -> Dict:
        """Sanitize metadata for ChromaDB compatibility.

        ChromaDB has restrictions on metadata values (e.g., no None values, must be simple types).

        Args:
            metadata: Raw metadata dictionary

        Returns:
            Sanitized metadata dictionary
        """
        sanitized = {}
        for key, value in metadata.items():
            # Convert None to empty string
            if value is None:
                sanitized[key] = ""
            # Convert lists to comma-separated strings
            elif isinstance(value, list):
                # Handle list of strings
                if all(isinstance(item, str) for item in value):
                    sanitized[key] = ",".join(value)
                # Handle list of numbers
                elif all(isinstance(item, (int, float)) for item in value):
                    sanitized[key] = ",".join(str(item) for item in value)
                else:
                    # For complex lists, convert to JSON string
                    import json
                    sanitized[key] = json.dumps(value)
            # Keep simple types as-is
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            # Convert other types to string
            else:
                sanitized[key] = str(value)

        return sanitized
