"""Embedding indexer for loading features into ChromaDB."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .graph_store import GraphStore
from .schema import EmbeddingRecord
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IndexingResult:
    """Result of an indexing operation."""

    total_indexed: int
    skipped: int
    time_taken_seconds: float
    embedding_dimension: int
    errors: List[str]

    def __str__(self) -> str:
        """Format as readable string."""
        lines = [
            "✅ Embedding indexing complete!",
            f"   Indexed: {self.total_indexed} conversations",
            f"   Skipped: {self.skipped} (invalid/missing embeddings)",
            f"   Vector dimension: {self.embedding_dimension}",
            f"   Time taken: {self.time_taken_seconds:.1f}s",
        ]
        if self.errors:
            lines.append(f"   Warnings: {len(self.errors)} issues")
        return "\n".join(lines)


class EmbeddingIndexer:
    """Indexes conversation embeddings from features.json into ChromaDB."""

    def __init__(self, vector_store: VectorStore):
        """Initialize indexer.

        Args:
            vector_store: VectorStore instance to index into
        """
        self.vector_store = vector_store

    def index_from_features(
        self,
        features_path: Path,
        graph_path: Optional[Path] = None,
        verbose: bool = True,
        user_id: str = "",
    ) -> IndexingResult:
        """Load embeddings from features.json and index into vector store.

        Args:
            features_path: Path to features.json file
            graph_path: Optional path to graph JSON for enriching metadata
            verbose: Show progress output

        Returns:
            IndexingResult with statistics

        Raises:
            FileNotFoundError: If features file doesn't exist
            ValueError: If features JSON is invalid
        """
        start_time = time.time()
        errors = []

        if verbose:
            print(f"Loading features from {features_path}...")

        # Load features
        features_data = self._load_features(features_path)
        conversations = features_data.get("conversations", [])
        embeddings = features_data.get("embeddings", [])

        if len(conversations) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(conversations)} conversations but {len(embeddings)} embeddings"
            )

        # Load graph for cluster metadata (optional)
        graph_store = None
        if graph_path and Path(graph_path).exists():
            if verbose:
                print(f"Loading graph from {graph_path} for metadata enrichment...")
            try:
                graph_store = GraphStore(graph_path)
            except Exception as e:
                errors.append(f"Failed to load graph: {e}")
                logger.warning(f"Could not load graph: {e}")

        # Build node lookup from graph
        node_lookup = {}
        if graph_store:
            for node in graph_store.get_all_nodes():
                # Match by both numeric ID and orig_id
                node_id = node.get("id")
                orig_id = node.get("orig_id")
                if node_id is not None:
                    node_lookup[str(node_id)] = node
                if orig_id:
                    node_lookup[orig_id] = node

        # Process and index embeddings
        records = []
        skipped = 0
        embedding_dim = 0

        for idx, (conv, embedding) in enumerate(zip(conversations, embeddings)):
            try:
                # Validate embedding
                if not embedding or not isinstance(embedding, list):
                    skipped += 1
                    errors.append(f"Conversation {idx}: Invalid embedding")
                    continue

                if not embedding_dim:
                    embedding_dim = len(embedding)

                # Check for zero vectors
                if all(v == 0 for v in embedding):
                    skipped += 1
                    errors.append(f"Conversation {idx}: Zero embedding vector")
                    continue

                # Get conversation ID
                conv_id = str(conv.get("id", idx))
                orig_id = conv.get("orig_id", "")

                # Build metadata
                metadata = self._build_metadata(conv, node_lookup, orig_id, user_id)

                # ID: "{user_id}_{orig_id}" when user_id is set, otherwise orig_id
                raw_id = orig_id if orig_id else f"conv_{conv_id}"
                record_id = f"{user_id}_{raw_id}" if user_id else raw_id

                # Create record
                record = EmbeddingRecord(
                    id=record_id,
                    embedding=embedding,
                    metadata=metadata,
                )
                records.append(record)

            except Exception as e:
                skipped += 1
                errors.append(f"Conversation {idx}: {str(e)}")
                logger.error(f"Error processing conversation {idx}: {e}")

        # Batch index
        if verbose:
            print(f"Indexing {len(records)} embeddings into vector store...")

        total_indexed = 0
        if records:
            total_indexed = self.vector_store.add_embeddings(records)

        time_taken = time.time() - start_time

        result = IndexingResult(
            total_indexed=total_indexed,
            skipped=skipped,
            time_taken_seconds=time_taken,
            embedding_dimension=embedding_dim,
            errors=errors[:10],  # Keep first 10 errors
        )

        if verbose:
            print(result)

        return result

    def reindex(
        self,
        features_path: Path,
        graph_path: Optional[Path] = None,
        verbose: bool = True,
        user_id: str = "",
    ) -> IndexingResult:
        """Clear existing index and rebuild from scratch.

        Args:
            features_path: Path to features.json file
            graph_path: Optional path to graph JSON
            verbose: Show progress output

        Returns:
            IndexingResult with statistics
        """
        if verbose:
            print("Clearing existing vector store...")

        self.vector_store.clear()

        return self.index_from_features(features_path, graph_path, verbose, user_id)

    def _load_features(self, features_path: Path) -> Dict[str, Any]:
        """Load and validate features JSON.

        Args:
            features_path: Path to features.json

        Returns:
            Features data dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid
        """
        features_path = Path(features_path)
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")

        try:
            with open(features_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                raise ValueError("Features JSON must be a dictionary")

            if "conversations" not in data:
                raise ValueError("Features JSON must contain 'conversations' field")

            if "embeddings" not in data:
                raise ValueError("Features JSON must contain 'embeddings' field")

            return data

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}") from e

    def _build_metadata(
        self,
        conversation: Dict[str, Any],
        node_lookup: Dict[str, Dict],
        orig_id: str,
        user_id: str = "",
    ) -> Dict[str, Any]:
        """Build metadata for embedding record.

        Args:
            conversation: Conversation data from features.json
            node_lookup: Lookup dict for graph nodes
            orig_id: Original conversation ID
            user_id: User identifier (required for multi-user filtering)

        Returns:
            Metadata dictionary matching the macro_node schema:
            {user_id, conversation_id, orig_id, cluster_id, cluster_name,
             keywords, create_time, num_sections}
        """
        # Start with basic conversation data
        metadata = {
            "user_id": user_id,
            "conversation_id": orig_id,
            "orig_id": orig_id,
            "num_sections": conversation.get("num_sections", 0),
            "create_time": conversation.get("create_time") or 0,
        }

        # Extract keywords
        keywords_list = conversation.get("keywords", [])
        if keywords_list:
            # Get top 5 keyword terms
            terms = []
            for kw in keywords_list[:5]:
                if isinstance(kw, dict):
                    term = kw.get("term", "")
                    if term:
                        terms.append(term)
                elif isinstance(kw, str):
                    terms.append(kw)

            metadata["keywords"] = ",".join(terms) if terms else ""

        # Enrich with graph data if available
        node = None
        # Try to find node by orig_id first, then by numeric conversation ID
        conv_numeric_id = str(conversation.get("id", ""))
        if orig_id and orig_id in node_lookup:
            node = node_lookup[orig_id]
        elif conv_numeric_id and conv_numeric_id in node_lookup:
            node = node_lookup[conv_numeric_id]

        if node:
            metadata["cluster_id"] = node.get("cluster_id", "")
            metadata["cluster_name"] = node.get("cluster_name", "")
            metadata["cluster_confidence"] = str(node.get("cluster_confidence", ""))

            # Use top_keywords from graph if available
            if "top_keywords" in node and node["top_keywords"]:
                metadata["keywords"] = ",".join(node["top_keywords"][:5])

        return metadata


def index_embeddings_cli(
    features_path: Path,
    graph_path: Optional[Path],
    output_dir: Path,
    collection_name: str = "conversation_embeddings",
    reindex: bool = False,
    verbose: bool = True,
    user_id: str = "",
) -> IndexingResult:
    """CLI function for indexing embeddings.

    Args:
        features_path: Path to features.json
        graph_path: Optional path to graph JSON
        output_dir: Directory for ChromaDB persistence
        collection_name: ChromaDB collection name
        reindex: Whether to clear and rebuild index
        verbose: Show progress output

    Returns:
        IndexingResult with statistics
    """
    from .schema import VectorStoreConfig
    from .vector_store import VectorStore

    # Create vector store config
    config = VectorStoreConfig(
        persist_directory=str(output_dir),
        collection_name=collection_name,
        embedding_dimension=384,  # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    )

    # Initialize vector store
    if verbose:
        print(f"Initializing vector store at {output_dir}...")

    vector_store = VectorStore(config)

    # Create indexer
    indexer = EmbeddingIndexer(vector_store)

    # Index or reindex
    if reindex:
        result = indexer.reindex(features_path, graph_path, verbose, user_id)
    else:
        result = indexer.index_from_features(
            features_path, graph_path, verbose, user_id
        )

    return result
