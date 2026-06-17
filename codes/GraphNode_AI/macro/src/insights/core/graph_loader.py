"""Unified interface for graph data and vector search."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..storage import GraphStore, VectorStore, VectorStoreConfig
from .config import InsightsConfig
from .schema import (
    ClusterData,
    ClusterWithNodes,
    GraphStats,
    NodeData,
    SearchResultWithNode,
)

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class GraphLoader:
    """
    Unified interface for graph data and vector search.
    Combines GraphStore and VectorStore for seamless access to the knowledge graph.
    """

    def __init__(
        self,
        graph_path: Path,
        vector_db_path: Optional[Path] = None,
        config: Optional[InsightsConfig] = None
    ):
        """Initialize graph loader with graph and vector store.

        Args:
            graph_path: Path to graph JSON file
            vector_db_path: Optional path to vector database directory
            config: Optional insights configuration

        Raises:
            FileNotFoundError: If graph file doesn't exist
        """
        self.config = config or InsightsConfig()
        self.graph_path = Path(graph_path)
        self.vector_db_path = Path(vector_db_path) if vector_db_path else None

        # Initialize graph store
        self.graph_store = GraphStore(self.graph_path)
        logger.info(f"Loaded graph from {self.graph_path}")

        # Initialize vector store (optional)
        self.vector_store: Optional[VectorStore] = None
        if self.vector_db_path:
            try:
                vector_config = VectorStoreConfig(
                    persist_directory=str(self.vector_db_path),
                    collection_name=self.config.vector_store.collection_name,
                    embedding_dimension=self.config.vector_store.embedding_dimension
                )
                self.vector_store = VectorStore(vector_config)
                logger.info(f"Loaded vector store from {self.vector_db_path}")
            except Exception as e:
                logger.warning(f"Could not load vector store: {e}")
                self.vector_store = None

        # Lazy-loaded embedding model
        self._embedding_model = None

    # === Graph Access Methods ===

    def get_node(self, node_id: str) -> Optional[NodeData]:
        """Get node by ID.

        Args:
            node_id: Node ID (numeric or string)

        Returns:
            NodeData instance or None if not found
        """
        node = self.graph_store.get_node(node_id)
        if not node:
            return None
        return NodeData.from_graph_node(node)

    def get_cluster(self, cluster_id: str) -> Optional[ClusterData]:
        """Get cluster metadata by ID.

        Args:
            cluster_id: Cluster ID

        Returns:
            ClusterData instance or None if not found
        """
        cluster = self.graph_store.get_cluster(cluster_id)
        if not cluster:
            return None

        # Get actual size from nodes
        nodes = self.graph_store.get_nodes_by_cluster(cluster_id)
        return ClusterData.from_graph_cluster(cluster, size=len(nodes))

    def get_all_nodes(self) -> List[NodeData]:
        """Get all nodes in the graph.

        Returns:
            List of NodeData instances
        """
        nodes = self.graph_store.get_all_nodes()
        return [NodeData.from_graph_node(node) for node in nodes]

    def get_all_clusters(self) -> List[ClusterData]:
        """Get all clusters.

        Returns:
            List of ClusterData instances
        """
        clusters = self.graph_store.get_all_clusters()
        result = []

        for cluster in clusters:
            cluster_id = cluster.get("id", "")
            nodes = self.graph_store.get_nodes_by_cluster(cluster_id)
            result.append(ClusterData.from_graph_cluster(cluster, size=len(nodes)))

        return result

    def get_graph_stats(self) -> GraphStats:
        """Get graph statistics.

        Returns:
            GraphStats instance
        """
        stats = self.graph_store.get_stats()
        return GraphStats(
            total_nodes=stats["total_nodes"],
            total_edges=stats["total_edges"],
            total_clusters=stats["total_clusters"],
            time_range=stats["time_range"],
            avg_cluster_size=stats["avg_cluster_size"],
            edge_density=0.0,  # Can be calculated if needed
            nodes_with_times=stats.get("nodes_with_times", 0)
        )

    # === Vector Search Methods ===

    def search_similar(
        self,
        query: str,
        top_k: int = 10,
        cluster_filter: Optional[str] = None,
        time_filter: Optional[Tuple[str, str]] = None
    ) -> List[SearchResultWithNode]:
        """Search for similar nodes using natural language query.

        Args:
            query: Natural language query text
            top_k: Maximum number of results
            cluster_filter: Optional cluster ID to filter results
            time_filter: Optional (start, end) timestamp filter

        Returns:
            List of SearchResultWithNode instances

        Raises:
            RuntimeError: If vector store is not available
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not available. Provide vector_db_path during initialization.")

        # Encode query to embedding
        query_embedding = self._encode_query(query)

        # Build filters
        filters = {}
        if cluster_filter:
            filters["cluster_id"] = cluster_filter

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters if filters else None
        )

        # Enrich with node data
        enriched_results = []
        for result in results:
            node_id = result.metadata.get("node_id", "")
            orig_id = result.metadata.get("orig_id", "")

            # Try to get node from graph
            node = self.graph_store.get_node(node_id) or self.graph_store.get_node(orig_id)

            if node:
                node_data = NodeData.from_graph_node(node)

                # Extract matched keywords (simple keyword matching)
                matched_keywords = self._extract_matched_keywords(query, node_data)

                enriched_results.append(
                    SearchResultWithNode(
                        node=node_data,
                        score=result.score,
                        matched_keywords=matched_keywords
                    )
                )

        return enriched_results

    def search_by_embedding(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[SearchResultWithNode]:
        """Search for similar nodes using embedding vector.

        Args:
            embedding: Query embedding vector
            top_k: Maximum number of results
            filters: Optional metadata filters

        Returns:
            List of SearchResultWithNode instances

        Raises:
            RuntimeError: If vector store is not available
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not available.")

        results = self.vector_store.search(
            query_embedding=embedding,
            top_k=top_k,
            filters=filters
        )

        # Enrich with node data
        enriched_results = []
        for result in results:
            node_id = result.metadata.get("node_id", "")
            orig_id = result.metadata.get("orig_id", "")

            node = self.graph_store.get_node(node_id) or self.graph_store.get_node(orig_id)

            if node:
                node_data = NodeData.from_graph_node(node)
                enriched_results.append(
                    SearchResultWithNode(
                        node=node_data,
                        score=result.score,
                        matched_keywords=[]
                    )
                )

        return enriched_results

    def get_node_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """Get embedding vector for a node.

        Args:
            node_id: Node ID

        Returns:
            Embedding vector or None if not found

        Raises:
            RuntimeError: If vector store is not available
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not available.")

        # Try to get by node_id or orig_id
        node = self.graph_store.get_node(node_id)
        if not node:
            return None

        orig_id = node.get("orig_id", "")
        search_ids = [node_id, orig_id] if orig_id else [node_id]

        records = self.vector_store.get_by_ids(search_ids)

        if records:
            return records[0].to_numpy()

        return None

    # === Combined Query Methods ===

    def get_cluster_with_nodes(self, cluster_id: str) -> Optional[ClusterWithNodes]:
        """Get cluster with all its nodes.

        Args:
            cluster_id: Cluster ID

        Returns:
            ClusterWithNodes instance or None if not found
        """
        cluster = self.get_cluster(cluster_id)
        if not cluster:
            return None

        # Get all nodes in cluster
        graph_nodes = self.graph_store.get_nodes_by_cluster(cluster_id)
        nodes = [NodeData.from_graph_node(node) for node in graph_nodes]

        # Count internal edges
        internal_edge_count = 0
        for node in graph_nodes:
            node_id = str(node.get("id", ""))
            edges = self.graph_store.get_edges_for_node(node_id)

            for edge in edges:
                # Check if edge is within cluster
                if edge.get("is_intra_cluster", False):
                    internal_edge_count += 1

        # Edges are counted twice (once for each endpoint)
        internal_edge_count = internal_edge_count // 2

        return ClusterWithNodes(
            cluster=cluster,
            nodes=nodes,
            internal_edge_count=internal_edge_count
        )

    def get_related_nodes(self, node_id: str, top_k: int = 5) -> List[NodeData]:
        """Get nodes most similar to the given node.

        Args:
            node_id: Source node ID
            top_k: Number of similar nodes to return

        Returns:
            List of related NodeData instances

        Raises:
            RuntimeError: If vector store is not available
        """
        # Get node's embedding
        embedding = self.get_node_embedding(node_id)
        if embedding is None:
            return []

        # Search for similar nodes
        results = self.search_by_embedding(embedding, top_k=top_k + 1)  # +1 to exclude self

        # Filter out the source node itself
        related = [r.node for r in results if r.node.id != node_id]

        return related[:top_k]

    def get_nodes_by_keywords(self, keywords: List[str], top_k: int = 10) -> List[NodeData]:
        """Find nodes by keyword matching.

        Args:
            keywords: List of keywords to search for
            top_k: Maximum number of results

        Returns:
            List of matching NodeData instances
        """
        all_matches = []

        for keyword in keywords:
            matches = self.graph_store.search_nodes_by_keyword(keyword, case_sensitive=False)
            all_matches.extend(matches)

        # Remove duplicates while preserving order
        seen = set()
        unique_matches = []
        for node in all_matches:
            node_id = str(node.get("id", ""))
            if node_id not in seen:
                seen.add(node_id)
                unique_matches.append(NodeData.from_graph_node(node))

        return unique_matches[:top_k]

    # === Private Helper Methods ===

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode text query to embedding vector.

        Args:
            query: Text query

        Returns:
            Embedding vector

        Raises:
            RuntimeError: If embedding model cannot be loaded
        """
        if self._embedding_model is None:
            self._load_embedding_model()

        # Encode query
        if hasattr(self._embedding_model, 'encode'):
            embedding = self._embedding_model.encode(query, convert_to_numpy=True)
            return np.array(embedding, dtype=np.float32)
        else:
            # Fallback for dummy model
            return self._embedding_model.embed([query])[0]

    def _load_embedding_model(self) -> None:
        """Load embedding model for query encoding."""
        # Try to get model name from graph metadata
        metadata = self.graph_store.get_metadata()
        params = metadata.get("params", {}) or metadata.get("pipeline_params", {})

        if isinstance(params, dict):
            model_name = params.get("embedding_model")
        else:
            model_name = None

        # Fallback to config
        if not model_name:
            model_name = self.config.embedding_model

        logger.info(f"Loading embedding model: {model_name}")

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            warnings.warn("sentence-transformers not available. Install it for embedding support.")
            raise RuntimeError("sentence-transformers not installed")

        try:
            self._embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Could not load embedding model: {e}") from e

    def _extract_matched_keywords(self, query: str, node: NodeData) -> List[str]:
        """Extract keywords from node that match the query.

        Args:
            query: Query string
            node: Node data

        Returns:
            List of matched keyword terms
        """
        query_lower = query.lower()
        matched = []

        for kw in node.keywords[:10]:  # Check top 10 keywords
            if isinstance(kw, dict):
                term = kw.get("term", "")
            else:
                term = str(kw)

            term_lower = term.lower()

            # Simple substring matching
            if term_lower in query_lower or query_lower in term_lower:
                matched.append(term)

        return matched


def load_graph(
    graph_path: Union[str, Path],
    vector_db_path: Optional[Union[str, Path]] = None,
    config_path: Optional[Path] = None
) -> GraphLoader:
    """Convenience function to create GraphLoader.

    Args:
        graph_path: Path to graph JSON file
        vector_db_path: Optional path to vector database
        config_path: Optional path to insights config YAML

    Returns:
        GraphLoader instance

    Example:
        >>> loader = load_graph("output/graph.json", "output/vector_db")
        >>> stats = loader.get_graph_stats()
        >>> print(f"Total nodes: {stats.total_nodes}")
    """
    config = None
    if config_path:
        config = InsightsConfig.from_yaml(config_path)

    return GraphLoader(
        graph_path=Path(graph_path),
        vector_db_path=Path(vector_db_path) if vector_db_path else None,
        config=config
    )
