"""Graph store for loading and querying knowledge graph JSON data."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GraphStore:
    """Store for loading and querying graph JSON data."""

    def __init__(self, graph_path: Path):
        """Load graph JSON from file.

        Args:
            graph_path: Path to graph JSON file

        Raises:
            FileNotFoundError: If graph file doesn't exist
            ValueError: If graph JSON is invalid
        """
        self.graph_path = Path(graph_path)
        if not self.graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_path}")

        self._data = self._load_graph()
        self._build_indices()

        logger.info(
            f"Loaded graph with {len(self._nodes_index)} nodes, "
            f"{len(self._data.get('edges', []))} edges, "
            f"{len(self._clusters_index)} clusters"
        )

    def _load_graph(self) -> Dict[str, Any]:
        """Load and validate graph JSON."""
        try:
            with open(self.graph_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate structure
            if not isinstance(data, dict):
                raise ValueError("Graph JSON must be a dictionary")

            if "nodes" not in data:
                raise ValueError("Graph JSON must contain 'nodes' field")

            return data

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}") from e

    def _build_indices(self) -> None:
        """Build internal indices for fast lookups."""
        # Index nodes by ID
        self._nodes_index: Dict[str, dict] = {}
        for node in self._data.get("nodes", []):
            # Support both string and int node IDs
            node_id = str(node.get("id", ""))
            if node_id:
                self._nodes_index[node_id] = node

        # Index nodes by cluster
        self._cluster_nodes_index: Dict[str, List[dict]] = {}
        for node in self._data.get("nodes", []):
            cluster_id = node.get("cluster_id")
            if cluster_id:
                if cluster_id not in self._cluster_nodes_index:
                    self._cluster_nodes_index[cluster_id] = []
                self._cluster_nodes_index[cluster_id].append(node)

        # Index clusters
        self._clusters_index: Dict[str, dict] = {}
        metadata = self._data.get("metadata", {})
        clusters_data = metadata.get("clusters", {})

        # Handle different cluster formats
        # Format 1: {"clusters": [{"id": "cluster_1", ...}, ...]}
        cluster_list = clusters_data.get("clusters", []) if isinstance(clusters_data, dict) and "clusters" in clusters_data else []

        # Format 2: [{"id": "cluster_1", ...}, ...]
        if not cluster_list and isinstance(clusters_data, list):
            cluster_list = clusters_data

        # Format 3: {"cluster_1": {...}, "cluster_2": {...}}
        if not cluster_list and isinstance(clusters_data, dict):
            for cluster_id, cluster_info in clusters_data.items():
                if cluster_id != "clusters" and isinstance(cluster_info, dict):
                    # Add id to cluster info if not present
                    cluster_with_id = {"id": cluster_id, **cluster_info}
                    self._clusters_index[cluster_id] = cluster_with_id
        else:
            # Process list format
            for cluster in cluster_list:
                cluster_id = cluster.get("id")
                if cluster_id:
                    self._clusters_index[cluster_id] = cluster

        # Index edges by node
        self._node_edges_index: Dict[str, List[dict]] = {}
        for edge in self._data.get("edges", []):
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))

            if source:
                if source not in self._node_edges_index:
                    self._node_edges_index[source] = []
                self._node_edges_index[source].append(edge)

            if target:
                if target not in self._node_edges_index:
                    self._node_edges_index[target] = []
                self._node_edges_index[target].append(edge)

    def get_node(self, node_id: str) -> Optional[dict]:
        """Get node by ID.

        Args:
            node_id: Node ID (can be int or string)

        Returns:
            Node dictionary or None if not found
        """
        return self._nodes_index.get(str(node_id))

    def get_nodes_by_cluster(self, cluster_id: str) -> List[dict]:
        """Get all nodes in a cluster.

        Args:
            cluster_id: Cluster ID

        Returns:
            List of node dictionaries
        """
        return self._cluster_nodes_index.get(cluster_id, [])

    def get_cluster(self, cluster_id: str) -> Optional[dict]:
        """Get cluster metadata by ID.

        Args:
            cluster_id: Cluster ID

        Returns:
            Cluster dictionary or None if not found
        """
        return self._clusters_index.get(cluster_id)

    def get_all_clusters(self) -> List[dict]:
        """Get all clusters.

        Returns:
            List of cluster dictionaries
        """
        return list(self._clusters_index.values())

    def get_edges_for_node(self, node_id: str) -> List[dict]:
        """Get all edges connected to a node.

        Args:
            node_id: Node ID

        Returns:
            List of edge dictionaries
        """
        return self._node_edges_index.get(str(node_id), [])

    def get_metadata(self) -> dict:
        """Get graph metadata.

        Returns:
            Metadata dictionary
        """
        return self._data.get("metadata", {})

    def get_all_nodes(self) -> List[dict]:
        """Get all nodes.

        Returns:
            List of node dictionaries
        """
        return self._data.get("nodes", [])

    def get_all_edges(self) -> List[dict]:
        """Get all edges.

        Returns:
            List of edge dictionaries
        """
        return self._data.get("edges", [])

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics.

        Returns:
            Dictionary with graph statistics
        """
        metadata = self.get_metadata()

        # Try to get stats from metadata first
        total_nodes = metadata.get("total_nodes", len(self._nodes_index))
        total_edges = metadata.get("total_edges", len(self._data.get("edges", [])))
        total_clusters = len(self._clusters_index)

        # Calculate additional stats
        nodes_with_times = [
            n for n in self._data.get("nodes", [])
            if n.get("create_time")
        ]
        create_times = [n["create_time"] for n in nodes_with_times]

        time_range = (
            min(create_times) if create_times else None,
            max(create_times) if create_times else None
        )

        # Calculate average cluster size
        cluster_sizes = [len(nodes) for nodes in self._cluster_nodes_index.values()]
        avg_cluster_size = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "total_clusters": total_clusters,
            "time_range": time_range,
            "avg_cluster_size": avg_cluster_size,
            "nodes_with_times": len(nodes_with_times)
        }

    def search_nodes_by_keyword(self, keyword: str, case_sensitive: bool = False) -> List[dict]:
        """Search nodes by keyword in their keywords list.

        Args:
            keyword: Keyword to search for
            case_sensitive: Whether to perform case-sensitive search

        Returns:
            List of nodes containing the keyword
        """
        matching_nodes = []
        search_term = keyword if case_sensitive else keyword.lower()

        for node in self._data.get("nodes", []):
            keywords_list = node.get("keywords", [])

            # Handle both list of dicts and list of strings
            for kw in keywords_list:
                if isinstance(kw, dict):
                    term = kw.get("term", "")
                else:
                    term = str(kw)

                compare_term = term if case_sensitive else term.lower()

                if search_term in compare_term:
                    matching_nodes.append(node)
                    break

        return matching_nodes
