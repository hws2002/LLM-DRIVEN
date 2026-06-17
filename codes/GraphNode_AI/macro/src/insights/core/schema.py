"""Data classes for graph loader and insights."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class NodeData:
    """Representation of a conversation node with full metadata."""

    id: str
    orig_id: str
    cluster_id: str
    cluster_name: str
    keywords: List[Dict[str, float]]  # [{"term": str, "score": float}, ...]
    create_time: Optional[int] = None
    update_time: Optional[int] = None
    num_sections: int = 0
    top_keywords: List[str] = field(default_factory=list)  # Top keyword terms only
    cluster_confidence: Optional[float] = None

    @classmethod
    def from_graph_node(cls, node: Dict) -> "NodeData":
        """Create NodeData from graph JSON node.

        Args:
            node: Node dictionary from graph JSON

        Returns:
            NodeData instance
        """
        # Extract keywords
        keywords = node.get("keywords", [])

        # Extract top keyword terms
        top_keywords = node.get("top_keywords", [])
        if not top_keywords and keywords:
            # Fallback: use first 3 keyword terms
            top_keywords = [
                kw.get("term", "") if isinstance(kw, dict) else str(kw)
                for kw in keywords[:3]
            ]

        return cls(
            id=str(node.get("id", "")),
            orig_id=node.get("orig_id", ""),
            cluster_id=node.get("cluster_id", ""),
            cluster_name=node.get("cluster_name", ""),
            keywords=keywords,
            create_time=node.get("create_time"),
            update_time=node.get("update_time"),
            num_sections=node.get("num_sections", 0),
            top_keywords=top_keywords,
            cluster_confidence=node.get("cluster_confidence")
        )

    def get_keyword_terms(self, top_n: int = 5) -> List[str]:
        """Extract top N keyword terms.

        Args:
            top_n: Number of keywords to return

        Returns:
            List of keyword terms
        """
        terms = []
        for kw in self.keywords[:top_n]:
            if isinstance(kw, dict):
                term = kw.get("term", "")
                if term:
                    terms.append(term)
            elif isinstance(kw, str):
                terms.append(kw)
        return terms


@dataclass
class ClusterData:
    """Representation of a cluster with metadata."""

    id: str
    name: str
    description: str
    size: int
    key_themes: List[str] = field(default_factory=list)

    @classmethod
    def from_graph_cluster(cls, cluster: Dict, size: Optional[int] = None) -> "ClusterData":
        """Create ClusterData from graph JSON cluster.

        Args:
            cluster: Cluster dictionary from graph JSON
            size: Override size (if not in cluster dict)

        Returns:
            ClusterData instance
        """
        # Handle different key_themes field names
        key_themes = cluster.get("key_themes") or cluster.get("themes", [])

        return cls(
            id=cluster.get("id", ""),
            name=cluster.get("name", ""),
            description=cluster.get("description", ""),
            size=size if size is not None else cluster.get("size", 0),
            key_themes=key_themes
        )


@dataclass
class GraphStats:
    """Statistics about the knowledge graph."""

    total_nodes: int
    total_edges: int
    total_clusters: int
    time_range: Tuple[Optional[int], Optional[int]]
    avg_cluster_size: float
    edge_density: float = 0.0
    nodes_with_times: int = 0


@dataclass
class SearchResultWithNode:
    """Search result combining similarity score with full node data."""

    node: NodeData
    score: float
    matched_keywords: List[str] = field(default_factory=list)  # Keywords contributing to match

    def __post_init__(self):
        """Validate score."""
        if not 0 <= self.score <= 1:
            raise ValueError(f"score must be between 0 and 1, got {self.score}")


@dataclass
class ClusterWithNodes:
    """Cluster with its member nodes."""

    cluster: ClusterData
    nodes: List[NodeData]
    internal_edge_count: int = 0

    @property
    def size(self) -> int:
        """Get cluster size."""
        return len(self.nodes)

    @property
    def avg_sections_per_node(self) -> float:
        """Calculate average sections per node."""
        if not self.nodes:
            return 0.0
        return sum(node.num_sections for node in self.nodes) / len(self.nodes)

    def get_all_keywords(self, top_n: int = 10) -> List[str]:
        """Get most common keywords across all nodes in cluster.

        Args:
            top_n: Number of top keywords to return

        Returns:
            List of keyword terms sorted by frequency
        """
        keyword_counts: Dict[str, int] = {}

        for node in self.nodes:
            for term in node.get_keyword_terms(top_n=5):
                keyword_counts[term] = keyword_counts.get(term, 0) + 1

        # Sort by frequency
        sorted_keywords = sorted(
            keyword_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [kw for kw, _ in sorted_keywords[:top_n]]
