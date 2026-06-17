"""Repetition pattern metrics collector."""

from collections import defaultdict
from typing import Any, Dict, List, Set

from .base import BaseMetricsCollector, PatternEvidence


class RepetitionMetricsCollector(BaseMetricsCollector):
    """
    Measures repetition patterns using existing graph data:
    - Keyword frequency across nodes
    - High-similarity edges (already computed)
    - Hub nodes with many connections
    - Cross-cluster keywords
    - Intra-cluster density
    """

    def __init__(self, graph_store: "GraphStore"):
        """
        Initialize the repetition metrics collector.

        Args:
            graph_store: GraphStore instance containing graph data
        """
        super().__init__(graph_store)
        self._metrics_cache = None

    def collect(self) -> Dict[str, Any]:
        """
        Collect all repetition metrics.

        Returns:
            Dictionary containing all metrics
        """
        if self._metrics_cache:
            return self._metrics_cache

        self._metrics_cache = {
            "keyword_frequency": self._compute_keyword_frequency(),
            "high_similarity_pairs": self._find_high_similarity_pairs(threshold=0.8),
            "hub_nodes": self._find_hub_nodes(min_edges=3),
            "cross_cluster_keywords": self._find_cross_cluster_keywords(),
            "intra_cluster_density": self._compute_intra_cluster_density(),
        }
        return self._metrics_cache

    def _compute_keyword_frequency(self) -> Dict[str, Dict]:
        """
        Count how many nodes each keyword appears in.
        Returns only keywords appearing in 2+ nodes (actual repetition).

        Returns:
            Dictionary mapping keywords to their occurrence data
        """
        keyword_nodes: Dict[str, List[str]] = defaultdict(list)
        keyword_clusters: Dict[str, Set[str]] = defaultdict(set)

        for node in self.graph_store.get_all_nodes():
            node_id = node.get("orig_id") or str(node.get("id"))
            cluster_id = node.get("cluster_id", "")

            # Get top 5 keywords per node
            for kw in node.get("keywords", [])[:5]:
                term = kw.get("term") if isinstance(kw, dict) else str(kw)
                if term:
                    keyword_nodes[term].append(node_id)
                    if cluster_id:
                        keyword_clusters[term].add(cluster_id)

        # Filter to keywords in 2+ nodes
        return {
            kw: {
                "count": len(nodes),
                "node_ids": nodes,
                "cluster_count": len(keyword_clusters[kw]),
                "clusters": list(keyword_clusters[kw])
            }
            for kw, nodes in keyword_nodes.items()
            if len(nodes) >= 2
        }

    def _find_high_similarity_pairs(self, threshold: float = 0.8) -> List[Dict]:
        """
        Find edges with weight >= threshold.
        These represent highly similar conversation pairs.

        Args:
            threshold: Minimum similarity score (default 0.8)

        Returns:
            List of high-similarity pair dictionaries
        """
        pairs = []

        for edge in self.graph_store.get_all_edges():
            weight = edge.get("weight", 0)
            if weight >= threshold:
                source_id = str(edge.get("source"))
                target_id = str(edge.get("target"))

                source_node = self.graph_store.get_node(source_id)
                target_node = self.graph_store.get_node(target_id)

                if source_node and target_node:
                    pairs.append({
                        "source_id": source_node.get("orig_id") or source_id,
                        "target_id": target_node.get("orig_id") or target_id,
                        "similarity": weight,
                        "is_intra_cluster": edge.get("is_intra_cluster", False),
                        "source_cluster": source_node.get("cluster_name", ""),
                        "target_cluster": target_node.get("cluster_name", ""),
                        "shared_keywords": self._get_shared_keywords(source_node, target_node)
                    })

        return sorted(pairs, key=lambda x: x["similarity"], reverse=True)

    def _find_hub_nodes(self, min_edges: int = 3) -> List[Dict]:
        """
        Find nodes connected to many similar conversations.
        These are central topics that recur frequently.

        Args:
            min_edges: Minimum number of edges to qualify as a hub (default 3)

        Returns:
            List of hub node dictionaries
        """
        node_connections: Dict[str, List[Dict]] = defaultdict(list)

        for edge in self.graph_store.get_all_edges():
            source_id = str(edge.get("source"))
            target_id = str(edge.get("target"))
            weight = edge.get("weight", 0)

            node_connections[source_id].append({"peer": target_id, "weight": weight})
            node_connections[target_id].append({"peer": source_id, "weight": weight})

        hubs = []
        for node_id, connections in node_connections.items():
            if len(connections) >= min_edges:
                node = self.graph_store.get_node(node_id)
                if node:
                    avg_similarity = sum(c["weight"] for c in connections) / len(connections)
                    keywords = node.get("keywords", [])

                    # Get peer node IDs (orig_id if available)
                    connected_to = []
                    for conn in connections[:5]:
                        peer_node = self.graph_store.get_node(conn["peer"])
                        if peer_node:
                            peer_id = peer_node.get("orig_id") or conn["peer"]
                            connected_to.append(peer_id)

                    hubs.append({
                        "node_id": node.get("orig_id") or node_id,
                        "edge_count": len(connections),
                        "avg_similarity": round(avg_similarity, 3),
                        "cluster_name": node.get("cluster_name", ""),
                        "top_keywords": [
                            kw.get("term") if isinstance(kw, dict) else str(kw)
                            for kw in keywords[:3]
                        ],
                        "connected_to": connected_to
                    })

        return sorted(hubs, key=lambda x: x["edge_count"], reverse=True)

    def _find_cross_cluster_keywords(self) -> List[Dict]:
        """
        Find keywords that appear across multiple clusters.

        Returns:
            List of cross-cluster keyword dictionaries
        """
        freq = self._compute_keyword_frequency()

        cross_cluster = [
            {
                "keyword": kw,
                "cluster_count": data["cluster_count"],
                "clusters": data["clusters"],
                "total_occurrences": data["count"]
            }
            for kw, data in freq.items()
            if data["cluster_count"] >= 2
        ]

        return sorted(cross_cluster, key=lambda x: x["cluster_count"], reverse=True)

    def _compute_intra_cluster_density(self) -> Dict[str, Dict]:
        """
        Compute edge density within each cluster.

        Returns:
            Dictionary mapping cluster IDs to density metrics
        """
        cluster_nodes: Dict[str, List[str]] = defaultdict(list)
        cluster_edges: Dict[str, int] = defaultdict(int)

        # Count nodes per cluster
        for node in self.graph_store.get_all_nodes():
            cluster_id = node.get("cluster_id", "unknown")
            cluster_nodes[cluster_id].append(str(node.get("id")))

        # Count intra-cluster edges
        for edge in self.graph_store.get_all_edges():
            if edge.get("is_intra_cluster", False):
                source = self.graph_store.get_node(str(edge.get("source")))
                if source:
                    cluster_id = source.get("cluster_id", "unknown")
                    cluster_edges[cluster_id] += 1

        # Calculate density
        result = {}
        for cluster_id, nodes in cluster_nodes.items():
            n = len(nodes)
            max_edges = n * (n - 1) // 2 if n > 1 else 0
            actual_edges = cluster_edges.get(cluster_id, 0)
            density = actual_edges / max_edges if max_edges > 0 else 0

            cluster = self.graph_store.get_cluster(cluster_id)
            result[cluster_id] = {
                "name": cluster.get("name", cluster_id) if cluster else cluster_id,
                "node_count": n,
                "edge_count": actual_edges,
                "density": round(density, 3)
            }

        return result

    def _get_shared_keywords(self, node1: Dict, node2: Dict) -> List[str]:
        """
        Extract keywords shared between two nodes.

        Args:
            node1: First node dictionary
            node2: Second node dictionary

        Returns:
            List of shared keyword terms
        """
        def extract_terms(node):
            return {
                kw.get("term") if isinstance(kw, dict) else str(kw)
                for kw in node.get("keywords", [])[:5]
            }

        return list(extract_terms(node1) & extract_terms(node2))

    def find_evidence(self) -> List[PatternEvidence]:
        """
        Generate evidence objects with confidence scores.

        Returns:
            List of PatternEvidence objects
        """
        metrics = self.collect()
        evidences = []

        # Evidence 1: High-frequency keywords
        freq = metrics["keyword_frequency"]
        if freq:
            top_keywords = sorted(freq.items(), key=lambda x: x[1]["count"], reverse=True)[:5]
            top_kw, top_data = top_keywords[0]

            evidences.append(PatternEvidence(
                node_ids=top_data["node_ids"][:5],
                keywords=[kw for kw, _ in top_keywords],
                metric_values={
                    "type": "keyword_frequency",
                    "top_keyword": top_kw,
                    "top_count": top_data["count"],
                    "repeated_keyword_count": len(freq)
                },
                confidence=min(1.0, top_data["count"] / 5),  # 5+ occurrences = 1.0
                description=f"'{top_kw}' appears in {top_data['count']} conversations"
            ))

        # Evidence 2: High similarity pairs
        pairs = metrics["high_similarity_pairs"]
        if pairs:
            top_pair = pairs[0]
            evidences.append(PatternEvidence(
                node_ids=[top_pair["source_id"], top_pair["target_id"]],
                keywords=top_pair["shared_keywords"],
                metric_values={
                    "type": "high_similarity_pairs",
                    "pair_count": len(pairs),
                    "highest_similarity": top_pair["similarity"],
                    "avg_similarity": sum(p["similarity"] for p in pairs) / len(pairs)
                },
                confidence=top_pair["similarity"],
                description=f"{len(pairs)} conversation pairs with ≥80% similarity"
            ))

        # Evidence 3: Hub nodes
        hubs = metrics["hub_nodes"]
        if hubs:
            top_hub = hubs[0]
            evidences.append(PatternEvidence(
                node_ids=[top_hub["node_id"]] + top_hub["connected_to"][:3],
                keywords=top_hub["top_keywords"],
                metric_values={
                    "type": "hub_nodes",
                    "hub_count": len(hubs),
                    "max_connections": top_hub["edge_count"]
                },
                confidence=min(1.0, top_hub["edge_count"] / 5),
                description=f"'{top_hub['node_id']}' connected to {top_hub['edge_count']} similar conversations"
            ))

        # Evidence 4: Cross-cluster keywords
        cross = metrics["cross_cluster_keywords"]
        if cross:
            top_cross = cross[0]
            evidences.append(PatternEvidence(
                node_ids=[],
                keywords=[c["keyword"] for c in cross[:5]],
                metric_values={
                    "type": "cross_cluster_keywords",
                    "keyword": top_cross["keyword"],
                    "cluster_count": top_cross["cluster_count"]
                },
                confidence=min(1.0, top_cross["cluster_count"] / 3),
                description=f"'{top_cross['keyword']}' spans {top_cross['cluster_count']} clusters"
            ))

        return sorted(evidences, key=lambda e: e.confidence, reverse=True)

    def get_summary_for_llm(self) -> str:
        """
        Generate formatted summary for LLM prompt.

        Returns:
            Markdown-formatted summary string
        """
        metrics = self.collect()
        lines = ["## Repetition Patterns (Measured from Graph Data)"]

        # 1. Keyword frequency
        freq = metrics["keyword_frequency"]
        if freq:
            top_keywords = sorted(freq.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
            lines.append("\n### Keywords Appearing in Multiple Conversations")
            for kw, data in top_keywords:
                node_preview = ", ".join(data["node_ids"][:3])
                if len(data["node_ids"]) > 3:
                    node_preview += "..."
                lines.append(f"- **{kw}**: {data['count']} conversations [{node_preview}]")
        else:
            lines.append("\n### Keywords Appearing in Multiple Conversations")
            lines.append("- No repeated keywords found")

        # 2. High similarity pairs
        pairs = metrics["high_similarity_pairs"]
        if pairs:
            lines.append(f"\n### Highly Similar Conversation Pairs (≥80% similarity): {len(pairs)} pairs")
            for p in pairs[:5]:
                shared = ", ".join(p["shared_keywords"]) if p["shared_keywords"] else "none identified"
                cluster_info = "[same cluster]" if p["is_intra_cluster"] else f"[cross-cluster: {p['source_cluster']} ↔ {p['target_cluster']}]"
                lines.append(f"- {p['source_id']} ↔ {p['target_id']}: **{p['similarity']:.1%}** {cluster_info}")
                lines.append(f"  - Shared keywords: {shared}")
        else:
            lines.append("\n### Highly Similar Conversation Pairs (≥80% similarity): 0 pairs")
            lines.append("- No highly similar pairs found")

        # 3. Hub nodes
        hubs = metrics["hub_nodes"]
        if hubs:
            lines.append(f"\n### Hub Conversations (Connected to 3+ Similar Conversations): {len(hubs)} hubs")
            for h in hubs[:5]:
                kw_str = ", ".join(h["top_keywords"]) if h["top_keywords"] else "none"
                lines.append(f"- **{h['node_id']}**: {h['edge_count']} connections (avg similarity: {h['avg_similarity']:.1%})")
                lines.append(f"  - Cluster: {h['cluster_name']}, Keywords: {kw_str}")
        else:
            lines.append("\n### Hub Conversations (Connected to 3+ Similar Conversations): 0 hubs")
            lines.append("- No hub nodes found")

        # 4. Cross-cluster keywords
        cross = metrics["cross_cluster_keywords"]
        if cross:
            lines.append(f"\n### Keywords Spanning Multiple Clusters: {len(cross)} keywords")
            for c in cross[:5]:
                clusters = ", ".join(c["clusters"][:3])
                if len(c["clusters"]) > 3:
                    clusters += "..."
                lines.append(f"- **{c['keyword']}**: {c['cluster_count']} clusters ({clusters})")
        else:
            lines.append("\n### Keywords Spanning Multiple Clusters: 0 keywords")
            lines.append("- No cross-cluster keywords found")

        # 5. Cluster density
        density = metrics["intra_cluster_density"]
        if density:
            lines.append("\n### Intra-Cluster Repetition Density")
            sorted_density = sorted(density.items(), key=lambda x: x[1]["density"], reverse=True)
            for cid, data in sorted_density[:5]:
                lines.append(f"- {data['name']}: {data['density']:.1%} ({data['edge_count']} edges among {data['node_count']} nodes)")
        else:
            lines.append("\n### Intra-Cluster Repetition Density")
            lines.append("- No cluster density data available")

        return "\n".join(lines)
