"""Sub-cluster generation module for grouping densely connected nodes within clusters.

This module identifies tightly connected node groups within each cluster using
graph-based community detection algorithms. Sub-clusters represent coherent
conversation topics that can be collapsed/expanded in the frontend.

Pipeline position: features → clusters → edges → subclusters → graph
"""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


@dataclass
class SubCluster:
    """Represents a sub-cluster (group of densely connected nodes)."""

    id: str
    cluster_id: str  # Parent cluster ID
    node_ids: List[int]  # Member node IDs
    size: int

    # Computed properties
    internal_edges: int = 0
    density: float = 0.0  # Internal edge density (0-1)
    cohesion_score: float = 0.0  # Average internal edge weight

    # Representative info (for display)
    top_keywords: List[str] = field(default_factory=list)
    representative_node_id: Optional[int] = None  # Most central node


@dataclass
class SubClusteringResult:
    """Result of sub-clustering operation."""

    subclusters: List[SubCluster]
    node_to_subcluster: Dict[int, str]  # node_id -> subcluster_id
    metadata: Dict[str, Any]


def build_cluster_subgraph(
    nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], cluster_id: str
) -> Tuple[List[int], List[Tuple[int, int, float]]]:
    """
    Extract nodes and edges belonging to a specific cluster.

    Args:
        nodes: All graph nodes
        edges: All graph edges
        cluster_id: Target cluster ID

    Returns:
        Tuple of (node_ids, edge_list) where edge_list contains (source, target, weight)
    """
    # Get node IDs in this cluster
    cluster_node_ids = set()
    for node in nodes:
        if node.get("cluster_id") == cluster_id:
            node_id = node.get("id")
            if node_id is not None:
                cluster_node_ids.add(int(node_id))

    # Get intra-cluster edges
    cluster_edges = []
    for edge in edges:
        source = int(edge.get("source", -1))
        target = int(edge.get("target", -1))

        # Only include edges where both endpoints are in this cluster
        if source in cluster_node_ids and target in cluster_node_ids:
            weight = float(edge.get("weight", 1.0))
            cluster_edges.append((source, target, weight))

    return list(cluster_node_ids), cluster_edges


def detect_communities_louvain(
    node_ids: List[int],
    edges: List[Tuple[int, int, float]],
    resolution: float = 1.0,
    min_community_size: int = 2,
) -> List[Set[int]]:
    """
    Detect communities using Louvain algorithm.

    Args:
        node_ids: List of node IDs in the subgraph
        edges: List of (source, target, weight) tuples
        resolution: Resolution parameter (higher = more communities)
        min_community_size: Minimum nodes per community

    Returns:
        List of node ID sets, each representing a community
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("networkx is required for community detection")

    if len(node_ids) < min_community_size:
        return []

    # Build networkx graph
    G = nx.Graph()
    G.add_nodes_from(node_ids)

    for source, target, weight in edges:
        G.add_edge(source, target, weight=weight)

    # Handle disconnected nodes - they won't form communities
    if G.number_of_edges() == 0:
        return []

    # Run Louvain community detection
    try:
        from networkx.algorithms.community import louvain_communities

        communities = louvain_communities(
            G, weight="weight", resolution=resolution, seed=42  # For reproducibility
        )
    except ImportError:
        # Fallback: use connected components if louvain not available
        communities = list(nx.connected_components(G))

    # Filter by minimum size
    valid_communities = [
        comm for comm in communities if len(comm) >= min_community_size
    ]

    return valid_communities


def detect_communities_connected_components(
    node_ids: List[int],
    edges: List[Tuple[int, int, float]],
    weight_threshold: float = 0.0,
    min_community_size: int = 2,
) -> List[Set[int]]:
    """
    Detect communities using connected components with edge weight filtering.

    Simpler alternative to Louvain - groups nodes that are directly or
    transitively connected via edges above the weight threshold.

    Args:
        node_ids: List of node IDs
        edges: List of (source, target, weight) tuples
        weight_threshold: Minimum edge weight to include
        min_community_size: Minimum nodes per community

    Returns:
        List of node ID sets
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("networkx is required for community detection")

    if len(node_ids) < min_community_size:
        return []

    # Build graph with filtered edges
    G = nx.Graph()
    G.add_nodes_from(node_ids)

    for source, target, weight in edges:
        if weight >= weight_threshold:
            G.add_edge(source, target, weight=weight)

    # Get connected components
    components = list(nx.connected_components(G))

    # Filter by minimum size
    valid_components = [comp for comp in components if len(comp) >= min_community_size]

    return valid_components


def detect_cliques(
    node_ids: List[int],
    edges: List[Tuple[int, int, float]],
    min_clique_size: int = 3,
    weight_threshold: float = 0.6,
) -> List[Set[int]]:
    """
    Find cliques (fully connected subgraphs) in the graph.

    Cliques represent the most tightly connected node groups.

    Args:
        node_ids: List of node IDs
        edges: List of (source, target, weight) tuples
        min_clique_size: Minimum nodes in a clique
        weight_threshold: Minimum edge weight for clique membership

    Returns:
        List of node ID sets (maximal cliques)
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("networkx is required for clique detection")

    if len(node_ids) < min_clique_size:
        return []

    # Build graph with strong edges only
    G = nx.Graph()
    G.add_nodes_from(node_ids)

    for source, target, weight in edges:
        if weight >= weight_threshold:
            G.add_edge(source, target, weight=weight)

    # Find all maximal cliques
    cliques = list(nx.find_cliques(G))

    # Filter by size and convert to sets
    valid_cliques = [
        set(clique) for clique in cliques if len(clique) >= min_clique_size
    ]

    # Remove cliques that are subsets of larger cliques
    # (already handled by find_cliques returning maximal cliques)

    return valid_cliques


def merge_overlapping_communities(
    communities: List[Set[int]], overlap_threshold: float = 0.5
) -> List[Set[int]]:
    """
    Merge communities that have significant overlap.

    Args:
        communities: List of node ID sets
        overlap_threshold: Jaccard similarity threshold for merging

    Returns:
        Merged community list
    """
    if not communities:
        return []

    # Sort by size (largest first)
    sorted_communities = sorted(communities, key=len, reverse=True)
    merged = []

    for community in sorted_communities:
        # Check overlap with existing merged communities
        merged_into = None

        for i, existing in enumerate(merged):
            intersection = len(community & existing)
            union = len(community | existing)
            jaccard = intersection / union if union > 0 else 0

            if jaccard >= overlap_threshold:
                # Merge into existing
                merged[i] = existing | community
                merged_into = i
                break

        if merged_into is None:
            merged.append(community.copy())

    return merged


def compute_subcluster_metrics(
    subcluster_nodes: Set[int], all_edges: List[Tuple[int, int, float]]
) -> Tuple[int, float, float]:
    """
    Compute metrics for a sub-cluster.

    Args:
        subcluster_nodes: Set of node IDs in the sub-cluster
        all_edges: All edges in the parent cluster

    Returns:
        Tuple of (internal_edges, density, cohesion_score)
    """
    n = len(subcluster_nodes)
    if n < 2:
        return 0, 0.0, 0.0

    # Count internal edges and sum weights
    internal_edges = 0
    weight_sum = 0.0

    for source, target, weight in all_edges:
        if source in subcluster_nodes and target in subcluster_nodes:
            internal_edges += 1
            weight_sum += weight

    # Maximum possible edges (undirected graph)
    max_edges = n * (n - 1) / 2

    # Density: fraction of possible edges that exist
    density = internal_edges / max_edges if max_edges > 0 else 0.0

    # Cohesion: average edge weight
    cohesion = weight_sum / internal_edges if internal_edges > 0 else 0.0

    return internal_edges, density, cohesion


def extract_top_keywords(
    node_ids: Set[int], nodes: List[Dict[str, Any]], top_n: int = 5
) -> List[str]:
    """
    Extract most common keywords from sub-cluster nodes.

    Args:
        node_ids: Node IDs in the sub-cluster
        nodes: All nodes with keyword data
        top_n: Number of keywords to return

    Returns:
        List of top keyword terms
    """
    keyword_counts: Dict[str, float] = defaultdict(float)

    # Build node lookup
    node_map = {int(n.get("id", -1)): n for n in nodes}

    for node_id in node_ids:
        node = node_map.get(node_id)
        if not node:
            continue

        keywords = node.get("keywords", [])
        for kw in keywords:
            if isinstance(kw, dict):
                term = kw.get("term", "")
                score = kw.get("score", 1.0)
            else:
                term = str(kw)
                score = 1.0

            if term:
                keyword_counts[term] += score

    # Sort by score and return top N
    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)

    return [term for term, _ in sorted_keywords[:top_n]]


def find_representative_node(
    node_ids: Set[int], edges: List[Tuple[int, int, float]]
) -> Optional[int]:
    """
    Find the most central node in a sub-cluster (highest degree/weight sum).

    Args:
        node_ids: Node IDs in the sub-cluster
        edges: Edges within the sub-cluster's parent cluster

    Returns:
        Node ID of the most central node, or None
    """
    if not node_ids:
        return None

    if len(node_ids) == 1:
        return list(node_ids)[0]

    # Calculate weighted degree for each node
    degree_scores: Dict[int, float] = defaultdict(float)

    for source, target, weight in edges:
        if source in node_ids and target in node_ids:
            degree_scores[source] += weight
            degree_scores[target] += weight

    if not degree_scores:
        # No internal edges, return first node
        return list(node_ids)[0]

    # Return node with highest weighted degree
    return max(degree_scores.items(), key=lambda x: x[1])[0]


def build_subclusters(
    graph_path: Path,
    output_path: Path,
    method: str = "louvain",
    min_subcluster_size: int = 2,
    resolution: float = 1.0,
    weight_threshold: float = 0.0,
    verbose: bool = True,
) -> SubClusteringResult:
    """
    Build sub-clusters from graph data.

    Args:
        graph_path: Path to graph JSON (post-processed)
        output_path: Path to output subclusters JSON
        method: Detection method ("louvain", "components", "cliques")
        min_subcluster_size: Minimum nodes per sub-cluster
        resolution: Resolution for Louvain (higher = more sub-clusters)
        weight_threshold: Minimum edge weight for inclusion
        verbose: Print progress

    Returns:
        SubClusteringResult with all sub-clusters and mappings
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError(
            "networkx is required for sub-clustering. "
            "Install with: pip install networkx"
        )

    start_time = time.perf_counter()

    if verbose:
        print(f"📊 Building sub-clusters using {method} method...")
        print(f"   Parameters: min_size={min_subcluster_size}, resolution={resolution}")

    # Load graph
    if verbose:
        print(f"   Loading graph from {graph_path}...")

    with open(graph_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    metadata = graph_data.get("metadata", {})

    if verbose:
        print(f"   Loaded {len(nodes)} nodes, {len(edges)} edges")

    # Get unique cluster IDs
    cluster_ids = set()
    for node in nodes:
        cluster_id = node.get("cluster_id")
        if cluster_id:
            cluster_ids.add(cluster_id)

    if verbose:
        print(f"   Processing {len(cluster_ids)} clusters...")

    # Process each cluster
    all_subclusters: List[SubCluster] = []
    node_to_subcluster: Dict[int, str] = {}
    subcluster_counter = 0

    cluster_stats = {}

    for cluster_id in sorted(cluster_ids):
        # Extract cluster subgraph
        cluster_node_ids, cluster_edges = build_cluster_subgraph(
            nodes, edges, cluster_id
        )

        if len(cluster_node_ids) < min_subcluster_size:
            if verbose:
                print(f"   Skipping {cluster_id}: only {len(cluster_node_ids)} nodes")
            continue

        # Detect communities based on method
        if method == "louvain":
            communities = detect_communities_louvain(
                cluster_node_ids,
                cluster_edges,
                resolution=resolution,
                min_community_size=min_subcluster_size,
            )
        elif method == "components":
            communities = detect_communities_connected_components(
                cluster_node_ids,
                cluster_edges,
                weight_threshold=weight_threshold,
                min_community_size=min_subcluster_size,
            )
        elif method == "cliques":
            communities = detect_cliques(
                cluster_node_ids,
                cluster_edges,
                min_clique_size=min_subcluster_size,
                weight_threshold=weight_threshold,
            )
            # Merge overlapping cliques
            communities = merge_overlapping_communities(communities)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Create SubCluster objects
        cluster_subclusters = []
        for community in communities:
            subcluster_counter += 1
            subcluster_id = f"subcluster_{subcluster_counter}"

            # Compute metrics
            internal_edges, density, cohesion = compute_subcluster_metrics(
                community, cluster_edges
            )

            # Extract keywords
            top_keywords = extract_top_keywords(community, nodes, top_n=5)

            # Find representative node
            representative = find_representative_node(community, cluster_edges)

            subcluster = SubCluster(
                id=subcluster_id,
                cluster_id=cluster_id,
                node_ids=sorted(community),
                size=len(community),
                internal_edges=internal_edges,
                density=round(density, 4),
                cohesion_score=round(cohesion, 4),
                top_keywords=top_keywords,
                representative_node_id=representative,
            )

            cluster_subclusters.append(subcluster)

            # Update node mapping
            for node_id in community:
                node_to_subcluster[node_id] = subcluster_id

        all_subclusters.extend(cluster_subclusters)

        # Track stats
        cluster_stats[cluster_id] = {
            "total_nodes": len(cluster_node_ids),
            "subclusters_found": len(cluster_subclusters),
            "nodes_in_subclusters": sum(sc.size for sc in cluster_subclusters),
            "isolated_nodes": len(cluster_node_ids)
            - sum(sc.size for sc in cluster_subclusters),
        }

    elapsed = time.perf_counter() - start_time

    # Build result
    result_metadata = {
        "method": method,
        "parameters": {
            "min_subcluster_size": min_subcluster_size,
            "resolution": resolution,
            "weight_threshold": weight_threshold,
        },
        "total_subclusters": len(all_subclusters),
        "total_nodes_in_subclusters": len(node_to_subcluster),
        "total_nodes": len(nodes),
        "coverage": round(len(node_to_subcluster) / len(nodes), 4) if nodes else 0,
        "cluster_stats": cluster_stats,
        "processing_time_seconds": round(elapsed, 2),
    }

    result = SubClusteringResult(
        subclusters=all_subclusters,
        node_to_subcluster=node_to_subcluster,
        metadata=result_metadata,
    )

    # Save to file
    output_data = {
        "subclusters": [asdict(sc) for sc in all_subclusters],
        "node_to_subcluster": {str(k): v for k, v in node_to_subcluster.items()},
        "metadata": result_metadata,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # Print summary
    if verbose:
        print(f"\n✅ Sub-clustering complete!")
        print(f"\n📊 Results:")
        print(f"   Total sub-clusters: {len(all_subclusters)}")
        print(
            f"   Nodes in sub-clusters: {len(node_to_subcluster)}/{len(nodes)} ({result_metadata['coverage']*100:.1f}%)"
        )
        print(f"   Processing time: {elapsed:.2f}s")

        print(f"\n📈 Per-cluster breakdown:")
        for cluster_id, stats in cluster_stats.items():
            print(f"   {cluster_id}:")
            print(
                f"      Nodes: {stats['total_nodes']}, Sub-clusters: {stats['subclusters_found']}"
            )
            print(
                f"      In sub-clusters: {stats['nodes_in_subclusters']}, Isolated: {stats['isolated_nodes']}"
            )

        # Show largest sub-clusters
        if all_subclusters:
            print(f"\n🔝 Largest sub-clusters:")
            sorted_sc = sorted(all_subclusters, key=lambda x: x.size, reverse=True)
            for sc in sorted_sc[:5]:
                kw_str = (
                    ", ".join(sc.top_keywords[:3])
                    if sc.top_keywords
                    else "(no keywords)"
                )
                print(
                    f"   {sc.id} ({sc.cluster_id}): {sc.size} nodes, density={sc.density:.2f}"
                )
                print(f"      Keywords: {kw_str}")

        print(f"\n💾 Saved to: {output_path.resolve()}")

    return result


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for sub-cluster generation."""
    parser = argparse.ArgumentParser(
        description="Generate sub-clusters from densely connected nodes within clusters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with Louvain method
  python build_subclusters.py \\
      --graph output/graph_postprocessed.json \\
      --output output/subclusters.json

  # Use connected components method
  python build_subclusters.py \\
      --graph output/graph_postprocessed.json \\
      --output output/subclusters.json \\
      --method components

  # Adjust resolution for more/fewer sub-clusters
  python build_subclusters.py \\
      --graph output/graph_postprocessed.json \\
      --output output/subclusters.json \\
      --resolution 1.5  # Higher = more sub-clusters

  # Find cliques (fully connected groups)
  python build_subclusters.py \\
      --graph output/graph_postprocessed.json \\
      --output output/subclusters.json \\
      --method cliques \\
      --weight-threshold 0.7
        """,
    )

    parser.add_argument(
        "--graph",
        type=Path,
        required=True,
        help="Path to input graph JSON (post-processed)",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to output subclusters JSON"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["louvain", "components", "cliques"],
        default="louvain",
        help="Community detection method (default: louvain)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=2,
        help="Minimum nodes per sub-cluster (default: 2)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Resolution for Louvain method (default: 1.0, higher = more sub-clusters)",
    )
    parser.add_argument(
        "--weight-threshold",
        type=float,
        default=0.0,
        help="Minimum edge weight for inclusion (default: 0.0)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args(argv)

    # Validate input
    if not args.graph.exists():
        print(f"❌ Error: Graph file not found: {args.graph}")
        return

    try:
        build_subclusters(
            graph_path=args.graph,
            output_path=args.output,
            method=args.method,
            min_subcluster_size=args.min_size,
            resolution=args.resolution,
            weight_threshold=args.weight_threshold,
            verbose=not args.quiet,
        )
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("   Install networkx: pip install networkx")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
