"""Merge pipeline outputs into the final knowledge graph."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from util.graph_utils import convert_to_frontend_format


def load_subclusters(subcluster_path: Optional[Path]) -> Dict[str, Any]:
    """
    Load subclusters data if available.

    Args:
        subcluster_path: Path to subclusters.json (optional)

    Returns:
        Dictionary with subclusters data or empty dict if not available
    """
    if subcluster_path is None or not subcluster_path.exists():
        return {}

    try:
        with open(subcluster_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load subclusters from {subcluster_path}: {e}")
        return {}


# Removed: Duplicate functions now imported from util.graph_utils
# - _normalize_cluster_entries -> normalize_cluster_entries
# - convert_to_frontend_format (imported above)


def save_graph(
    graph_data: Dict[str, Any],
    output_path: Path,
    frontend_output_path: Optional[Path] = None,
    subcluster_data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Persist graph data to disk and optionally write the frontend format.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)

    if frontend_output_path:
        frontend_output_path.parent.mkdir(parents=True, exist_ok=True)
        frontend_data = convert_to_frontend_format(graph_data, subcluster_data)
        with open(frontend_output_path, "w", encoding="utf-8") as f:
            json.dump(frontend_data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file with error handling."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")


def merge_graph_data(
    features_path: Path,
    cluster_path: Path,
    edges_path: Path,
    output_path: Path,
    frontend_output_path: Optional[Path] = None,
    subcluster_path: Optional[Path] = None,
    verbose: bool = True,
) -> None:
    """
    Merge pipeline outputs into final graph.json.

    This function combines:
    - features.json: conversation features with keywords and embeddings
    - clusters.json: LLM-generated cluster assignments
    - edges.json: similarity-based edges with metadata
    - subclusters.json (optional): sub-cluster assignments

    Into a unified graph structure with nodes, edges, and comprehensive metadata.

    Args:
        features_path: Path to features.json (from extract_features.py)
        cluster_path: Path to clusters.json (from cluster_with_llm.py)
        edges_path: Path to edges.json (from build_edges.py)
        output_path: Path to output graph.json
        frontend_output_path: Path to save frontend-formatted JSON (optional)
        subcluster_path: Path to subclusters.json (from build_subclusters.py, optional)
        verbose: Print progress messages
    """
    if verbose:
        print("🔗 Merging graph data...")

    # Load all input files
    if verbose:
        print("  📂 Loading features.json...")
    features_data = load_json(features_path)
    conversations = features_data.get("conversations", [])
    features_metadata = features_data.get("metadata", {})

    if verbose:
        print("  📂 Loading clusters.json...")
    cluster_data = load_json(cluster_path)
    assignments = cluster_data.get("assignments", [])
    clusters = cluster_data.get("clusters", [])
    cluster_metadata = cluster_data.get("metadata", {})

    if verbose:
        print("  📂 Loading edges.json...")
    edges_data = load_json(edges_path)
    edges = edges_data.get("edges", [])
    edge_metadata = edges_data.get("metadata", {})

    # Load subclusters if provided
    subcluster_data = {}
    node_to_subcluster = {}
    if subcluster_path:
        if verbose:
            print("  📂 Loading subclusters.json...")
        subcluster_data = load_subclusters(subcluster_path)
        if subcluster_data:
            node_to_subcluster = {
                int(k): v
                for k, v in subcluster_data.get("node_to_subcluster", {}).items()
            }
            if verbose:
                sc_count = len(subcluster_data.get("subclusters", []))
                print(f"     Loaded {sc_count} subclusters")

    # Create lookup maps
    if verbose:
        print("  🔗 Building node mappings...")

    # Map: conversation_id -> cluster assignment info
    assignment_map = {
        assign["conversation_id"]: {
            "cluster_id": assign["cluster_id"],
            "confidence": assign["confidence"],
            "top_keywords": assign.get("top_keywords", []),
        }
        for assign in assignments
    }

    # Map: cluster_id -> cluster details
    cluster_map = {
        cluster["id"]: {
            "name": cluster["name"],
            "description": cluster["description"],
            "key_themes": cluster.get("key_themes", []),
            "size": cluster.get("size", 0),
        }
        for cluster in clusters
    }

    # Build enriched nodes
    if verbose:
        print("  🔨 Building enriched nodes...")

    nodes = []
    for conv in conversations:
        conv_id = conv["id"]
        assignment = assignment_map.get(conv_id, {})
        cluster_id = assignment.get("cluster_id", "unknown")
        cluster_info = cluster_map.get(cluster_id, {})

        # Get subcluster assignment
        subcluster_id = node_to_subcluster.get(conv_id)

        node = {
            "id": conv_id,
            "orig_id": conv["orig_id"],
            "title": conv.get("title"),  # NEW: conversation or note title
            "cluster_id": cluster_id,
            "cluster_name": cluster_info.get("name", "Unknown"),
            "cluster_confidence": assignment.get("confidence", 0.0),
            "subcluster_id": subcluster_id,  # None if not in any subcluster
            "keywords": conv["keywords"],
            "top_keywords": assignment.get("top_keywords", []),
            "timestamp": conv.get("timestamp"),
            "num_sections": conv.get("num_sections", 0),
            "source_type": conv.get("source_type", "chat"),  # track source type
        }
        nodes.append(node)

    # Generate comprehensive metadata
    if verbose:
        print("  📊 Generating metadata...")

    # Cluster statistics with details
    cluster_stats = {}
    for cluster in clusters:
        cluster_stats[cluster["id"]] = {
            "name": cluster["name"],
            "description": cluster["description"],
            "size": cluster.get("size", 0),
            "key_themes": cluster.get("key_themes", []),
        }

    # Edge statistics
    edge_stats = {
        "total_edges": edge_metadata.get("total_edges", len(edges)),
        "intra_cluster_edges": edge_metadata.get("intra_cluster_edges", 0),
        "inter_cluster_edges": edge_metadata.get("inter_cluster_edges", 0),
        "high_confidence_edges": edge_metadata.get("high_confidence_edges", 0),
        "llm_verified_edges": edge_metadata.get("llm_verified_edges", 0),
        "thresholds": edge_metadata.get("thresholds", {}),
    }

    # Calculate edge density
    num_nodes = len(nodes)
    max_possible_edges = (num_nodes * (num_nodes - 1)) // 2 if num_nodes > 1 else 0
    edge_density = len(edges) / max_possible_edges if max_possible_edges > 0 else 0.0
    edge_stats["edge_density"] = round(edge_density, 4)

    # Timing information from all steps
    timing_info = {
        "feature_extraction": features_metadata.get("timing", {}),
        "clustering": {
            "total_seconds": cluster_metadata.get("clustering_time_seconds", 0)
        },
        "edge_generation": {
            "total_seconds": edge_metadata.get("edge_generation_time_seconds", 0)
        },
    }

    # Calculate total pipeline time
    total_time = (
        timing_info["feature_extraction"].get("total_seconds", 0)
        + timing_info["clustering"]["total_seconds"]
        + timing_info["edge_generation"]["total_seconds"]
    )
    timing_info["total_pipeline_seconds"] = round(total_time, 2)

    # Subcluster statistics
    subcluster_stats = {}
    if subcluster_data:
        sc_metadata = subcluster_data.get("metadata", {})
        subcluster_stats = {
            "total_subclusters": sc_metadata.get("total_subclusters", 0),
            "nodes_in_subclusters": sc_metadata.get("total_nodes_in_subclusters", 0),
            "coverage": sc_metadata.get("coverage", 0),
            "method": sc_metadata.get("method", "unknown"),
            "parameters": sc_metadata.get("parameters", {}),
            "cluster_breakdown": sc_metadata.get("cluster_stats", {}),
        }

    # Final metadata structure
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "total_clusters": len(clusters),
        "total_subclusters": subcluster_stats.get("total_subclusters", 0),
        "clusters": cluster_stats,
        "edge_statistics": edge_stats,
        "subcluster_statistics": subcluster_stats,
        "timing": timing_info,
        "pipeline_params": {
            "embedding_model": features_metadata.get("embedding_model", "unknown"),
            "clustering_model": cluster_metadata.get("clustering_model", "unknown"),
            "keyword_params": features_metadata.get("keyword_params", {}),
            "preprocess_params": features_metadata.get("preprocess_params", {}),
        },
        "source_files": {
            "features": str(features_path),
            "clusters": str(cluster_path),
            "edges": str(edges_path),
            "subclusters": str(subcluster_path) if subcluster_path else None,
        },
    }

    # Create final graph structure
    graph = {"nodes": nodes, "edges": edges, "metadata": metadata}

    # Add subclusters to graph if available
    if subcluster_data and subcluster_data.get("subclusters"):
        graph["subclusters"] = subcluster_data["subclusters"]

    # Save to file(s)
    if verbose:
        print(f"  💾 Writing to {output_path}...")
        if frontend_output_path:
            print(f"  💾 Writing frontend graph to {frontend_output_path}...")

    save_graph(graph, output_path, frontend_output_path, subcluster_data)

    # Print summary
    if verbose:
        print(f"\n✅ Graph merged successfully!")
        print(f"\n📊 Final Graph Statistics:")
        print(f"  Nodes:                  {len(nodes)}")
        print(f"  Edges:                  {len(edges)}")
        print(f"  Clusters:               {len(clusters)}")
        if subcluster_stats:
            print(
                f"  Sub-clusters:           {subcluster_stats.get('total_subclusters', 0)}"
            )
            coverage = subcluster_stats.get("coverage", 0)
            nodes_in_sc = subcluster_stats.get("nodes_in_subclusters", 0)
            print(f"  Nodes in sub-clusters:  {nodes_in_sc} ({coverage*100:.1f}%)")
        print(
            f"  Intra-cluster edges:    {edge_stats['intra_cluster_edges']} ({edge_stats['intra_cluster_edges']/len(edges)*100:.1f}%)"
            if edges
            else "  Intra-cluster edges:    0"
        )
        print(
            f"  Inter-cluster edges:    {edge_stats['inter_cluster_edges']} ({edge_stats['inter_cluster_edges']/len(edges)*100:.1f}%)"
            if edges
            else "  Inter-cluster edges:    0"
        )
        print(f"  Edge density:           {edge_stats['edge_density']:.4f}")
        print(f"\n💾 Saved to: {output_path.resolve()}")


def validate_input_files(
    features_path: Path,
    cluster_path: Path,
    edges_path: Path,
    subcluster_path: Optional[Path] = None,
) -> None:
    """
    Validate that all required input files exist.

    Raises:
        FileNotFoundError: If any required file is missing
    """
    files = [
        (features_path, "features.json"),
        (cluster_path, "clusters.json"),
        (edges_path, "edges.json"),
    ]

    missing = []
    for path, name in files:
        if not path.exists():
            missing.append(f"{name} at {path}")

    if missing:
        raise FileNotFoundError(
            f"Missing required files:\n" + "\n".join(f"  - {f}" for f in missing)
        )

    # Subcluster file is optional, just warn if specified but missing
    if subcluster_path and not subcluster_path.exists():
        print(f"⚠️  Warning: Subcluster file not found: {subcluster_path}")
        print("   Proceeding without subclusters.")


def main() -> int:
    """CLI entry point for merge_graph.py"""
    parser = argparse.ArgumentParser(
        description="Merge pipeline outputs into final graph.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python merge_graph.py \\
    --features output/features.json \\
    --clusters output/clusters.json \\
    --edges output/edges.json \\
    --output output/graph.json

  # With subclusters
  python merge_graph.py \\
    --features output/features.json \\
    --clusters output/clusters.json \\
    --edges output/edges.json \\
    --subclusters output/subclusters.json \\
    --output output/graph.json

  # With frontend output
  python merge_graph.py \\
    --features output/features.json \\
    --clusters output/clusters.json \\
    --edges output/edges.json \\
    --subclusters output/subclusters.json \\
    --output output/graph.json \\
    --frontend-output output/frontend_graph.json

  # Quiet mode
  python merge_graph.py \\
    --features output/features.json \\
    --clusters output/clusters.json \\
    --edges output/edges.json \\
    --output output/graph.json \\
    --quiet
        """,
    )

    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Path to features.json (from extract_features.py)",
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        required=True,
        help="Path to clusters.json (from cluster_with_llm.py)",
    )
    parser.add_argument(
        "--edges",
        type=Path,
        required=True,
        help="Path to edges.json (from build_edges.py)",
    )
    parser.add_argument(
        "--subclusters",
        type=Path,
        default=None,
        help="Path to subclusters.json (from build_subclusters.py, optional)",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to output graph.json"
    )
    parser.add_argument(
        "--frontend-output",
        type=Path,
        default=None,
        help="Path to save frontend-formatted JSON (optional)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress messages"
    )

    args = parser.parse_args()

    try:
        # Validate input files
        validate_input_files(args.features, args.clusters, args.edges, args.subclusters)

        # Merge graph data
        merge_graph_data(
            features_path=args.features,
            cluster_path=args.clusters,
            edges_path=args.edges,
            output_path=args.output,
            frontend_output_path=args.frontend_output,
            subcluster_path=args.subclusters,
            verbose=not args.quiet,
        )

        return 0

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
