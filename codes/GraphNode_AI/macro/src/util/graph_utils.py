"""Common graph transformation utilities.

This module provides shared functions for converting graph data between
internal and frontend formats, eliminating code duplication across
merge_graph.py and run_pipeline.py.
"""

from typing import Any, Dict, List, Optional


def normalize_cluster_entries(raw_clusters: Any) -> List[Dict[str, Any]]:
    """Normalize cluster entries from various input formats to a flat list.

    Handles multiple input formats:
    - List of cluster dicts: [{"id": "cluster_1", ...}, ...]
    - Nested dict: {"clusters": [...]}
    - Dict with cluster_id keys: {"cluster_1": {...}, ...}

    Args:
        raw_clusters: Raw cluster data in any supported format

    Returns:
        Flat list of cluster dictionaries
    """
    if isinstance(raw_clusters, list):
        return [entry for entry in raw_clusters if isinstance(entry, dict)]

    if isinstance(raw_clusters, dict):
        # Check for nested "clusters" key
        nested = raw_clusters.get("clusters")
        if isinstance(nested, list):
            return [entry for entry in nested if isinstance(entry, dict)]

        # Treat dict keys as cluster IDs
        coerced: List[Dict[str, Any]] = []
        for cluster_id, details in raw_clusters.items():
            entry = {"id": cluster_id}
            if isinstance(details, dict):
                entry.update(details)
            coerced.append(entry)
        return coerced

    return []


def convert_to_frontend_format(
    graph_data: Dict[str, Any], subcluster_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convert internal graph structure into the frontend rendering format.

    This is the single source of truth for frontend format conversion,
    used by both merge_graph.py and run_pipeline.py.

    Args:
        graph_data: Internal graph data with nodes, edges, and metadata
        subcluster_data: Optional subcluster data for enrichment

    Returns:
        Frontend-compatible graph structure with simplified fields
    """
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    metadata = graph_data.get("metadata", {})

    # Build node_to_subcluster mapping from subcluster data
    node_to_subcluster: Dict[int, str] = {}
    if subcluster_data:
        node_to_subcluster = {
            int(k): v for k, v in subcluster_data.get("node_to_subcluster", {}).items()
        }

    # Convert nodes to frontend format (lightweight)
    frontend_nodes = []
    for node in nodes:
        node_id = node.get("id")
        frontend_node = {
            "id": node_id,
            "orig_id": node.get("orig_id"),
            "cluster_id": node.get("cluster_id"),
            "cluster_name": node.get("cluster_name"),
            "subcluster_id": (
                node_to_subcluster.get(node_id) or node.get("subcluster_id")
            ),
            "timestamp": node.get("timestamp"),
            "num_sections": node.get("num_sections"),
        }
        frontend_nodes.append(frontend_node)

    # Convert edges with confidence type mapping
    confidence_to_type = {
        "high": "hard",
        "llm_verified": "insight",
        "medium": "insight",
    }
    frontend_edges = []
    for edge in edges:
        confidence = edge.get("confidence", "")
        frontend_edge = {
            "source": edge.get("source"),
            "target": edge.get("target"),
            "weight": edge.get("weight"),
            "type": confidence_to_type.get(confidence, "insight"),
            "intraCluster": edge.get("is_intra_cluster", False),
        }
        frontend_edges.append(frontend_edge)

    # Convert clusters from metadata
    cluster_entries = normalize_cluster_entries(metadata.get("clusters"))
    frontend_clusters = []
    for cluster in cluster_entries:
        frontend_cluster = {
            "id": cluster.get("id"),
            "name": cluster.get("name"),
            "description": cluster.get("description"),
            "size": cluster.get("size", 0),
            "themes": cluster.get("key_themes") or cluster.get("themes") or [],
        }
        frontend_clusters.append(frontend_cluster)

    # Process subclusters for frontend
    frontend_subclusters = []
    subclusters_list = (
        subcluster_data.get("subclusters", [])
        if subcluster_data
        else graph_data.get("subclusters", [])
    )
    for sc in subclusters_list:
        frontend_subcluster = {
            "id": sc.get("id"),
            "cluster_id": sc.get("cluster_id"),
            "node_ids": sc.get("node_ids", []),
            "size": sc.get("size", 0),
            "representative_node_id": sc.get("representative_node_id"),
        }
        # Include optional metrics if present
        if "density" in sc:
            frontend_subcluster["density"] = sc["density"]
        if "cohesion_score" in sc:
            frontend_subcluster["cohesion_score"] = sc["cohesion_score"]
        if "top_keywords" in sc:
            frontend_subcluster["top_keywords"] = sc["top_keywords"]

        frontend_subclusters.append(frontend_subcluster)

    # Build stats
    stats = {
        "nodes": metadata.get("total_nodes", len(nodes)),
        "edges": metadata.get("total_edges", len(edges)),
        "clusters": metadata.get("total_clusters", len(frontend_clusters)),
        "subclusters": len(frontend_subclusters),
    }

    result = {
        "nodes": frontend_nodes,
        "edges": frontend_edges,
        "clusters": frontend_clusters,
        "stats": stats,
    }

    if frontend_subclusters:
        result["subclusters"] = frontend_subclusters

    return result


def extract_keywords_metadata(keywords_list: List[Any], top_n: int = 5) -> str:
    """Extract keyword terms from a list and format as comma-separated string.

    Handles both dict format [{"term": "python", "score": 0.9}]
    and string format ["python", "flask"].

    Args:
        keywords_list: List of keywords in either format
        top_n: Maximum number of keywords to extract

    Returns:
        Comma-separated string of keyword terms
    """
    terms = []
    for kw in keywords_list[:top_n]:
        if isinstance(kw, dict):
            term = kw.get("term", "")
            if term:
                terms.append(term)
        elif isinstance(kw, str):
            terms.append(kw)
    return ",".join(terms) if terms else ""


def build_node_metadata(
    conversation: Dict[str, Any], node_lookup: Dict[str, Dict[str, Any]], orig_id: str
) -> Dict[str, Any]:
    """Build comprehensive metadata for a node/embedding record.

    Combines data from conversation features with optional graph enrichment.

    Args:
        conversation: Conversation data from features.json
        node_lookup: Lookup dict for graph nodes (by ID and orig_id)
        orig_id: Original conversation ID

    Returns:
        Metadata dictionary suitable for vector store
    """
    # Base metadata from conversation
    metadata = {
        "node_id": str(conversation.get("id", "")),
        "orig_id": orig_id,
        "num_sections": conversation.get("num_sections", 0),
        "create_time": conversation.get("create_time"),
        "update_time": conversation.get("update_time"),
    }

    # Extract keywords
    keywords_list = conversation.get("keywords", [])
    metadata["keywords"] = extract_keywords_metadata(keywords_list, top_n=5)

    # Enrich with graph data if available
    node = node_lookup.get(orig_id) or node_lookup.get(metadata["node_id"])

    if node:
        metadata["cluster_id"] = node.get("cluster_id", "")
        metadata["cluster_name"] = node.get("cluster_name", "")
        metadata["cluster_confidence"] = str(node.get("cluster_confidence", ""))

        # Use top_keywords from graph if available (usually more refined)
        if node.get("top_keywords"):
            metadata["keywords"] = ",".join(node["top_keywords"][:5])

    return metadata
