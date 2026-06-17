"""Retrieval strategies for RAG."""
from typing import List, Dict, Any
from infra.repositories.graph.graphnode_repository import GraphNodeDBHandler
from infra.repositories.neo4j.handler import Neo4jHandler
from infra.repositories.vectordb.chunks_store import VectorDBHandler


def retrieve_vector_chunks(
    vector_db: VectorDBHandler,
    query: str,
    user_id: str,
    group_id: str,
    top_k: int = 5,
    source_name: str = None,
) -> List[Dict[str, Any]]:
    """Retrieve chunks using vector search."""
    return vector_db.retrieve_chunks(
        query=query,
        user_id=user_id,
        group_id=group_id,
        n_results=top_k,
        source_name=source_name,
    )



def retrieve_graph_neighbors(
    graph_db: Neo4jHandler,
    entity_names: List[str],
    user_id: str,
    group_id: str,
    depth: int = 1,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Retrieve neighbors for multiple entities."""
    all_neighbors = []
    for entity_name in entity_names:
        neighbors = graph_db.get_neighbors(
            entity_name=entity_name,
            user_id=user_id,
            group_id=group_id,
            depth=depth,
            limit=limit,
        )
        all_neighbors.extend(neighbors)
    return all_neighbors


def retrieve_entity_chunks(
    graph_db: Neo4jHandler,
    entity_names: List[str],
    user_id: str,
    group_id: str,
) -> List[Dict[str, Any]]:
    """Get all chunks for given entities."""
    all_chunks = []
    for entity_name in entity_names:
        chunks = graph_db.get_chunks_for_entity(
            entity_name=entity_name,
            user_id=user_id,
            group_id=group_id,
        )
        all_chunks.extend(chunks)
    return all_chunks


def retrieve_with_graph_expansion(
    db: GraphNodeDBHandler,
    query: str,
    user_id: str,
    group_id: str,
    top_k: int = 5,
    hop_depth: int = 1,
) -> List[Dict[str, Any]]:
    """Hybrid retrieval: vector search + graph expansion.

    1. Vector search for initial chunks
    2. Extract entity names from chunks
    3. Get neighbors from graph
    4. Get chunks for neighbor entities
    5. Merge and return
    """
    from .context_builder import merge_chunks

    # English comment.
    initial_chunks = retrieve_vector_chunks(
        db.vector_db, query, user_id, group_id, top_k
    )

    # English comment.
    entity_names = set()
    for chunk in initial_chunks:
        names = chunk.get("entity_names", [])
        if isinstance(names, list):
            entity_names.update(names)
        elif isinstance(names, str):
            entity_names.update(n.strip() for n in names.split(",") if n.strip())

    if not entity_names:
        return initial_chunks

    # English comment.
    neighbors = retrieve_graph_neighbors(
        db.graph_db, list(entity_names), user_id, group_id, hop_depth
    )

    neighbor_names = [n.get("name") for n in neighbors if n.get("name")]

    # English comment.
    graph_chunks = retrieve_entity_chunks(
        db.graph_db, neighbor_names, user_id, group_id
    )

    # English comment.
    return merge_chunks(initial_chunks, graph_chunks)


