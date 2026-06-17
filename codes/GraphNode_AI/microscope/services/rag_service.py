"""Core RAG service logic — shared between HTTP API (main.py) and SQS worker (worker.py).

Each function takes plain parameters (not DTO objects) so this module has no dependency
on server-layer DTOs, and can be used from any interface (HTTP, SQS, CLI).
"""

from __future__ import annotations

from typing import Any, Dict, List

from infra.repositories.graph.graphnode_repository import GraphNodeDBHandler
from microscope.rag.retrieval_strategies import (
    retrieve_with_graph_expansion,
    retrieve_vector_chunks,
    retrieve_graph_neighbors,
)
from microscope.rag.context_builder import build_context
from microscope.rag.prompt_builder import PromptFactory
from microscope.rag.answer_gen import get_response
from microscope.rag.macro_context import build_macro_profile
from microscope.utils.io_utils import save_service_output
from shared.api_provider import ApiProvider


def run_query(  # English comment.
    db: GraphNodeDBHandler,
    provider: ApiProvider,
    *,
    query: str,
    user_id: str,
    group_id: str,
    top_k: int = 5,
    hop_depth: int = 1,
    no_rag: bool = False,
    macro_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Hybrid RAG query.

    When ``macro_summary`` (a stored macro GraphSummary dict) is provided, the
    answer is personalized: the user's Macro-Graph patterns and learning
    tendencies are fused into the prompt. Falls back to plain RAG otherwise.

    Returns:
        {"answer": str, "context": str, "chunks": list}
    """
    if no_rag:
        chunks: List[Dict[str, Any]] = []
        context = ""
        prompt = PromptFactory.no_context(query)
    else:
        chunks = retrieve_with_graph_expansion(
            db, query, user_id, group_id, top_k=top_k, hop_depth=hop_depth
        )
        context = build_context(chunks)
        profile = build_macro_profile(macro_summary)
        prompt = PromptFactory.rag_with_profile(context, query, profile)

    answer = get_response(prompt, api_provider=provider)
    save_service_output("query", {
        "query": query, "user_id": user_id, "group_id": group_id,
        "no_rag": no_rag, "context": context, "answer": answer,
    })
    return {"answer": answer, "context": context, "chunks": chunks}


def run_synthesize( # English comment.
    db: GraphNodeDBHandler,
    provider: ApiProvider,
    *,
    topic: str,
    user_id: str,
    group_id: str,
    top_k: int = 5,
    hop_depth: int = 1,
    no_rag: bool = False,
) -> Dict[str, Any]:
    """Topic synthesis.

    Returns:
        {"answer": str, "context": str, "chunks": list}
    """
    if no_rag:
        chunks: List[Dict[str, Any]] = []
        context = ""
        prompt = PromptFactory.no_context(topic)
    else:
        chunks = retrieve_with_graph_expansion(
            db, topic, user_id, group_id, top_k=top_k, hop_depth=hop_depth
        )
        context = build_context(chunks)
        prompt = PromptFactory.summary(context, topic)

    answer = get_response(prompt, api_provider=provider)
    save_service_output("synthesize", {
        "topic": topic, "user_id": user_id, "group_id": group_id,
        "no_rag": no_rag, "context": context, "answer": answer,
    })
    return {"answer": answer, "context": context, "chunks": chunks}


def run_related_questions( # English comment.
    db: GraphNodeDBHandler,
    provider: ApiProvider,
    *,
    query: str,
    user_id: str,
    group_id: str,
    top_k: int = 5,
    hop_depth: int = 1,
) -> Dict[str, Any]:
    """Generate follow-up questions via vector search + graph expansion.

    Returns:
        {"questions": str, "entities": list[str]}
    """
    seed_chunks = retrieve_vector_chunks(
        db.vector_db, query, user_id, group_id, top_k=top_k
    )

    entity_names: List[str] = []
    seen: set = set()
    for chunk in seed_chunks:
        raw = chunk.get("entity_names", [])
        if isinstance(raw, str):
            raw = [n.strip() for n in raw.split(",") if n.strip()]
        for name in (raw if isinstance(raw, list) else []):
            if name and name not in seen:
                seen.add(name)
                entity_names.append(name)

    if entity_names:
        neighbors = retrieve_graph_neighbors(
            db.graph_db, entity_names, user_id, group_id, depth=hop_depth
        )
        seen_n: set = set()
        neighbor_names: List[str] = []
        for node in neighbors:
            name = node.get("name")
            if name and name not in seen_n:
                seen_n.add(name)
                neighbor_names.append(name)
        all_entities = entity_names + [n for n in neighbor_names if n not in set(entity_names)]
    else:
        all_entities = []

    prompt = PromptFactory.related_questions(all_entities, query)
    questions = get_response(prompt, api_provider=provider)
    save_service_output("related_questions", {
        "query": query, "user_id": user_id, "group_id": group_id,
        "entities": all_entities, "questions": questions,
    })
    return {"questions": questions, "entities": all_entities}
