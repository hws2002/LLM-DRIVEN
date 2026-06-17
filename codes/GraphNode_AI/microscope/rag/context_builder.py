"""Context building utilities for RAG."""
from typing import List, Dict, Any


def build_context(chunks: List[Dict[str, Any]], max_chars: int = 4000) -> str:
    """Build text context from chunks with character limit.

    Args:
        chunks: List of chunk dicts with "text" field
        max_chars: Maximum total characters

    Returns:
        Combined context string
    """
    texts = []
    total = 0
    for chunk in chunks:
        text = chunk.get("text", "")
        if not text:
            continue
        if total + len(text) > max_chars:
            break
        texts.append(text)
        total += len(text)
    return "\n\n".join(texts)


def merge_chunks(
    vector_chunks: List[Dict[str, Any]],
    graph_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge vector and graph chunks, deduplicating by uuid."""
    merged = {}

    for ch in vector_chunks:
        chunk_id = ch.get("uuid")
        if chunk_id:
            merged[chunk_id] = ch

    for ch in graph_chunks:
        chunk_id = ch.get("uuid")
        if chunk_id and chunk_id not in merged:
            merged[chunk_id] = ch

    return list(merged.values())


def serialize_context(chunks: List[Dict[str, Any]], separator: str = "\n\n---\n\n") -> str:
    """Serialize chunks to text with custom separator."""
    texts = [ch.get("text", "") for ch in chunks if ch.get("text")]
    return separator.join(texts)
