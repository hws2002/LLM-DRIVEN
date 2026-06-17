"""English documentation."""

from typing import Any, Dict, List

from .logger import logger
from shared.text_core import canonicalize_text
import shared.config as cfg

_FALLBACK_EMBEDDER = None


def build_cluster_prompt(
    existing_clusters: List[Dict[str, Any]],
    conversation_keywords: List[str],
    conversation_title: str = ""
) -> str:
    """English documentation."""

    clusters_text = ""
    for i, cluster in enumerate(existing_clusters):
        cluster_id = cluster.get('clusterId') or cluster.get('id')
        clusters_text += f"""
Cluster {i+1}:
  - ID: {cluster_id}
  - Name: {cluster.get('name')}
  - Description: {cluster.get('description', 'N/A')}
  - Themes: {', '.join(cluster.get('themes', [])[:5])}
"""

    prompt = f"""You are a conversation classifier. Given existing clusters and a new conversation's keywords, determine which cluster the conversation belongs to.

## Existing Clusters:
{clusters_text}

## New Conversation:
- Title: {conversation_title}
- Keywords: {', '.join(conversation_keywords[:15])}

## Task:
Analyze the keywords and determine which existing cluster this conversation best fits into.
If none of the existing clusters are a good match (similarity < 50%), respond with "NEW_CLUSTER".

## Response Format (JSON only):
{{
  "cluster_id": "<cluster_id or NEW_CLUSTER>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>"
}}

Respond with JSON only, no additional text.
"""
    return prompt


def fallback_cluster_assignment(
    existing_clusters: List[Dict[str, Any]],
    conversation_keywords: List[str],
    *,
    use_embedding_fallback: bool = False,
) -> Dict[str, Any]:
    """English documentation."""
    logger.info("Using fallback keyword-based cluster assignment")

    if not existing_clusters:
        return {
            "cluster_id": "NEW_CLUSTER",
            "confidence": 1.0,
            "reasoning": "No existing clusters",
            "is_new_cluster": True
        }

    def _canonical_phrases(texts: List[str]) -> List[str]:
        out: List[str] = []
        for text in texts:
            canon = canonicalize_text(str(text or ""))
            if canon:
                out.append(canon)
        return out

    def _token_set(phrases: List[str]) -> set[str]:
        out: set[str] = set()
        for phrase in phrases:
            out.update(phrase.split())
        return out

    def _jaccard(a: set[str], b: set[str]) -> float:
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    conv_phrases = _canonical_phrases(conversation_keywords)
    conv_tokens = _token_set(conv_phrases)
    if not conv_tokens:
        return {
            "cluster_id": "NEW_CLUSTER",
            "confidence": 1.0,
            "reasoning": "Conversation keywords are empty after normalization",
            "is_new_cluster": True,
        }

    best_cluster = None
    best_score = 0.0
    best_breakdown: Dict[str, float] = {}

    for cluster in existing_clusters:
        texts: List[str] = []
        name = str(cluster.get("name", "") or "")
        desc = str(cluster.get("description", "") or "")
        themes = [str(t) for t in (cluster.get("themes", []) or [])]
        if name:
            texts.append(name)
        if desc:
            texts.append(desc)
        texts.extend(themes)

        cluster_phrases = _canonical_phrases(texts)
        cluster_tokens = _token_set(cluster_phrases)
        if not cluster_tokens:
            continue

        overlap = len(conv_tokens & cluster_tokens)
        union = len(conv_tokens | cluster_tokens)
        token_jaccard = overlap / union if union else 0.0
        containment = overlap / len(conv_tokens) if conv_tokens else 0.0

        # English comment.
        phrase_scores: List[float] = []
        cluster_phrase_tokens = [_token_set([p]) for p in cluster_phrases]
        for conv_phrase in conv_phrases:
            conv_phrase_tokens = _token_set([conv_phrase])
            if not conv_phrase_tokens:
                continue
            max_sim = 0.0
            for cpt in cluster_phrase_tokens:
                sim = _jaccard(conv_phrase_tokens, cpt)
                if sim > max_sim:
                    max_sim = sim
            phrase_scores.append(max_sim)
        phrase_soft = (
            sum(sorted(phrase_scores, reverse=True)[:3]) / min(3, len(phrase_scores))
            if phrase_scores else 0.0
        )

        score = 0.50 * containment + 0.30 * token_jaccard + 0.20 * phrase_soft
        if score > best_score:
            best_score = score
            best_cluster = cluster
            best_breakdown = {
                "containment": containment,
                "jaccard": token_jaccard,
                "phrase_soft": phrase_soft,
            }

    # English comment.
    # English comment.
    if best_cluster is None:
        return {
            "cluster_id": "NEW_CLUSTER",
            "confidence": 1.0,
            "reasoning": "No valid existing cluster representation",
            "is_new_cluster": True,
        }

    if best_score < 0.08 and use_embedding_fallback:
        try:
            global _FALLBACK_EMBEDDER
            if _FALLBACK_EMBEDDER is None:
                from sentence_transformers import SentenceTransformer

                _FALLBACK_EMBEDDER = SentenceTransformer(cfg.ADDNODE_EMBEDDING_MODEL)

            conv_text = " ".join(conversation_keywords[:15]).strip()
            cluster_records: List[tuple[Dict[str, Any], str]] = []
            for cluster in existing_clusters:
                text_parts: List[str] = []
                name = str(cluster.get("name", "") or "")
                desc = str(cluster.get("description", "") or "")
                themes = [str(t) for t in (cluster.get("themes", []) or [])]
                if name:
                    text_parts.append(name)
                if desc:
                    text_parts.append(desc)
                text_parts.extend(themes[:10])
                text = " ".join(text_parts).strip()
                if text:
                    cluster_records.append((cluster, text))

            if conv_text and cluster_records:
                encoded = _FALLBACK_EMBEDDER.encode(
                    [conv_text] + [text for _, text in cluster_records],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                conv_vec = encoded[0]
                cluster_vecs = encoded[1:]
                scores = cluster_vecs @ conv_vec
                best_idx = int(scores.argmax())
                emb_best_score = float(scores[best_idx])
                emb_best_cluster = cluster_records[best_idx][0]

                if emb_best_score >= 0.20:
                    return {
                        "cluster_id": emb_best_cluster.get("clusterId") or emb_best_cluster.get("id"),
                        "confidence": round(max(0.30, min(0.95, emb_best_score)), 3),
                        "reasoning": f"Embedding fallback similarity={emb_best_score:.2f}",
                        "is_new_cluster": False,
                    }
        except Exception as exc:
            logger.warning(f"Embedding fallback disabled due to error: {exc}")

    if best_score < 0.08:
        return {
            "cluster_id": "NEW_CLUSTER",
            "confidence": round(1.0 - best_score, 3),
            "reasoning": f"Very low similarity after soft matching (best score: {best_score:.2f})",
            "is_new_cluster": True,
        }

    return {
        "cluster_id": best_cluster.get("clusterId") or best_cluster.get("id"),
        "confidence": round(max(0.30, min(0.95, best_score)), 3),
        "reasoning": (
            f"Soft similarity score={best_score:.2f} "
            f"(contain={best_breakdown.get('containment', 0.0):.2f}, "
            f"jacc={best_breakdown.get('jaccard', 0.0):.2f}, "
            f"phrase={best_breakdown.get('phrase_soft', 0.0):.2f})"
        ),
        "is_new_cluster": False,
    }
