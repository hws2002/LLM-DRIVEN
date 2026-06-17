"""English documentation."""

import json
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

from .utils import logger
import shared.config as cfg
from shared.text_core import canonicalize_text
from shared.token_usage import TokenUsageTracker
from shared.cost_calculator import save_token_run

DEFAULT_EMBEDDING_MODEL   = cfg.ADDNODE_EMBEDDING_MODEL
KEYWORD_METHOD            = cfg.ADDNODE_KEYWORD_METHOD
KEYWORD_TOP_N             = cfg.ADDNODE_KEYWORD_TOP_N
NGRAM_MAX                 = cfg.ADDNODE_NGRAM_MAX
QA_CLUSTERING_MODE        = cfg.ADDNODE_QA_CLUSTERING_MODE
MIN_CLUSTER_SIZE          = cfg.ADDNODE_MIN_CLUSTER_SIZE
MERGE_DISTANCE_THRESHOLD  = cfg.ADDNODE_MERGE_DISTANCE_THRESHOLD
EDGE_SIMILARITY_THRESHOLD = cfg.ADDNODE_EDGE_SIMILARITY_THRESHOLD
EDGE_TOP_K                = cfg.ADDNODE_EDGE_TOP_K
EDGE_FETCH_TOP_K          = int(getattr(cfg, "ADDNODE_EDGE_FETCH_TOP_K", 20))
EDGE_FALLBACK_ENABLED     = bool(getattr(cfg, "ADDNODE_EDGE_FALLBACK_ENABLED", True))
EDGE_FALLBACK_TOP_K       = int(getattr(cfg, "ADDNODE_EDGE_FALLBACK_TOP_K", 20))
NEW_CLUSTER_GUARD_ENABLED = bool(getattr(cfg, "ADDNODE_NEW_CLUSTER_GUARD_ENABLED", True))
NEW_CLUSTER_GUARD_THRESHOLD = float(getattr(cfg, "ADDNODE_NEW_CLUSTER_GUARD_THRESHOLD", 0.30))
NEW_CLUSTER_GUARD_EMBED_WEIGHT = float(getattr(cfg, "ADDNODE_NEW_CLUSTER_GUARD_EMBED_WEIGHT", 0.6))
NEW_CLUSTER_GUARD_KEYWORD_WEIGHT = float(getattr(cfg, "ADDNODE_NEW_CLUSTER_GUARD_KEYWORD_WEIGHT", 0.4))

MISC_CLUSTER_ID = "cluster_misc"
MISC_CLUSTER_NAMES = {"ko": "Miscellaneous Cluster", "zh": "其他", "en": "Others"}
MISC_CLUSTER_DESCRIPTIONS = {
    "ko": "Conversations that do not clearly belong to any topic",
    "zh": "不明确属于任何主题的对话集合",
    "en": "Conversations that do not clearly belong to any topic",
}

from .steps import (
    build_qa_pairs,
    build_note_sections,
    extract_keywords_for_conv,
    extract_note_embeddings,
    cluster_qa_single_conv,
    pool_embeddings,
    assign_cluster_with_llm,
    create_hard_edges_for_new_nodes,
)
from shared.api_provider import ApiProvider
from infra.repositories.vectordb.macro_node_store import MacroNodeStore

# English comment.
_PROJECT_ROOT = Path(__file__).parent
TMP_DIR = _PROJECT_ROOT / "tmp"
_NEW_CLUSTER_GUARD_EMBEDDER = None


def _has_embedding(vec: Any) -> bool:
    """Return True if embedding-like value is non-empty."""
    if vec is None:
        return False
    if isinstance(vec, np.ndarray):
        return vec.size > 0
    if isinstance(vec, (list, tuple)):
        return len(vec) > 0
    return True


def _embedding_to_list(vec: Any) -> Optional[List[float]]:
    """Normalize embedding to list[float] when possible."""
    if vec is None:
        return None
    if isinstance(vec, np.ndarray):
        return vec.astype(np.float32).tolist()
    if isinstance(vec, list):
        return vec
    if isinstance(vec, tuple):
        return list(vec)
    return None


def _select_keywords_from_clusters(
    keywords_data: List[Dict[str, Any]],
    cluster_embeddings: Dict[int, Dict[str, Any]],
    min_total: int = 6,
    per_cluster: int = 2
) -> List[str]:
    """English documentation."""
    qa_keywords_map: Dict[str, List[Dict[str, Any]]] = {}
    for item in keywords_data:
        qa_id = str(item.get("qa_id", ""))
        kws = item.get("keywords", [])
        if qa_id and kws:
            qa_keywords_map[qa_id] = kws

    n_clusters = len(cluster_embeddings)
    if n_clusters == 0:
        all_kws = []
        for item in keywords_data:
            for kw in item.get("keywords", []):
                all_kws.append((kw.get("keyword", ""), kw.get("similarity", 0)))
        all_kws.sort(key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in all_kws[:min_total]]

    adjusted_per_cluster = max(per_cluster, (min_total + n_clusters - 1) // n_clusters)
    selected_keywords = []

    for cluster_id, cluster_data in cluster_embeddings.items():
        qa_ids = cluster_data.get("qa_ids", [])
        cluster_kws = []
        for qa_id in qa_ids:
            qa_id_str = str(qa_id)
            if qa_id_str in qa_keywords_map:
                for kw in qa_keywords_map[qa_id_str]:
                    cluster_kws.append((kw.get("keyword", ""), kw.get("similarity", 0)))
        cluster_kws.sort(key=lambda x: x[1], reverse=True)
        for kw, _ in cluster_kws[:adjusted_per_cluster]:
            if kw and kw not in selected_keywords:
                selected_keywords.append(kw)

    if len(selected_keywords) < min_total:
        all_kws = []
        for item in keywords_data:
            for kw in item.get("keywords", []):
                keyword = kw.get("keyword", "")
                if keyword and keyword not in selected_keywords:
                    all_kws.append((keyword, kw.get("similarity", 0)))
        all_kws.sort(key=lambda x: x[1], reverse=True)
        for kw, _ in all_kws:
            if len(selected_keywords) >= min_total:
                break
            if kw not in selected_keywords:
                selected_keywords.append(kw)

    return selected_keywords


def _select_top_keywords(
    keywords_data: List[Dict[str, Any]],
    top_n: int = 5,
) -> List[str]:
    """Pick unique top-N keywords across all QA pairs by similarity."""
    best_scores: Dict[str, float] = {}

    for item in keywords_data:
        for kw in item.get("keywords", []):
            keyword = str(kw.get("keyword", "")).strip()
            if not keyword:
                continue
            try:
                similarity = float(kw.get("similarity", 0.0))
            except (TypeError, ValueError):
                similarity = 0.0
            prev = best_scores.get(keyword)
            if prev is None or similarity > prev:
                best_scores[keyword] = similarity

    ordered = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
    return [keyword for keyword, _ in ordered[:top_n]]


def _flatten_unique_keywords(
    keywords_data: List[Dict[str, Any]],
    limit: int = 50,
) -> List[str]:
    """Flatten per-QA keyword objects into unique keyword strings."""
    out: List[str] = []
    seen: set[str] = set()
    for item in keywords_data:
        for kw in item.get("keywords", []):
            k = str(kw.get("keyword", "")).strip()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(k)
            if len(out) >= limit:
                return out
    return out


def _extract_cluster_number(cluster_id: Any) -> Optional[int]:
    """Extract numeric suffix from cluster id, e.g. cluster_12 -> 12."""
    if cluster_id is None:
        return None
    text = str(cluster_id).strip()
    if not text:
        return None
    m = re.search(r"cluster_(\d+)$", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)$", text)
    if m:
        return int(m.group(1))
    return None


def _normalize_create_time(raw: Any) -> int:
    """Normalize various timestamp shapes to unix epoch seconds."""
    if raw is None:
        return int(time.time())
    if isinstance(raw, (int, float)):
        val = int(raw)
        # Heuristic: milliseconds -> seconds
        if val > 10_000_000_000:
            val = val // 1000
        return val if val > 0 else int(time.time())
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return int(time.time())
        if s.isdigit():
            val = int(s)
            if val > 10_000_000_000:
                val = val // 1000
            return val if val > 0 else int(time.time())
    return int(time.time())


def _next_cluster_id(existing_clusters: List[Dict[str, Any]]) -> str:
    """Return next cluster id based on max cluster number in existing clusters."""
    max_num = 0
    for c in existing_clusters:
        cid = c.get("clusterId") or c.get("id")
        num = _extract_cluster_number(cid)
        if num is not None and num > max_num:
            max_num = num
    return f"cluster_{max_num + 1}"


def _cluster_profile_text(cluster: Dict[str, Any]) -> str:
    parts: List[str] = []
    name = str(cluster.get("name", "") or "").strip()
    desc = str(cluster.get("description", "") or "").strip()
    themes = [str(t).strip() for t in (cluster.get("themes", []) or []) if str(t).strip()]
    if name:
        parts.append(name)
    if desc:
        parts.append(desc)
    parts.extend(themes[:10])
    return " ".join(parts).strip()


def _token_set_from_text(text: str) -> set[str]:
    canon = canonicalize_text(text)
    if not canon:
        return set()
    return {tok for tok in canon.split() if tok}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _get_new_cluster_guard_embedder():
    global _NEW_CLUSTER_GUARD_EMBEDDER
    if _NEW_CLUSTER_GUARD_EMBEDDER is not None:
        return _NEW_CLUSTER_GUARD_EMBEDDER

    try:
        from sentence_transformers import SentenceTransformer

        _NEW_CLUSTER_GUARD_EMBEDDER = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    except Exception as exc:
        logger.warning(f"NEW_CLUSTER guard embedder load failed: {exc}")
        _NEW_CLUSTER_GUARD_EMBEDDER = None

    return _NEW_CLUSTER_GUARD_EMBEDDER


def _select_existing_cluster_for_new_guard(
    existing_clusters: List[Dict[str, Any]],
    selected_keywords: List[str],
    conversation_embedding: Optional[List[float]],
    conversation_title: str = "",
) -> Optional[Dict[str, Any]]:
    """When LLM returns NEW_CLUSTER, find the closest existing cluster."""
    if not existing_clusters:
        return None

    conv_signal_text = " ".join(
        part for part in [conversation_title.strip(), " ".join(selected_keywords)] if part
    )
    conv_kw_tokens = _token_set_from_text(conv_signal_text)
    conv_vec: Optional[np.ndarray] = None
    if _has_embedding(conversation_embedding):
        try:
            vec = np.asarray(conversation_embedding, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vec))
            if vec.size > 0 and norm > 0:
                conv_vec = vec / norm
        except Exception:
            conv_vec = None

    candidates: List[Dict[str, Any]] = []
    for cluster in existing_clusters:
        cluster_id = cluster.get("clusterId") or cluster.get("id")
        if not cluster_id:
            continue
        profile_text = _cluster_profile_text(cluster)
        kw_sim = _jaccard(conv_kw_tokens, _token_set_from_text(profile_text))
        candidates.append(
            {
                "cluster_id": str(cluster_id),
                "name": str(cluster.get("name", "") or ""),
                "profile_text": profile_text,
                "keyword_sim": float(kw_sim),
                "embedding_sim": 0.0,
            }
        )

    if not candidates:
        return None

    if conv_vec is not None:
        embedder = _get_new_cluster_guard_embedder()
        if embedder is not None:
            profile_texts = [c["profile_text"] or c["name"] for c in candidates]
            try:
                profile_vecs = embedder.encode(
                    profile_texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                if isinstance(profile_vecs, np.ndarray) and profile_vecs.ndim == 2:
                    for idx, row in enumerate(profile_vecs):
                        row_vec = np.asarray(row, dtype=np.float32).reshape(-1)
                        if row_vec.size == conv_vec.size:
                            candidates[idx]["embedding_sim"] = float(np.dot(conv_vec, row_vec))
            except Exception as exc:
                logger.warning(f"NEW_CLUSTER guard embedding similarity failed: {exc}")

    use_embed = any(c["embedding_sim"] > 0 for c in candidates)
    embed_w = max(0.0, NEW_CLUSTER_GUARD_EMBED_WEIGHT if use_embed else 0.0)
    kw_w = max(0.0, NEW_CLUSTER_GUARD_KEYWORD_WEIGHT)
    denom = embed_w + kw_w if (embed_w + kw_w) > 0 else 1.0

    best: Optional[Dict[str, Any]] = None
    best_score = -1.0
    for c in candidates:
        final_score = (embed_w * c["embedding_sim"] + kw_w * c["keyword_sim"]) / denom
        c["final_score"] = float(final_score)
        if final_score > best_score:
            best_score = float(final_score)
            best = c

    return best


def _build_output_dev(
    conversation_embedding: Optional[List[float]],
    target_nodes: List[Dict[str, Any]],
    edge_threshold: float,
    edge_top_k: int,
    top_n: int = 10,
) -> Dict[str, Any]:
    """Build debug info for candidate retrieval and similarities."""
    out: Dict[str, Any] = {
        "retrievedCandidates": len(target_nodes),
        "edgeThreshold": edge_threshold,
        "edgeTopK": edge_top_k,
        "similarityTop": [],
    }

    if not _has_embedding(conversation_embedding) or not target_nodes:
        return out

    valid_nodes = [n for n in target_nodes if _has_embedding(n.get("embedding"))]
    if not valid_nodes:
        return out

    query = np.array([conversation_embedding], dtype=np.float32)
    cand_emb = np.array([_embedding_to_list(n["embedding"]) for n in valid_nodes], dtype=np.float32)

    q_norm = np.linalg.norm(query, axis=1, keepdims=True) + 1e-12
    c_norm = np.linalg.norm(cand_emb, axis=1, keepdims=True).T + 1e-12
    sims = (query @ cand_emb.T) / (q_norm * c_norm)
    sim_arr = sims[0]
    order = np.argsort(-sim_arr)[:top_n]

    for idx in order:
        node = valid_nodes[int(idx)]
        out["similarityTop"].append(
            {
                "targetId": node.get("id"),
                "clusterId": node.get("clusterId"),
                "similarity": float(sim_arr[int(idx)]),
            }
        )
    return out


def _normalize_macro_search_results(raw_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize MacroNodeStore search results for edge creation."""
    return [
        {
            "id": n.get("id"),
            "clusterId": (n.get("metadata") or {}).get("cluster_id"),
            "embedding": _embedding_to_list(n.get("embedding")),
        }
        for n in raw_nodes
        if n.get("id") and _has_embedding(n.get("embedding"))
    ]


def _search_macro_nodes(
    macro_node_store: Optional[MacroNodeStore],
    *,
    query_embedding: Optional[List[float]],
    user_id: str,
    cluster_id: Optional[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Search macro_node safely and return normalized nodes."""
    if macro_node_store is None or not _has_embedding(query_embedding):
        return []
    try:
        raw = macro_node_store.search(
            query_embedding=query_embedding,
            user_id=user_id,
            cluster_id=cluster_id,
            top_k=top_k,
        )
    except Exception as exc:
        logger.warning(f"macro_node search failed (cluster_id={cluster_id}): {exc}")
        return []
    return _normalize_macro_search_results(raw)


def _merge_unique_nodes(*node_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge node groups preserving first occurrence by id."""
    merged: List[Dict[str, Any]] = []
    seen = set()
    for group in node_groups:
        for node in group:
            node_id = node.get("id")
            if not node_id or node_id in seen:
                continue
            merged.append(node)
            seen.add(node_id)
    return merged


def _create_edges_with_fallback(
    *,
    new_nodes: List[Dict[str, Any]],
    primary_nodes: List[Dict[str, Any]],
    query_embedding: Optional[List[float]],
    macro_node_store: Optional[MacroNodeStore],
    user_id: str,
    assigned_cluster_id: Optional[str],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Create cluster-scoped edges, then fallback to user-wide search if needed."""
    output_dev = _build_output_dev(
        conversation_embedding=query_embedding,
        target_nodes=primary_nodes,
        edge_threshold=EDGE_SIMILARITY_THRESHOLD,
        edge_top_k=EDGE_TOP_K,
    )
    output_dev.update({
        "edgeFallbackEnabled": EDGE_FALLBACK_ENABLED,
        "edgeFallbackUsed": False,
        "edgeFallbackRetrievedCandidates": 0,
        "edgeFallbackSimilarityTop": [],
    })

    edges: List[Dict[str, Any]] = []
    if primary_nodes and _has_embedding(query_embedding):
        edges = create_hard_edges_for_new_nodes(
            new_nodes=new_nodes,
            existing_nodes=primary_nodes,
            threshold=EDGE_SIMILARITY_THRESHOLD,
            top_k=EDGE_TOP_K,
        )
        logger.info(f"Created {len(edges)} edges to existing nodes in cluster {assigned_cluster_id}")

    if edges or not EDGE_FALLBACK_ENABLED or macro_node_store is None or not _has_embedding(query_embedding):
        return edges, output_dev

    fallback_nodes = _search_macro_nodes(
        macro_node_store,
        query_embedding=query_embedding,
        user_id=user_id,
        cluster_id=None,
        top_k=EDGE_FALLBACK_TOP_K,
    )
    fallback_nodes = _merge_unique_nodes(primary_nodes, fallback_nodes)
    fallback_dev = _build_output_dev(
        conversation_embedding=query_embedding,
        target_nodes=fallback_nodes,
        edge_threshold=EDGE_SIMILARITY_THRESHOLD,
        edge_top_k=EDGE_TOP_K,
    )
    output_dev["edgeFallbackRetrievedCandidates"] = fallback_dev.get("retrievedCandidates", 0)
    output_dev["edgeFallbackSimilarityTop"] = fallback_dev.get("similarityTop", [])

    if fallback_nodes:
        edges = create_hard_edges_for_new_nodes(
            new_nodes=new_nodes,
            existing_nodes=fallback_nodes,
            threshold=EDGE_SIMILARITY_THRESHOLD,
            top_k=EDGE_TOP_K,
        )
        output_dev["edgeFallbackUsed"] = bool(edges)
        if edges:
            logger.info(
                "Edge fallback created %d user-wide edges for cluster %s",
                len(edges),
                assigned_cluster_id,
            )
    return edges, output_dev


def run_add_node_pipeline(
    conv_id: str,
    user_id: str,
    api_provider: ApiProvider,
    existing_clusters: Optional[List[Dict[str, Any]]] = None,
    macro_node_store: Optional[MacroNodeStore] = None,
    tmp_dir: Path = TMP_DIR,
    tracker: Optional[TokenUsageTracker] = None,
    language: str = "zh",
) -> Dict[str, Any]:
    """English documentation."""
    logger.info(f"Starting add_node pipeline for conversation_{conv_id}")
    existing_clusters = existing_clusters or []

    logger.info(f"Conversation ID: {conv_id}")

    # Step 1: Build Q-A pairs
    logger.info("Step 1/7: Building Q-A pairs...")
    qa_pairs_path = build_qa_pairs(conv_id, tmp_dir)
    logger.info(f"Q-A pairs saved to: {qa_pairs_path}")

    # English comment.
    with open(qa_pairs_path, "r", encoding="utf-8") as f:
        _qa_check = json.load(f)
    if not _qa_check:
        logger.warning(f"[{conv_id}] Q-A pairs text — text text")
        return {
            "nodes": [], "edges": [], "skipped": True,
            "assignedCluster": None, "qaKeywords": [], "selectedKeywords": [],
        }

    # Step 2: Extract keywords and embeddings
    logger.info("Step 2/7: Extracting keywords and embeddings...")
    keywords_output = tmp_dir / f"qa_keywords_{conv_id}.json"
    extract_keywords_for_conv(
        qa_pairs_path=tmp_dir / f"qa_pairs_{conv_id}.json",
        conv_id_target=conv_id,
        model_name=DEFAULT_EMBEDDING_MODEL,
        ngram_max=NGRAM_MAX,
        max_candidates=100,
        top_n=KEYWORD_TOP_N,
        output_path=keywords_output,
        keyword_method=KEYWORD_METHOD,
        keybert_use_tokenizer=True
    )
    logger.info(f"Keywords saved to: {keywords_output}")

    with open(keywords_output, "r", encoding="utf-8") as f:
        keywords_data = json.load(f)

    # Step 3: Q-A clustering mode
    qa_emb_path = tmp_dir / "embeddings" / f"qa_keyword_embeddings_{conv_id}.pkl"
    cluster_output = None
    if QA_CLUSTERING_MODE == "hdbscan":
        logger.info("Step 3/7: Clustering Q-A pairs (HDBSCAN mode)...")
        cluster_output = tmp_dir / "cluster" / f"qa_clusters_{conv_id}.json"
        cluster_qa_single_conv(
            qa_emb_path=qa_emb_path,
            conversation_id=conv_id,
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=1,
            metric="euclidean",
            output_path=cluster_output,
            merge_distance_threshold=MERGE_DISTANCE_THRESHOLD
        )
        logger.info(f"Clusters saved to: {cluster_output}")
    else:
        logger.info("Step 3/7: Skipping local Q-A clustering (all_qa mode)...")

    # Step 4: Pool embeddings
    logger.info("Step 4/7: Pooling embeddings (length-weighted over QA pairs)...")
    pooled_result = pool_embeddings(
        qa_emb_path=qa_emb_path,
        conv_id=conv_id,
        cluster_path=cluster_output,
    )
    cluster_embeddings = pooled_result["cluster_embeddings"]
    conversation_embedding = pooled_result["conversation_embedding"]
    logger.info(f"Pooled {len(cluster_embeddings)} cluster embeddings + 1 conversation embedding")

    # Step 5: Assign cluster using LLM
    logger.info("Step 5/7: Assigning cluster via LLM...")
    conversation_path = tmp_dir / f"conversation_{conv_id}.json"
    conv_title = ""
    conv_create_time = int(time.time())
    if conversation_path.exists():
        with open(conversation_path, "r", encoding="utf-8") as f:
            conv_data = json.load(f)
        if isinstance(conv_data, list) and len(conv_data) > 0:
            conv_title = conv_data[0].get("title", "")
            conv_create_time = _normalize_create_time(
                conv_data[0].get("create_time", conv_data[0].get("createdAt"))
            )
        elif isinstance(conv_data, dict):
            conv_title = conv_data.get("title", "")
            conv_create_time = _normalize_create_time(
                conv_data.get("create_time", conv_data.get("createdAt"))
            )

    if QA_CLUSTERING_MODE == "hdbscan":
        selected_keywords = _select_keywords_from_clusters(
            keywords_data=keywords_data,
            cluster_embeddings=cluster_embeddings,
            min_total=6,
            per_cluster=2
        )
        logger.info(f"Selected {len(selected_keywords)} keywords from {len(cluster_embeddings)} clusters")
    else:
        selected_keywords = _select_top_keywords(
            keywords_data=keywords_data,
            top_n=5
        )
        logger.info(f"Selected top-{len(selected_keywords)} keywords for LLM assignment")

    cluster_assignment = assign_cluster_with_llm(
        existing_clusters=existing_clusters,
        conversation_keywords=selected_keywords,
        conversation_title=conv_title,
        api_provider=api_provider,
        tracker=tracker,
    )
    logger.info(f"Cluster assignment: {cluster_assignment}")

    assigned_cluster_id = cluster_assignment.get("cluster_id")
    is_new_cluster = cluster_assignment.get("is_new_cluster", False)

    if is_new_cluster and existing_clusters and NEW_CLUSTER_GUARD_ENABLED:
        guard_best = _select_existing_cluster_for_new_guard(
            existing_clusters=existing_clusters,
            selected_keywords=selected_keywords,
            conversation_embedding=conversation_embedding,
            conversation_title=conv_title,
        )
        if guard_best is not None:
            logger.info(
                "NEW_CLUSTER guard candidate: %s (final=%.3f, emb=%.3f, kw=%.3f)",
                guard_best.get("cluster_id"),
                float(guard_best.get("final_score", 0.0)),
                float(guard_best.get("embedding_sim", 0.0)),
                float(guard_best.get("keyword_sim", 0.0)),
            )
            if float(guard_best.get("final_score", 0.0)) >= NEW_CLUSTER_GUARD_THRESHOLD:
                assigned_cluster_id = guard_best.get("cluster_id")
                is_new_cluster = False
                prev_reason = str(cluster_assignment.get("reasoning", "") or "").strip()
                guard_reason = (
                    "NEW_CLUSTER guard override -> "
                    f"{assigned_cluster_id} "
                    f"(final={float(guard_best.get('final_score', 0.0)):.3f}, "
                    f"emb={float(guard_best.get('embedding_sim', 0.0)):.3f}, "
                    f"kw={float(guard_best.get('keyword_sim', 0.0)):.3f}, "
                    f"threshold={NEW_CLUSTER_GUARD_THRESHOLD:.3f})"
                )
                cluster_assignment["cluster_id"] = assigned_cluster_id
                cluster_assignment["is_new_cluster"] = False
                cluster_assignment["confidence"] = round(
                    float(guard_best.get("final_score", 0.0)),
                    3,
                )
                cluster_assignment["reasoning"] = (
                    f"{prev_reason} | {guard_reason}" if prev_reason else guard_reason
                )
                logger.info("NEW_CLUSTER guard applied: assigned to %s", assigned_cluster_id)

    cluster_name = ""
    cluster_description = ""
    cluster_themes = []
    if is_new_cluster:
        assigned_cluster_id = MISC_CLUSTER_ID
        cluster_name = MISC_CLUSTER_NAMES.get(language, MISC_CLUSTER_NAMES["en"])
        cluster_description = MISC_CLUSTER_DESCRIPTIONS.get(language, MISC_CLUSTER_DESCRIPTIONS["en"])
        cluster_themes = []
        logger.info(f"Assigned to misc cluster: {assigned_cluster_id}, name: {cluster_name}")
    else:
        for c in existing_clusters:
            if (c.get("clusterId") or c.get("id")) == assigned_cluster_id:
                cluster_name = c.get("name", "")
                cluster_themes = c.get("themes", [])
                break

    # Step 6: Fetch cluster nodes
    logger.info("Step 6/7: Fetching cluster nodes...")
    target_nodes = []
    if is_new_cluster:
        logger.info("New cluster. Skipping node fetch.")
    elif macro_node_store is not None:
        target_nodes = _search_macro_nodes(
            macro_node_store,
            query_embedding=conversation_embedding,
            user_id=user_id,
            cluster_id=assigned_cluster_id,
            top_k=EDGE_FETCH_TOP_K,
        )
    else:
        logger.warning("macro_node_store is None. Skipping cluster node fetch.")
    logger.info(f"Fetched {len(target_nodes)} cluster nodes")
    if target_nodes:
        logger.info(f"sample node : {target_nodes[0]}")

    # Build new node
    total_num_messages = int(pooled_result.get("total_qa_count", 0))
    record_id = f"{user_id}_{conv_id}"
    new_node = {
        "id": record_id,
        "userId": user_id,
        "origId": conv_id,
        "clusterId": assigned_cluster_id,
        "clusterName": cluster_name,
        "numMessages": total_num_messages,
        "embedding": conversation_embedding,
        "timestamp": None,
        "createdAt": None,
        "updatedAt": None,
    }
    nodes_output = [new_node]

    # Step 7: Create edges
    logger.info("Step 7/7: Creating edges...")
    edges_output, output_dev = _create_edges_with_fallback(
        new_nodes=nodes_output,
        primary_nodes=target_nodes,
        query_embedding=conversation_embedding,
        macro_node_store=macro_node_store,
        user_id=user_id,
        assigned_cluster_id=assigned_cluster_id,
    )

    # After Step 7: store new node embedding in ChromaDB
    if macro_node_store is not None and _has_embedding(conversation_embedding):
        flat_qa_keywords = _flatten_unique_keywords(keywords_data)
        macro_node_store.add_embeddings([
            {
                "id": record_id,
                "embedding": _embedding_to_list(conversation_embedding),
                "metadata": {
                    "user_id": user_id,
                    "conversation_id": conv_id,
                    "orig_id": conv_id,
                    "cluster_id": assigned_cluster_id,
                    "cluster_name": new_node["clusterName"],
                    "num_messages": total_num_messages,
                    "create_time": conv_create_time,
                    "selected_keywords": ",".join(selected_keywords),
                    "selected_keywords_count": len(selected_keywords),
                    "qa_keywords": ",".join(flat_qa_keywords),
                    "qa_keywords_count": len(flat_qa_keywords),
                },
            }
        ])
        logger.info(f"Stored conversation embedding in macro_node: {record_id} (cluster={assigned_cluster_id})")

    logger.info(f"Pipeline complete. Generated {len(nodes_output)} node, {len(edges_output)} edges")

    # Do not expose raw embedding vectors in API/SQS output payloads.
    public_nodes_output = [{k: v for k, v in n.items() if k != "embedding"} for n in nodes_output]

    return {
        "nodes": public_nodes_output,
        "edges": edges_output,
        "outputDev": output_dev,
        "qaKeywords": keywords_data,
        "selectedKeywords": selected_keywords,
        "assignedCluster": {
            "clusterId": assigned_cluster_id,
            "isNewCluster": is_new_cluster,
            "confidence": cluster_assignment.get("confidence", 0),
            "reasoning": cluster_assignment.get("reasoning", ""),
            "name": cluster_name,
            "description": cluster_description,
            "themes": cluster_themes
        }
    }


def run_add_note_pipeline(
    note_id: str,
    user_id: str,
    api_provider: ApiProvider,
    existing_clusters: Optional[List[Dict[str, Any]]] = None,
    macro_node_store: Optional[MacroNodeStore] = None,
    tmp_dir: Path = TMP_DIR,
    tracker: Optional[TokenUsageTracker] = None,
    language: str = "zh",
) -> Dict[str, Any]:
    """English documentation."""
    logger.info(f"Starting add_note pipeline for note_{note_id}")
    existing_clusters = existing_clusters or []

    # English comment.
    logger.info("Step 1/7: Parsing note sections...")
    build_note_sections(note_id, tmp_dir)

    sections_path = tmp_dir / f"note_sections_{note_id}.json"
    with open(sections_path, "r", encoding="utf-8") as f:
        sections = json.load(f)

    if not sections:
        logger.warning(f"[{note_id}] No sections found — skipping pipeline")
        return {
            "nodes": [], "edges": [], "skipped": True,
            "assignedCluster": None, "qaKeywords": [], "selectedKeywords": [],
        }

    # English comment.
    note_file = tmp_dir / f"note_{note_id}.json"
    note_title = ""
    if note_file.exists():
        note_data = json.loads(note_file.read_text(encoding="utf-8"))
        note_title = note_data.get("title", "")

    # English comment.
    # English comment.
    # English comment.
    logger.info("Step 2/7: Extracting section embeddings and keywords...")
    keywords_output = tmp_dir / f"qa_keywords_{note_id}.json"
    extract_note_embeddings(
        note_id=note_id,
        tmp_dir=tmp_dir,
        model_name=DEFAULT_EMBEDDING_MODEL,
        ngram_max=NGRAM_MAX,
        max_candidates=100,
        top_n=KEYWORD_TOP_N,
        keyword_method=KEYWORD_METHOD,
        keybert_use_tokenizer=True,
    )

    with open(keywords_output, "r", encoding="utf-8") as f:
        keywords_data = json.load(f)

    # English comment.
    qa_emb_path = tmp_dir / "embeddings" / f"qa_keyword_embeddings_{note_id}.pkl"
    cluster_output = None
    if QA_CLUSTERING_MODE == "hdbscan":
        logger.info("Step 3/7: Clustering sections (HDBSCAN mode)...")
        cluster_output = tmp_dir / "cluster" / f"qa_clusters_{note_id}.json"
        cluster_qa_single_conv(
            qa_emb_path=qa_emb_path,
            conversation_id=note_id,
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=1,
            metric="euclidean",
            output_path=cluster_output,
            merge_distance_threshold=MERGE_DISTANCE_THRESHOLD,
        )
    else:
        logger.info("Step 3/7: Skipping section clustering (all_qa mode)...")

    # English comment.
    logger.info("Step 4/7: Pooling section embeddings (length-weighted)...")
    pooled_result = pool_embeddings(
        qa_emb_path=qa_emb_path,
        conv_id=note_id,
        cluster_path=cluster_output,
    )
    cluster_embeddings = pooled_result["cluster_embeddings"]
    note_embedding = pooled_result["conversation_embedding"]

    # English comment.
    logger.info("Step 5/7: Assigning cluster via LLM...")
    if QA_CLUSTERING_MODE == "hdbscan":
        selected_keywords = _select_keywords_from_clusters(
            keywords_data=keywords_data,
            cluster_embeddings=cluster_embeddings,
            min_total=6,
            per_cluster=2,
        )
    else:
        selected_keywords = _select_top_keywords(keywords_data=keywords_data, top_n=5)

    cluster_assignment = assign_cluster_with_llm(
        existing_clusters=existing_clusters,
        conversation_keywords=selected_keywords,
        conversation_title=note_title,
        api_provider=api_provider,
        tracker=tracker,
    )
    logger.info(f"Cluster assignment: {cluster_assignment}")

    assigned_cluster_id = cluster_assignment.get("cluster_id")
    is_new_cluster = cluster_assignment.get("is_new_cluster", False)

    if is_new_cluster and existing_clusters and NEW_CLUSTER_GUARD_ENABLED:
        guard_best = _select_existing_cluster_for_new_guard(
            existing_clusters=existing_clusters,
            selected_keywords=selected_keywords,
            conversation_embedding=note_embedding,
            conversation_title=note_title,
        )
        if guard_best is not None and float(guard_best.get("final_score", 0.0)) >= NEW_CLUSTER_GUARD_THRESHOLD:
            assigned_cluster_id = guard_best.get("cluster_id")
            is_new_cluster = False
            cluster_assignment["cluster_id"] = assigned_cluster_id
            cluster_assignment["is_new_cluster"] = False

    cluster_name = ""
    cluster_description = ""
    cluster_themes = []
    if is_new_cluster:
        assigned_cluster_id = MISC_CLUSTER_ID
        cluster_name = MISC_CLUSTER_NAMES.get(language, MISC_CLUSTER_NAMES["en"])
        cluster_description = MISC_CLUSTER_DESCRIPTIONS.get(language, MISC_CLUSTER_DESCRIPTIONS["en"])
        cluster_themes = []
        logger.info(f"Assigned to misc cluster: {assigned_cluster_id}, name: {cluster_name}")
    else:
        for c in existing_clusters:
            if (c.get("clusterId") or c.get("id")) == assigned_cluster_id:
                cluster_name = c.get("name", "")
                cluster_themes = c.get("themes", [])
                break

    # English comment.
    logger.info("Step 6/7: Fetching cluster nodes...")
    target_nodes = []
    if is_new_cluster:
        logger.info("New cluster. Skipping node fetch.")
    elif macro_node_store is not None:
        target_nodes = _search_macro_nodes(
            macro_node_store,
            query_embedding=note_embedding,
            user_id=user_id,
            cluster_id=assigned_cluster_id,
            top_k=EDGE_FETCH_TOP_K,
        )
    else:
        logger.warning("macro_node_store is None. Skipping cluster node fetch.")
    logger.info(f"Fetched {len(target_nodes)} cluster nodes")
    if target_nodes:
        logger.info(f"sample node : {target_nodes[0]}")

    # English comment.
    logger.info("Step 7/7: Creating edges...")
    record_id = f"{user_id}_{note_id}"
    new_node = {
        "id": record_id,
        "userId": user_id,
        "origId": note_id,
        "clusterId": assigned_cluster_id,
        "clusterName": cluster_name,
        "numSections": len(sections),
        "embedding": note_embedding,
    }
    nodes_output = [new_node]

    edges_output, output_dev = _create_edges_with_fallback(
        new_nodes=nodes_output,
        primary_nodes=target_nodes,
        query_embedding=note_embedding,
        macro_node_store=macro_node_store,
        user_id=user_id,
        assigned_cluster_id=assigned_cluster_id,
    )

    # English comment.
    if macro_node_store is not None and _has_embedding(note_embedding):
        flat_keywords = _flatten_unique_keywords(keywords_data)
        macro_node_store.add_embeddings([{
            "id": record_id,
            "embedding": _embedding_to_list(note_embedding),
            "metadata": {
                "user_id": user_id,
                "note_id": note_id,
                "orig_id": note_id,
                "cluster_id": assigned_cluster_id,
                "cluster_name": cluster_name,
                "num_sections": len(sections),
                "selected_keywords": ",".join(selected_keywords),
                "selected_keywords_count": len(selected_keywords),
                "qa_keywords": ",".join(flat_keywords),
                "qa_keywords_count": len(flat_keywords),
            },
        }])
        logger.info(f"Stored note embedding in macro_node: {record_id} (cluster={assigned_cluster_id})")

    logger.info(f"Note pipeline complete. 1 node, {len(edges_output)} edges")

    public_nodes = [{k: v for k, v in n.items() if k != "embedding"} for n in nodes_output]

    return {
        "nodes": public_nodes,
        "edges": edges_output,
        "outputDev": output_dev,
        "qaKeywords": keywords_data,
        "selectedKeywords": selected_keywords,
        "assignedCluster": {
            "clusterId": assigned_cluster_id,
            "isNewCluster": is_new_cluster,
            "confidence": cluster_assignment.get("confidence", 0),
            "reasoning": cluster_assignment.get("reasoning", ""),
            "name": cluster_name,
            "description": cluster_description,
            "themes": cluster_themes,
        },
        "skipped": False,
    }


def run_add_node_batch_pipeline(
    batch_data: Dict[str, Any],
    api_provider: ApiProvider,
    macro_node_store: Optional[MacroNodeStore],
    dev_output_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    language: str = "zh",
) -> Dict[str, Any]:
    """English documentation."""
    user_id = batch_data["userId"]
    existing_clusters = batch_data.get("existingClusters", [])
    conversations = batch_data.get("conversations", [])
    notes = batch_data.get("notes", [])

    tracker = TokenUsageTracker(
        model_name=api_provider.model,
        provider_name=api_provider.provider or "unknown",
    )

    # English comment.
    _task_scope = run_id or uuid.uuid4().hex[:8]
    user_tmp_dir = TMP_DIR / user_id / _task_scope
    user_tmp_dir.mkdir(parents=True, exist_ok=True)

    results = []
    try:
        for conv in conversations:
            conv_id = conv["conversationId"]
            try:
                conv_file = user_tmp_dir / f"conversation_{conv_id}.json"
                conv_file.write_text(json.dumps([conv], ensure_ascii=False), encoding="utf-8")

                result = run_add_node_pipeline(
                    conv_id=conv_id,
                    user_id=user_id,
                    api_provider=api_provider,
                    existing_clusters=existing_clusters,
                    macro_node_store=macro_node_store,
                    tmp_dir=user_tmp_dir,
                    tracker=tracker,
                    language=language,
                )
                results.append({"conversationId": conv_id, **result})

                # English comment.
                assigned = result.get("assignedCluster") or {}
                assigned_cid = assigned.get("clusterId")
                if assigned.get("isNewCluster") and assigned_cid:
                    if not any(
                        (c.get("clusterId") or c.get("id")) == assigned_cid
                        for c in existing_clusters
                    ):
                        existing_clusters.append({
                            "clusterId": assigned_cid,
                            "name": assigned.get("name", ""),
                            "description": assigned.get("reasoning", ""),
                            "themes": assigned.get("themes", []),
                        })
                        logger.info(f"Added new cluster to in-batch context: {assigned_cid}")

            except Exception as e:
                # English comment.
                logger.error(f"[{conv_id}] text text (text): {e}", exc_info=True)
                results.append({
                    "conversationId": conv_id,
                    "nodes": [], "edges": [],
                    "error": str(e), "skipped": True,
                })

        # English comment.
        for note in notes:
            note_id = note.get("noteId")
            if not note_id:
                logger.warning("Note missing noteId, skipping")
                continue
            try:
                note_file = user_tmp_dir / f"note_{note_id}.json"
                note_file.write_text(json.dumps(note, ensure_ascii=False), encoding="utf-8")

                result = run_add_note_pipeline(
                    note_id=note_id,
                    user_id=user_id,
                    api_provider=api_provider,
                    existing_clusters=existing_clusters,
                    macro_node_store=macro_node_store,
                    tmp_dir=user_tmp_dir,
                    tracker=tracker,
                    language=language,
                )
                results.append({"noteId": note_id, **result})

                # English comment.
                assigned = result.get("assignedCluster") or {}
                assigned_cid = assigned.get("clusterId")
                if assigned.get("isNewCluster") and assigned_cid:
                    if not any(
                        (c.get("clusterId") or c.get("id")) == assigned_cid
                        for c in existing_clusters
                    ):
                        existing_clusters.append({
                            "clusterId": assigned_cid,
                            "name": assigned.get("name", ""),
                            "description": assigned.get("reasoning", ""),
                            "themes": assigned.get("themes", []),
                        })
                        logger.info(f"Added new cluster from note to in-batch context: {assigned_cid}")

            except Exception as e:
                logger.error(f"[note:{note_id}] text text (text): {e}", exc_info=True)
                results.append({
                    "noteId": note_id,
                    "nodes": [], "edges": [],
                    "error": str(e), "skipped": True,
                })
    finally:
        # English comment.
        if dev_output_dir is not None and user_tmp_dir.exists():
            dev_tmp_dst = dev_output_dir / "tmp"
            shutil.copytree(str(user_tmp_dir), str(dev_tmp_dst), dirs_exist_ok=True)
            logger.info(f"Dev tmp saved to: {dev_tmp_dst}")
        if user_tmp_dir.exists():
            shutil.rmtree(user_tmp_dir)
            logger.info(f"Cleaned up tmp: {user_tmp_dir}")

    if run_id:
        save_token_run(tracker, run_id, service_name="add_node", user_id=user_id)

    return {
        "userId": user_id,
        "processedCount": len(results),
        "results": results,
    }
