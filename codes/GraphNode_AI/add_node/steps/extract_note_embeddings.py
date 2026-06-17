"""English documentation."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

import shared.config as cfg
from shared.text_core import (
    build_shared_vectorizer,
    canonicalize_text_list,
    prepare_keyword_pairs,
)

from ..utils import logger

HUGGINGFACE_CACHE_DIR = os.getenv("HUGGINGFACE_CACHE_DIR")
STOPWORD_LANGS = ("en", "zh", "ko")

# English comment.
MAX_CHARS_PER_CHUNK = 1500


def _split_text(text: str, max_chars: int) -> List[str]:
    """English documentation."""
    chunks: List[str] = []
    cur = ""
    for word in text.split():
        if len(cur) + len(word) + 1 > max_chars and cur:
            chunks.append(cur.strip())
            cur = word
        else:
            cur += (" " + word) if cur else word
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def _embed_section(model: SentenceTransformer, text: str) -> np.ndarray:
    """English documentation."""
    if len(text) <= MAX_CHARS_PER_CHUNK:
        return model.encode([text], normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False)[0]

    chunks = _split_text(text, MAX_CHARS_PER_CHUNK)
    chunk_embs = model.encode(chunks, normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False)  # (N, dim)
    chunk_lengths = np.array([len(c) for c in chunks], dtype=np.float32)
    total_weight = chunk_lengths.sum()

    # English comment.
    pooled = (chunk_lengths[:, None] * chunk_embs).sum(axis=0) / total_weight
    norm = np.linalg.norm(pooled)
    return pooled / (norm + 1e-9)


def extract_note_embeddings(
    note_id: str,
    tmp_dir: Path,
    model_name: str,
    ngram_max: int,
    max_candidates: int,
    top_n: int,
    keyword_method: str,
    keybert_use_tokenizer: bool,
) -> None:
    """English documentation."""
    text_core_profile = getattr(cfg, "TEXT_CORE_PROFILE_ADDNODE", "balanced")

    sections_file = tmp_dir / f"note_sections_{note_id}.json"
    sections: List[Dict[str, Any]] = json.loads(sections_file.read_text(encoding="utf-8"))

    if not sections:
        logger.warning(f"[{note_id}] No sections to embed")
        return

    logger.info(f"[{note_id}] Loading model: {model_name}")
    if HUGGINGFACE_CACHE_DIR:
        model = SentenceTransformer(model_name, cache_folder=HUGGINGFACE_CACHE_DIR)
    else:
        model = SentenceTransformer(model_name)

    keybert_model = None
    keybert_vectorizer = None
    if keyword_method == "keybert":
        keybert_model = KeyBERT(model=model)
        if keybert_use_tokenizer:
            keybert_vectorizer = build_shared_vectorizer(ngram_max, stopword_langs=STOPWORD_LANGS)

    keyword_results: List[Dict[str, Any]] = []
    emb_records: List[Dict[str, Any]] = []

    for sec in sections:
        section_id = sec["qa_id"]          # English comment.
        section_index = sec["qa_index"]
        text = sec["content"]

        # English comment.
        sec_vec = _embed_section(model, text)

        # Keyword extraction
        top_keywords: List[Dict[str, float]] = []
        candidates_for_store: List[str] = []
        cand_vecs = np.zeros((0, sec_vec.shape[0]), dtype=np.float32)

        if keyword_method == "keybert" and keybert_model:
            keywords_raw = keybert_model.extract_keywords(
                text,
                top_n=max(top_n * 2, top_n),
                vectorizer=keybert_vectorizer,
            )
            normalized_pairs = prepare_keyword_pairs(
                keywords_raw,
                source_text=text,
                top_n=top_n,
                min_keywords=min(3, top_n),
                dedup_threshold=0.8,
                max_formula_keywords=2,
                profile=text_core_profile,
            )
            if normalized_pairs:
                top_keywords = [{"keyword": kw, "similarity": float(score)}
                                for kw, score in normalized_pairs if kw]
                candidate_texts = [kw for kw, _ in normalized_pairs]
                deduped = canonicalize_text_list(candidate_texts, stopword_langs=STOPWORD_LANGS)
                if deduped:
                    cand_vecs = model.encode(deduped, normalize_embeddings=True,
                                            convert_to_numpy=True, show_progress_bar=False)
                    candidates_for_store = deduped
        else:
            from ..utils.ngram_utils import build_candidates_for_text
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            raw_candidates = build_candidates_for_text(text, ngram_max=ngram_max,
                                                       max_candidates=max_candidates)
            candidates = canonicalize_text_list(raw_candidates, stopword_langs=STOPWORD_LANGS)
            if candidates:
                cand_vecs = model.encode(candidates, normalize_embeddings=True,
                                         convert_to_numpy=True, show_progress_bar=False)
                sims = cos_sim(sec_vec.reshape(1, -1), cand_vecs).flatten()
                order = np.argsort(-sims)[:top_n]
                top_keywords = [{"keyword": candidates[i], "similarity": float(sims[i])}
                                for i in order]
                candidates_for_store = candidates

        keyword_results.append({
            "qa_id": section_id,
            "conversation_id": note_id,
            "qa_index": section_index,
            "keywords": top_keywords,
        })

        candidates_list = []
        for i, cand_text in enumerate(candidates_for_store):
            if i < len(cand_vecs):
                candidates_list.append({
                    "text": cand_text,
                    "embedding": cand_vecs[i].tolist(),
                })

        emb_records.append({
            "qa_id": section_id,            # English comment.
            "conversation_id": note_id,
            "qa_index": section_index,
            "qa_embedding": sec_vec.tolist(),   # English comment.
            "qa_length": len(text),             # English comment.
            "candidates": candidates_list,
        })

    # English comment.
    keywords_out = tmp_dir / f"qa_keywords_{note_id}.json"
    keywords_out.write_text(json.dumps(keyword_results, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    logger.info(f"[{note_id}] Keywords saved → {keywords_out}")

    # English comment.
    emb_dir = tmp_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    emb_out = emb_dir / f"qa_keyword_embeddings_{note_id}.pkl"
    with open(emb_out, "wb") as f:
        pickle.dump(emb_records, f)
    logger.info(f"[{note_id}] Embeddings saved → {emb_out}  ({len(emb_records)} sections)")
