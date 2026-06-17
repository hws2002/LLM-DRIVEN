"""English documentation."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

# KeyBERT imports transformers.pipeline. In local worker environments where
# TensorFlow/protobuf versions drift, disabling TF keeps the text stack on torch.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import shared.config as cfg
from shared.text_core import (
    build_shared_vectorizer,
    canonicalize_text_list,
    prepare_keyword_pairs,
)

from ..utils import logger
from ..utils.io_helpers import load_qa_pairs
from ..utils.ngram_utils import build_candidates_for_text

HUGGINGFACE_CACHE_DIR = os.getenv("HUGGINGFACE_CACHE_DIR")
STOPWORD_LANGS = ("en", "zh", "ko")


def _dedupe_keyword_dicts(pairs: List[tuple[str, float]]) -> List[Dict[str, float]]:
    return [{"keyword": kw, "similarity": float(score)} for kw, score in pairs if kw]


def _candidate_embeddings(
    model: SentenceTransformer,
    qa_vec: np.ndarray,
    candidate_texts: List[str],
) -> tuple[np.ndarray, List[str]]:
    deduped = canonicalize_text_list(candidate_texts, stopword_langs=STOPWORD_LANGS)
    if not deduped:
        return np.zeros((0, qa_vec.shape[0]), dtype=np.float32), []

    cand_vecs = model.encode(
        deduped,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return cand_vecs, deduped


def extract_keywords_for_conv(
    qa_pairs_path: Path,
    conv_id_target: str,
    model_name: str,
    ngram_max: int,
    max_candidates: int,
    top_n: int,
    output_path: Path,
    keyword_method: str,
    keybert_use_tokenizer: bool,
) -> None:
    """English documentation."""
    text_core_profile = getattr(cfg, "TEXT_CORE_PROFILE_ADDNODE", "balanced")
    qa_pairs_file = Path(qa_pairs_path)
    qa_pairs = load_qa_pairs(qa_pairs_file)

    if not qa_pairs:
        logger.warning(f"No QA pairs in {qa_pairs_file}")
        return

    logger.info(f"Conversation {conv_id_target}: {len(qa_pairs)} QA pairs")

    logger.info(f"Loading model: {model_name}")
    if HUGGINGFACE_CACHE_DIR:
        logger.info(f"Using cache directory: {HUGGINGFACE_CACHE_DIR}")
        model = SentenceTransformer(model_name, cache_folder=HUGGINGFACE_CACHE_DIR)
    else:
        logger.info("Using default cache directory (~/.cache/huggingface)")
        model = SentenceTransformer(model_name)

    keybert_model = None
    keybert_vectorizer = None
    if keyword_method == "keybert":
        keybert_model = KeyBERT(model=model)
        if keybert_use_tokenizer:
            keybert_vectorizer = build_shared_vectorizer(
                ngram_max,
                stopword_langs=STOPWORD_LANGS,
            )

    results: List[Dict[str, Any]] = []
    emb_records: List[Dict[str, Any]] = []

    for pair in qa_pairs:
        qa_id = pair.get("qa_id")
        question = pair.get("question", "")
        answer = pair.get("answer", "")
        qa_text = f"Q: {question} A: {answer}"

        qa_vec = model.encode(
            [qa_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        top_keywords: List[Dict[str, float]] = []
        candidates_for_store: List[str] = []
        cand_vecs = np.zeros((0, qa_vec.shape[0]), dtype=np.float32)

        if keyword_method == "keybert":
            keywords_raw = keybert_model.extract_keywords(
                qa_text,
                top_n=max(top_n * 2, top_n),
                vectorizer=keybert_vectorizer,
            )
            normalized_pairs = prepare_keyword_pairs(
                keywords_raw,
                source_text=qa_text,
                top_n=top_n,
                min_keywords=min(3, top_n),
                dedup_threshold=0.8,
                max_formula_keywords=2,
                profile=text_core_profile,
            )

            if normalized_pairs:
                top_keywords = _dedupe_keyword_dicts(normalized_pairs)
                candidate_texts = [kw for kw, _ in normalized_pairs]
                cand_vecs, candidates_for_store = _candidate_embeddings(
                    model,
                    qa_vec,
                    candidate_texts,
                )
        else:
            raw_candidates = build_candidates_for_text(
                qa_text,
                ngram_max=ngram_max,
                max_candidates=max_candidates,
            )
            candidates = canonicalize_text_list(
                raw_candidates,
                stopword_langs=STOPWORD_LANGS,
            )

            if candidates:
                cand_vecs = model.encode(
                    candidates,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                sims = cosine_similarity(qa_vec.reshape(1, -1), cand_vecs).flatten()
                order = np.argsort(-sims)
                top_indices = order[:top_n]
                ngram_pairs = [
                    (candidates[i], float(sims[i])) for i in top_indices
                ]
                normalized_pairs = prepare_keyword_pairs(
                    ngram_pairs,
                    source_text=qa_text,
                    top_n=top_n,
                    min_keywords=min(3, top_n),
                    dedup_threshold=0.8,
                    max_formula_keywords=2,
                    profile=text_core_profile,
                )
                top_keywords = _dedupe_keyword_dicts(normalized_pairs)
                candidates_for_store = candidates

        results.append(
            {
                "qa_id": qa_id,
                "conversation_id": conv_id_target,
                "qa_index": pair.get("qa_index"),
                "keywords": top_keywords,
            }
        )

        qa_length = len(question) + len(answer)

        emb_records.append(
            {
                "qa_id": qa_id,
                "conversation_id": conv_id_target,
                "qa_index": pair.get("qa_index"),
                "qa_embedding": qa_vec.tolist(),
                "qa_length": qa_length,
                "candidates": [
                    {
                        "text": candidates_for_store[i],
                        "embedding": cand_vecs[i].tolist(),
                        "similarity": (
                            float(
                                cosine_similarity(
                                    qa_vec.reshape(1, -1),
                                    cand_vecs[i].reshape(1, -1),
                                )[0][0]
                            )
                            if cand_vecs.size
                            else 0.0
                        ),
                    }
                    for i in range(len(candidates_for_store))
                ],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Keywords saved: {output_path}")

    emb_output_path = (
        output_path.parent
        / "embeddings"
        / f"qa_keyword_embeddings_{conv_id_target}.pkl"
    )
    emb_output_path.parent.mkdir(parents=True, exist_ok=True)
    with emb_output_path.open("wb") as f:
        pickle.dump(emb_records, f)
    logger.info(f"Embeddings saved: {emb_output_path}")
