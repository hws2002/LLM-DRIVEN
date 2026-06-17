"""CLI entry point for extracting embeddings and keywords from chat history."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import time

import numpy as np
import yaml
from pydantic import ValidationError
from sklearn.metrics.pairwise import cosine_similarity
try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shared.config as shared_cfg
from shared.text_core import (
    build_shared_vectorizer,
    preprocess_text_for_pipeline,
    prepare_keyword_pairs,
)

from util.io_schemas import (
    InputData,
    Keyword,
    Section,
    SourceNode,
)

np.random.seed(42)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - handled by fallback
    SentenceTransformer = None  # type: ignore

try:
    from keybert import KeyBERT
except ImportError as exc:  # pragma: no cover - library required
    raise SystemExit(
        "KeyBERT is required. Install dependencies via requirements.txt"
    ) from exc

from stopwordsiso import stopwords as stopwords_iso

MAX_CHARS_PER_CHUNK = 512
SENTENCE_END_RE = re.compile(r"(?<=[\.\!\?。？！])\s+")
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
FILLER_TOKENS = {
    "text",
    "text",
    "lol",
    "haha",
    "uh",
    "um",
    "mmm",
    "text",
    "text",
    "text",
}


@dataclass
class KeywordConfig:
    top_n: int
    max_ngram: int
    dedup_thresh: float


@dataclass
class PreprocessConfig:
    lower: bool
    strip_urls: bool
    strip_code: bool
    strip_punct: bool
    stopwords_langs: List[str]


@dataclass
class Config:
    embedding_model: str
    keyword: KeywordConfig
    preprocess: PreprocessConfig


@dataclass
class NodeFeatures:
    """Lightweight node feature summary for LLM-based clustering."""

    id: int
    orig_id: str
    keywords: List[Keyword]
    num_sections: int
    # Fields with defaults must come after fields without defaults
    title: Optional[str] = None  # conversation title or markdown note title
    source_type: str = "chat"  # "chat" | "markdown" | "notion"
    # Add create_time and update_time from original conversation
    create_time: Optional[int] = None
    update_time: Optional[int] = None


@dataclass
class FeatureData:
    """Feature dataset containing inputs needed for LLM clustering and edge generation."""

    conversations: List[NodeFeatures]
    embeddings: np.ndarray
    metadata: Dict[str, Any]


class DummySentenceTransformer:
    """Deterministic embedding fallback when the real model cannot be loaded."""

    def __init__(self, model_name: str, dimension: int = 384) -> None:
        self.model_name = model_name
        self._dimension = dimension

    def encode(
        self,
        sentences: Sequence[str],
        *,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for sentence in sentences:
            if not sentence:
                vec = np.zeros(self._dimension, dtype=np.float32)
            else:
                digest = hashlib.sha1(sentence.encode("utf-8")).hexdigest()
                seed = int(digest[:16], 16)
                rng = np.random.default_rng(seed)
                vec = rng.standard_normal(self._dimension).astype(np.float32)
            if normalize_embeddings:
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    vec = vec / norm
            vectors.append(vec)
        stacked = np.vstack(vectors)
        return stacked if convert_to_numpy else stacked.tolist()

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension


# Type alias for embedding models
if SentenceTransformer is not None:
    EmbeddingModel = Union[SentenceTransformer, DummySentenceTransformer]
else:
    EmbeddingModel = DummySentenceTransformer  # type: ignore


def load_embedding_model(model_name: str) -> EmbeddingModel:
    """Load a sentence transformer model with a deterministic fallback."""
    if SentenceTransformer is None:
        warnings.warn(
            "sentence-transformers not available; using deterministic fallback embeddings."
        )
        return DummySentenceTransformer(model_name)
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:  # pragma: no cover - network/cache failures
        warnings.warn(
            f"Failed to load {model_name}: {exc}. Using deterministic fallback embeddings."
        )
        return DummySentenceTransformer(model_name)


def load_config(path: Path) -> Config:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    keyword_cfg = KeywordConfig(**data["keyword"])
    preprocess_cfg = PreprocessConfig(**data["preprocess"])
    return Config(
        embedding_model=data["embedding_model"],
        keyword=keyword_cfg,
        preprocess=preprocess_cfg,
    )


def _is_message_like(item: Dict[str, Any]) -> bool:
    required = {"id", "role", "content"}
    return required.issubset(item.keys())


def _group_chatgpt_export(payload: Iterable[Dict[str, Any]]) -> List[SourceNode]:
    """Group ChatGPT export data into SourceNode objects."""
    conversations: List[SourceNode] = []

    def _normalize_epoch(value: Any) -> Optional[int]:
        """Cast floats/strings coming from exports into ints for Pydantic."""
        if value is None:
            return None
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            try:
                return int(float(value))
            except ValueError:
                return None
        return None

    for conv_idx, conversation in enumerate(payload):
        mapping = conversation.get("mapping")
        if not isinstance(mapping, dict):
            continue

        # English comment.
        explicit_conv_id = conversation.get("conversation_id")
        title = conversation.get("title", f"Conversation {conv_idx}")
        create_time = _normalize_epoch(conversation.get("create_time"))
        update_time = _normalize_epoch(conversation.get("update_time"))
        sections: List[Section] = []

        def iter_nodes() -> Iterable[Dict[str, Any]]:
            visited: set[str] = set()
            roots = [node for node in mapping.values() if node.get("parent") is None]

            def sort_key(node: Dict[str, Any]) -> str:
                message = node.get("message")
                if isinstance(message, dict):
                    return str(message.get("id", ""))
                return ""

            roots.sort(key=sort_key)
            queue: deque[Dict[str, Any]] = deque(roots)

            while queue:
                node = queue.popleft()
                node_id = node.get("id")
                if node_id in visited:
                    continue
                visited.add(node_id)
                yield node

                for child_id in node.get("children") or []:
                    child = mapping.get(child_id)
                    if child:
                        queue.append(child)

            for node in mapping.values():
                node_id = node.get("id")
                if node_id not in visited:
                    yield node

        for node in iter_nodes():
            message = node.get("message") or {}
            message_id = message.get("id")
            author = message.get("author") or {}
            role = author.get("role")
            content_obj = message.get("content") or {}

            content: str = ""
            if isinstance(content_obj, dict):
                parts = content_obj.get("parts")
                if isinstance(parts, list):
                    content = "\n".join(str(part) for part in parts)
                elif isinstance(content_obj.get("text"), str):
                    content = content_obj["text"]
            elif isinstance(content_obj, str):
                content = content_obj

            if not message_id or role is None or content is None:
                continue

            sections.append(
                Section(
                    id=str(message_id),
                    role=str(role),
                    content=str(content),
                )
            )

        if sections:
            conv_id: str
            if explicit_conv_id is not None:
                conv_id = str(explicit_conv_id)
            else:
                conv_id = f"conv_{conv_idx}"

            conversations.append(
                SourceNode(
                    id=conv_id,
                    title=title,
                    sections=sections,
                    source_type="chat",
                    create_time=create_time,
                    update_time=update_time,
                )
            )

    return conversations


def _group_sections_by_source(sections: List[Section]) -> List[SourceNode]:
    """Group simple section list by source ID prefix."""
    from collections import defaultdict

    grouped: Dict[str, List[Section]] = defaultdict(list)

    for sec in sections:
        source_id = sec.id.rsplit("_", 1)[0] if "_" in sec.id else "src_0"
        grouped[source_id].append(sec)

    source_nodes: List[SourceNode] = []
    for source_id, source_sections in grouped.items():
        source_nodes.append(
            SourceNode(
                id=source_id,
                title=f"Source {source_id}",
                sections=source_sections,
                source_type="chat",
            )
        )

    return source_nodes


def load_messages(path: Path) -> InputData:
    """
    Load sections and group them into source nodes.

    Supports ChatGPT export payloads, flat section lists, markdown files/directories,
    and raw document formats (PDF, PPTX, DOCX).
    """
    # English comment.
    if path.is_dir():
        all_data = []
        for f in sorted(path.rglob("*")):
            if not f.is_file():
                continue
            try:
                data = load_messages(f)
                if data.source_nodes:
                    all_data.append(data)
            except Exception:
                pass  # English comment.
        return merge_inputs(*all_data) if all_data else InputData(source_nodes=[])

    # PDF
    if path.suffix.lower() == ".pdf":
        from util.raw_file_loader import load_pdf
        return load_pdf(path)

    # DOCX / DOC
    if path.suffix.lower() in (".docx", ".doc"):
        from util.raw_file_loader import load_docx
        return load_docx(path)

    # PPTX / PPT
    if path.suffix.lower() in (".pptx", ".ppt"):
        from util.raw_file_loader import load_pptx
        return load_pptx(path)

    # Markdown
    if path.suffix.lower() == ".md":
        from util.markdown_loader import load_markdown_dir
        return load_markdown_dir(path)

    # NEW: JSON list of .md paths
    if path.suffix.lower() == ".json":
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(candidate, list) and all(
                isinstance(p, str) and p.endswith(".md") for p in candidate
            ):
                from util.markdown_loader import load_markdown_dir

                return load_markdown_dir(path)
        except Exception:
            pass  # fall through to existing logic

    payload = json.loads(path.read_text(encoding="utf-8"))

    # NEW: Check for merged JSON format with "source_nodes" key
    if isinstance(payload, dict) and "source_nodes" in payload:
        source_nodes_data = payload["source_nodes"]
        source_nodes = []

        for node_dict in source_nodes_data:
            # Convert sections from dict to Section objects
            sections = [
                Section(
                    id=s.get("id", f"sec_{i}"),
                    content=s.get("content", ""),
                    role=s.get("role"),
                    section_title=s.get("section_title"),
                )
                for i, s in enumerate(node_dict.get("sections", []))
            ]

            # Create SourceNode
            source_node = SourceNode(
                id=node_dict.get("id", ""),
                title=node_dict.get("title"),
                sections=sections,
                source_type=node_dict.get("source_type", "chat"),
                create_time=node_dict.get("create_time"),
                update_time=node_dict.get("update_time"),
            )
            source_nodes.append(source_node)

        return InputData.from_source_nodes(source_nodes)

    if not isinstance(payload, list):
        if isinstance(payload, dict):
            source_nodes = _group_chatgpt_export([payload])
            if not source_nodes:
                raise ValueError(
                    "Input JSON must be a list of section objects or a ChatGPT export mapping."
                )
            return InputData.from_source_nodes(source_nodes)
        raise ValueError("Input JSON must be a list of section objects.")

    if not payload:
        return InputData.from_source_nodes([])

    if payload and all(
        isinstance(item, dict) and _is_message_like(item) for item in payload
    ):
        sections = [Section(**item) for item in payload]
        source_nodes = _group_sections_by_source(sections)
        return InputData.from_source_nodes(source_nodes)

    try:
        source_nodes = _group_chatgpt_export(payload)
        if not source_nodes:
            raise ValueError("Unable to interpret input JSON as chat sections.")
        return InputData.from_source_nodes(source_nodes)
    except Exception as exc:
        raise ValueError(
            "Input must be either a list of sections or ChatGPT export format."
        ) from exc


def preprocess_text(text: str, cfg: PreprocessConfig) -> str:
    text_core_profile = getattr(shared_cfg, "TEXT_CORE_PROFILE_MACRO", "balanced")
    return preprocess_text_for_pipeline(
        text,
        lower=cfg.lower,
        strip_code=cfg.strip_code,
        strip_urls=cfg.strip_urls,
        strip_html=True,
        strip_citations=True,
        strip_punct=cfg.strip_punct,
        strip_inline_code=True,
        strip_emoji=True,
        segment_cjk=False,
        profile=text_core_profile,
    )


def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    if len(text) <= max_chars:
        return [text] if text else []
    parts = SENTENCE_END_RE.split(text)
    parts = [part.strip() for part in parts if part.strip()]
    if not parts:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    chunks: List[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current} {part}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current.strip())
        if len(part) <= max_chars:
            current = part
        else:
            for start in range(0, len(part), max_chars):
                segment = part[start : start + max_chars].strip()
                if segment:
                    chunks.append(segment)
            current = ""
    if current:
        chunks.append(current.strip())
    return chunks


def get_embedding_dimension(model: EmbeddingModel) -> int:
    if hasattr(model, "get_sentence_embedding_dimension"):
        dim = model.get_sentence_embedding_dimension()
        if isinstance(dim, int):
            return dim
    sample = model.encode(["dimension_probe"], convert_to_numpy=True)
    if isinstance(sample, list):
        sample = np.array(sample)
    return int(sample.shape[1])


def mean_pool_embeddings(model: EmbeddingModel, texts: Sequence[str]) -> np.ndarray:
    embeddings = model.encode(list(texts), convert_to_numpy=True)
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    return embeddings.mean(axis=0)


def generate_embeddings(
    cleaned_texts: Sequence[str],
    model: EmbeddingModel,
    *,
    chunk_cache: Optional[Sequence[Sequence[str]]] = None,
) -> np.ndarray:
    """Generate embeddings for a sequence of texts using a transformer model.

    Texts are chunked to stay within model limits, and chunk embeddings are
    mean-pooled to create a single vector per text.

    Args:
        cleaned_texts: Sequence of preprocessed text strings to embed.
        model: Embedding model (SentenceTransformer or fallback).
        chunk_cache: Optional pre-computed chunked texts. If not provided,
            texts will be chunked automatically using MAX_CHARS_PER_CHUNK.

    Returns:
        2D numpy array of shape (len(cleaned_texts), embedding_dimension)
        containing the mean-pooled embeddings for each input text.
    """
    dimension = get_embedding_dimension(model)
    vectors: List[np.ndarray] = []
    total = len(cleaned_texts)
    iterable = range(total)
    if _tqdm is not None:
        iterable = _tqdm(iterable, total=total, desc="Embedding", file=sys.stdout, dynamic_ncols=True)
    for idx in iterable:
        text = cleaned_texts[idx]
        cached_chunks: Optional[Sequence[str]] = None
        if chunk_cache is not None:
            try:
                cached_chunks = chunk_cache[idx]
            except (IndexError, TypeError):
                cached_chunks = None
        if cached_chunks is None:
            chunks = chunk_text(text) if text else []
        else:
            chunks = list(cached_chunks)
        if chunks:
            vec = mean_pool_embeddings(model, chunks)
        else:
            vec = np.zeros(dimension, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        vectors.append(vec.astype(np.float32))
        pct = int((idx + 1) / total * 100)
        prev_pct = int(idx / total * 100)
        if pct != prev_pct:
            elapsed = getattr(iterable, "format_dict", {}).get("elapsed", 0)
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta_sec = int((total - idx - 1) / rate) if rate > 0 else 0
            print(f"PROGRESS:embedding:{pct}:{idx+1}:{total}:{eta_sec}", flush=True)
    return np.vstack(vectors) if vectors else np.zeros((0, dimension), dtype=np.float32)


def build_stopwords(langs: Iterable[str]) -> List[str]:
    stoplist = set()
    for lang in langs:
        try:
            stoplist.update(stopwords_iso(lang))
        except KeyError:
            warnings.warn(f"No stopwords found for language '{lang}'.")
    stoplist.update(FILLER_TOKENS)
    cleaned = {word.strip().lower() for word in stoplist if word and word.strip()}
    return sorted(cleaned)


def keyword_token_set(term: str) -> set:
    tokens = TOKEN_RE.findall(term.lower())
    if not tokens:
        tokens = list(term.lower())
    return set(tokens)


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def deduplicate_keywords(
    candidates: Sequence[Tuple[str, float]],
    *,
    limit: int,
    threshold: float,
) -> List[Keyword]:
    """Remove semantically redundant keywords using Jaccard similarity.

    Iterates through candidates and selects keywords that are sufficiently
    different from already-selected ones, up to the specified limit.

    Args:
        candidates: Sequence of (term, score) tuples sorted by relevance.
        limit: Maximum number of keywords to return.
        threshold: Jaccard similarity threshold; candidates above this
            compared to any selected keyword are considered duplicates.

    Returns:
        List of Keyword objects (term and score) with duplicates removed.
    """
    selected: List[Keyword] = []
    seen: List[set] = []
    for term, score in candidates:
        token_set = keyword_token_set(term)
        if any(jaccard(token_set, prev) >= threshold for prev in seen):
            continue
        seen.append(token_set)
        selected.append(Keyword(term=term, score=float(score)))
        if len(selected) >= limit:
            break
    return selected


def extract_keywords(
    texts: Sequence[str],
    config: KeywordConfig,
    stoplist: Optional[List[str]],
    keybert_model: KeyBERT,
    doc_embeddings: Optional[np.ndarray] = None,
) -> List[List[Keyword]]:
    """English documentation."""
    text_core_profile = getattr(shared_cfg, "TEXT_CORE_PROFILE_MACRO", "balanced")
    results: List[List[Keyword]] = []
    doc_embeddings_arr = doc_embeddings
    shared_vectorizer = build_shared_vectorizer(
        config.max_ngram,
        stopword_langs=("en", "zh", "ko"),
    )

    if doc_embeddings_arr is not None:
        if (
            not isinstance(doc_embeddings_arr, np.ndarray)
            or doc_embeddings_arr.ndim < 1
        ):
            warnings.warn(
                "Document embeddings must be a numpy array; ignoring provided embeddings."
            )
            doc_embeddings_arr = None
        elif len(doc_embeddings_arr) != len(texts):
            warnings.warn(
                "Document embeddings count does not match number of texts; ignoring provided embeddings."
            )
            doc_embeddings_arr = None

    total = len(texts)
    iterable = range(total)
    if _tqdm is not None:
        iterable = _tqdm(iterable, total=total, desc="Keywords", file=sys.stdout, dynamic_ncols=True)

    try:
        count = shared_vectorizer.fit(list(texts))
        words = count.get_feature_names_out()
        doc_term_matrix = count.transform(list(texts))
        word_embeddings = keybert_model.model.embed(words)
        if isinstance(word_embeddings, list):
            word_embeddings = np.array(word_embeddings)
    except Exception as exc:  # pragma: no cover - defensive fallback
        warnings.warn(
            "Batched keyword candidate embedding failed; falling back to "
            f"per-document KeyBERT extraction: {exc}"
        )
        word_embeddings = None
        doc_term_matrix = None
        words = None

    for idx in iterable:
        text = texts[idx]
        raw_keywords: List[Tuple[str, float]] = []

        if text and word_embeddings is not None and doc_term_matrix is not None and words is not None:
            try:
                candidate_indices = doc_term_matrix[idx].nonzero()[1]
                if len(candidate_indices):
                    candidates = [words[candidate_idx] for candidate_idx in candidate_indices]
                    candidate_embeddings = word_embeddings[candidate_indices]

                    doc_embedding: Optional[np.ndarray] = None
                    if doc_embeddings_arr is not None:
                        doc_embedding = doc_embeddings_arr[idx]
                        if isinstance(doc_embedding, np.ndarray):
                            if doc_embedding.ndim == 1:
                                doc_embedding = doc_embedding.reshape(1, -1)
                            elif doc_embedding.ndim != 2:
                                doc_embedding = None
                        else:
                            doc_embedding = None
                    if doc_embedding is None:
                        doc_embedding = keybert_model.model.embed([text])

                    distances = cosine_similarity(doc_embedding, candidate_embeddings)
                    raw_keywords = [
                        (candidates[candidate_idx], round(float(distances[0][candidate_idx]), 4))
                        for candidate_idx in distances.argsort()[0][-config.top_n * 4 :]
                    ][::-1]
            except Exception as exc:  # pragma: no cover - defensive fallback per item
                warnings.warn(f"Batched keyword extraction failed for item {idx}: {exc}")
                raw_keywords = []

        elif text:
            doc_embedding: Optional[np.ndarray] = None
            if doc_embeddings_arr is not None:
                doc_embedding = doc_embeddings_arr[idx]
                if isinstance(doc_embedding, np.ndarray):
                    if doc_embedding.ndim == 1:
                        doc_embedding = doc_embedding.reshape(1, -1)
                    elif doc_embedding.ndim != 2:
                        warnings.warn(
                            "Unexpected document embedding shape; falling back to KeyBERT's internal embeddings."
                        )
                        doc_embedding = None
                else:
                    doc_embedding = None
            try:
                raw_keywords = keybert_model.extract_keywords(
                    text,
                    top_n=config.top_n * 4,
                    vectorizer=shared_vectorizer,
                    doc_embeddings=doc_embedding,
                )
            except Exception as exc:  # pragma: no cover - defensive
                warnings.warn(f"Keyword extraction failed: {exc}")
                raw_keywords = []

        prepared_pairs = prepare_keyword_pairs(
            raw_keywords,
            source_text=text,
            top_n=config.top_n,
            min_keywords=min(3, config.top_n),
            dedup_threshold=config.dedup_thresh,
            max_formula_keywords=2,
            profile=text_core_profile,
        )
        deduped = [Keyword(term=term, score=float(score)) for term, score in prepared_pairs]
        results.append(deduped)
        pct = int((idx + 1) / total * 100)
        prev_pct = int(idx / total * 100)
        if pct != prev_pct:
            elapsed = getattr(iterable, "format_dict", {}).get("elapsed", 0)
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta_sec = int((total - idx - 1) / rate) if rate > 0 else 0
            print(f"PROGRESS:keywords:{pct}:{idx+1}:{total}:{eta_sec}", flush=True)
    return results


def extract_keywords_and_embeddings(
    input_data: InputData, config: Config
) -> Tuple[List[NodeFeatures], np.ndarray, Dict[str, Any]]:
    """
    Execute pipeline up to keyword extraction (before clustering).
    Returns conversation features, embeddings, and metadata with timing.
    """
    overall_start = time.perf_counter()

    text_core_profile = getattr(shared_cfg, "TEXT_CORE_PROFILE_MACRO", "balanced")
    model = load_embedding_model(config.embedding_model)
    stoplist = build_stopwords(config.preprocess.stopwords_langs)

    source_nodes = input_data.source_nodes
    if not source_nodes:
        # Handle flat section lists
        sections = input_data.sections
        source_nodes = [
            SourceNode(id=sec.id, sections=[sec], source_type="chat")
            for sec in sections
        ]

    merged_texts = [node.get_merged_content() for node in source_nodes]
    cleaned_texts = [preprocess_text(text, config.preprocess) for text in merged_texts]
    chunked_texts = [chunk_text(text) if text else [] for text in cleaned_texts]
    total_chunk_segments = sum(len(chunks) for chunks in chunked_texts)
    avg_chunks_per_conversation = (
        total_chunk_segments / len(chunked_texts) if chunked_texts else 0.0
    )

    # === STEP 1: Embedding Generation (preprocessing, chunking, mean pooling) ===
    print("STEP_START:embedding", flush=True)
    embedding_start = time.perf_counter()
    embeddings = generate_embeddings(cleaned_texts, model, chunk_cache=chunked_texts)
    embedding_time = time.perf_counter() - embedding_start
    num_embeddings = embeddings.shape[0]
    embedding_dim = embeddings.shape[1] if embeddings.ndim > 1 else 0
    print(f"  ⏱️  Embedding generation: {embedding_time:.1f}s")
    print(
        f"      └─ Generated {num_embeddings} embeddings "
        f"({embedding_dim}-dimensional vectors)"
    )
    print(
        f"      └─ Encoded {total_chunk_segments} chunk segments before mean pooling "
        f"(avg {avg_chunks_per_conversation:.2f} per conversation)"
    )
    print("STEP_DONE:embedding", flush=True)

    # === STEP 2: Keyword Extraction ===
    print("STEP_START:keywords", flush=True)
    keyword_start = time.perf_counter()
    keybert_model = KeyBERT(model=model)
    keywords = extract_keywords(
        cleaned_texts,
        config.keyword,
        stoplist,
        keybert_model,
        doc_embeddings=embeddings,
    )
    keyword_time = time.perf_counter() - keyword_start
    print(f"  ⏱️  Keyword extraction: {keyword_time:.1f}s")
    print("STEP_DONE:keywords", flush=True)

    # Build node-level feature summaries
    node_features: List[NodeFeatures] = []
    for idx, (node, keyword_list) in enumerate(zip(source_nodes, keywords)):
        num_sections = len(node.sections)

        node_features.append(
            NodeFeatures(
                id=idx,
                orig_id=node.id,
                title=node.title,
                keywords=keyword_list,
                num_sections=num_sections,
                source_type=node.source_type,
                create_time=node.create_time,
                update_time=node.update_time,
            )
        )

    # Calculate total time
    total_time = time.perf_counter() - overall_start

    # Prepare metadata
    metadata = {
        "total_conversations": len(node_features),
        "embedding_model": config.embedding_model,
        "keyword_params": {
            "top_n": config.keyword.top_n,
            "max_ngram": config.keyword.max_ngram,
            "dedup_thresh": config.keyword.dedup_thresh,
        },
        "preprocess_params": {
            "lower": config.preprocess.lower,
            "strip_urls": config.preprocess.strip_urls,
            "strip_code": config.preprocess.strip_code,
            "strip_punct": config.preprocess.strip_punct,
            "stopwords_langs": config.preprocess.stopwords_langs,
            "text_core_profile": text_core_profile,
        },
        "timing": {
            "embedding_seconds": round(embedding_time, 2),
            "keyword_seconds": round(keyword_time, 2),
            "total_seconds": round(total_time, 2),
        },
        "embedding_stats": {
            "conversations": len(node_features),
            "chunk_segments": total_chunk_segments,
            "avg_chunks_per_conversation": avg_chunks_per_conversation,
            "embedding_dimension": int(embedding_dim),
        },
    }

    return node_features, embeddings, metadata


def merge_inputs(*input_data_list: InputData) -> InputData:
    """
    Merge multiple InputData objects into one.

    Re-indexes all SourceNode IDs to avoid collisions.
    Preserves source_type on each SourceNode.

    Args:
        *input_data_list: Variable number of InputData objects to merge

    Returns:
        Combined InputData with all source nodes from all inputs
    """
    all_source_nodes = []

    for i, data in enumerate(input_data_list):
        nodes = data.source_nodes

        for node in nodes:
            # Create a copy and prefix ID with source index to guarantee uniqueness
            node_dict = node.dict() if hasattr(node, "dict") else node
            node_dict["id"] = f"src{i}_{node_dict['id']}"

            # Recreate SourceNode to ensure proper validation
            all_source_nodes.append(SourceNode(**node_dict))

    return InputData.from_source_nodes(all_source_nodes)


def extract_and_save_features(
    input_path: Path,
    output_path: Path,
    config_path: Path,
) -> FeatureData:
    """
    Build feature data for downstream processing.
    Extracts keywords and embeddings, saves to JSON.
    """
    config = load_config(config_path)
    input_data = load_messages(input_path)
    try:
        # Trigger validation
        _ = input_data.sections
        _ = input_data.source_nodes
    except ValidationError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Invalid input data: {exc}") from exc

    node_features, embeddings, metadata = extract_keywords_and_embeddings(
        input_data, config
    )

    # Create FeatureData
    feature_data = FeatureData(
        conversations=node_features, embeddings=embeddings, metadata=metadata
    )

    # Convert to dictionary for JSON serialization
    output_data = {
        "conversations": [
            {
                "id": node.id,
                "orig_id": node.orig_id,
                "title": node.title,
                "keywords": [
                    {"term": kw.term, "score": kw.score} for kw in node.keywords
                ],
                "num_sections": node.num_sections,
                "source_type": node.source_type,
                "create_time": node.create_time,
                "update_time": node.update_time,
            }
            for node in feature_data.conversations
        ],
        "embeddings": feature_data.embeddings.tolist(),
        "metadata": feature_data.metadata,
    }

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Print summary with timing breakdown
    timing = metadata.get("timing", {})
    total_time = timing.get("total_seconds", 0)
    embedding_time = timing.get("embedding_seconds", 0)
    keyword_time = timing.get("keyword_seconds", 0)

    print(f"\n⏱️  Feature extraction completed in {total_time:.1f}s")
    print(f"    └─ Embedding: {embedding_time:.1f}s, Keyword: {keyword_time:.1f}s")
    print(
        f"    └─ Embeddings count: {feature_data.embeddings.shape[0]} "
        f"({feature_data.embeddings.shape[1]} dims)"
    )
    embedding_stats = metadata.get("embedding_stats", {})
    chunk_segments = embedding_stats.get("chunk_segments")
    avg_chunks = embedding_stats.get("avg_chunks_per_conversation")
    if chunk_segments is not None:
        if avg_chunks is None:
            total_convs = (
                embedding_stats.get("conversations")
                or len(feature_data.conversations)
                or 0
            )
            avg_chunks = (chunk_segments / total_convs) if total_convs else 0.0
        print(
            f"    └─ Chunk segments encoded before pooling: {chunk_segments} "
            f"(avg {avg_chunks:.2f} per conversation)"
        )

    return feature_data


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract keywords and embeddings from chat conversations"
    )
    parser.add_argument(
        "--input",
        "--in",
        dest="input_path",
        required=True,
        help="Path to chat history JSON file.",
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="output_path",
        required=True,
        help="Path to write intermediate results JSON.",
    )
    parser.add_argument(
        "--cfg", dest="config_path", required=True, help="Path to YAML config."
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    config_path = Path(args.config_path)

    # Extract feature data (embeddings and keywords)
    feature_data = extract_and_save_features(input_path, output_path, config_path)
    print(
        f"Feature data extracted for {len(feature_data.conversations)} conversations. "
        f"Output saved to {output_path.resolve()}."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
