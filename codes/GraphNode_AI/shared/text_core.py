from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import re
import unicodedata

import jieba
import stopwordsiso as stopwordsiso
from sklearn.feature_extraction.text import CountVectorizer

try:
    import emoji
except Exception:  # pragma: no cover - optional at runtime
    emoji = None


@dataclass
class CanonicalKeyword:
    raw: str
    canonical: str
    score: float


# ---------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
TILDE_CODE_BLOCK_RE = re.compile(r"~~~.*?~~~", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
WS_RE = re.compile(r"\s+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
CITE_TURN_RE = re.compile(r"cite\s*turn\d+search\d+", re.IGNORECASE)
TURN_REF_RE = re.compile(r"turn\d+search\d+", re.IGNORECASE)
CONTAINS_CITE_RE = re.compile(r"(?:\\uCD9C\\uCC98\\s*)?cite", re.IGNORECASE)
BRACKET_SOURCE_RE = re.compile(r"\[\d+†source\]")
FULLWIDTH_BRACKET_RE = re.compile(r"【.*?】")
DAGGER_BRACKET_RE = re.compile(r"\[.*?†.*?\]")
BARE_CITE_RE = re.compile(r"\bcite\b", re.IGNORECASE)
MARKDOWN_HR_RE = re.compile(r"(?m)^\s*(?:-{3,}|_{3,}|\*{3,})\s*$")
PUNCT_STRIP_RE = re.compile(
    r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~，。！？；：、《》「」『』（）【】…·]"
)
CJK_SPAN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")

# English comment.
OUTER_PUNCT_RE = re.compile(r"^[^\w\u4e00-\u9fff\uAC00-\uD7A3]+|[^\w\u4e00-\u9fff\uAC00-\uD7A3]+$")
LOW_SIGNAL_CHARS_RE = re.compile(r"[\\{}|^=`]")
LATEX_ARTIFACT_RE = re.compile(
    r"(?:\\[a-zA-Z]+|int_|sum_|prod_|frac|mathbb|\\pi|\\theta|\\oint)",
    re.IGNORECASE,
)
MARKDOWN_ARTIFACT_RE = re.compile(r"^[*_`]+|[*_`]+$")
# English comment.
# English comment.
INLINE_MARKDOWN_GLUE_RE = re.compile(
    r"(?<=[A-Za-z0-9\uAC00-\uD7A3\u4e00-\u9fff])[*`]{2,}(?=[A-Za-z0-9\uAC00-\uD7A3\u4e00-\u9fff])"
)
HEXISH_TOKEN_RE = re.compile(r"^[0-9a-f]{2,8}$", re.IGNORECASE)
SHORT_MIXED_ID_TOKEN_RE = re.compile(
    r"^(?=.*[a-z])(?=.*\d)[a-z0-9]{2,6}$",
    re.IGNORECASE,
)
LONG_ASCII_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9]{14,}$", re.IGNORECASE)
CODE_CHAIN_TOKEN_RE = re.compile(
    r"^[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*){1,}$",
    re.IGNORECASE,
)
PATH_LIKE_TOKEN_RE = re.compile(
    r"(?:^|/)[\w\-.]+\.(?:py|js|ts|jsx|tsx|dart|java|kt|go|rs|cpp|c|h|html|css|json|ya?ml|md|txt|csv|tsv|pdf|ipynb|docx?|pptx?|xlsx?)"
    r"(?::\d+(?::\d+)?)?$",
    re.IGNORECASE,
)
CODE_META_TOKEN_RE = re.compile(r"(?:^|/)(?:src|lib|app|pages?|components?|widgets?)/", re.IGNORECASE)
ALNUM_OR_LETTER_RE = re.compile(r"[A-Za-z0-9\uAC00-\uD7A3\u4e00-\u9fff]")
CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
DIGIT_RE = re.compile(r"\d")
CAMEL_CASE_RE = re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]+){1,}\b")
SNAKE_CASE_RE = re.compile(r"\b[a-z]+_[a-z0-9_]{2,}\b")
CODE_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
INLINE_MATH_RE = re.compile(r"\$[^$\n]{2,}\$")
DISPLAY_MATH_RE = re.compile(r"\$\$[\s\S]{2,}?\$\$")
LATEX_PAREN_RE = re.compile(r"\\\([^)]+\\\)")
LATEX_BRACKET_RE = re.compile(r"\\\[[^\]]+\\\]")
LATEX_COMMAND_RE = re.compile(r"\\([a-zA-Z]+)")
Z_SUBSCRIPT_RE = re.compile(r"\bz[_\s]*\{?(\d+)\}?\b", re.IGNORECASE)
Z_POWER_RE = re.compile(r"\bz\^\{?(\d+)\}?\b", re.IGNORECASE)
MATH_SIGNAL_RE = re.compile(
    r"(?:\\(?:int|sum|prod|frac|pi|theta|oint|tan|sin|cos|log|exp)|\b(?:integral|residue|theorem|contour|laurent|taylor)\b)",
    re.IGNORECASE,
)
META_ROLE_TOKENS = {"assistant", "user", "system"}
CODE_NOISE_TOKENS = {
    "from",
    "import",
    "class",
    "def",
    "return",
    "widget",
    "setstate",
    "buildcontext",
    "textfield",
    "requestfocus",
}
CODE_IDENTIFIER_BLOCKLIST = {
    "final",
    "void",
    "const",
    "var",
    "let",
    "static",
    "public",
    "private",
    "protected",
    "async",
    "await",
    "true",
    "false",
    "null",
    "none",
    "message",
    "node",
    "container",
}
METADATA_FIELD_TOKENS = {
    "asset_pointer",
    "asset_pointer_link",
    "image_asset_pointer",
    "watermarked_asset_pointer",
    "size_bytes",
    "content_type",
    "request_id",
    "message_type",
    "model_slug",
    "parent_id",
    "conversation_id",
    "conversation_template_id",
    "qa_id",
    "qa_index",
    "cluster_id",
    "cluster_ids",
    "user_id",
    "timestamp",
    "timestamp_",
    "create_time",
    "update_time",
    "created_at",
    "updated_at",
    "safe_urls",
    "blocked_urls",
    "metadata",
}
METADATA_FIELD_SUFFIXES = (
    "_id",
    "_ids",
    "_type",
    "_path",
    "_paths",
    "_time",
    "_times",
    "_url",
    "_urls",
    "_slug",
    "_bytes",
    "_pointer",
)
LOW_VALUE_PHRASE_TOKENS = {
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
}
PHRASE_CONNECTOR_TOKENS = {
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "facing",
    "issue",
    "based",
}
EN_PHRASE_GLUE_TOKENS = {
    "a",
    "an",
    "and",
    "or",
    "the",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "from",
    "by",
    "is",
    "are",
    "was",
    "were",
}
META_DISCOURSE_TOKENS = {
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
}
META_DISCOURSE_STEM_RE = re.compile(
    r"^(?:text|text|text|text|text|text|text|text)(?:[\uAC00-\uD7A3]{0,4})$"
)
KOREAN_PARTICLE_SUFFIXES = (
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
    "text",
)
GENERIC_CJK_META_TOKENS = {
    "段落",
    "文章",
    "句子",
    "写",
    "请",
    "总结",
    "概括",
    "大概",
    "上面",
    "结合",
    "三个",
}
CJK_TOPIC_HINT_TOKENS = {
    "香港",
    "教育",
    "暴乱",
    "反修例",
    "社会",
    "政治",
    "媒体",
    "矫正",
    "牙齿",
    "牙科",
    "保持器",
    "环绕",
    "正畸",
    "医院",
}
MATH_HINT_TOKENS = {
    "residue theorem",
    "contour integral",
    "integral",
    "series expansion",
    "tan(pi z)",
}
STRICT_MATH_CONTEXT_HINTS = {
    "integral",
    "residue",
    "contour",
    "laurent",
    "taylor",
    "maclaurin",
    "singular",
    "pole",
    "cauchy",
    "holomorphic",
    "complex",
    "text",
    "text",
}
ML_CONTEXT_HINTS = {
    "gan",
    "generator",
    "discriminator",
    "transformer",
    "encoder",
    "decoder",
    "attention",
    "embedding",
    "gradient",
    "optimizer",
    "loss",
    "minimax",
    "latent",
    "diffusion",
    "llm",
    "text",
    "text",
}
LATEX_COMMAND_CANONICAL = {
    "int": "integral",
    "oint": "contour integral",
    "sum": "series expansion",
    "prod": "series expansion",
    "frac": "fraction",
    "tan": "tan(pi z)",
}
LATEX_RAW_COMMAND_TOKENS = {"int", "oint", "sum", "prod", "frac"}

MAX_KEYWORD_TOKENS = 8
MAX_KEYWORD_CHARS = 96

# English comment.
TECH_TOKEN_RE = re.compile(
    r"^[a-z0-9]+([._/\-+][a-z0-9]+)*([<>]=?|->|::[a-z0-9]+)*$",
    re.IGNORECASE,
)

VALID_TEXT_CORE_PROFILES = {"balanced"}
DEFAULT_TEXT_CORE_PROFILE = "balanced"


def _resolve_text_core_profile(profile: Optional[str]) -> str:
    candidate = (profile or DEFAULT_TEXT_CORE_PROFILE or "balanced").strip().lower()
    return candidate if candidate in VALID_TEXT_CORE_PROFILES else "balanced"


# ---------------------------------------------------------------------
# Rule loading
# ---------------------------------------------------------------------
RULES_DIR = Path(__file__).resolve().parent / "text_rules"


def _require_rule_file(filename: str) -> Path:
    path = RULES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"[shared.text_core] Required rule file not found: {path}"
        )
    return path


def _load_txt_set(filename: str) -> set[str]:
    path = _require_rule_file(filename)
    values: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values.add(line.lower())

    return values


def _load_mapping_file(filename: str) -> Dict[str, str]:
    path = _require_rule_file(filename)
    mapping: Dict[str, str] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                raise ValueError(
                    f"[shared.text_core] Invalid mapping format in {path} "
                    f"(line {line_no}): '{line}'. Expected 'source=target'."
                )

            src, dst = line.split("=", 1)
            src = src.strip().lower()
            dst = dst.strip().lower()

            if not src or not dst:
                raise ValueError(
                    f"[shared.text_core] Empty source/target in {path} "
                    f"(line {line_no}): '{line}'"
                )

            mapping[src] = dst

    return mapping


def _safe_stopwords(lang: str) -> set[str]:
    try:
        return set(stopwordsiso.stopwords(lang))
    except Exception:
        return set()


# ---------------------------------------------------------------------
# Loaded rules (single source of truth = txt files)
# ---------------------------------------------------------------------
FOREIGN_KO_SUFFIXES = sorted(
    _load_txt_set("foreign_ko_suffixes.txt"),
    key=len,
    reverse=True,
)

KOREAN_ENDINGS_TO_STRIP = sorted(
    _load_txt_set("korean_endings.txt"),
    key=len,
    reverse=True,
)

BASE_STOPWORDS = _load_txt_set("base_stopwords.txt")
FORCE_KEEP_TOKENS = _load_txt_set("force_keep_tokens.txt")
KOREAN_STRIP_ONLY = _load_mapping_file("korean_strip_only.txt")

# English comment.
# English comment.
STOPWORDS_KO = _load_txt_set("stopwords_ko.txt")
STOPWORDS_ZH = _load_txt_set("stopwords_zh.txt")
EN_STOPWORDS = _safe_stopwords("en")

PROJECT_STOPWORDS = set(BASE_STOPWORDS)
ALL_STOPWORDS = BASE_STOPWORDS | STOPWORDS_KO | STOPWORDS_ZH | EN_STOPWORDS


# ---------------------------------------------------------------------
# Basic normalization
# ---------------------------------------------------------------------
def _strip_citation_artifacts(text: str) -> str:
    text = CITE_TURN_RE.sub(" ", text)
    text = TURN_REF_RE.sub(" ", text)
    text = CONTAINS_CITE_RE.sub(" ", text)
    text = BRACKET_SOURCE_RE.sub(" ", text)
    text = FULLWIDTH_BRACKET_RE.sub(" ", text)
    text = DAGGER_BRACKET_RE.sub(" ", text)
    text = BARE_CITE_RE.sub(" ", text)
    return text


def clean_text_common(
    text: str,
    *,
    lower: bool = True,
    strip_code: bool = True,
    strip_urls: bool = True,
    strip_html: bool = False,
    strip_citations: bool = False,
) -> str:
    """English documentation."""
    if not text:
        return ""

    cleaned = unicodedata.normalize("NFKC", text)

    if strip_html:
        cleaned = HTML_TAG_RE.sub(" ", cleaned)
    if strip_code:
        cleaned = CODE_BLOCK_RE.sub(" ", cleaned)
    if strip_urls:
        cleaned = URL_RE.sub(" ", cleaned)
    if strip_citations:
        cleaned = _strip_citation_artifacts(cleaned)
    if lower:
        cleaned = cleaned.lower()

    return WS_RE.sub(" ", cleaned).strip()


def _segment_cjk_span(match: re.Match[str]) -> str:
    text = match.group(0)
    parts = [p.strip() for p in jieba.lcut(text) if p and p.strip()]
    return " ".join(parts)


def preprocess_text_for_pipeline(
    text: str,
    *,
    lower: bool = True,
    strip_code: bool = True,
    strip_urls: bool = True,
    strip_html: bool = True,
    strip_citations: bool = True,
    strip_punct: bool = False,
    strip_inline_code: bool = True,
    strip_emoji: bool = True,
    segment_cjk: bool = False,
    profile: Optional[str] = None,
) -> str:
    """English documentation."""
    cleaned = clean_text_common(
        text,
        lower=lower,
        strip_code=strip_code,
        strip_urls=strip_urls,
        strip_html=strip_html,
        strip_citations=strip_citations,
    )

    if strip_code:
        cleaned = TILDE_CODE_BLOCK_RE.sub(" ", cleaned)
    if strip_inline_code:
        cleaned = INLINE_CODE_RE.sub(" ", cleaned)

    cleaned = MARKDOWN_HR_RE.sub(" ", cleaned)

    if strip_emoji and emoji is not None:
        cleaned = emoji.replace_emoji(cleaned, replace="")

    if segment_cjk:
        cleaned = CJK_SPAN_RE.sub(_segment_cjk_span, cleaned)

    if strip_punct:
        cleaned = PUNCT_STRIP_RE.sub(" ", cleaned)

    return WS_RE.sub(" ", cleaned).strip()


def normalize_text_basic(
    text: str,
    *,
    lower: bool = True,
    strip_code: bool = True,
    strip_urls: bool = True,
) -> str:
    return preprocess_text_for_pipeline(
        text,
        lower=lower,
        strip_code=strip_code,
        strip_urls=strip_urls,
        strip_html=False,
        strip_citations=False,
        strip_punct=False,
        strip_inline_code=False,
        strip_emoji=False,
        segment_cjk=False,
    )


def strip_outer_punct(token: str) -> str:
    return OUTER_PUNCT_RE.sub("", token or "").strip()


def contains_hangul(text: str) -> bool:
    return any("\\uAC00" <= ch <= "\\uD7A3" for ch in text)


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def contains_latin(text: str) -> bool:
    return any(("a" <= ch <= "z") or ("A" <= ch <= "Z") for ch in text)


def is_pure_cjk(text: str) -> bool:
    return text != "" and all("\u4e00" <= ch <= "\u9fff" for ch in text)


def is_korean_only(text: str) -> bool:
    return (
        text != ""
        and contains_hangul(text)
        and not contains_latin(text)
        and not contains_cjk(text)
    )


def is_metadata_token(token: str) -> bool:
    token = (token or "").strip().lower()
    if not token:
        return False

    if token in METADATA_FIELD_TOKENS:
        return True

    if token.startswith(("file-service://", "sediment://", "file_")):
        return True

    if any(token.endswith(suffix) for suffix in METADATA_FIELD_SUFFIXES):
        return True

    if token.count("_") >= 2:
        parts = set(token.split("_"))
        if parts & {"asset", "pointer", "request", "model", "timestamp", "metadata"}:
            return True

    return False


def _contains_any_hint(text: str, hints: set[str]) -> bool:
    if not text:
        return False
    low = text.lower()
    for hint in hints:
        if hint in low:
            return True
    return False


def _has_strict_math_context(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    if "\\int" in text or "\\oint" in text:
        return True
    return _contains_any_hint(low, STRICT_MATH_CONTEXT_HINTS)


def _has_ml_context(text: str) -> bool:
    return _contains_any_hint(text.lower() if text else "", ML_CONTEXT_HINTS)


# ---------------------------------------------------------------------
# Token normalization
# ---------------------------------------------------------------------
def strip_korean_suffix_from_foreign_token(token: str) -> str:
    token = strip_outer_punct(token)
    if not token:
        return ""

    # English comment.
    if contains_hangul(token) and (
        contains_latin(token)
        or contains_cjk(token)
        or any(ch.isdigit() for ch in token)
        or any(sym in token for sym in "._/-+><:")
    ):
        for suffix in FOREIGN_KO_SUFFIXES:
            if token.endswith(suffix) and len(token) > len(suffix):
                stem = strip_outer_punct(token[: -len(suffix)])
                if stem:
                    return stem

    return token


def normalize_korean_token(token: str) -> str:
    token = strip_outer_punct(token)
    if not token:
        return ""

    if token in KOREAN_STRIP_ONLY:
        return KOREAN_STRIP_ONLY[token]

    if token in ALL_STOPWORDS:
        return ""

    if len(token) <= 1:
        return ""

    for ending in KOREAN_ENDINGS_TO_STRIP:
        if token.endswith(ending) and len(token) > len(ending) + 1:
            stem = token[: -len(ending)]
            if stem in ALL_STOPWORDS:
                return ""
            return stem

    # English comment.
    for suffix in KOREAN_PARTICLE_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix) + 1:
            stem = token[: -len(suffix)]
            stem = strip_outer_punct(stem)
            if not stem:
                return ""
            if stem in ALL_STOPWORDS:
                return ""
            return stem

    # English comment.
    for suffix in ("text", "text", "text", "text", "text", "text", "text", "text"):
        if token.endswith(suffix) and len(token) > 2:
            stem = token[: -len(suffix)]
            if stem and stem not in ALL_STOPWORDS:
                return stem

    return token


def canonicalize_token(token: str) -> str:
    token = unicodedata.normalize("NFKC", token or "")
    token = token.strip().lower()
    token = INLINE_MARKDOWN_GLUE_RE.sub("", token)
    token = MARKDOWN_ARTIFACT_RE.sub("", token)
    token = strip_outer_punct(token)

    if not token:
        return ""

    token = strip_korean_suffix_from_foreign_token(token)
    token = strip_outer_punct(token)

    if not token:
        return ""

    if is_metadata_token(token):
        return ""

    # English comment.
    if token in FORCE_KEEP_TOKENS:
        return token

    # English comment.
    if is_low_signal_token(token):
        return ""

    # English comment.
    if TECH_TOKEN_RE.match(token):
        if token in ALL_STOPWORDS:
            return ""
        return token

    # English comment.
    if is_pure_cjk(token):
        if token in ALL_STOPWORDS:
            return ""
        return token

    # English comment.
    if is_korean_only(token):
        token = normalize_korean_token(token)
        if not token:
            return ""

    # English comment.
    if token in ALL_STOPWORDS:
        return ""

    # English comment.
    if len(token) == 1 and token not in {"x", "y", "z"}:
        return ""

    return token


def _segment_single_token(token: str) -> List[str]:
    token = strip_outer_punct(token)
    token = strip_korean_suffix_from_foreign_token(token)
    token = strip_outer_punct(token)

    if not token:
        return []

    if is_pure_cjk(token):
        parts = [canonicalize_token(p) for p in jieba.lcut(token) if p.strip()]
        return [p for p in parts if p]

    canon = canonicalize_token(token)
    return [canon] if canon else []


# ---------------------------------------------------------------------
# Public tokenizer API
# ---------------------------------------------------------------------
def shared_multilingual_tokenize(text: str) -> List[str]:
    """English documentation."""
    if not text:
        return []

    text = normalize_text_basic(text)
    rough_tokens = text.split()

    out: List[str] = []
    for tok in rough_tokens:
        out.extend(_segment_single_token(tok))

    return out


def canonicalize_text(text: str) -> str:
    return " ".join(shared_multilingual_tokenize(text))


def is_low_signal_token(token: str) -> bool:
    token = token.strip()
    if not token:
        return True

    if is_metadata_token(token):
        return True

    if token in META_ROLE_TOKENS:
        return True

    if len(token) > 48:
        return True

    if PATH_LIKE_TOKEN_RE.search(token):
        return True

    # Generic local/cloud path-like token fallback (e.g., mnt/data/file_name.ext)
    if token.count("/") >= 2 and ("." in token or "_" in token):
        return True

    if CODE_META_TOKEN_RE.search(token):
        return True

    if "/" in token and ":" in token and any(ch.isdigit() for ch in token):
        return True

    if LOW_SIGNAL_CHARS_RE.search(token):
        return True

    if "(" in token or ")" in token or "[" in token or "]" in token:
        return True

    if LATEX_ARTIFACT_RE.search(token):
        return True

    if not ALNUM_OR_LETTER_RE.search(token):
        return True

    if not contains_hangul(token) and not contains_cjk(token):
        alnum = sum(ch.isalnum() for ch in token)
        if alnum / max(len(token), 1) < 0.30:
            return True

    return False


def _normalize_phrase_token(token: str) -> str:
    token = INLINE_MARKDOWN_GLUE_RE.sub("", token or "")
    token = MARKDOWN_ARTIFACT_RE.sub("", token or "")
    token = strip_outer_punct(token)
    if not token:
        return ""

    token = strip_korean_suffix_from_foreign_token(token)
    token = strip_outer_punct(token)
    if not token:
        return ""

    if is_metadata_token(token):
        return ""

    if token in LATEX_RAW_COMMAND_TOKENS:
        return ""

    if is_korean_only(token):
        token = normalize_korean_token(token)
        if not token:
            return ""

    if token in ALL_STOPWORDS and token not in FORCE_KEEP_TOKENS:
        return ""

    # English comment.
    if contains_cjk(token) and len(token) == 1 and token not in CJK_TOPIC_HINT_TOKENS:
        return ""

    return token


def _compress_phrase_tokens(tokens: Sequence[str]) -> List[str]:
    if not tokens:
        return []

    compact: List[str] = []
    n = len(tokens)

    for idx, tok in enumerate(tokens):
        if tok in LOW_VALUE_PHRASE_TOKENS:
            continue

        if n >= 3 and tok in GENERIC_CJK_META_TOKENS:
            continue

        if n >= 3 and tok in PHRASE_CONNECTOR_TOKENS:
            continue

        if n >= 3 and tok in EN_PHRASE_GLUE_TOKENS:
            continue

        # English comment.
        if (idx == 0 or idx == n - 1) and tok in GENERIC_CJK_META_TOKENS:
            continue

        compact.append(tok)

    return compact if compact else list(tokens)


def _contains_code_signal(text: str) -> bool:
    if not text:
        return False
    return (
        "```" in text
        or "~~~" in text
        or "import " in text
        or "class " in text
        or "def " in text
        or "function " in text
        or "=>" in text
        or "::" in text
        or "{ " in text
        or " }" in text
    )


def _is_meta_discourse_token(token: str) -> bool:
    tok = (token or "").strip().lower()
    if not tok:
        return False
    if tok in META_DISCOURSE_TOKENS:
        return True
    if tok in PHRASE_CONNECTOR_TOKENS:
        return True
    if tok in LOW_VALUE_PHRASE_TOKENS:
        return True
    if META_DISCOURSE_STEM_RE.fullmatch(tok):
        return True
    return False


def _meta_discourse_ratio(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    hits = sum(1 for tok in tokens if _is_meta_discourse_token(tok))
    return hits / max(len(tokens), 1)


def _has_strong_topic_anchor(term: str, tokens: Sequence[str]) -> bool:
    low = (term or "").lower()
    if _is_formula_term(low):
        return True
    if CAMEL_CASE_RE.search(term) or SNAKE_CASE_RE.search(term):
        return True
    if TECH_TOKEN_RE.search(term):
        return True
    if any(DIGIT_RE.search(tok) for tok in tokens):
        return True
    for tok in tokens:
        if tok in FORCE_KEEP_TOKENS:
            return True
        if tok in CJK_TOPIC_HINT_TOKENS:
            return True
        if tok in ALL_STOPWORDS:
            continue
        if _is_meta_discourse_token(tok):
            continue
        if contains_hangul(tok) and len(tok) >= 3:
            return True
        if contains_cjk(tok) and len(tok) >= 2:
            return True
        if contains_latin(tok) and len(tok) >= 4:
            return True
    return False


def normalize_keyword_phrase(
    phrase: str,
    *,
    allow_generic_cjk_meta: bool = False,
    profile: Optional[str] = None,
) -> str:
    if not phrase:
        return ""

    original_tokens = [tok for tok in phrase.split() if tok]
    if not original_tokens:
        return ""

    tokens: List[str] = []
    for tok in original_tokens:
        tok = _normalize_phrase_token(tok)
        if not tok:
            continue
        if is_low_signal_token(tok):
            continue
        tokens.append(tok)
    if not tokens:
        return ""

    # English comment.
    if len(original_tokens) >= 3 and len(tokens) < 2:
        return ""
    if any(len(tok) > 64 for tok in original_tokens) and len(tokens) < 2:
        return ""

    max_tokens = MAX_KEYWORD_TOKENS
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    cjk_only = [
        tok for tok in tokens if contains_cjk(tok) and not contains_hangul(tok) and not contains_latin(tok)
    ]
    if cjk_only and not allow_generic_cjk_meta:
        generic_hits = sum(tok in GENERIC_CJK_META_TOKENS for tok in cjk_only)
        if len(cjk_only) <= 2 and generic_hits >= 1:
            return ""
        if generic_hits >= 2 and generic_hits >= len(cjk_only) * 0.4:
            return ""
        if all(len(tok) == 1 for tok in cjk_only) and len(cjk_only) <= 3:
            return ""

    # English comment.
    if len(tokens) >= 2:
        tokens = _compress_phrase_tokens(tokens)

    out_tokens: List[str] = []
    for tok in tokens:
        candidate = " ".join(out_tokens + [tok])
        if len(candidate) > MAX_KEYWORD_CHARS:
            break
        out_tokens.append(tok)

    return " ".join(out_tokens).strip()


def is_noise_keyword_phrase(
    phrase: str,
    *,
    phrase_is_canonical: bool = False,
) -> bool:
    """English documentation."""
    canonical = phrase if phrase_is_canonical else normalize_keyword_phrase(canonicalize_text(phrase))
    if not canonical:
        return True

    tokens = [tok for tok in canonical.split() if tok]
    if not tokens:
        return True

    code_hits = 0
    id_like_hits = 0
    code_chain_hits = 0
    metadata_hits = 0

    for tok in tokens:
        if tok in CODE_NOISE_TOKENS:
            code_hits += 1
            continue
        if is_metadata_token(tok):
            metadata_hits += 1
            continue
        if CODE_CHAIN_TOKEN_RE.fullmatch(tok):
            code_chain_hits += 1
            continue
        if tok in FORCE_KEEP_TOKENS:
            continue
        if HEXISH_TOKEN_RE.fullmatch(tok):
            id_like_hits += 1
            continue
        if SHORT_MIXED_ID_TOKEN_RE.fullmatch(tok):
            id_like_hits += 1
            continue
        if LONG_ASCII_IDENTIFIER_RE.fullmatch(tok):
            id_like_hits += 1
            continue

    has_cjk_or_ko = any(contains_cjk(tok) or contains_hangul(tok) for tok in tokens)
    has_long_ascii_token = any(
        len(tok) >= 18 and tok.isascii() and tok.isalnum() for tok in tokens
    )

    if code_hits >= 2 and has_long_ascii_token:
        return True
    if metadata_hits >= 2:
        return True
    if metadata_hits >= 1 and (code_hits >= 1 or id_like_hits >= 1):
        return True
    if code_hits >= 1 and id_like_hits >= 1:
        return True
    if code_chain_hits >= 1 and (code_hits >= 1 or id_like_hits >= 1):
        return True
    if code_chain_hits >= 2 and not has_cjk_or_ko:
        return True
    if not has_cjk_or_ko and id_like_hits >= 2 and len(tokens) <= 4:
        return True

    # English comment.
    if _meta_discourse_ratio(tokens) >= 0.66 and not _has_strong_topic_anchor(canonical, tokens):
        return True

    return False


def _token_set_for_dedup(term: str) -> set[str]:
    tokens = {tok for tok in term.split() if tok}
    if tokens:
        return tokens
    return {ch for ch in term if ch.strip()}


def _token_jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _cjk_generic_ratio(tokens: Sequence[str]) -> float:
    cjk_tokens = [
        tok
        for tok in tokens
        if contains_cjk(tok) and not contains_hangul(tok) and not contains_latin(tok)
    ]
    if not cjk_tokens:
        return 0.0
    generic_hits = sum(tok in GENERIC_CJK_META_TOKENS for tok in cjk_tokens)
    return generic_hits / len(cjk_tokens)


def _has_cjk_topic_anchor(tokens: Sequence[str]) -> bool:
    for tok in tokens:
        if tok in CJK_TOPIC_HINT_TOKENS:
            return True
        if contains_cjk(tok) and DIGIT_RE.search(tok):
            return True
        if contains_cjk(tok) and len(tok) >= 2 and tok not in GENERIC_CJK_META_TOKENS:
            return True
    return False


def _is_cjk_phrase_term(term: str) -> bool:
    tokens = [tok for tok in term.split() if tok]
    cjk_tokens = [tok for tok in tokens if contains_cjk(tok)]
    return len(cjk_tokens) >= 2


def _recover_cjk_phrase(
    raw_phrase: str,
    *,
    profile: Optional[str] = None,
) -> str:
    resolved_profile = _resolve_text_core_profile(profile)
    strict = normalize_keyword_phrase(
        canonicalize_text(raw_phrase),
        profile=resolved_profile,
    )
    if strict:
        return strict

    relaxed = normalize_keyword_phrase(
        canonicalize_text(raw_phrase),
        allow_generic_cjk_meta=True,
        profile=resolved_profile,
    )
    if not relaxed:
        return ""

    tokens = [tok for tok in relaxed.split() if tok]
    if not tokens:
        return ""

    generic_ratio = _cjk_generic_ratio(tokens)
    if generic_ratio >= 0.34:
        return ""
    if not _has_cjk_topic_anchor(tokens):
        return ""
    return relaxed


def _is_formula_term(term: str) -> bool:
    low = term.lower()
    if low in MATH_HINT_TOKENS:
        return True
    if MATH_SIGNAL_RE.search(low):
        return True
    if Z_SUBSCRIPT_RE.search(low) or Z_POWER_RE.search(low):
        return True
    return False


def _score_keyword_term(
    term: str,
    base_score: float,
    *,
    source_text_lower: str,
    profile: Optional[str] = None,
) -> float:
    score = float(base_score)
    tokens = [tok for tok in term.split() if tok]
    source_has_ml_context = _has_ml_context(source_text_lower)
    source_has_strict_math_context = _has_strict_math_context(source_text_lower)

    if _is_formula_term(term):
        score += 0.05
    if CAMEL_CASE_RE.search(term) or SNAKE_CASE_RE.search(term):
        score += 0.05
    if any(tok in CJK_TOPIC_HINT_TOKENS for tok in tokens):
        score += 0.04
    if any(tok in CODE_NOISE_TOKENS for tok in tokens):
        score -= 0.06
    if _cjk_generic_ratio(tokens) >= 0.4:
        score -= 0.12
    if is_noise_keyword_phrase(term, phrase_is_canonical=True) and not _is_formula_term(term):
        score -= 0.18
    if source_text_lower.startswith("user:") and term.startswith("user "):
        score -= 0.05
    if source_text_lower.startswith("assistant:") and term.startswith("assistant "):
        score -= 0.05

    # English comment.
    # English comment.
    meta_ratio = _meta_discourse_ratio(tokens)
    if meta_ratio >= 0.66 and not _is_formula_term(term):
        score -= 0.16
    elif meta_ratio >= 0.50 and not _is_formula_term(term):
        score -= 0.10

    # English comment.
    if _is_formula_term(term) and source_has_ml_context:
        score -= 0.10
        if not source_has_strict_math_context:
            score -= 0.06

    # English comment.
    if source_has_ml_context and _contains_any_hint(term.lower(), ML_CONTEXT_HINTS):
        score += 0.06

    return score


def _dedupe_ranked_pairs(
    ranked_pairs: Sequence[Tuple[str, float]],
    *,
    top_n: int,
    dedup_threshold: float,
    max_formula_keywords: int,
) -> List[Tuple[str, float]]:
    selected: List[Tuple[str, float]] = []
    selected_sets: List[set[str]] = []
    formula_count = 0

    for term, score in ranked_pairs:
        if not term:
            continue
        token_set = _token_set_for_dedup(term)
        if any(_token_jaccard(token_set, prev) >= dedup_threshold for prev in selected_sets):
            continue

        if _is_formula_term(term):
            if formula_count >= max_formula_keywords:
                continue
            formula_count += 1

        selected.append((term, float(score)))
        selected_sets.append(token_set)
        if len(selected) >= top_n:
            break

    return selected


def extract_code_keyword_candidates(
    text: str,
    *,
    limit: int = 12,
) -> List[str]:
    if not text:
        return []

    source = unicodedata.normalize("NFKC", text)
    lowered = source.lower()
    if not _contains_code_signal(source):
        return []

    counter: Counter[str] = Counter()

    # English comment.
    for match in CAMEL_CASE_RE.finditer(source):
        token = _normalize_phrase_token(match.group(0).lower())
        if (
            token
            and token not in CODE_NOISE_TOKENS
            and token not in CODE_IDENTIFIER_BLOCKLIST
            and not is_metadata_token(token)
        ):
            counter[token] += 2

    for match in SNAKE_CASE_RE.finditer(source):
        token = _normalize_phrase_token(match.group(0).lower())
        if (
            token
            and token not in CODE_NOISE_TOKENS
            and token not in CODE_IDENTIFIER_BLOCKLIST
            and not is_metadata_token(token)
        ):
            counter[token] += 2

    # English comment.
    for match in CODE_IDENTIFIER_RE.finditer(source):
        raw = match.group(0)
        token = _normalize_phrase_token(raw.lower())
        if not token:
            continue
        if token in CODE_NOISE_TOKENS:
            continue
        if token in CODE_IDENTIFIER_BLOCKLIST:
            continue
        if is_metadata_token(token):
            continue
        if token in EN_STOPWORDS or token in BASE_STOPWORDS:
            continue
        if len(token) < 4 and token not in FORCE_KEEP_TOKENS:
            continue
        if token.isalpha() and token == token.lower() and len(token) <= 6 and token not in FORCE_KEEP_TOKENS:
            continue

        bonus = 0
        if raw.lower() in FORCE_KEEP_TOKENS:
            bonus += 2
        if "_" in raw or CAMEL_CASE_RE.fullmatch(raw):
            bonus += 1
        if token in lowered:
            bonus += 1

        counter[token] += max(1, bonus)

    # English comment.
    for keep in FORCE_KEEP_TOKENS:
        if keep in lowered and keep not in CODE_NOISE_TOKENS:
            counter[keep] += 1

    out: List[str] = []
    for term, _ in counter.most_common(limit * 2):
        canon = normalize_keyword_phrase(term)
        if not canon or canon in out:
            continue
        out.append(canon)
        if len(out) >= limit:
            break

    return out


def extract_formula_keyword_candidates(
    text: str,
    *,
    limit: int = 8,
) -> List[str]:
    if not text:
        return []

    source = unicodedata.normalize("NFKC", text)
    low = source.lower()
    strict_math_context = _has_strict_math_context(source)
    series_context = _contains_any_hint(
        low,
        {"series", "taylor", "laurent", "maclaurin", "text", "text"},
    )
    candidates: List[str] = []

    def _push(value: str) -> None:
        canon = normalize_keyword_phrase(canonicalize_text(value), allow_generic_cjk_meta=True)
        if not canon:
            return
        if canon in candidates:
            return
        candidates.append(canon)

    for pattern in (DISPLAY_MATH_RE, INLINE_MATH_RE, LATEX_PAREN_RE, LATEX_BRACKET_RE):
        for match in pattern.finditer(source):
            expr = match.group(0)
            for cmd in LATEX_COMMAND_RE.findall(expr):
                cmd_l = cmd.lower()
                mapped = LATEX_COMMAND_CANONICAL.get(cmd_l)
                if mapped:
                    if mapped == "series expansion" and not (strict_math_context or series_context):
                        continue
                    _push(mapped)

    if "residue" in low and "theorem" in low:
        _push("residue theorem")
    if "\\oint" in source or "contour" in low:
        _push("contour integral")
    if "\\int" in source or "integral" in low:
        _push("integral")
    if ("\\sum" in source and (strict_math_context or series_context)) or "laurent" in low or "taylor" in low:
        _push("series expansion")
    if "\\tan" in source or "tan(" in low or "tan pi z" in low:
        _push("tan(pi z)")

    for match in Z_SUBSCRIPT_RE.finditer(low):
        _push(f"z_{match.group(1)}")
    for match in Z_POWER_RE.finditer(low):
        _push(f"z^{match.group(1)}")

    # English comment.
    # English comment.
    if _has_ml_context(low) and not _contains_any_hint(
        low,
        {"residue", "contour", "laurent", "taylor", "maclaurin", "singular", "cauchy", "holomorphic", "text", "text"},
    ):
        candidates = [
            item
            for item in candidates
            if item not in {"series expansion", "integral", "contour integral"}
        ]

    return candidates[:limit]


def extract_topic_fallback_keywords(
    text: str,
    *,
    limit: int = 20,
) -> List[str]:
    if not text:
        return []

    tokens = shared_multilingual_tokenize(text)
    tokens = [
        tok
        for tok in tokens
        if tok
        and tok not in GENERIC_CJK_META_TOKENS
        and not tok.isdigit()
        and not is_low_signal_token(tok)
    ]
    cjk_counter: Counter[str] = Counter()
    cjk_bigram: Counter[str] = Counter()
    cjk_spans = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    for span in cjk_spans:
        span_tokens = [tok.strip() for tok in jieba.lcut(span) if tok.strip()]
        span_tokens = [
            tok
            for tok in span_tokens
            if len(tok) >= 2
            and tok not in GENERIC_CJK_META_TOKENS
            and tok not in STOPWORDS_ZH
        ]
        for tok in span_tokens:
            cjk_counter[tok] += 1
        for left, right in zip(span_tokens, span_tokens[1:]):
            phrase = normalize_keyword_phrase(
                f"{left} {right}",
                allow_generic_cjk_meta=True,
            )
            if phrase:
                cjk_bigram[phrase] += 1
    for hint in CJK_TOPIC_HINT_TOKENS:
        if hint in text:
            cjk_counter[hint] += 2

    if not tokens and not cjk_counter and not cjk_bigram:
        return []

    unigram = Counter(tokens)
    bigram: Counter[str] = Counter()
    for left, right in zip(tokens, tokens[1:]):
        if not left or not right:
            continue
        if left in GENERIC_CJK_META_TOKENS or right in GENERIC_CJK_META_TOKENS:
            continue
        phrase = normalize_keyword_phrase(f"{left} {right}", allow_generic_cjk_meta=True)
        if phrase:
            bigram[phrase] += 1

    scored: List[Tuple[str, float]] = []
    for term, freq in unigram.items():
        base = float(freq)
        if contains_cjk(term):
            base += 0.25
        if DIGIT_RE.search(term):
            base += 0.10
        scored.append((term, base))
    for term, freq in bigram.items():
        base = float(freq) + 0.35
        if contains_cjk(term):
            base += 0.15
        scored.append((term, base))
    for term, freq in cjk_counter.items():
        base = float(freq) + 0.55
        if term in CJK_TOPIC_HINT_TOKENS:
            base += 0.20
        scored.append((term, base))
    for term, freq in cjk_bigram.items():
        base = float(freq) + 0.75
        scored.append((term, base))

    seen: set[str] = set()
    out: List[str] = []
    for term, _ in sorted(scored, key=lambda item: item[1], reverse=True):
        canon = normalize_keyword_phrase(term, allow_generic_cjk_meta=True)
        if not canon or canon in seen:
            continue
        if _cjk_generic_ratio(canon.split()) >= 0.5:
            continue
        seen.add(canon)
        out.append(canon)
        if len(out) >= limit:
            break
    return out


def prepare_keyword_pairs(
    raw_keywords: Sequence[Tuple[str, float]],
    *,
    source_text: str,
    top_n: int,
    min_keywords: int = 3,
    dedup_threshold: float = 0.8,
    max_formula_keywords: int = 2,
    profile: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """English documentation."""
    resolved_profile = _resolve_text_core_profile(profile)
    top_n = max(1, int(top_n))
    min_keywords = max(1, min(int(min_keywords), top_n))
    source_text = source_text or ""
    source_text_lower = source_text.lower()

    merged_scores: Dict[str, float] = {}
    order: List[str] = []

    for raw, score in raw_keywords:
        canonical = normalize_keyword_phrase(
            canonicalize_text(raw),
            profile=resolved_profile,
        )
        if not canonical:
            continue
        if canonical not in merged_scores:
            order.append(canonical)
            merged_scores[canonical] = float(score)
        else:
            merged_scores[canonical] = max(merged_scores[canonical], float(score))

    if len(merged_scores) < top_n:
        for raw, score in raw_keywords:
            recovered = _recover_cjk_phrase(raw, profile=resolved_profile)
            if not recovered:
                continue
            if recovered not in merged_scores:
                order.append(recovered)
                merged_scores[recovered] = float(score) * 0.98
            else:
                merged_scores[recovered] = max(
                    merged_scores[recovered],
                    float(score) * 0.98,
                )

    for idx, formula_kw in enumerate(extract_formula_keyword_candidates(source_text, limit=8)):
        base = 0.62 - (idx * 0.015)
        if formula_kw not in merged_scores:
            order.append(formula_kw)
            merged_scores[formula_kw] = base
        else:
            merged_scores[formula_kw] = max(merged_scores[formula_kw], base)

    if len(merged_scores) < top_n or _contains_code_signal(source_text):
        for idx, code_kw in enumerate(extract_code_keyword_candidates(source_text, limit=12)):
            base = 0.60 - (idx * 0.015)
            if code_kw not in merged_scores:
                order.append(code_kw)
                merged_scores[code_kw] = base
            else:
                merged_scores[code_kw] = max(merged_scores[code_kw], base)

    if len(merged_scores) < min_keywords:
        for idx, fallback_kw in enumerate(extract_topic_fallback_keywords(source_text, limit=24)):
            base = 0.58 - (idx * 0.01)
            if fallback_kw not in merged_scores:
                order.append(fallback_kw)
                merged_scores[fallback_kw] = base
            else:
                merged_scores[fallback_kw] = max(merged_scores[fallback_kw], base)
            if len(merged_scores) >= max(top_n * 2, min_keywords + 3):
                break

    source_cjk_chars = len(CJK_CHAR_RE.findall(source_text))
    if source_cjk_chars >= 40:
        has_cjk_candidate = any(contains_cjk(term) for term in merged_scores)
        if not has_cjk_candidate:
            cjk_fallback_terms = [
                kw
                for kw in extract_topic_fallback_keywords(source_text, limit=30)
                if contains_cjk(kw)
                and all(len(tok) >= 2 for tok in kw.split() if contains_cjk(tok))
            ]
            for idx, term in enumerate(cjk_fallback_terms[:2]):
                base = 0.66 - (idx * 0.02)
                if term not in merged_scores:
                    order.append(term)
                    merged_scores[term] = base
                else:
                    merged_scores[term] = max(merged_scores[term], base)

    ranked_pairs: List[Tuple[str, float]] = []
    for term in order:
        if is_noise_keyword_phrase(term, phrase_is_canonical=True) and not _is_formula_term(term):
            continue
        base_score = merged_scores.get(term, 0.0)
        adjusted = _score_keyword_term(
            term,
            base_score,
            source_text_lower=source_text_lower,
            profile=resolved_profile,
        )
        ranked_pairs.append((term, adjusted))
    ranked_pairs.sort(key=lambda item: item[1], reverse=True)

    selected = _dedupe_ranked_pairs(
        ranked_pairs,
        top_n=top_n,
        dedup_threshold=dedup_threshold,
        max_formula_keywords=max_formula_keywords,
    )

    if len(selected) < min_keywords:
        selected_terms = {term for term, _ in selected}
        fallback_ranked = sorted(
            (
                (
                    term,
                    _score_keyword_term(
                        term,
                        0.5,
                        source_text_lower=source_text_lower,
                        profile=resolved_profile,
                    ),
                )
                for term in extract_topic_fallback_keywords(source_text, limit=30)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        for term, score in fallback_ranked:
            if term in selected_terms:
                continue
            if is_noise_keyword_phrase(term, phrase_is_canonical=True):
                continue
            selected.append((term, score))
            selected_terms.add(term)
            if len(selected) >= min_keywords:
                break

    if source_cjk_chars >= 40 and not any(contains_cjk(term) for term, _ in selected):
        selected_terms = {term for term, _ in selected}
        cjk_candidates = [
            (term, score)
            for term, score in ranked_pairs
            if contains_cjk(term)
            and term not in selected_terms
            and all(len(tok) >= 2 for tok in term.split() if contains_cjk(tok))
        ]
        if not cjk_candidates:
            cjk_candidates = [
                (
                    term,
                    _score_keyword_term(
                        term,
                        0.55,
                        source_text_lower=source_text_lower,
                        profile=resolved_profile,
                    ),
                )
                for term in extract_topic_fallback_keywords(source_text, limit=30)
                if contains_cjk(term)
                and term not in selected_terms
                and all(len(tok) >= 2 for tok in term.split() if contains_cjk(tok))
            ]
        if cjk_candidates:
            best_cjk = max(cjk_candidates, key=lambda item: item[1])
            if len(selected) >= top_n:
                selected[-1] = best_cjk
            else:
                selected.append(best_cjk)

    # English comment.
    # English comment.
    if source_cjk_chars >= 40 and not any(_is_cjk_phrase_term(term) for term, _ in selected):
        selected_terms = {term for term, _ in selected}
        phrase_candidates: List[Tuple[str, float]] = [
            (term, score)
            for term, score in ranked_pairs
            if _is_cjk_phrase_term(term)
            and _cjk_generic_ratio(term.split()) < 0.5
            and term not in selected_terms
        ]
        if not phrase_candidates:
            phrase_candidates = [
                (
                    term,
                    _score_keyword_term(
                        term,
                        0.57,
                        source_text_lower=source_text_lower,
                        profile=resolved_profile,
                    ),
                )
                for term in extract_topic_fallback_keywords(source_text, limit=36)
                if _is_cjk_phrase_term(term)
                and _cjk_generic_ratio(term.split()) < 0.5
                and term not in selected_terms
            ]
        if phrase_candidates:
            best_phrase = max(phrase_candidates, key=lambda item: item[1])
            if len(selected) >= top_n:
                selected[-1] = best_phrase
            else:
                selected.append(best_phrase)

    return selected[:top_n]


def canonicalize_text_list(
    texts: Sequence[str],
    *,
    drop_empty: bool = True,
    **kwargs,
) -> List[str]:
    """English documentation."""
    results: List[str] = []
    for text in texts:
        canon = normalize_keyword_phrase(canonicalize_text(text))
        if canon:
            results.append(canon)
        elif not drop_empty:
            results.append("")
    return results


# ---------------------------------------------------------------------
# Vectorizer builders
# ---------------------------------------------------------------------
def build_shared_vectorizer(
    ngram_max: int | None = None,
    *,
    stopword_langs=None,
    extra_stopwords=None,
    stop_words=None,
    min_df: int = 1,
    ngram_range=None,
    **kwargs,
) -> CountVectorizer:
    """English documentation."""
    if ngram_range is not None:
        final_ngram_range = ngram_range
    else:
        if ngram_max is None:
            ngram_max = 1
        final_ngram_range = (1, ngram_max)

    # NOTE:
    # English comment.
    # English comment.
    # English comment.
    # English comment.
    vectorizer_stop_words = sorted(PROJECT_STOPWORDS) if PROJECT_STOPWORDS else None

    return CountVectorizer(
        analyzer="word",
        tokenizer=shared_multilingual_tokenize,
        token_pattern=None,
        stop_words=vectorizer_stop_words,
        ngram_range=final_ngram_range,
        min_df=min_df,
    )


def build_vectorizer(*args, **kwargs):
    return build_shared_vectorizer(*args, **kwargs)


# ---------------------------------------------------------------------
# Keyword canonicalization
# ---------------------------------------------------------------------
def canonicalize_keywords(
    raw_keywords: Sequence[Tuple[str, float]],
    *,
    stopword_langs=None,
    extra_stopwords=None,
    stop_words=None,
    **kwargs,
) -> List[CanonicalKeyword]:
    """English documentation."""
    merged: Dict[str, CanonicalKeyword] = {}
    order: List[str] = []

    for raw, score in raw_keywords:
        canonical = normalize_keyword_phrase(canonicalize_text(raw))
        if not canonical:
            continue

        if canonical not in merged:
            merged[canonical] = CanonicalKeyword(
                raw=raw,
                canonical=canonical,
                score=float(score),
            )
            order.append(canonical)
        else:
            if float(score) > merged[canonical].score:
                merged[canonical] = CanonicalKeyword(
                    raw=raw,
                    canonical=canonical,
                    score=float(score),
                )

    return [merged[key] for key in order]


def canonicalize_keyword_pairs(
    raw_keywords: Sequence[Tuple[str, float]],
    *,
    stopword_langs=None,
    extra_stopwords=None,
    stop_words=None,
    **kwargs,
) -> List[Tuple[str, float]]:
    """English documentation."""
    normalized = canonicalize_keywords(
        raw_keywords,
        stopword_langs=stopword_langs,
        extra_stopwords=extra_stopwords,
        stop_words=stop_words,
        **kwargs,
    )
    return [(item.canonical, item.score) for item in normalized]
