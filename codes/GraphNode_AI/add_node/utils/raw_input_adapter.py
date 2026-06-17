"""Adapt raw files and ChatGPT exports into add_node batch payloads."""

from __future__ import annotations

import copy
import json
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


REPO_ROOT = Path(__file__).resolve().parents[2]
MACRO_SRC = REPO_ROOT / "macro" / "src"
if str(MACRO_SRC) not in sys.path:
    sys.path.insert(0, str(MACRO_SRC))


RAW_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".md", ".txt"}
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | {".json"}


def build_add_node_batch_from_input(
    path: Path,
    *,
    user_id: str,
    base_batch: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return an add_node batch payload from a file or directory.

    Existing add_node batch JSON remains supported. Raw documents and ChatGPT
    exports are normalized into the existing `notes`/`conversations` fields so
    the core add_node pipeline can run unchanged.
    """
    path = Path(path)
    batch = _normalize_batch(base_batch, user_id=user_id)

    if path.is_dir():
        consumed: Set[Path] = set()
        files = [p for p in sorted(path.rglob("*")) if p.is_file()]

        # Load explicit add_node batch JSON first, so its existingClusters are preserved.
        for candidate in files:
            if candidate.suffix.lower() != ".json":
                continue
            loaded = _try_load_add_node_batch_json(candidate)
            if loaded is None:
                continue
            _merge_batch(batch, _normalize_batch(loaded, user_id=user_id))
            consumed.add(candidate)

        for candidate in files:
            if candidate in consumed or candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            _merge_batch(batch, _batch_from_source_file(candidate, user_id=user_id))
        return batch

    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.suffix.lower() == ".json":
        loaded = _try_load_add_node_batch_json(path)
        if loaded is not None:
            _merge_batch(batch, _normalize_batch(loaded, user_id=user_id))
            return batch

    _merge_batch(batch, _batch_from_source_file(path, user_id=user_id))
    return batch


def merge_add_node_batches(
    base: Dict[str, Any],
    extra: Dict[str, Any],
    *,
    user_id: str,
) -> Dict[str, Any]:
    """Merge two normalized add_node batch payloads."""
    out = _normalize_batch(base, user_id=user_id)
    _merge_batch(out, _normalize_batch(extra, user_id=user_id))
    return out


def _empty_batch(user_id: str) -> Dict[str, Any]:
    return {
        "userId": user_id,
        "existingClusters": [],
        "conversations": [],
        "notes": [],
    }


def _normalize_batch(batch: Optional[Dict[str, Any]], *, user_id: str) -> Dict[str, Any]:
    out = _empty_batch(user_id)
    if not batch:
        return out

    out["userId"] = user_id
    for key in ("existingClusters", "conversations", "notes"):
        value = batch.get(key, [])
        out[key] = copy.deepcopy(value) if isinstance(value, list) else []
    return out


def _merge_batch(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    target.setdefault("existingClusters", [])
    target.setdefault("conversations", [])
    target.setdefault("notes", [])

    known_clusters = {
        str(c.get("clusterId") or c.get("id"))
        for c in target["existingClusters"]
        if isinstance(c, dict) and (c.get("clusterId") or c.get("id"))
    }
    for cluster in source.get("existingClusters", []):
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("clusterId") or cluster.get("id") or "")
        if cluster_id and cluster_id in known_clusters:
            continue
        target["existingClusters"].append(copy.deepcopy(cluster))
        if cluster_id:
            known_clusters.add(cluster_id)

    target["conversations"].extend(copy.deepcopy(source.get("conversations", [])))
    target["notes"].extend(copy.deepcopy(source.get("notes", [])))


def _try_load_add_node_batch_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    batch_keys = {"existingClusters", "conversations", "notes"}
    if not any(key in payload for key in batch_keys):
        return None

    if "conversations" in payload and not isinstance(payload.get("conversations"), list):
        return None
    if "notes" in payload and not isinstance(payload.get("notes"), list):
        return None
    if "existingClusters" in payload and not isinstance(payload.get("existingClusters"), list):
        return None

    return payload


def _batch_from_source_file(path: Path, *, user_id: str) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _batch_from_text_file(path, user_id=user_id)

    if suffix not in SUPPORTED_EXTENSIONS:
        return _empty_batch(user_id)

    input_data = _load_input_data(path)
    batch = _empty_batch(user_id)
    for node in input_data.source_nodes:
        if _is_chat_source(node):
            batch["conversations"].append(_source_node_to_conversation(node))
        else:
            batch["notes"].append(_source_node_to_note(node))
    return batch


def _load_input_data(path: Path) -> Any:
    """Load a supported source file without importing macro extract_features."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        from util.raw_file_loader import load_pdf
        return load_pdf(path)
    if suffix in {".docx", ".doc"}:
        from util.raw_file_loader import load_docx
        return load_docx(path)
    if suffix in {".pptx", ".ppt"}:
        from util.raw_file_loader import load_pptx
        return load_pptx(path)
    if suffix == ".md":
        from util.markdown_loader import load_markdown_dir
        return load_markdown_dir(path)
    if suffix == ".json":
        return _load_json_input_data(path)

    from util.io_schemas import InputData
    return InputData(source_nodes=[])


def _load_json_input_data(path: Path) -> Any:
    from util.io_schemas import InputData, Section, SourceNode

    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, list) and all(isinstance(p, str) and p.endswith(".md") for p in payload):
        from util.markdown_loader import load_markdown_dir
        return load_markdown_dir(path)

    if isinstance(payload, dict) and "source_nodes" in payload:
        nodes = []
        for node_dict in payload.get("source_nodes", []):
            if not isinstance(node_dict, dict):
                continue
            sections = [
                Section(
                    id=str(section.get("id", f"sec_{idx}")),
                    content=str(section.get("content", "")),
                    role=section.get("role"),
                    section_title=section.get("section_title"),
                )
                for idx, section in enumerate(node_dict.get("sections", []))
                if isinstance(section, dict)
            ]
            if sections:
                nodes.append(
                    SourceNode(
                        id=str(node_dict.get("id", f"node_{len(nodes)}")),
                        title=node_dict.get("title"),
                        sections=sections,
                        source_type=node_dict.get("source_type", "chat"),
                        create_time=node_dict.get("create_time"),
                        update_time=node_dict.get("update_time"),
                    )
                )
        return InputData(source_nodes=nodes)

    if isinstance(payload, dict):
        nodes = _group_chatgpt_export([payload])
        return InputData(source_nodes=nodes)

    if isinstance(payload, list):
        if all(isinstance(item, dict) and _is_message_like(item) for item in payload):
            sections = [
                Section(
                    id=str(item.get("id", f"msg_{idx}")),
                    content=str(item.get("content", "")),
                    role=item.get("role"),
                )
                for idx, item in enumerate(payload)
            ]
            return InputData(
                source_nodes=[
                    SourceNode(
                        id=_safe_id("json", path.stem),
                        title=path.stem,
                        sections=sections,
                        source_type="chat",
                    )
                ]
            )
        nodes = _group_chatgpt_export(payload)
        return InputData(source_nodes=nodes)

    return InputData(source_nodes=[])


def _is_message_like(item: Dict[str, Any]) -> bool:
    return {"id", "role", "content"}.issubset(item.keys())


def _group_chatgpt_export(payload: Iterable[Dict[str, Any]]) -> List[Any]:
    from util.io_schemas import Section, SourceNode

    conversations: List[Any] = []

    def _normalize_epoch(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value.strip()))
            except ValueError:
                return None
        return None

    for conv_idx, conversation in enumerate(payload):
        if not isinstance(conversation, dict):
            continue
        mapping = conversation.get("mapping")
        if not isinstance(mapping, dict):
            continue

        conv_id = str(conversation.get("conversation_id") or f"conv_{conv_idx}")
        title = conversation.get("title", f"Conversation {conv_idx}")
        create_time = _normalize_epoch(conversation.get("create_time"))
        update_time = _normalize_epoch(conversation.get("update_time"))
        sections = []

        def iter_nodes() -> Iterable[Dict[str, Any]]:
            visited: set[str] = set()
            roots = [node for node in mapping.values() if isinstance(node, dict) and node.get("parent") is None]
            queue: deque[Dict[str, Any]] = deque(roots)

            while queue:
                node = queue.popleft()
                node_id = str(node.get("id", ""))
                if node_id and node_id in visited:
                    continue
                if node_id:
                    visited.add(node_id)
                yield node

                for child_id in node.get("children") or []:
                    child = mapping.get(child_id)
                    if isinstance(child, dict):
                        queue.append(child)

            for node in mapping.values():
                if not isinstance(node, dict):
                    continue
                node_id = str(node.get("id", ""))
                if node_id not in visited:
                    yield node

        for node in iter_nodes():
            message = node.get("message") or {}
            if not isinstance(message, dict):
                continue
            message_id = message.get("id")
            role = ((message.get("author") or {}).get("role"))
            content_obj = message.get("content") or {}

            content = ""
            if isinstance(content_obj, dict):
                parts = content_obj.get("parts")
                if isinstance(parts, list):
                    content = "\n".join(str(part) for part in parts if part is not None)
                elif isinstance(content_obj.get("text"), str):
                    content = content_obj["text"]
            elif isinstance(content_obj, str):
                content = content_obj

            if not message_id or role not in {"user", "assistant"} or not content.strip():
                continue

            sections.append(
                Section(
                    id=str(message_id),
                    role=str(role),
                    content=content,
                )
            )

        if sections:
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


def _batch_from_text_file(path: Path, *, user_id: str) -> Dict[str, Any]:
    text = _read_text_best_effort(path)
    if not text.strip():
        return _empty_batch(user_id)
    batch = _empty_batch(user_id)
    batch["notes"].append(
        {
            "noteId": _safe_id("txt", path.stem),
            "title": path.stem,
            "content": _ensure_markdown_heading(path.stem, text),
        }
    )
    return batch


def _is_chat_source(node: Any) -> bool:
    source_type = str(getattr(node, "source_type", "") or "").lower()
    sections = list(getattr(node, "sections", []) or [])
    if source_type != "chat" or not sections:
        return False
    return any(str(getattr(section, "role", "") or "").lower() in {"user", "assistant"} for section in sections)


def _source_node_to_conversation(node: Any) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = []
    for idx, section in enumerate(getattr(node, "sections", []) or []):
        role = str(getattr(section, "role", "") or "").lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(getattr(section, "content", "") or "").strip()
        if content:
            messages.append({"role": role, "content": content, "createdAt": idx})

    return {
        "conversationId": _safe_id("conv", str(getattr(node, "id", "") or "conversation")),
        "title": getattr(node, "title", None) or str(getattr(node, "id", "") or "Conversation"),
        "messages": messages,
        "create_time": getattr(node, "create_time", None),
        "update_time": getattr(node, "update_time", None),
    }


def _source_node_to_note(node: Any) -> Dict[str, Any]:
    title = getattr(node, "title", None) or str(getattr(node, "id", "") or "Document")
    sections = list(getattr(node, "sections", []) or [])
    content_parts: List[str] = []

    for idx, section in enumerate(sections):
        section_title = getattr(section, "section_title", None) or f"Section {idx + 1}"
        content = str(getattr(section, "content", "") or "").strip()
        if not content:
            continue
        content_parts.append(f"# {section_title}\n\n{content}")

    if not content_parts:
        content_parts.append(f"# {title}\n\n{str(getattr(node, 'get_merged_content', lambda: '')()).strip()}")

    source_type = str(getattr(node, "source_type", "") or "document")
    return {
        "noteId": _safe_id(source_type, str(getattr(node, "id", "") or title)),
        "title": title,
        "content": "\n\n".join(part for part in content_parts if part.strip()),
        "sourceType": source_type,
    }


def _safe_id(prefix: str, raw: str) -> str:
    safe = re.sub(r"[^\w\-]+", "_", raw, flags=re.UNICODE).strip("_")
    safe = safe[:80] if safe else "item"
    return f"{prefix}_{safe}"


def _ensure_markdown_heading(title: str, text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("# "):
        return stripped
    return f"# {title}\n\n{stripped}"


def _read_text_best_effort(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp949", "gbk", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")
