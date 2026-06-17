"""English documentation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from ..utils import logger
from ..utils.preprocess import preprocess_content


def _split_by_h1(text: str) -> List[Dict[str, str]]:
    """English documentation."""
    parts = re.split(r"^# (.+)$", text, flags=re.MULTILINE)
    # parts: ['pre-content', 'Heading1', 'body1', 'Heading2', 'body2', ...]
    if len(parts) == 1:
        # English comment.
        return [{"heading": "", "body": parts[0].strip()}]

    sections = []
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append({"heading": heading, "body": body})
    return sections


def build_note_sections(note_id: str, tmp_dir: Path) -> Path:
    """English documentation."""
    note_file = tmp_dir / f"note_{note_id}.json"
    note_data: Dict[str, Any] = json.loads(note_file.read_text(encoding="utf-8"))
    content: str = note_data.get("content", "")
    note_title: str = note_data.get("title", "")

    raw_sections = _split_by_h1(content)
    logger.info(f"[{note_id}] {len(raw_sections)} raw H1 section(s) found")

    records = []
    for idx, sec in enumerate(raw_sections):
        cleaned = preprocess_content(sec["body"])
        if len(cleaned) < 20:
            logger.debug(f"[{note_id}] Section {idx} too short after preprocess, skipping")
            continue
        heading = sec["heading"] or note_title
        records.append({
            "qa_id": f"{note_id}_{idx}",        # English comment.
            "conversation_id": note_id,           # English comment.
            "qa_index": idx,
            "note_id": note_id,
            "note_title": note_title,
            "heading": heading,
            "content": cleaned,
            "section_index": idx,
        })

    logger.info(f"[{note_id}] {len(records)} valid section(s) after filtering")

    out = tmp_dir / f"note_sections_{note_id}.json"
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
