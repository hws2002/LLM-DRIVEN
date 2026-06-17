"""Macro-graph context for personalized RAG.

The Micro RAG answers questions over a single document's concept network.
To fulfill GraphNode's promise of *personalized* answers, we additionally fuse
the user's **Macro Graph** insights — the patterns and learning tendencies that
the macro `GraphSummary` already discovers across the whole knowledge graph —
into the RAG prompt. This module turns a stored `GraphSummary` (summary.json)
into a compact "user knowledge profile" text block.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def build_macro_profile(summary: Optional[Dict[str, Any]], max_items: int = 3) -> str:
    """Format a compact user knowledge profile from a macro GraphSummary dict.

    Pulls the user's learning tendency (primary interests, conversation style)
    and the most significant cross-graph patterns / recommendations. Returns an
    empty string when no usable summary is available, so callers can fall back
    to plain RAG transparently.
    """
    if not summary:
        return ""

    lines: List[str] = []

    overview = summary.get("overview") or {}
    interests = overview.get("primary_interests") or []
    style = overview.get("conversation_style")
    if interests:
        lines.append(f"- Primary interests: {', '.join(interests[:5])}")
    if style:
        lines.append(f"- Learning style: {style}")

    patterns = summary.get("patterns") or []
    significant = _top_by_significance(patterns, max_items)
    if significant:
        lines.append("- Recurring patterns across the user's knowledge graph:")
        for p in significant:
            desc = (p.get("description") or "").strip()
            if desc:
                lines.append(f"    * [{p.get('pattern_type', 'pattern')}] {desc}")

    recommendations = summary.get("recommendations") or []
    top_recs = _top_by_priority(recommendations, max_items)
    if top_recs:
        lines.append("- Relevant suggestions for the user:")
        for r in top_recs:
            title = (r.get("title") or "").strip()
            if title:
                lines.append(f"    * {title}")

    if not lines:
        return ""
    return "User knowledge profile (from the user's Macro Graph):\n" + "\n".join(lines)


def _top_by_significance(items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    rank = {"high": 0, "medium": 1, "low": 2}
    ordered = sorted(items, key=lambda x: rank.get(x.get("significance", "low"), 3))
    return ordered[:limit]


def _top_by_priority(items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    rank = {"high": 0, "medium": 1, "low": 2}
    ordered = sorted(items, key=lambda x: rank.get(x.get("priority", "low"), 3))
    return ordered[:limit]


def load_macro_summary(
    user_id: str,
    group_id: str,
    base_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load a stored macro GraphSummary (summary.json) for a user/group.

    Looks under ``base_dir`` (or the ``MACRO_SUMMARY_DIR`` env var) for, in order:
        <base>/<group_id>/<user_id>/summary.json
        <base>/<group_id>/summary.json
    Returns None when nothing is found, so personalization degrades gracefully.
    """
    base = base_dir or os.getenv("MACRO_SUMMARY_DIR")
    if not base:
        return None

    root = Path(base)
    candidates = [
        root / group_id / user_id / "summary.json",
        root / group_id / "summary.json",
    ]
    for path in candidates:
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return None
    return None
