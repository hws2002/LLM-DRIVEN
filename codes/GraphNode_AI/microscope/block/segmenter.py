"""LLM-based block segmentation for chat and note sources."""
from __future__ import annotations
import logging
import re
from typing import List

from microscope.block.models import Block
from microscope.block.prompts.segmentation import build_chat_prompt, build_note_prompt
from microscope.utils.io_utils import extract_json_from_text

logger = logging.getLogger(__name__)

GRANULARITY_OPTIONS = ("coarse", "medium", "fine")


class BlockSegmenter:
    def __init__(self, api_provider):
        self.api_provider       = api_provider
        self.last_raw           = ""
        self.last_system_prompt = ""
        self.last_user_prompt   = ""

    def segment(
        self,
        text: str,
        source_type: str = "chat",
        granularity: str = "medium",
    ) -> List[Block]:
        if granularity not in GRANULARITY_OPTIONS:
            raise ValueError(f"granularity must be one of {GRANULARITY_OPTIONS}")

        if source_type == "chat":
            return self._segment_chat(text, granularity)
        elif source_type == "note":
            return self._segment_note(text)
        else:
            raise ValueError(f"source_type must be 'chat' or 'note', got {source_type!r}")

    def _segment_chat(self, text: str, granularity: str) -> List[Block]:
        system_prompt, user_prompt = build_chat_prompt(text, granularity)
        raw = self._call_llm(system_prompt, user_prompt)
        self.last_system_prompt = system_prompt
        self.last_user_prompt   = user_prompt
        self.last_raw           = raw
        data = extract_json_from_text(raw)
        if not data or "blocks" not in data:
            logger.warning("Block segmentation returned no blocks; treating whole text as one block")
            return [_fallback_block(text, "chat")]

        blocks: List[Block] = []
        for raw_block in data["blocks"]:
            block_text = _resolve_text(text, raw_block)
            blocks.append(Block(
                block_id      = raw_block["block_id"],
                title         = raw_block.get("title", raw_block["block_id"]),
                summary       = raw_block.get("summary", ""),
                key_concepts  = raw_block.get("key_concepts", []),
                raw_text      = block_text,
                order_index   = len(blocks),
                source_type   = "chat",
            ))
        return blocks

    def _segment_note(self, text: str) -> List[Block]:
        system_prompt, user_prompt = build_note_prompt(text)
        raw = self._call_llm(system_prompt, user_prompt)
        self.last_system_prompt = system_prompt
        self.last_user_prompt   = user_prompt
        self.last_raw           = raw
        data = extract_json_from_text(raw)
        if not data or "blocks" not in data:
            logger.warning("Block segmentation returned no blocks; treating whole text as one block")
            return [_fallback_block(text, "note")]

        blocks: List[Block] = []
        for raw_block in data["blocks"]:
            block_text = _resolve_text(text, raw_block)
            blocks.append(Block(
                block_id      = raw_block["block_id"],
                title         = raw_block.get("title", raw_block["block_id"]),
                summary       = raw_block.get("summary", ""),
                key_concepts  = raw_block.get("key_concepts", []),
                raw_text      = block_text,
                order_index   = len(blocks),
                source_type   = "note",
                heading_path  = raw_block.get("heading_path"),
            ))
        return blocks

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        return self.api_provider.chat_completion_text(messages=messages)


def _find_anchor(full_text: str, anchor: str, search_from: int = 0) -> tuple[int, int]:
    """English documentation."""
    anchor = anchor.strip()
    if not anchor:
        return -1, -1

    # English comment.
    pos = full_text.find(anchor, search_from)
    if pos != -1:
        return pos, pos + len(anchor)

    # English comment.
    def ws_search(text: str) -> tuple[int, int]:
        tokens = text.split()
        if not tokens:
            return -1, -1
        pattern = r"\s+".join(re.escape(t) for t in tokens)
        m = re.search(pattern, full_text[search_from:])
        if m:
            return search_from + m.start(), search_from + m.end()
        return -1, -1

    s, e = ws_search(anchor)
    if s != -1:
        return s, e

    # English comment.
    return ws_search(anchor[:15])


def _resolve_text(full_text: str, raw_block: dict) -> str:
    """English documentation."""
    start_anchor = raw_block.get("start_anchor", "")
    end_anchor   = raw_block.get("end_anchor", "")

    if start_anchor and end_anchor:
        start, _ = _find_anchor(full_text, start_anchor)
        if start != -1:
            _, end = _find_anchor(full_text, end_anchor, start + 1)
            if end == -1:
                # English comment.
                end_char = raw_block.get("end_char")
                end = end_char if isinstance(end_char, int) and end_char > start else len(full_text)
            return full_text[start:end]
        logger.warning(
            "start_anchor not found for %s, falling back to char offsets",
            raw_block.get("block_id"),
        )

    # fallback: start_char / end_char
    start = raw_block.get("start_char", 0)
    end   = raw_block.get("end_char", len(full_text))
    if not isinstance(start, int):
        start = 0
    if not isinstance(end, int):
        end = len(full_text)
    start = max(0, min(start, len(full_text)))
    end   = max(start, min(end, len(full_text)))
    return full_text[start:end]


def _fallback_block(text: str, source_type: str) -> Block:
    return Block(
        block_id     = "block_001",
        title        = "Full Content",
        summary      = "Entire content treated as a single block.",
        key_concepts = [],
        raw_text     = text,
        order_index  = 0,
        source_type  = source_type,
    )
