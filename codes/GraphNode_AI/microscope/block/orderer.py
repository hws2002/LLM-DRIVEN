"""LLM-based block ordering and dependency DAG generation."""
from __future__ import annotations
import logging
from typing import List, Tuple

from microscope.block.models import Block, BlockEdge
from microscope.block.prompts.ordering import build_ordering_prompt
from microscope.utils.io_utils import extract_json_from_text

logger = logging.getLogger(__name__)

VALID_EDGE_TYPES = {"PREREQUISITE_OF", "FOLLOWS", "CONTRASTS", "ELABORATES", "PARALLEL"}


class BlockOrderer:
    def __init__(self, api_provider):
        self.api_provider       = api_provider
        self.last_raw           = ""
        self.last_system_prompt = ""
        self.last_user_prompt   = ""

    def order(self, blocks: List[Block]) -> Tuple[List[BlockEdge], List[List[str]], str]:
        """Analyze dependencies between blocks.

        Returns:
            edges       : List[BlockEdge]
            paths       : List[List[str]] — recommended traversal paths
            rationale   : str — ordering explanation
        """
        if len(blocks) <= 1:
            return [], [[b.block_id for b in blocks]], ""

        system_prompt, user_prompt = build_ordering_prompt(blocks)
        raw = self._call_llm(system_prompt, user_prompt)
        self.last_system_prompt = system_prompt
        self.last_user_prompt   = user_prompt
        self.last_raw           = raw
        data = extract_json_from_text(raw)

        if not data:
            logger.warning("Block ordering LLM returned no valid JSON; using original order")
            return [], [[b.block_id for b in blocks]], "Original order preserved (LLM unavailable)"

        edges     = self._parse_edges(data.get("edges", []))
        paths     = data.get("recommended_paths", [[b.block_id for b in blocks]])
        rationale = data.get("ordering_rationale", "")

        self._apply_order_indices(blocks, paths)

        return edges, paths, rationale

    def _parse_edges(self, raw_edges: list) -> List[BlockEdge]:
        edges = []
        for e in raw_edges:
            edge_type = e.get("type", "FOLLOWS").upper()
            if edge_type not in VALID_EDGE_TYPES:
                logger.debug("Unknown edge type %r — skipping", edge_type)
                continue
            edges.append(BlockEdge(
                source      = e["source"],
                target      = e["target"],
                edge_type   = edge_type,
                description = e.get("description", ""),
                confidence  = float(e.get("confidence", 0.8)),
            ))
        return edges

    def _apply_order_indices(self, blocks: List[Block], paths: List[List[str]]) -> None:
        if not paths:
            return
        primary_path = paths[0]
        order_map = {bid: i for i, bid in enumerate(primary_path)}
        for block in blocks:
            if block.block_id in order_map:
                block.order_index = order_map[block.block_id]

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        return self.api_provider.chat_completion_text(messages=messages)
