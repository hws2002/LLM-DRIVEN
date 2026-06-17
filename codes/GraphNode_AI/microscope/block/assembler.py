"""Per-block micro extraction and final BlockGraph assembly."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document

from microscope.block.models import Block, BlockEdge, BlockGraph, MicroGraph
from microscope.utils.document_utils import chunk_document

logger = logging.getLogger(__name__)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class BlockAssembler:

    def __init__(self, graph_generator, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.generator     = graph_generator
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_micro_graphs(
        self,
        blocks: List[Block],
        save_dir: Optional[Path] = None,
    ) -> None:
        """Runs micro extraction on each block in-place (fills block.micro_graph).

        If save_dir is provided, writes per-block intermediate files immediately
        after each block completes.
        """
        for block in blocks:
            logger.info("[assembler] Extracting micro graph for %s: %s", block.block_id, block.title)
            try:
                micro, inter = self._extract_one(block.raw_text)
                block.micro_graph = micro
            except Exception as exc:
                logger.warning("[assembler] Micro extraction failed for %s: %s", block.block_id, exc)
                block.micro_graph = MicroGraph()
                inter = {
                    "raw_text": block.raw_text,
                    "extracted": [], "standardized": [],
                    "name_mapping": {}, "raw_llm_outputs": [],
                    "error": str(exc),
                }

            if save_dir:
                self._save_block(save_dir / block.block_id, block, inter)
                logger.info("[assembler] Saved intermediates for %s → %s", block.block_id, save_dir / block.block_id)

    def _extract_one(self, text: str):
        doc    = Document(page_content=text)
        chunks = chunk_document([doc], self.chunk_size, self.chunk_overlap)

        extracted, raw_llm_outputs = self.generator.extract_entity_relation_from_chunks(chunks)
        standardized, name_mapping = self.generator.standardize_extracted_graph(
            extracted, existing_nodes=[]
        )

        all_nodes, all_edges = [], []
        seen_nodes: set[tuple] = set()
        for batch in standardized:
            for node in batch.get("nodes", []):
                key = (node.get("name", ""), node.get("type", ""))
                if key not in seen_nodes:
                    seen_nodes.add(key)
                    all_nodes.append(node)
            all_edges.extend(batch.get("edges", []))

        micro = MicroGraph(nodes=all_nodes, edges=all_edges)
        inter = {
            "raw_text":        text,
            "extracted":       extracted,
            "standardized":    standardized,
            "name_mapping":    name_mapping,
            "raw_llm_outputs": raw_llm_outputs,
        }
        return micro, inter

    @staticmethod
    def _save_block(blk_dir: Path, block: Block, inter: dict) -> None:
        block_summary = {
            "block_id":     block.block_id,
            "title":        block.title,
            "summary":      block.summary,
            "order_index":  block.order_index,
            "key_concepts": block.key_concepts,
            "turn_range":   block.turn_range,
            "heading_path": block.heading_path,
            "nodes": block.micro_graph.nodes if block.micro_graph else [],
            "edges": block.micro_graph.edges if block.micro_graph else [],
        }
        _write_json(blk_dir / "block_summary.json",    block_summary)
        _write_text(blk_dir / "raw_text.txt",          inter.get("raw_text", ""))
        _write_json(blk_dir / "extracted.json",        inter.get("extracted", []))
        _write_json(blk_dir / "standardized.json",     inter.get("standardized", []))
        _write_json(blk_dir / "name_mapping.json",     inter.get("name_mapping", {}))
        _write_json(blk_dir / "raw_llm_outputs.json",  inter.get("raw_llm_outputs", []))
        if "error" in inter:
            _write_text(blk_dir / "error.txt", inter["error"])

    @staticmethod
    def assemble(
        blocks: List[Block],
        edges: List[BlockEdge],
        paths: List[List[str]],
        source_type: str,
        ordering_rationale: str = "",
    ) -> BlockGraph:
        return BlockGraph(
            blocks             = blocks,
            edges              = edges,
            paths              = paths,
            source_type        = source_type,
            ordering_rationale = ordering_rationale,
        )
