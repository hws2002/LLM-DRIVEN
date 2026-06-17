"""BLOCK-based graph data models for consciousness-flow-aware microscope."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class MicroGraph:
    nodes: List[dict] = field(default_factory=list)
    edges: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"nodes": self.nodes, "edges": self.edges}


@dataclass
class Block:
    block_id: str
    title: str
    summary: str
    key_concepts: List[str]
    raw_text: str
    order_index: int
    source_type: str                              # "chat" | "note"
    turn_range: Optional[Tuple[int, int]] = None  # English comment.
    heading_path: Optional[List[str]] = None      # English comment.
    micro_graph: Optional[MicroGraph] = None      # English comment.

    def to_dict(self) -> dict:
        d = {
            "block_id":    self.block_id,
            "title":       self.title,
            "summary":     self.summary,
            "key_concepts":self.key_concepts,
            "raw_text":    self.raw_text,
            "order_index": self.order_index,
            "source_type": self.source_type,
            "micro_graph": self.micro_graph.to_dict() if self.micro_graph else {"nodes": [], "edges": []},
        }
        if self.turn_range is not None:
            d["turn_range"] = list(self.turn_range)
        if self.heading_path is not None:
            d["heading_path"] = self.heading_path
        return d


@dataclass
class BlockEdge:
    source: str       # block_id
    target: str       # block_id
    edge_type: str    # PREREQUISITE_OF | FOLLOWS | CONTRASTS | ELABORATES | PARALLEL
    description: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "source":      self.source,
            "target":      self.target,
            "type":        self.edge_type,
            "description": self.description,
            "confidence":  self.confidence,
        }


@dataclass
class BlockGraph:
    blocks: List[Block]
    edges: List[BlockEdge]
    paths: List[List[str]]   # English comment.
    source_type: str         # "chat" | "note"
    ordering_rationale: str = ""

    def to_dict(self) -> dict:
        sorted_blocks = sorted(self.blocks, key=lambda b: b.order_index)
        return {
            "source_type": self.source_type,
            "block_graph": {
                "blocks":              [b.to_dict() for b in sorted_blocks],
                "edges":               [e.to_dict() for e in self.edges],
                "paths":               self.paths,
                "ordering_rationale":  self.ordering_rationale,
            },
        }
