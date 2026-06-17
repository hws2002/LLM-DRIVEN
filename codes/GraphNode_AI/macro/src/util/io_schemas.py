"""Pydantic schemas for validating chat graph input and output."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator


class Section(BaseModel):
    """Single section of content (can be a chat message or a markdown section)."""

    id: str
    content: str
    role: Optional[str] = None  # For ChatGPT backward compat
    section_title: Optional[str] = None  # For markdown headings


class SourceNode(BaseModel):
    """A source node containing multiple sections (can be chat conversation or markdown note)."""

    id: str
    title: Optional[str] = None
    sections: List[Section]
    source_type: str = "chat"  # "chat" | "markdown" | "notion"
    create_time: Optional[int] = None
    update_time: Optional[int] = None

    def get_merged_content(self) -> str:
        """Merge all section contents into a single text."""
        merged_parts = []
        for section in self.sections:
            if section.role:
                merged_parts.append(f"{section.role}: {section.content}")
            else:
                merged_parts.append(section.content)
        return "\n".join(merged_parts)


class InputData(BaseModel):
    """Input payload: list of sections or source nodes."""

    sections: List[Section] = Field(default_factory=list)
    source_nodes: List[SourceNode] = Field(default_factory=list)

    @classmethod
    def from_raw(cls, payload: List[dict]) -> "InputData":
        return cls(sections=[Section(**item) for item in payload])

    @classmethod
    def from_source_nodes(cls, source_nodes: List[SourceNode]) -> "InputData":
        return cls(source_nodes=source_nodes)

    @classmethod
    def from_conversations(cls, conversations: List[SourceNode]) -> "InputData":
        """Deprecated alias for from_source_nodes."""
        return cls.from_source_nodes(conversations)


class Keyword(BaseModel):
    term: str
    score: float


class GraphNode(BaseModel):
    id: int
    orig_id: str = Field(..., alias="orig_id")
    role: str
    text: str
    create_time: Optional[int] = None
    update_time: Optional[int] = None
    cluster: int
    keywords: List[Keyword]
    num_sections: int = 0
    source_type: str = "chat"
    message_ids: List[str] = Field(default_factory=list)


class Edge(BaseModel):
    source: int
    target: int
    weight: float
    type: str


class KeywordParams(BaseModel):
    top_n: int
    max_ngram: int
    dedup_thresh: float


class GraphParams(BaseModel):
    sim_top_k: Optional[int]
    sim_threshold: Optional[float]


class ClusterParams(BaseModel):
    min_cluster_size: int
    min_samples: int
    metric: str


class PreprocessParams(BaseModel):
    lower: bool
    strip_urls: bool
    strip_code: bool
    strip_punct: bool
    stopwords_langs: List[str]


class Params(BaseModel):
    embedding_model: str
    embedding_model_digest: str
    keyword: KeywordParams
    cluster: ClusterParams
    graph: GraphParams
    preprocess: PreprocessParams


class ClusterSummary(BaseModel):
    size: int
    top_terms: List[Tuple[str, float]]


class Counts(BaseModel):
    nodes: int
    edges: int
    clusters: int
    outliers: int


class Metadata(BaseModel):
    clusters: Dict[str, ClusterSummary]
    params: Params
    counts: Counts


class OutputGraph(BaseModel):
    nodes: List[GraphNode]
    edges: List[Edge]
    metadata: Metadata

    @validator("edges", each_item=True)
    def validate_edge_type(cls, value: Edge) -> Edge:
        if value.type != "similarity":
            raise ValueError('edge type must be "similarity"')
        return value
