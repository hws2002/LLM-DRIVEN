"""Data schemas for graph summarization and insights."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OverviewSection:
    """High-level overview of the conversation graph."""

    total_source_nodes: int
    time_span: str  # "2024-01 ~ 2025-01" or "N/A"
    primary_interests: List[str]  # Top 3-5 topics
    conversation_style: str  # "technical deep-dive", "exploratory", etc.
    most_active_period: str  # English comment.
    summary_text: str  # LLM-generated natural language summary


@dataclass
class ClusterAnalysis:
    """Detailed analysis of a single cluster."""

    cluster_id: str
    name: str
    size: int

    # Computed metrics
    density: float  # Internal edge density (0-1)
    centrality: float  # How connected to other clusters (0-1)
    recency: str  # "active", "dormant", "new", "unknown"

    # Content analysis
    top_keywords: List[str]
    key_themes: List[str]
    common_question_types: List[str]  # ["debugging", "concept explanation", "comparison"]

    # LLM insights
    insight_text: str
    notable_conversations: List[str] = field(default_factory=list)  # Node IDs


@dataclass
class Pattern:
    """Identified pattern across the graph."""

    pattern_type: str  # "repetition", "progression", "gap"
    description: str
    evidence: List[str]  # Node IDs supporting this pattern
    significance: str  # "high", "medium", "low"


@dataclass
class ClusterConnection:
    """Connection between two clusters."""

    source_cluster: str
    target_cluster: str
    connection_strength: float  # 0-1
    bridge_keywords: List[str]  # Keywords that connect them
    description: str


@dataclass
class Recommendation:
    """Actionable recommendation for the user."""

    type: str  # "consolidate", "explore", "review", "connect"
    title: str
    description: str
    related_nodes: List[str]  # Node IDs
    priority: str  # "high", "medium", "low"


@dataclass
class GraphSummary:
    """Complete graph summary with insights."""

    # Overview
    overview: OverviewSection

    # Cluster analyses
    clusters: List[ClusterAnalysis]

    # Cross-cutting insights
    patterns: List[Pattern]
    connections: List[ClusterConnection]

    # Actionable items
    recommendations: List[Recommendation]

    # Meta
    generated_at: str
    detail_level: str  # "brief", "standard", "detailed"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        from dataclasses import asdict
        return asdict(self)

    def __str__(self) -> str:
        """Format as readable text."""
        lines = []
        lines.append("=" * 60)
        lines.append("📊 GRAPH SUMMARY")
        lines.append("=" * 60)
        lines.append("")

        # Overview
        lines.append("## Overview")
        lines.append(f"Total Source Nodes: {self.overview.total_source_nodes}")
        lines.append(f"Time Span: {self.overview.time_span}")
        lines.append(f"Primary Interests: {', '.join(self.overview.primary_interests)}")
        lines.append(f"Conversation Style: {self.overview.conversation_style}")
        lines.append("")
        lines.append(self.overview.summary_text)
        lines.append("")

        # Clusters
        lines.append("## Clusters")
        for cluster in self.clusters:
            lines.append(f"\n### {cluster.name} ({cluster.cluster_id})")
            lines.append(f"Size: {cluster.size} | Density: {cluster.density:.2f} | Recency: {cluster.recency}")
            lines.append(f"Keywords: {', '.join(cluster.top_keywords[:5])}")
            lines.append(f"\n{cluster.insight_text}")

        # Patterns
        if self.patterns:
            lines.append("\n## Patterns")
            for pattern in self.patterns:
                lines.append(f"\n- [{pattern.significance.upper()}] {pattern.pattern_type}")
                lines.append(f"  {pattern.description}")

        # Recommendations
        if self.recommendations:
            lines.append("\n## Recommendations")
            for rec in self.recommendations:
                lines.append(f"\n[{rec.priority.upper()}] {rec.title}")
                lines.append(f"  {rec.description}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
