"""Base classes for pattern metrics collectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PatternEvidence:
    """Concrete evidence for a pattern."""

    node_ids: List[str]
    keywords: List[str]
    metric_values: Dict[str, Any]
    confidence: float  # 0-1
    description: str


class BaseMetricsCollector(ABC):
    """Base class for pattern metrics collectors."""

    def __init__(self, graph_store: "GraphStore"):
        """
        Initialize the metrics collector.

        Args:
            graph_store: GraphStore instance containing graph data
        """
        self.graph_store = graph_store

    @abstractmethod
    def collect(self) -> Dict[str, Any]:
        """
        Collect all metrics.

        Returns:
            Dictionary containing all collected metrics
        """
        pass

    @abstractmethod
    def find_evidence(self) -> List[PatternEvidence]:
        """
        Find pattern evidence with confidence scores.

        Returns:
            List of PatternEvidence objects
        """
        pass

    @abstractmethod
    def get_summary_for_llm(self) -> str:
        """
        Generate summary string for LLM prompt.

        Returns:
            Formatted string summarizing the metrics for LLM consumption
        """
        pass
