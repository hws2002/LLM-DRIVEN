"""Pydantic schemas for vector storage and search results."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field, validator


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""

    persist_directory: str = Field(
        ..., description="Directory path for ChromaDB persistence"
    )
    collection_name: str = Field(
        default="conversation_embeddings",
        description="Name of the ChromaDB collection"
    )
    embedding_dimension: int = Field(
        default=384, description="Dimension of embedding vectors"
    )

    @validator("persist_directory")
    def validate_directory(cls, value: str) -> str:
        """Ensure directory path is not empty."""
        if not value or not value.strip():
            raise ValueError("persist_directory cannot be empty")
        return value.strip()

    @validator("embedding_dimension")
    def validate_dimension(cls, value: int) -> int:
        """Ensure embedding dimension is positive."""
        if value <= 0:
            raise ValueError("embedding_dimension must be positive")
        return value


class EmbeddingRecord(BaseModel):
    """Record for storing embedding with metadata."""

    id: str = Field(..., description="Unique identifier for the record")
    embedding: List[float] = Field(
        ..., description="Embedding vector as list of floats"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Associated metadata"
    )

    @validator("id")
    def validate_id(cls, value: str) -> str:
        """Ensure ID is not empty."""
        if not value or not value.strip():
            raise ValueError("id cannot be empty")
        return value.strip()

    @validator("embedding")
    def validate_embedding(cls, value: List[float]) -> List[float]:
        """Ensure embedding is not empty."""
        if not value:
            raise ValueError("embedding cannot be empty")
        return value

    @classmethod
    def from_numpy(
        cls,
        id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "EmbeddingRecord":
        """Create record from numpy array."""
        return cls(
            id=id,
            embedding=embedding.tolist(),
            metadata=metadata or {}
        )

    def to_numpy(self) -> np.ndarray:
        """Convert embedding to numpy array."""
        return np.array(self.embedding, dtype=np.float32)


class SearchResult(BaseModel):
    """Result from vector similarity search."""

    id: str = Field(..., description="ID of the matched record")
    score: float = Field(..., description="Similarity score (0-1, higher is better)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata of the matched record"
    )

    @validator("score")
    def validate_score(cls, value: float) -> float:
        """Ensure score is between 0 and 1."""
        if not 0 <= value <= 1:
            raise ValueError("score must be between 0 and 1")
        return value

    class Config:
        """Pydantic config."""
        validate_assignment = True
