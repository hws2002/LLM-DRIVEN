"""Configuration management for GraphNode Insights."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, validator


class VectorStoreSettings(BaseModel):
    """Vector store configuration settings."""

    persist_directory: str = Field(
        default="./vector_db",
        description="Directory for ChromaDB persistence"
    )
    collection_name: str = Field(
        default="conversation_embeddings",
        description="ChromaDB collection name"
    )
    embedding_dimension: int = Field(
        default=384,
        description="Embedding vector dimension"
    )


class SearchSettings(BaseModel):
    """Search configuration settings."""

    default_top_k: int = Field(
        default=10,
        description="Default number of search results"
    )
    similarity_threshold: float = Field(
        default=0.5,
        description="Minimum similarity score for results"
    )
    semantic_weight: float = Field(
        default=0.6,
        description="Weight for semantic similarity in scoring"
    )
    keyword_weight: float = Field(
        default=0.25,
        description="Weight for keyword overlap in scoring"
    )
    recency_weight: float = Field(
        default=0.1,
        description="Weight for recency bonus in scoring"
    )
    cluster_match_weight: float = Field(
        default=0.05,
        description="Weight for cluster match bonus in scoring"
    )

    @validator("similarity_threshold")
    def validate_threshold(cls, value: float) -> float:
        """Ensure threshold is between 0 and 1."""
        if not 0 <= value <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        return value

    @validator("semantic_weight", "keyword_weight", "recency_weight", "cluster_match_weight")
    def validate_weight(cls, value: float) -> float:
        """Ensure weight is between 0 and 1."""
        if not 0 <= value <= 1:
            raise ValueError("weight must be between 0 and 1")
        return value


class LLMSettings(BaseModel):
    """LLM configuration settings."""

    default_provider: str = Field(
        default="openai",
        description="Default LLM provider (openai, qwen, groq, gemini)"
    )
    default_model: str = Field(
        default="gpt-4o-mini",
        description="Default model name"
    )
    max_tokens: int = Field(
        default=4000,
        description="Maximum tokens for LLM responses"
    )
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature"
    )
    batch_size: int = Field(
        default=10,
        description="Batch size for bulk LLM operations"
    )

    @validator("default_provider")
    def validate_provider(cls, value: str) -> str:
        """Ensure provider is supported."""
        allowed = {"openai", "qwen", "groq", "gemini"}
        if value not in allowed:
            raise ValueError(f"provider must be one of {allowed}")
        return value


class PathSettings(BaseModel):
    """Path configuration settings."""

    graph_path: Optional[str] = Field(
        default=None,
        description="Default path to graph JSON file"
    )
    vector_db_path: Optional[str] = Field(
        default=None,
        description="Default path to vector database"
    )
    output_directory: str = Field(
        default="./output",
        description="Default output directory"
    )


class InsightsConfig(BaseModel):
    """Main configuration for GraphNode Insights."""

    vector_store: VectorStoreSettings = Field(
        default_factory=VectorStoreSettings,
        description="Vector store settings"
    )
    search: SearchSettings = Field(
        default_factory=SearchSettings,
        description="Search settings"
    )
    llm: LLMSettings = Field(
        default_factory=LLMSettings,
        description="LLM settings"
    )
    paths: PathSettings = Field(
        default_factory=PathSettings,
        description="Path settings"
    )

    # Embedding model configuration
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Embedding model name"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose output"
    )

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "InsightsConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            InsightsConfig instance

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML is invalid
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                data = {}

            return cls(**data)

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}") from e

    @classmethod
    def from_env(cls) -> "InsightsConfig":
        """Load configuration from environment variables.

        Environment variables follow the pattern: INSIGHTS_<SECTION>_<KEY>
        Example: INSIGHTS_VECTOR_STORE_PERSIST_DIRECTORY

        Returns:
            InsightsConfig instance
        """
        config_dict: Dict[str, Any] = {
            "vector_store": {},
            "search": {},
            "llm": {},
            "paths": {}
        }

        # Vector store settings
        if val := os.getenv("INSIGHTS_VECTOR_STORE_PERSIST_DIRECTORY"):
            config_dict["vector_store"]["persist_directory"] = val
        if val := os.getenv("INSIGHTS_VECTOR_STORE_COLLECTION_NAME"):
            config_dict["vector_store"]["collection_name"] = val
        if val := os.getenv("INSIGHTS_VECTOR_STORE_EMBEDDING_DIMENSION"):
            config_dict["vector_store"]["embedding_dimension"] = int(val)

        # Search settings
        if val := os.getenv("INSIGHTS_SEARCH_DEFAULT_TOP_K"):
            config_dict["search"]["default_top_k"] = int(val)
        if val := os.getenv("INSIGHTS_SEARCH_SIMILARITY_THRESHOLD"):
            config_dict["search"]["similarity_threshold"] = float(val)

        # LLM settings
        if val := os.getenv("INSIGHTS_LLM_DEFAULT_PROVIDER"):
            config_dict["llm"]["default_provider"] = val
        if val := os.getenv("INSIGHTS_LLM_DEFAULT_MODEL"):
            config_dict["llm"]["default_model"] = val
        if val := os.getenv("INSIGHTS_LLM_MAX_TOKENS"):
            config_dict["llm"]["max_tokens"] = int(val)
        if val := os.getenv("INSIGHTS_LLM_TEMPERATURE"):
            config_dict["llm"]["temperature"] = float(val)

        # Path settings
        if val := os.getenv("INSIGHTS_PATHS_GRAPH_PATH"):
            config_dict["paths"]["graph_path"] = val
        if val := os.getenv("INSIGHTS_PATHS_VECTOR_DB_PATH"):
            config_dict["paths"]["vector_db_path"] = val
        if val := os.getenv("INSIGHTS_PATHS_OUTPUT_DIRECTORY"):
            config_dict["paths"]["output_directory"] = val

        # General settings
        if val := os.getenv("INSIGHTS_EMBEDDING_MODEL"):
            config_dict["embedding_model"] = val
        if val := os.getenv("INSIGHTS_LOG_LEVEL"):
            config_dict["log_level"] = val
        if val := os.getenv("INSIGHTS_VERBOSE"):
            config_dict["verbose"] = val.lower() in ("true", "1", "yes")

        return cls(**config_dict)

    @classmethod
    def load(
        cls,
        yaml_path: Optional[Path] = None,
        use_env: bool = True
    ) -> "InsightsConfig":
        """Load configuration with priority: YAML > ENV > Defaults.

        Args:
            yaml_path: Optional path to YAML file
            use_env: Whether to override with environment variables

        Returns:
            InsightsConfig instance
        """
        # Start with defaults
        config = cls()

        # Load from YAML if provided
        if yaml_path:
            config = cls.from_yaml(yaml_path)

        # Override with environment variables if enabled
        if use_env:
            env_config = cls.from_env()
            # Merge env_config into config
            config = cls(**{**config.dict(), **env_config.dict(exclude_unset=True)})

        return config

    def to_yaml(self, yaml_path: Path) -> None:
        """Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML file
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                self.dict(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False
            )


def get_default_config() -> InsightsConfig:
    """Get default configuration instance.

    Returns:
        InsightsConfig with default values
    """
    return InsightsConfig()
