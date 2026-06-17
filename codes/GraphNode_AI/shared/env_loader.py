"""Shared environment loader for GraphNode_AI.

Single source of truth: graphnode_ai/.env
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_root_env(*, override: bool = False) -> None:
    """Load environment variables from graphnode_ai/.env."""
    root_env = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=root_env, override=override)
