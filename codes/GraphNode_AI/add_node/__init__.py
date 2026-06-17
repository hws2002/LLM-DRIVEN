"""add_node package - single conversation node addition pipeline."""

from __future__ import annotations

from typing import Any

__all__ = ["run_add_node_pipeline", "run_add_node_batch_pipeline"]


def __getattr__(name: str) -> Any:
    if name in {"run_add_node_pipeline", "run_add_node_batch_pipeline"}:
        from .call import run_add_node_batch_pipeline, run_add_node_pipeline

        mapping = {
            "run_add_node_pipeline": run_add_node_pipeline,
            "run_add_node_batch_pipeline": run_add_node_batch_pipeline,
        }
        return mapping[name]
    raise AttributeError(f"module 'add_node' has no attribute '{name}'")
