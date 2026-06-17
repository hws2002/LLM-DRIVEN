"""Shared logger helpers."""

from __future__ import annotations

import logging
import sys


def get_shared_logger(name: str) -> logging.Logger:
    """Return a logger with a shared default console configuration."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
    return logging.getLogger(name)

