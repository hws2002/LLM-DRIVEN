"""English documentation."""

import logging
import sys
from datetime import datetime

# English comment.
logger = logging.getLogger("add_node")
logger.setLevel(logging.DEBUG)
logger.propagate = False  # English comment.

# English comment.
if not logger.handlers:
    # English comment.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # English comment.
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

# English comment.
def info(msg: str):
    logger.info(msg)

def debug(msg: str):
    logger.debug(msg)

def warning(msg: str):
    logger.warning(msg)

def error(msg: str):
    logger.error(msg)
