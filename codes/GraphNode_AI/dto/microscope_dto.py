from dataclasses import dataclass
from typing import Optional


@dataclass
class ToMicroObjectContext:
    """English documentation."""

    file_path: str   # English comment.
    file_name: str   # English comment.
    user_id: str
    group_id: str
    source_id: Optional[str] = None
    schema_name: Optional[str] = None
