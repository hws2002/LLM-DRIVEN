"""English documentation."""

import json
import os
from typing import Dict, List, Any
from pathlib import Path

class ConversationLoader:
    """English documentation."""

    def __init__(self, data_path: str = None):
        """English documentation."""
        if data_path is None:
            raise ValueError("data_path is required")
        self.data_path = Path(data_path)

    def load(self) -> List[Dict[str, Any]]:
        """English documentation."""
        print(f"text processing: {self.data_path}")

        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                conversations = json.load(f)

            print(f"text {len(conversations)}text text text.")
            return conversations

        except json.JSONDecodeError as e:
            raise ValueError(f"JSON text text: {e}")
        except Exception as e:
            raise RuntimeError(f"text processing text text: {e}")

    def load_sample(self, n: int = 10) -> List[Dict[str, Any]]:
        """English documentation."""
        return self.load()
        # return conversations[:min(len(conversations),n)]

    @staticmethod
    def get_conversation_info(conversation: Dict[str, Any]) -> Dict[str, Any]:
        """English documentation."""
        return {
            'title': conversation.get('title', 'text text'),
            'create_time': conversation.get('create_time'),
            'update_time': conversation.get('update_time'),
            'message_count': len(conversation.get('mapping', {}))
        }
