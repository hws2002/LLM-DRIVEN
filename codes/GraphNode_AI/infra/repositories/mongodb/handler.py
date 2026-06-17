from __future__ import annotations

from typing import Any, Dict, List, Optional

from pymongo import MongoClient, ASCENDING


class MongoDBHandler:
    CONVERSATIONS = "conversations"
    MESSAGES = "messages"
    NOTES = "notes"

    def __init__(self, url: str, db_name: str = "test") -> None:
        self._client = MongoClient(url)
        self._db = self._client[db_name]

    def get_conversation_messages(self, conversation_id: str, user_id: str) -> List[Dict[str, Any]]:
        """English documentation."""
        cursor = self._db[self.MESSAGES].find(
            {
                "conversationId": conversation_id,
                "ownerUserId": user_id,
                "deletedAt": None,
            },
            {"role": 1, "content": 1, "createdAt": 1, "_id": 0},
        ).sort("createdAt", ASCENDING)
        return list(cursor)

    def get_note(self, note_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """English documentation."""
        return self._db[self.NOTES].find_one(
            {"_id": note_id, "ownerUserId": user_id, "deletedAt": None},
            {"title": 1, "content": 1, "_id": 0},
        )

    def close(self) -> None:
        self._client.close()
