"""English documentation."""

from typing import Dict, List, Any
import re


class QAPairParser:
    """English documentation."""

    def __init__(self, min_answer_length: int = 20, min_question_length: int = 5):
        """English documentation."""
        self.min_answer_length = min_answer_length
        self.min_question_length = min_question_length

    def parse_qa_pairs(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """English documentation."""
        qa_pairs = []

        for conv_idx, conversation in enumerate(conversations, start=1):
            try:
                pairs = self._extract_qa_pairs_from_conversation(
                    conversation,
                    conversation_id=conv_idx
                )
                qa_pairs.extend(pairs)
            except Exception as e:
                print(f"text: text {conv_idx} Q-A text text text - {e}")
                continue

        print(f"text {len(qa_pairs)}text Q-A text text.")
        return qa_pairs

    def _extract_qa_pairs_from_conversation(
        self,
        conversation: Dict[str, Any],
        conversation_id: int
    ) -> List[Dict[str, Any]]:
        """English documentation."""
        title = conversation.get('title', f'text {conversation_id}')
        messages_raw = conversation.get('messages', [])

        if messages_raw:
            # MongoDB format: messages[]
            messages = []
            for m in messages_raw:
                if m.get('role') in ('user', 'assistant') and m.get('content'):
                    messages.append({
                        'role': m['role'],
                        'content': str(m['content']).strip(),
                        'create_time': m.get('createdAt', 0),
                    })
            messages.sort(key=lambda x: x['create_time'])
        else:
            # fallback: ChatGPT mapping format
            mapping = conversation.get('mapping', {})
            messages = self._extract_messages(mapping)

        # English comment.
        qa_pairs = []
        qa_index = 0

        i = 0
        while i < len(messages):
            # English comment.
            if messages[i]['role'] == 'user':
                question = messages[i]['content']

                # English comment.
                if i + 1 < len(messages) and messages[i + 1]['role'] == 'assistant':
                    answer = messages[i + 1]['content']

                    # English comment.
                    if len(question) >= self.min_question_length and len(answer) >= self.min_answer_length:
                        qa_id = f"{conversation_id}_{qa_index}"

                        qa_pairs.append({
                            'qa_id': qa_id,
                            'conversation_id': conversation_id,
                            'conversation_title': title,
                            'question': question,
                            'answer': answer,
                            'qa_index': qa_index,
                            'timestamp': messages[i + 1].get('create_time')
                        })

                        qa_index += 1

                    i += 2  # English comment.
                else:
                    i += 1
            else:
                i += 1

        return qa_pairs

    def _extract_messages(self, mapping: Dict[str, Any]) -> List[Dict[str, str]]:
        """English documentation."""
        messages = []

        for node_id, node_data in mapping.items():
            message = node_data.get('message')

            if not message:
                continue

            # English comment.
            author = message.get('author', {})
            role = author.get('role')

            # English comment.
            if role not in ['user', 'assistant']:
                continue

            content_obj = message.get('content', {})
            parts = content_obj.get('parts', [])

            # English comment.
            if not parts or not parts[0]:
                continue

            text = str(parts[0]).strip()
            text = self._remove_emoji(text)

            if not text:
                continue

            messages.append({
                'role': role,
                'content': text,
                'create_time': message.get('create_time', 0)
            })

        # English comment.
        messages.sort(key=lambda x: x['create_time'])

        return messages

    def _remove_emoji(self, text: str) -> str:
        """English documentation."""
        # English comment.
        try:
            emoji_pattern = re.compile(r"[\U0001F300-\U0001FAFF\U0001F600-\U0001F64F\u2600-\u27BF]+", flags=re.UNICODE)
            text = emoji_pattern.sub("", text)
        except re.error:
            text = re.sub(r"[\u2600-\u27BF]+", "", text)
        return text.strip()
