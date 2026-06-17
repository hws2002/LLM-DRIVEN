"""English documentation."""

import json
import time
from pathlib import Path

from ..analyze.loader import ConversationLoader
from ..analyze.parser import QAPairParser
from ..utils.preprocess import preprocess_content
from ..utils import logger


def build_qa_pairs(conv_id: str, tmp_dir: Path = 'add_node/tmp') -> str:
    """English documentation."""

    start = time.time()

    # English comment.
    input_path = tmp_dir / f"conversation_{conv_id}.json"
    loader = ConversationLoader(data_path=str(input_path))

    load_t0 = time.time()
    conversations = loader.load_sample()
    load_elapsed = time.time() - load_t0

    parser = QAPairParser(min_answer_length=20, min_question_length=5)
    parse_t0 = time.time()
    qa_pairs = parser.parse_qa_pairs(conversations)
    parse_elapsed = time.time() - parse_t0

    # English comment.
    qa_pairs_file = tmp_dir / f"qa_pairs_{conv_id}.json"
    qa_pairs_file.parent.mkdir(parents=True, exist_ok=True)

    # English comment.
    processed_pairs = []
    for pair in qa_pairs:
        # English comment.
        cleaned_question = preprocess_content(pair['question'])
        cleaned_answer = preprocess_content(pair['answer'])

        # English comment.
        if len(cleaned_question) < 5 or len(cleaned_answer) < 10:
            continue

        processed_pairs.append({
            "qa_id": pair['qa_id'],
            "conversation_id": pair['conversation_id'],
            "conversation_title": pair['conversation_title'],
            "question": cleaned_question,
            "answer": cleaned_answer,
            "qa_index": pair['qa_index'],
            "timestamp": pair.get('timestamp')
        })

    with open(qa_pairs_file, "w", encoding="utf-8") as f:
        json.dump(processed_pairs, f, ensure_ascii=False, indent=2)

    total_elapsed = time.time() - start

    logger.info("=" * 60)
    logger.info("Q-A Pairs JSON Builder")
    logger.info("=" * 60)
    logger.info(f"Conversation loading: {load_elapsed:.2f}s")
    logger.info(f"Q-A parsing: {parse_elapsed:.2f}s")
    logger.info(f"Total time: {total_elapsed:.2f}s")
    logger.info(f"Total Q-A pairs: {len(processed_pairs)}")
    logger.info(f"Saved to: {qa_pairs_file}")

    return str(qa_pairs_file)
