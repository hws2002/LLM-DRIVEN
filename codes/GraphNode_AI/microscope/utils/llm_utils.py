from shared.api_provider import ApiProvider
import time
import logging
from typing import List, Dict

logger = logging.getLogger("microscope")


def llm_call(
    api_provider: ApiProvider,
    system_prompt: str,
    user_prompt: str,
    messages: List[Dict[str, str]] | None = None,
) -> str:
    """English documentation."""
    max_retries = 2
    backoff_sec = 2
    for attempt in range(1, max_retries + 1):
        try:
            if messages is None:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            return api_provider.chat_completion_text(
                messages=messages,
            )
        except Exception as exc:
            if attempt == max_retries:
                raise
            sleep_for = backoff_sec * attempt
            logger.warning(
                "LLM call failed (attempt %s/%s): %s. Retrying in %ss",
                attempt,
                max_retries,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)
    return ""
