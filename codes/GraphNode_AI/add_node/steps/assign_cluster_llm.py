"""English documentation."""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..utils import logger
from ..utils.prompt_builder import build_cluster_prompt, fallback_cluster_assignment
from shared.api_provider import ApiProvider

if TYPE_CHECKING:
    from shared.token_usage import TokenUsageTracker


def assign_cluster_with_llm(
    existing_clusters: List[Dict[str, Any]],
    conversation_keywords: List[str],
    conversation_title: str = "",
    api_provider: Optional[ApiProvider] = None,
    tracker: Optional["TokenUsageTracker"] = None,
) -> Dict[str, Any]:
    """English documentation."""
    logger.info(f"Assigning cluster for conversation with {len(conversation_keywords)} keywords")

    # English comment.
    if not existing_clusters:
        logger.info("No existing clusters. Will create new cluster.")
        return {
            "cluster_id": "NEW_CLUSTER",
            "confidence": 1.0,
            "reasoning": "No existing clusters available",
            "is_new_cluster": True
        }

    if api_provider is None:
        logger.warning("No ApiProvider provided. Using fallback clustering.")
        return fallback_cluster_assignment(
            existing_clusters,
            conversation_keywords,
            use_embedding_fallback=True,
        )

    logger.info(f"Using LLM provider: {api_provider.provider}, model: {api_provider.model}")

    try:
        prompt = build_cluster_prompt(
            existing_clusters,
            conversation_keywords,
            conversation_title
        )

        system_prompt = "You are a precise JSON-only responder."
        content = api_provider.chat_completion_text(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
        )
        content = content.strip()

        if tracker is not None:
            tracker.record_call(
                stage="assign_cluster",
                system_prompt=system_prompt,
                user_prompt=prompt,
                response=content,
                max_tokens=4096,
                temperature=api_provider.temperature,
            )

        # English comment.
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        result = json.loads(content)

        is_new = result.get("cluster_id") == "NEW_CLUSTER"

        logger.info(
            "LLM assigned cluster: %s (confidence: %s, provider: %s)",
            result.get("cluster_id"),
            result.get("confidence"),
            api_provider.provider,
        )

        return {
            "cluster_id": result.get("cluster_id"),
            "confidence": result.get("confidence", 0.5),
            "reasoning": result.get("reasoning", ""),
            "is_new_cluster": is_new
        }

    except Exception as e:
        logger.error(f"LLM cluster assignment failed: {e}")
        return fallback_cluster_assignment(
            existing_clusters,
            conversation_keywords,
            use_embedding_fallback=True,
        )
