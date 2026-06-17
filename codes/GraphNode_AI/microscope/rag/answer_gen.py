"""LLM response for RAG services."""

import os
from typing import Optional

from shared.api_provider import ApiProvider, normalize_provider


def get_response(rag_prompt: str, api_provider: Optional[ApiProvider] = None) -> str:
    """Send a prompt. Uses the provided api_provider, or builds one from env."""
    if api_provider is None:
        llm_provider = normalize_provider(os.getenv("MICROSCOPE_LLM_PROVIDER", "groq"))
        env_key = "DEV_" + llm_provider.upper().replace(".", "") + "_API_KEY"
        api_provider = ApiProvider(
            provider=llm_provider,
            model=os.getenv("MICROSCOPE_LLM_MODEL", "llama-3.3-70b-versatile"),
            temperature=float(os.getenv("MICROSCOPE_LLM_TEMPERATURE", "0")),
            timeout_seconds=60.0,
            api_key=os.getenv(env_key, os.getenv("OPENAI_API_KEY", "")),
            max_tokens=int(os.getenv("MICROSCOPE_LLM_MAX_TOKENS", "8000")),
        )
    return api_provider.chat_completion_text(
        messages=[{"role": "user", "content": rag_prompt}],
    )
