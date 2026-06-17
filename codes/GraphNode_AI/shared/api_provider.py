"""Unified LangChain-based API provider module.

Supported providers:
- openai
- z.ai
- groq
- openrouter
"""

from __future__ import annotations

import logging
import re
import urllib.request
import urllib.error
import json
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

import shared.config as cfg


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    default_base_url: Optional[str]
    fallback_model: str


_PROVIDERS = {
    "openai": ProviderConfig(
        provider="openai",
        default_base_url=cfg.OPENAI_BASE_URL,
        fallback_model="gpt-4o-mini",
    ),
    "z.ai": ProviderConfig(
        provider="z.ai",
        default_base_url=cfg.ZAI_BASE_URL,
        fallback_model="glm-4-flash-250414",
    ),
    "groq": ProviderConfig(
        provider="groq",
        default_base_url=cfg.GROQ_BASE_URL,
        fallback_model="llama-3.3-70b-versatile",
    ),
    "openrouter": ProviderConfig(
        provider="openrouter",
        default_base_url=cfg.OPENROUTER_BASE_URL,
        fallback_model="nvidia/nemotron-3-super-120b-a12b:free",
    ),
}

_ALIASES = {
    "zai": "z.ai",
    "z.ai": "z.ai",
    "openai": "openai",
    "groq": "groq",
    "openrouter": "openrouter",
}

def check_openrouter_key(api_key: str) -> dict | None:
    """English documentation."""
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())["data"]
        remaining = data.get("limit_remaining")
        limit     = data.get("limit")
        usage     = data.get("usage", 0)
        daily     = data.get("usage_daily", 0)
        free_tier = data.get("is_free_tier", True)
        tier_tag  = "free-tier" if free_tier else "paid"
        if remaining is None:
            logger.info("[OpenRouter] %s | usage=%.4f$ (today=%.4f$) | limit=unlimited", tier_tag, usage, daily)
        else:
            logger.info(
                "[OpenRouter] %s | remaining=%.4f$ / %.4f$ | usage=%.4f$ (today=%.4f$)",
                tier_tag, remaining, limit or 0, usage, daily,
            )
        return data
    except Exception as exc:
        logger.warning("[OpenRouter] key check failed: %s", exc)
        return None


def normalize_provider(provider: str) -> str:
    key = provider.strip().lower()
    return _ALIASES.get(key, key)

def _is_reasoning_model(model: str) -> bool:
    """English documentation."""
    m = model.strip().lower()
    return bool(re.match(r"o\d", m)) or bool(re.match(r"gpt-5", m))

class ApiProvider:
    """Per-request LangChain-based API provider. Instantiate once per request."""

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
        max_tokens: int = 8000,
        api_key: str,
    ) -> None:
        self.provider = normalize_provider(provider or "")
        cfg = _PROVIDERS[self.provider]
        self.model = model or cfg.fallback_model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        _reasoning = _is_reasoning_model(self.model)
        if self.provider == "groq":
            from langchain_groq import ChatGroq

            self.client = ChatGroq(
                api_key=api_key,
                model=self.model,
                temperature=self.temperature,
                timeout=self.timeout_seconds,
            )
        else:
            from langchain_openai import ChatOpenAI

            self.client = ChatOpenAI(
                api_key=api_key,
                base_url=cfg.default_base_url,
                model=self.model,
                temperature=None if _reasoning else self.temperature,
                timeout=self.timeout_seconds if not _reasoning else max(self.timeout_seconds, 300.0),
            )

    def set_maxtokens(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens

    def chat_completion_text(
        self,
        messages: List[dict],
        max_tokens: Optional[int] = None,
    ) -> str:
        """Invoke chat model and return text response."""
        lc_messages = self._to_langchain_messages(messages)
        # English comment.
        # English comment.
        # English comment.
        if _is_reasoning_model(self.model):
            bind_kwargs: dict = {"max_tokens": 32000}
        else:
            bind_kwargs = {
                "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
                "temperature": self.temperature,
            }
        response = self.client.bind(**bind_kwargs).invoke(lc_messages)
        content = getattr(response, "content", "")
        return content if isinstance(content, str) else str(content)

    @staticmethod
    def _to_langchain_messages(messages: List[dict]) -> List[BaseMessage]:
        out: List[BaseMessage] = []
        for m in messages:
            role = (m.get("role") or "").lower()
            content = m.get("content", "")
            if role == "system":
                out.append(SystemMessage(content=content))
            elif role == "assistant":
                out.append(AIMessage(content=content))
            else:
                out.append(HumanMessage(content=content))
        return out
