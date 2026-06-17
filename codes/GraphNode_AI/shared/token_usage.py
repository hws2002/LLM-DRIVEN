from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tiktoken
except ImportError as exc:
    raise ImportError(
        "tiktoken is required for token counting. Install it with: pip install tiktoken"
    ) from exc


@dataclass
class TokenUsageRecord:
    stage: str
    call_index: int
    provider_name: str
    model_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    system_tokens: int
    user_tokens: int
    response_tokens: int
    usage_source: str
    max_tokens: int
    temperature: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageTokenSummary:
    stage: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_calls: int = 0
    provider_reported_calls: int = 0

    def add(self, record: TokenUsageRecord) -> None:
        self.calls += 1
        self.input_tokens += record.input_tokens
        self.output_tokens += record.output_tokens
        self.total_tokens += record.total_tokens
        if record.usage_source == "provider":
            self.provider_reported_calls += 1
        else:
            self.estimated_calls += 1


class TokenCounter:
    """
    Token counter using tiktoken.
    For non-OpenAI models, exact billing tokens may differ unless provider usage
    metadata is returned by the API and passed into the tracker.
    """

    def __init__(self, model_name: str, fallback_encoding: str = "o200k_base"):
        self.model_name = model_name
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding(fallback_encoding)

    def count_text(self, text: Optional[str]) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def count_prompt(self, system_prompt: str, user_prompt: str) -> Dict[str, int]:
        system_tokens = self.count_text(system_prompt)
        user_tokens = self.count_text(user_prompt)
        return {
            "system_tokens": system_tokens,
            "user_tokens": user_tokens,
            "input_tokens": system_tokens + user_tokens,
        }

    def count_response(self, response: Optional[str]) -> int:
        return self.count_text(response)


class TokenUsageTracker:
    def __init__(
        self,
        model_name: str,
        provider_name: str = "unknown",
        fallback_encoding: str = "cl100k_base",
    ):
        self.model_name = model_name
        self.provider_name = provider_name
        self.counter = TokenCounter(
            model_name=model_name,
            fallback_encoding=fallback_encoding,
        )
        self.records: List[TokenUsageRecord] = []
        self._call_index = 0

    @staticmethod
    def _normalize_provider_usage(
        provider_usage: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, int]]:
        if not provider_usage:
            return None

        prompt_tokens = provider_usage.get("prompt_tokens")
        completion_tokens = provider_usage.get("completion_tokens")

        if prompt_tokens is None:
            prompt_tokens = provider_usage.get("input_tokens")
        if completion_tokens is None:
            completion_tokens = provider_usage.get("output_tokens")

        total_tokens = provider_usage.get("total_tokens")

        if prompt_tokens is None or completion_tokens is None:
            return None

        prompt_tokens = int(prompt_tokens)
        completion_tokens = int(completion_tokens)
        total_tokens = (
            int(total_tokens)
            if total_tokens is not None
            else prompt_tokens + completion_tokens
        )

        return {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def record_call(
        self,
        *,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        response: Optional[str],
        max_tokens: int,
        temperature: float,
        metadata: Optional[Dict[str, Any]] = None,
        provider_usage: Optional[Dict[str, Any]] = None,
    ) -> TokenUsageRecord:
        self._call_index += 1

        prompt_counts = self.counter.count_prompt(system_prompt, user_prompt)
        response_tokens = self.counter.count_response(response)

        normalized_provider_usage = self._normalize_provider_usage(provider_usage)

        if normalized_provider_usage is not None:
            input_tokens = normalized_provider_usage["input_tokens"]
            output_tokens = normalized_provider_usage["output_tokens"]
            total_tokens = normalized_provider_usage["total_tokens"]
            usage_source = "provider"
        else:
            input_tokens = prompt_counts["input_tokens"]
            output_tokens = response_tokens
            total_tokens = input_tokens + output_tokens
            usage_source = "estimated"

        record = TokenUsageRecord(
            stage=stage,
            call_index=self._call_index,
            provider_name=self.provider_name,
            model_name=self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            system_tokens=prompt_counts["system_tokens"],
            user_tokens=prompt_counts["user_tokens"],
            response_tokens=response_tokens,
            usage_source=usage_source,
            max_tokens=max_tokens,
            temperature=temperature,
            metadata=metadata or {},
        )
        self.records.append(record)
        return record

    def summarize(self) -> Dict[str, StageTokenSummary]:
        summaries: Dict[str, StageTokenSummary] = {}
        for record in self.records:
            if record.stage not in summaries:
                summaries[record.stage] = StageTokenSummary(stage=record.stage)
            summaries[record.stage].add(record)
        return summaries

    def to_dict(self) -> Dict[str, Any]:
        summaries = self.summarize()
        return {
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "records": [asdict(r) for r in self.records],
            "summary": {
                stage: asdict(summary)
                for stage, summary in summaries.items()
            },
        }

    def save_json(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
