from __future__ import annotations

import argparse
import io
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from shared.token_usage import TokenUsageTracker

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]  # shared/ → GraphNode_AI/

DEFAULT_PRICING_PATH = Path(__file__).with_name("llm_pricing.json")
DEFAULT_ALIAS_PATH = Path(__file__).with_name("llm_model_aliases.json")


@dataclass(frozen=True)
class PriceEntry:
    provider: str
    model: str
    input_cost_per_million: float
    output_cost_per_million: float
    currency: str = "USD"
    cached_input_cost_per_million: Optional[float] = None
    source: Optional[str] = None
    notes: Optional[str] = None


def _normalize_provider(provider_name: Optional[str]) -> str:
    if not provider_name:
        return "unknown"
    provider = provider_name.strip().lower()
    if provider == "zai":
        return "z.ai"
    return provider


def _normalize_model_key(model_name: Optional[str]) -> str:
    if not model_name:
        return "unknown"
    normalized = model_name.strip().lower()
    normalized = normalized.replace("_", "-")
    normalized = "-".join(normalized.split())
    return normalized


class PricingCatalog:
    def __init__(self, entries: Dict[tuple[str, str], PriceEntry], aliases: Dict[str, Dict[str, str]]):
        self.entries = entries
        self.aliases = aliases

    @classmethod
    def load(
        cls,
        pricing_path: Optional[str | Path] = None,
        alias_path: Optional[str | Path] = None,
    ) -> "PricingCatalog":
        pricing_file = Path(pricing_path) if pricing_path else DEFAULT_PRICING_PATH
        alias_file = Path(alias_path) if alias_path else DEFAULT_ALIAS_PATH

        with open(pricing_file, "r", encoding="utf-8") as pricing_handle:
            pricing_rows = json.load(pricing_handle)

        with open(alias_file, "r", encoding="utf-8") as alias_handle:
            alias_rows = json.load(alias_handle)

        entries: Dict[tuple[str, str], PriceEntry] = {}
        for row in pricing_rows:
            provider = _normalize_provider(row["provider"])
            model = _normalize_model_key(row["model"])
            entries[(provider, model)] = PriceEntry(
                provider=provider,
                model=model,
                input_cost_per_million=float(row["input_cost_per_million"]),
                output_cost_per_million=float(row["output_cost_per_million"]),
                currency=row.get("currency", "USD"),
                cached_input_cost_per_million=(
                    float(row["cached_input_cost_per_million"])
                    if row.get("cached_input_cost_per_million") is not None
                    else None
                ),
                source=row.get("source"),
                notes=row.get("notes"),
            )

        aliases: Dict[str, Dict[str, str]] = {}
        for provider_name, provider_aliases in alias_rows.items():
            normalized_provider = _normalize_provider(provider_name)
            aliases[normalized_provider] = {
                _normalize_model_key(alias): _normalize_model_key(target)
                for alias, target in provider_aliases.items()
            }

        return cls(entries=entries, aliases=aliases)

    def resolve_model(self, provider_name: str, model_name: str) -> str:
        provider = _normalize_provider(provider_name)
        normalized_model = _normalize_model_key(model_name)
        provider_aliases = self.aliases.get(provider, {})
        return provider_aliases.get(normalized_model, normalized_model)

    def get_entry(self, provider_name: str, model_name: str) -> Optional[PriceEntry]:
        provider = _normalize_provider(provider_name)
        resolved_model = self.resolve_model(provider, model_name)
        return self.entries.get((provider, resolved_model))


def load_token_usage(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _calculate_record_cost(record: Dict[str, Any], price_entry: PriceEntry) -> Dict[str, Any]:
    input_tokens = int(record.get("input_tokens", 0))
    output_tokens = int(record.get("output_tokens", 0))
    input_cost = (input_tokens / 1_000_000) * price_entry.input_cost_per_million
    output_cost = (output_tokens / 1_000_000) * price_entry.output_cost_per_million
    total_cost = input_cost + output_cost

    return {
        "call_index": record.get("call_index"),
        "stage": record.get("stage"),
        "usage_source": record.get("usage_source", "estimated"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": int(record.get("total_tokens", input_tokens + output_tokens)),
        "input_cost": round(input_cost, 8),
        "output_cost": round(output_cost, 8),
        "total_cost": round(total_cost, 8),
        "currency": price_entry.currency,
        "metadata": record.get("metadata", {}),
    }


def build_cost_report(
    token_usage_data: Dict[str, Any],
    *,
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    pricing_path: Optional[str | Path] = None,
    alias_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    catalog = PricingCatalog.load(pricing_path=pricing_path, alias_path=alias_path)

    resolved_provider = _normalize_provider(provider_name or token_usage_data.get("provider_name"))
    resolved_model_input = model_name or token_usage_data.get("model_name")
    resolved_model = catalog.resolve_model(resolved_provider, resolved_model_input)
    price_entry = catalog.get_entry(resolved_provider, resolved_model)

    if price_entry is None:
        available_models = sorted(
            model for provider, model in catalog.entries.keys() if provider == resolved_provider
        )
        raise ValueError(
            f"No pricing entry for provider/model: {resolved_provider}/{resolved_model}. "
            f"Available models for provider: {available_models}"
        )

    records = token_usage_data.get("records", [])
    record_costs = [_calculate_record_cost(record, price_entry) for record in records]

    stages: Dict[str, Dict[str, Any]] = {}
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    estimated_calls = 0
    provider_reported_calls = 0

    for item in record_costs:
        total_input_tokens += item["input_tokens"]
        total_output_tokens += item["output_tokens"]
        total_cost += item["total_cost"]
        if item["usage_source"] == "provider":
            provider_reported_calls += 1
        else:
            estimated_calls += 1

        stage_name = item["stage"]
        stage_bucket = stages.setdefault(
            stage_name,
            {
                "stage": stage_name,
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "total_cost": 0.0,
                "estimated_calls": 0,
                "provider_reported_calls": 0,
                "currency": price_entry.currency,
            },
        )
        stage_bucket["calls"] += 1
        stage_bucket["input_tokens"] += item["input_tokens"]
        stage_bucket["output_tokens"] += item["output_tokens"]
        stage_bucket["total_tokens"] += item["total_tokens"]
        stage_bucket["input_cost"] += item["input_cost"]
        stage_bucket["output_cost"] += item["output_cost"]
        stage_bucket["total_cost"] += item["total_cost"]
        if item["usage_source"] == "provider":
            stage_bucket["provider_reported_calls"] += 1
        else:
            stage_bucket["estimated_calls"] += 1

    for stage_bucket in stages.values():
        stage_bucket["input_cost"] = round(stage_bucket["input_cost"], 8)
        stage_bucket["output_cost"] = round(stage_bucket["output_cost"], 8)
        stage_bucket["total_cost"] = round(stage_bucket["total_cost"], 8)

    return {
        "provider_name": resolved_provider,
        "model_name": resolved_model,
        "pricing": asdict(price_entry),
        "totals": {
            "calls": len(record_costs),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "input_cost": round((total_input_tokens / 1_000_000) * price_entry.input_cost_per_million, 8),
            "output_cost": round((total_output_tokens / 1_000_000) * price_entry.output_cost_per_million, 8),
            "total_cost": round(total_cost, 8),
            "estimated_calls": estimated_calls,
            "provider_reported_calls": provider_reported_calls,
            "currency": price_entry.currency,
        },
        "stages": {stage: data for stage, data in sorted(stages.items())},
        "records": record_costs,
    }


def save_cost_report(report: Dict[str, Any], output_path: str | Path) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def save_token_run(
    tracker: "TokenUsageTracker",
    run_id: str,
    service_name: str = "",
    user_id: str = "",
    s3_bucket: Optional[str] = None,
) -> str:
    """English documentation."""
    import boto3

    bucket = s3_bucket or os.getenv("S3_BUCKET", "")
    if not bucket:
        logger.warning("S3_BUCKET not set — token_usage upload skipped")
        return ""

    parts = ["token_usage"]
    if service_name:
        parts.append(service_name)
    if user_id:
        parts.append(user_id)
    parts.append(run_id)
    prefix = "/".join(parts)

    aws_profile = os.getenv("AWS_PROFILE")
    session = (
        boto3.Session(profile_name=aws_profile)
        if aws_profile
        else boto3.Session()
    )
    s3 = session.client("s3")

    # token_usage.json
    usage_data = tracker.to_dict()
    usage_bytes = json.dumps(usage_data, ensure_ascii=False, indent=2).encode("utf-8")
    usage_key = f"{prefix}/token_usage.json"
    s3.put_object(Bucket=bucket, Key=usage_key, Body=usage_bytes, ContentType="application/json")
    logger.info("Token usage saved → s3://%s/%s", bucket, usage_key)

    # cost_summary.json
    try:
        report = build_cost_report(usage_data)
        report_bytes = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
        summary_key = f"{prefix}/cost_summary.json"
        s3.put_object(Bucket=bucket, Key=summary_key, Body=report_bytes, ContentType="application/json")
        totals = report["totals"]
        logger.info(
            "Cost summary saved → s3://%s/%s  (calls=%s, total=%.8f %s)",
            bucket,
            summary_key,
            totals["calls"],
            totals["total_cost"],
            totals["currency"],
        )
    except ValueError as exc:
        logger.warning("Cost summary skipped: %s", exc)

    return f"s3://{bucket}/{prefix}"


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Calculate LLM cost from token_usage.json")
    parser.add_argument("--token-usage", type=Path, required=True, help="Path to token_usage.json")
    parser.add_argument("--output", type=Path, default=None, help="Path to save cost summary JSON")
    parser.add_argument("--provider", type=str, default=None, help="Provider override")
    parser.add_argument("--model", type=str, default=None, help="Model override")
    parser.add_argument("--pricing-catalog", type=Path, default=None, help="Custom pricing catalog JSON")
    parser.add_argument("--model-aliases", type=Path, default=None, help="Custom model alias JSON")
    args = parser.parse_args(argv)

    token_usage_data = load_token_usage(args.token_usage)
    report = build_cost_report(
        token_usage_data,
        provider_name=args.provider,
        model_name=args.model,
        pricing_path=args.pricing_catalog,
        alias_path=args.model_aliases,
    )

    if args.output:
        save_cost_report(report, args.output)

    totals = report["totals"]
    print(f"provider: {report['provider_name']}")
    print(f"model: {report['model_name']}")
    print(f"calls: {totals['calls']}")
    print(f"input_tokens: {totals['input_tokens']}")
    print(f"output_tokens: {totals['output_tokens']}")
    print(f"total_cost: {totals['total_cost']} {totals['currency']}")


if __name__ == "__main__":
    main()
