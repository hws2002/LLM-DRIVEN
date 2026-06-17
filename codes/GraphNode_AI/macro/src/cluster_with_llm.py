"""LLM-based conversation clustering using various LLM providers (OpenAI, Z.AI, Qwen, Groq, Gemini)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from util.llm_clients import BaseLLMClient, create_llm_client
from insights.discovery.prompts.summarizer_prompts import get_language_instruction

from cost_calculator import save_token_run
from token_usage import TokenUsageTracker

@dataclass
class Cluster:
    """Represents a topic cluster."""

    id: str
    name: str
    description: str
    key_themes: List[str]


@dataclass
class Assignment:
    """Represents a conversation-to-cluster assignment."""

    conversation_id: int
    orig_id: str
    cluster_id: str
    confidence: float
    top_keywords: List[str]


class LLMClusteringClient:
    """Client for LLM-based clustering (supports OpenAI, Qwen, Groq, Gemini)."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        language: str = "en",
        model_name: Optional[str] = None,
        provider_name: Optional[str] = None,
    ):
        """Initialize the LLM clustering client.

        Args:
            llm_client: An instance of BaseLLMClient (OpenAI, Qwen, Groq, or Gemini)
            language: Output language for cluster names and descriptions (en/ko/zh)
        """
        self.llm_client = llm_client
        self.language = language
        self.call_count = 0
        self.last_selected_num_clusters: Optional[int] = None
        self.model_name = model_name or llm_client.model_name or "unknown"
        self.provider_name = provider_name or getattr(llm_client, "provider_name", "unknown")
        self.token_tracker = TokenUsageTracker(
            model_name=self.model_name,
            provider_name=self.provider_name,
        )
    def _get_provider_usage(self) -> Optional[Dict[str, Any]]:
        """
        Optional hook:
        If your llm_client stores usage from the most recent API call, return it here.
        Example expected format:
        {
            "prompt_tokens": 123,
            "completion_tokens": 45,
            "total_tokens": 168
        }
        """
        usage = getattr(self.llm_client, "last_usage", None)
        if isinstance(usage, dict):
            return usage
        return None

    def _call_llm_with_token_tracking(
        self,
        *,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        self.call_count += 1

        response = self.llm_client.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if response is not None and not isinstance(response, str):
            raise TypeError(
                f"Expected LLM response to be str, got {type(response).__name__}"
            )

        self.token_tracker.record_call(
            stage=stage,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=response,
            max_tokens=max_tokens,
            temperature=temperature,
            metadata=metadata,
            provider_usage=self._get_provider_usage(),
        )
        return response

    def _is_json_truncated(self, text: str) -> bool:
        """Check if JSON appears to be truncated.

        Args:
            text: The JSON string to check

        Returns:
            True if the JSON appears truncated, False otherwise
        """
        if not text:
            return True

        text = text.strip()

        # JSON should end with } or ]
        if not (text.endswith("}") or text.endswith("]")):
            return True

        # Quick bracket balance check
        open_braces = text.count("{")
        close_braces = text.count("}")
        open_brackets = text.count("[")
        close_brackets = text.count("]")

        if open_braces != close_braces or open_brackets != close_brackets:
            return True

        return False

    def _clean_json_response(self, response: str) -> str:
        """Clean markdown code blocks from response.

        Args:
            response: Raw LLM response

        Returns:
            Cleaned JSON string
        """
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        return response.strip()
    
    def assign_conversations(
        self,
        conversations: List[Dict[str, Any]],
        clusters: List[Cluster],
        batch_size: int = 20,
        verbose: bool = False,
    ) -> List[Assignment]:
        print(
            f"\n🔗 Phase 2: Assigning {len(conversations)} conversations to clusters..."
        )

        all_assignments = []
        num_batches = (len(conversations) + batch_size - 1) // batch_size
        _cluster_loop_start = time.time()

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(conversations))
            batch = conversations[start_idx:end_idx]

            print(
                f"  Processing batch {batch_idx + 1}/{num_batches} ({len(batch)} conversations)..."
            )

            clusters_json = json.dumps(
                [
                    {
                        "id": c.id,
                        "name": c.name,
                        "description": c.description,
                        "key_themes": c.key_themes,
                    }
                    for c in clusters
                ],
                indent=2,
                ensure_ascii=False,
            )

            system_prompt = (
                "You are an expert at assigning conversations to relevant topic clusters. "
                "Respond with plain text only. For each conversation output exactly one line formatted as "
                "'conversation_id=<ID> | cluster_id=<CLUSTER_ID> | confidence=<0.00-1.00>'. "
                "Do not include any additional commentary, explanations, JSON, or markdown."
            )

            assignments_for_batch: dict[int, Assignment] = {}
            pending_conversations = {conv["id"]: conv for conv in batch}
            max_batch_attempts = 10
            batch_attempt = 0

            while pending_conversations:
                batch_attempt += 1
                if batch_attempt > max_batch_attempts:
                    raise ValueError(
                        "LLM failed to return assignments for all conversations in the batch.\n"
                        f"Remaining conversation IDs: {sorted(pending_conversations.keys())}"
                    )

                valid_conversation_ids = set(pending_conversations.keys())
                ids_list_str = ", ".join(str(i) for i in sorted(valid_conversation_ids))

                conversation_summaries = []
                for conv in pending_conversations.values():
                    keywords_str = ", ".join(kw["term"] for kw in conv["keywords"][:5])
                    conversation_summaries.append(
                        f'{{"id": {conv["id"]}, "keywords": "{keywords_str}"}}'
                    )
                conversations_text = "\n".join(conversation_summaries)

                user_prompt = f"""Assign each conversation to the most appropriate cluster.

Available Clusters:
{clusters_json}

Conversations to Assign (remaining {len(valid_conversation_ids)}):
{conversations_text}

Requirements:
- Assign each conversation ID listed above to exactly one cluster
- Provide confidence score (0.00 to 1.00) rounded to two decimals
- Base assignment on keyword relevance to cluster themes
- The conversation IDs to assign in this response are: {ids_list_str}
- Produce exactly {len(valid_conversation_ids)} lines, one per conversation ID
- Each line in your reply must follow this exact format with pipe separators:
  conversation_id=<ID> | cluster_id=<CLUSTER_ID> | confidence=<CONFIDENCE>
- Do not include any extra text before or after the lines"""

                response = self._call_llm_with_token_tracking(
                    stage="assign_conversations",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=3000,
                    temperature=0.7,
                    metadata={
                        "batch_index": batch_idx + 1,
                        "num_batches": num_batches,
                        "batch_attempt": batch_attempt,
                        "batch_size_requested": batch_size,
                        "num_clusters": len(clusters),
                        "pending_count": len(valid_conversation_ids),
                    },
                )

                if not response or not response.strip():
                    if batch_attempt >= max_batch_attempts:
                        raise ValueError(
                            "LLM returned empty content while assigning conversations. "
                            "Check provider logs for details."
                        )
                    time.sleep(1.0)
                    continue

                try:
                    if "```json" in response:
                        response = response.split("```json")[1].split("```")[0].strip()
                    elif "```" in response:
                        response = response.split("```")[1].split("```")[0].strip()

                    assignments_payload = self._parse_assignment_lines(
                        response, valid_conversation_ids
                    )

                    if not assignments_payload:
                        if batch_attempt >= max_batch_attempts:
                            raise ValueError(
                                "LLM returned no parsable assignments for the batch."
                            )
                        time.sleep(1.0)
                        continue

                    for assign in assignments_payload:
                        conv_id = assign["conversation_id"]
                        if conv_id not in pending_conversations:
                            continue

                        conv = pending_conversations.pop(conv_id)
                        top_keywords = [kw["term"] for kw in conv["keywords"][:3]]

                        assignments_for_batch[conv_id] = Assignment(
                            conversation_id=conv_id,
                            orig_id=conv["orig_id"],
                            cluster_id=assign["cluster_id"],
                            confidence=assign["confidence"],
                            top_keywords=top_keywords,
                        )

                    if pending_conversations:
                        remaining_ids = sorted(pending_conversations.keys())
                        print(
                            f"ℹ️ {len(remaining_ids)} conversations still pending for batch {batch_idx + 1} "
                            f"(attempt {batch_attempt}/{max_batch_attempts}). "
                            f"Retrying only the remaining IDs: {remaining_ids}"
                        )
                        time.sleep(1.0)

                except json.JSONDecodeError as exc:
                    if batch_attempt >= max_batch_attempts:
                        raise ValueError(
                            f"Failed to parse LLM response for batch {batch_idx + 1}: {exc}"
                        )
                    time.sleep(1.0)

            for conv_id in sorted(assignments_for_batch.keys()):
                all_assignments.append(assignments_for_batch[conv_id])

            pct = int((batch_idx + 1) / num_batches * 100)
            _elapsed = time.time() - _cluster_loop_start
            _rate = (batch_idx + 1) / _elapsed if _elapsed > 0 else 0
            _eta = int((num_batches - batch_idx - 1) / _rate) if _rate > 0 else 0
            print(f"PROGRESS:clustering:{pct}:{batch_idx + 1}:{num_batches}:{_eta}", flush=True)

        print(f"✅ Assigned all {len(all_assignments)} conversations")
        return all_assignments
    def generate_clusters(
        self,
        conversations: List[Dict[str, Any]],
        num_clusters: Optional[int] = None,
        min_clusters: int = 3,
        max_clusters: int = 5,
        verbose: bool = False,
    ) -> List[Cluster]:
        if num_clusters is not None and num_clusters <= 0:
            raise ValueError("num_clusters must be a positive integer.")
        if num_clusters is None:
            if min_clusters <= 0 or max_clusters <= 0:
                raise ValueError(
                    "min_clusters and max_clusters must be positive integers."
                )
            if min_clusters > max_clusters:
                raise ValueError(
                    f"min_clusters ({min_clusters}) cannot be greater than max_clusters ({max_clusters})."
                )

        if num_clusters is None:
            print(
                f"\n📊 Phase 1: Letting LLM choose between {min_clusters}-{max_clusters} topic clusters..."
            )
        else:
            print(f"\n📊 Phase 1: Generating {num_clusters} topic clusters...")

        conversation_summaries = []
        for conv in conversations:
            keywords_str = ", ".join(
                f"{kw['term']} ({kw['score']:.2f})" for kw in conv["keywords"][:5]
            )
            conversation_summaries.append(f"ID {conv['id']}: {keywords_str}")

        conversations_text = "\n".join(conversation_summaries)

        lang_instruction = get_language_instruction(self.language)
        system_prompt = (
            "You are an expert at analyzing conversation topics and creating meaningful clusters. "
            "Return ONLY valid JSON without any markdown formatting or code blocks. "
            f"{lang_instruction}"
        )

        if num_clusters is None:
            cluster_instruction = (
                f"Determine the optimal number of clusters between {min_clusters} and {max_clusters}. "
                'Provide the chosen number as "num_clusters" and create that many major topic clusters '
                "that best categorize these conversations."
            )
            return_format = """{
  "num_clusters": 3,
  "clusters": [
    {
      "id": "cluster_1",
      "name": "Descriptive Name",
      "description": "One sentence description",
      "key_themes": ["theme1", "theme2", "theme3"]
    }
  ]
}"""
        else:
            cluster_instruction = (
                f"Create {num_clusters} major topic clusters that best categorize these conversations."
            )
            return_format = """{
  "clusters": [
    {
      "id": "cluster_1",
      "name": "Descriptive Name",
      "description": "One sentence description",
      "key_themes": ["theme1", "theme2", "theme3"]
    }
  ]
}"""

        user_prompt = f"""Analyze these {len(conversations)} conversations and their extracted keywords.

{cluster_instruction}

Conversations:
{conversations_text}

Requirements:
- Each cluster should represent a distinct topic/domain
- Cluster names should be clear and descriptive (2-5 words)
- Provide a brief description for each cluster
- Identify 3-5 key themes/terms for each cluster

Return ONLY valid JSON in this exact format:
{return_format}"""

        if verbose:
            print(f"  Sending {len(conversations)} conversations to LLM...")

        max_attempts = 3
        base_max_tokens = 4000
        data = None
        last_error = None
        response = None

        for attempt in range(max_attempts):
            current_max_tokens = base_max_tokens + (attempt * 2000)

            if attempt > 0:
                print(
                    f"  ⚠️ processing... (text {attempt + 1}/{max_attempts}, max_tokens={current_max_tokens})"
                )
                time.sleep(1.0)

            response = self._call_llm_with_token_tracking(
                stage="generate_clusters",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=current_max_tokens,
                temperature=0.7,
                metadata={
                    "attempt": attempt + 1,
                    "num_conversations": len(conversations),
                    "num_clusters_fixed": num_clusters,
                    "min_clusters": min_clusters,
                    "max_clusters": max_clusters,
                },
            )

            if response is None:
                last_error = ValueError(
                    "LLM returned no content while generating clusters. "
                    "Check provider logs for details."
                )
                continue

            cleaned_response = self._clean_json_response(response)

            if self._is_json_truncated(cleaned_response):
                print(
                    f"  ⚠️ JSONtext text text text (ends with: '...{cleaned_response[-50:]}')"
                )
                last_error = json.JSONDecodeError(
                    "JSON appears truncated", cleaned_response, len(cleaned_response)
                )
                if attempt < max_attempts - 1:
                    continue

            try:
                data = json.loads(cleaned_response)
                break

            except json.JSONDecodeError as exc:
                last_error = exc
                print(f"  ⚠️ JSON text text: {exc}")
                if attempt < max_attempts - 1:
                    continue

        if data is None:
            print(
                f"❌ Failed to parse LLM response as JSON after {max_attempts} attempts"
            )
            if response:
                print(f"Response (last 500 chars): ...{response[-500:]}")
            raise last_error or ValueError("Failed to get valid JSON from LLM")

        clusters_payload = data.get("clusters", [])
        clusters = [
            Cluster(
                id=c["id"],
                name=c["name"],
                description=c["description"],
                key_themes=c["key_themes"],
            )
            for c in clusters_payload
        ]

        selected_num_clusters: Optional[int]
        if num_clusters is not None:
            selected_num_clusters = num_clusters
        else:
            selected_num_clusters_raw = data.get("num_clusters")
            try:
                selected_num_clusters = (
                    int(selected_num_clusters_raw)
                    if selected_num_clusters_raw is not None
                    else len(clusters)
                )
            except (TypeError, ValueError):
                warnings.warn(
                    f"LLM returned invalid num_clusters {selected_num_clusters_raw!r}; "
                    "falling back to using cluster count."
                )
                selected_num_clusters = len(clusters)

        self.last_selected_num_clusters = selected_num_clusters

        if (
            num_clusters is None
            and selected_num_clusters is not None
            and not (min_clusters <= selected_num_clusters <= max_clusters)
        ):
            warnings.warn(
                f"LLM selected {selected_num_clusters} clusters outside target range "
                f"[{min_clusters}, {max_clusters}]."
            )

        if selected_num_clusters is not None and selected_num_clusters != len(clusters):
            warnings.warn(
                f"LLM reported {selected_num_clusters} clusters but returned {len(clusters)} definitions."
            )

        if num_clusters is None:
            print(
                f"✅ Generated {len(clusters)} clusters (LLM selected {selected_num_clusters})"
            )
        else:
            print(f"✅ Generated {len(clusters)} clusters")

        for cluster in clusters:
            print(f"  - {cluster.name}: {cluster.description}")

        return clusters    

    @staticmethod
    def _parse_assignment_lines(
        response_text: str, valid_ids: set[int]
    ) -> List[Dict[str, Any]]:
        """Parse plain-text assignment lines of the form `conversation_id=... | cluster_id=... | confidence=...`."""
        if not response_text:
            return []

        pattern = re.compile(
            r"""
            ^\s*
            (?:[-*•]\s*)?
            conversation_id
            \s*=\s*(?P<conv>\d+)
            \s*\|\s*
            cluster_id
            \s*=\s*(?P<cluster>[^|]+?)
            \s*\|\s*
            confidence
            \s*=\s*(?P<confidence>[0-9]+(?:\.[0-9]+)?%?)
            \s*$""",
            re.IGNORECASE | re.VERBOSE,
        )

        assignments: List[Dict[str, Any]] = []
        seen: set[int] = set()

        for raw_line in response_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            match = pattern.match(line)
            if not match:
                continue

            conv_id = int(match.group("conv"))
            if conv_id not in valid_ids or conv_id in seen:
                continue

            cluster_id = match.group("cluster").strip()
            if cluster_id.isdigit():
                cluster_id = f"cluster_{cluster_id}"

            confidence_str = match.group("confidence").strip()

            if confidence_str.endswith("%"):
                try:
                    confidence_val = float(confidence_str[:-1]) / 100.0
                except ValueError:
                    continue
            else:
                try:
                    confidence_val = float(confidence_str)
                except ValueError:
                    continue
                if confidence_val > 1.0:
                    confidence_val = confidence_val / 100.0

            assignments.append(
                {
                    "conversation_id": conv_id,
                    "cluster_id": cluster_id,
                    "confidence": max(0.0, min(1.0, confidence_val)),
                }
            )
            seen.add(conv_id)

        if assignments:
            return assignments

        # Fallback: attempt to parse JSON payloads if the model returned structured data.
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            return assignments

        json_assignments = data.get("assignments")
        if not isinstance(json_assignments, list):
            return assignments

        for item in json_assignments:
            if not isinstance(item, dict):
                continue
            conv_id = item.get("conversation_id")
            cluster_id = item.get("cluster_id")
            confidence_val = item.get("confidence")
            try:
                conv_id_int = int(conv_id)
                confidence_float = float(confidence_val)
            except (TypeError, ValueError):
                continue
            if conv_id_int not in valid_ids or conv_id_int in seen:
                continue
            assignments.append(
                {
                    "conversation_id": conv_id_int,
                    "cluster_id": str(cluster_id),
                    "confidence": max(0.0, min(1.0, confidence_float)),
                }
            )
            seen.add(conv_id_int)

        return assignments

    def print_token_usage_summary(self) -> None:
        summary = self.token_tracker.summarize()

        print("\n🧮 Token Usage Summary")
        for stage, item in summary.items():
            print(f"\n[{stage}]")
            print(f"  calls         : {item.calls}")
            print(f"  input_tokens  : {item.input_tokens}")
            print(f"  output_tokens : {item.output_tokens}")
            print(f"  total_tokens  : {item.total_tokens}")

    def save_token_usage(self, output_path: str | Path) -> None:
        self.token_tracker.save_json(output_path)


def load_input_data(input_path: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load conversation data from JSON file.

    Args:
        input_path: Path to input JSON file

    Returns:
        Tuple of (conversations list, metadata dict)

    Raises:
        ValueError: If input file format is invalid
    """
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "conversations" not in data:
            raise ValueError("Input file must contain 'conversations' field")

        conversations = data["conversations"]
        metadata = data.get("metadata", {})

        print(f"📂 Loaded {len(conversations)} conversations from {input_path}")
        return conversations, metadata

    except Exception as exc:
        print(f"❌ Failed to load input file: {exc}")
        raise

def save_output(
    output_path: Path,
    clusters: List[Cluster],
    assignments: List[Assignment],
    input_metadata: Dict[str, Any],
    model_name: str,
    selected_num_clusters: Optional[int] = None,
    cluster_range: Optional[tuple[int, int]] = None,
) -> None:
    """Save clustering results to JSON file.

    Args:
        output_path: Path to output JSON file
        clusters: List of generated clusters
        assignments: List of conversation assignments
        input_metadata: Metadata from input file
        model_name: Name of the clustering model used
    """
    # Calculate statistics
    cluster_sizes = {}
    confidence_scores = []

    for assign in assignments:
        cluster_sizes[assign.cluster_id] = cluster_sizes.get(assign.cluster_id, 0) + 1
        confidence_scores.append(assign.confidence)

    avg_confidence = (
        sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    )

    metadata: Dict[str, Any] = {
        "total_conversations": len(assignments),
        "num_clusters": len(clusters),
        "clustering_model": model_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "average_confidence": round(avg_confidence, 4),
        "input_metadata": input_metadata,
    }

    if selected_num_clusters is not None:
        metadata["selected_num_clusters"] = selected_num_clusters
    if cluster_range is not None:
        metadata["target_cluster_range"] = {
            "min": cluster_range[0],
            "max": cluster_range[1],
        }

    output_data = {
        "clusters": [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "key_themes": c.key_themes,
                "size": cluster_sizes.get(c.id, 0),
            }
            for c in clusters
        ],
        "assignments": [
            {
                "conversation_id": a.conversation_id,
                "orig_id": a.orig_id,
                "cluster_id": a.cluster_id,
                "confidence": round(a.confidence, 4),
                "top_keywords": a.top_keywords,
            }
            for a in assignments
        ],
        "metadata": metadata,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Results saved to {output_path}")
    print(f"\n📈 Clustering Statistics:")
    print(f"  Total conversations: {len(assignments)}")
    print(f"  Number of clusters: {len(clusters)}")
    if selected_num_clusters is not None:
        print(f"  Selected cluster count: {selected_num_clusters}")
    if cluster_range is not None:
        print(f"  Requested cluster range: {cluster_range[0]}-{cluster_range[1]}")
    print(f"  Average confidence: {avg_confidence:.4f}")
    print(f"\n  Cluster sizes:")
    for cluster in clusters:
        size = cluster_sizes.get(cluster.id, 0)
        print(f"    {cluster.name}: {size} conversations")


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point for the clustering script."""
    parser = argparse.ArgumentParser(
        description="Cluster conversations using LLM-based topic analysis."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default="test_output.json",
        help="Path to input JSON file (default: test_output.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="clustered_output.json",
        help="Path to output JSON file (default: clustered_output.json)",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=None,
        help="Fixed number of clusters to create. If omitted, the LLM selects within the provided range.",
    )
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=3,
        help="Minimum number of clusters for the LLM to consider (default: 3). Ignored if --num-clusters is set.",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=5,
        help="Maximum number of clusters for the LLM to consider (default: 5). Ignored if --num-clusters is set.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "z.ai", "qwen", "groq", "gemini"],
        default="openai",
        help="LLM provider to use (default: openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (default: provider-specific default)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for Phase 2 (default: 20)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "ko", "zh"],
        help="Output language for cluster names and descriptions (default: en)",
    )
    parser.add_argument(
        "--pricing-catalog",
        type=Path,
        default=None,
        help="Optional pricing catalog JSON for cost calculation",
    )
    parser.add_argument(
        "--model-aliases",
        type=Path,
        default=None,
        help="Optional model alias JSON for cost calculation",
    )

    args = parser.parse_args(argv)

    print("🚀 Starting LLM-based conversation clustering")
    print(f"  Provider: {args.provider}")
    if args.num_clusters is not None:
        print(f"  Target clusters: {args.num_clusters}")
    else:
        print(f"  Target cluster range: {args.min_clusters}-{args.max_clusters}")

    if args.num_clusters is None and args.min_clusters > args.max_clusters:
        print("❌ Error: --min-clusters cannot be greater than --max-clusters.")
        sys.exit(1)
    if args.num_clusters is None and (args.min_clusters <= 0 or args.max_clusters <= 0):
        print("❌ Error: --min-clusters and --max-clusters must be positive integers.")
        sys.exit(1)
    if args.num_clusters is not None and args.num_clusters <= 0:
        print("❌ Error: --num-clusters must be a positive integer.")
        sys.exit(1)

    # Load input data
    conversations, input_metadata = load_input_data(args.input)

    # Initialize LLM client
    try:
        llm_client = create_llm_client(
            provider=args.provider,
            model_name=args.model,
        )
        print(f"  Model: {llm_client.model_name}")
    except (ValueError, ImportError) as exc:
        print(f"❌ Error: {exc}")
        sys.exit(1)

    # Initialize clustering client
    client = LLMClusteringClient(
        llm_client=llm_client,
        language=args.language,
        model_name=llm_client.model_name,
        provider_name=args.provider,
    )

    # Phase 1: Generate clusters
    clusters = client.generate_clusters(
        conversations=conversations,
        num_clusters=args.num_clusters,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        verbose=args.verbose,
    )
    selected_num_clusters = client.last_selected_num_clusters or len(clusters)
    cluster_range = (
        (args.min_clusters, args.max_clusters) if args.num_clusters is None else None
    )
    if args.num_clusters is None:
        print(f"  LLM-selected cluster count: {selected_num_clusters}")

    # Phase 2: Assign conversations
    assignments = client.assign_conversations(
        conversations=conversations,
        clusters=clusters,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    # Save results
    save_output(
        output_path=args.output,
        clusters=clusters,
        assignments=assignments,
        input_metadata=input_metadata,
        model_name=f"{args.provider}/{llm_client.model_name}",
        selected_num_clusters=selected_num_clusters,
        cluster_range=cluster_range,
    )
    client.print_token_usage_summary()
    run_id = args.output.parent.name
    token_run_dir = save_token_run(client.token_tracker, run_id, service_name="macro")
    print(f"\n💾 Token usage + cost summary saved to {token_run_dir}")

    print(f"\n✨ Clustering complete! Total API calls: {client.call_count}")


if __name__ == "__main__":
    main()
