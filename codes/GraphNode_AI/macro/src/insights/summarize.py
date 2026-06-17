"""CLI tool for generating graph summaries and insights.

Usage:
    python -m insights.summarize \\
        --graph output/graph_postprocessed.json \\
        --vector-db output/vector_db \\
        --output output/summary.json \\
        --provider openai \\
        --model gpt-4o-mini
"""

import argparse
import json
import sys
from pathlib import Path

from util.llm_clients import BaseLLMClient, create_llm_client

from .core import load_graph
from .discovery import GraphSummarizer


def get_llm_client(provider: str, model: str) -> BaseLLMClient:
    """Get LLM client for the specified provider.

    Args:
        provider: LLM provider name
        model: Model name

    Returns:
        BaseLLMClient instance
    """

    return create_llm_client(provider=provider, model_name=model)


def main():
    """Main CLI entry point."""
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / ".env")
    parser = argparse.ArgumentParser(
        description="Generate graph summary and insights using LLM analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate standard summary
  python -m insights.summarize \\
      --graph output/graph_postprocessed.json \\
      --vector-db output/vector_db \\
      --output output/summary.json

  # Detailed summary with specific provider
  python -m insights.summarize \\
      --graph output/graph_postprocessed.json \\
      --vector-db output/vector_db \\
      --output output/summary.json \\
      --detail-level detailed \\
      --provider openai \\
      --model gpt-4o

  # Focus on specific clusters
  python -m insights.summarize \\
      --graph output/graph_postprocessed.json \\
      --vector-db output/vector_db \\
      --focus-clusters cluster_1,cluster_2 \\
      --output output/summary.json
        """
    )

    parser.add_argument(
        "--graph",
        type=Path,
        required=True,
        help="Path to graph JSON file (required)"
    )

    parser.add_argument(
        "--vector-db",
        type=Path,
        default=None,
        help="Path to vector database directory (optional)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for summary JSON (optional, prints to stdout if omitted)"
    )

    parser.add_argument(
        "--detail-level",
        type=str,
        choices=["brief", "standard", "detailed"],
        default="standard",
        help="Level of detail in summary (default: standard)"
    )

    parser.add_argument(
        "--focus-clusters",
        type=str,
        default=None,
        help="Comma-separated cluster IDs to focus on (optional)"
    )

    parser.add_argument(
        "--no-recommendations",
        action="store_true",
        help="Skip generating recommendations"
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "qwen", "groq", "gemini"],
        help="LLM provider (default: openai)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model name (default: gpt-4o-mini)"
    )

    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["ko", "en", "zh"],
        help="Output language for insights (ko=Korean, en=English, zh=Chinese; default: en)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress output"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.graph.exists():
        print(f"❌ Error: Graph file not found: {args.graph}", file=sys.stderr)
        sys.exit(1)

    if args.vector_db and not args.vector_db.exists():
        print(f"❌ Error: Vector DB directory not found: {args.vector_db}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.verbose:
            print("=" * 60)
            print("🔮 GraphNode Insights - Graph Summarizer")
            print("=" * 60)
            print()

        # Load graph
        if args.verbose:
            print(f"Loading graph from {args.graph}...")
            if args.vector_db:
                print(f"Loading vector DB from {args.vector_db}...")

        loader = load_graph(
            graph_path=args.graph,
            vector_db_path=args.vector_db
        )

        if args.verbose:
            stats = loader.get_graph_stats()
            print(f"✓ Graph loaded: {stats.total_nodes} nodes, {stats.total_clusters} clusters\n")

        # Initialize LLM client
        if args.verbose:
            print(f"Initializing {args.provider} client with model {args.model}...")

        llm_client = get_llm_client(args.provider, args.model)

        if args.verbose:
            print("✓ LLM client initialized\n")

        # Create summarizer
        summarizer = GraphSummarizer(
            graph_loader=loader,
            llm_client=llm_client,
            language=args.language
        )

        # Parse focus clusters
        focus_areas = None
        if args.focus_clusters:
            focus_areas = [c.strip() for c in args.focus_clusters.split(",")]

        # Generate summary
        print("STEP_START:summary", flush=True)
        if args.verbose:
            print(f"Generating {args.detail_level} summary...")
            print("This may take a moment...\n")

        summary = summarizer.generate_summary(
            detail_level=args.detail_level,
            focus_areas=focus_areas,
            include_recommendations=not args.no_recommendations
        )

        # Output summary
        if args.output:
            # Save to JSON file
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)

            if args.verbose:
                print(f"\n✅ Summary saved to {args.output}")
                print(f"\n📊 API Calls: {llm_client.get_call_count()}")
                print("=" * 60)

            # Also print to console
            print(summary)

        else:
            # Print to stdout
            if args.verbose:
                print("=" * 60)
                print()

            print(summary)

            if args.verbose:
                print()
                print(f"📊 API Calls: {llm_client.get_call_count()}")

        sys.exit(0)

    except KeyboardInterrupt:
        print("\n❌ Interrupted by user", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"❌ Summarization failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
