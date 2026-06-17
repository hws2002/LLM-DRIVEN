"""CLI tool for indexing conversation embeddings into ChromaDB.

Usage:
    python -m insights.index_embeddings \\
        --features output/features.json \\
        --graph output/graph_postprocessed.json \\
        --output-dir output/vector_db \\
        --verbose
"""

import argparse
import sys
from pathlib import Path

from .storage.indexer import index_embeddings_cli


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Index conversation embeddings into ChromaDB vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index embeddings from features.json
  python -m insights.index_embeddings \\
      --features output/features.json \\
      --output-dir output/vector_db

  # Index with graph metadata enrichment
  python -m insights.index_embeddings \\
      --features output/features.json \\
      --graph output/graph_postprocessed.json \\
      --output-dir output/vector_db

  # Reindex (clear and rebuild)
  python -m insights.index_embeddings \\
      --features output/features.json \\
      --graph output/graph_postprocessed.json \\
      --output-dir output/vector_db \\
      --reindex

  # Custom collection name
  python -m insights.index_embeddings \\
      --features output/features.json \\
      --output-dir output/vector_db \\
      --collection my_conversations
        """
    )

    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Path to features.json file (required)"
    )

    parser.add_argument(
        "--graph",
        type=Path,
        default=None,
        help="Path to graph JSON for metadata enrichment (optional)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for ChromaDB persistence (required)"
    )

    parser.add_argument(
        "--collection",
        type=str,
        default="conversation_embeddings",
        help="ChromaDB collection name (default: conversation_embeddings)"
    )

    parser.add_argument(
        "--user-id",
        type=str,
        default="",
        help="User ID prefix for embedding record IDs (e.g. 'user123' → 'user123_conv_0')"
    )

    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Clear existing index and rebuild from scratch"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress output"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.features.exists():
        print(f"❌ Error: Features file not found: {args.features}", file=sys.stderr)
        sys.exit(1)

    if args.graph and not args.graph.exists():
        print(f"❌ Error: Graph file not found: {args.graph}", file=sys.stderr)
        sys.exit(1)

    # Run indexing
    try:
        if args.verbose:
            print("=" * 60)
            print("GraphNode Insights - Embedding Indexing")
            print("=" * 60)
            print()

        result = index_embeddings_cli(
            features_path=args.features,
            graph_path=args.graph,
            output_dir=args.output_dir,
            collection_name=args.collection,
            reindex=args.reindex,
            verbose=args.verbose,
            user_id=args.user_id,
        )

        if args.verbose:
            print()
            print("=" * 60)
            print(f"📊 Persist directory: {args.output_dir}")
            print(f"📁 Collection: {args.collection}")
            if result.errors:
                print(f"⚠️  {len(result.errors)} warnings/errors encountered")
            print("=" * 60)

        sys.exit(0)

    except Exception as e:
        print(f"❌ Indexing failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
