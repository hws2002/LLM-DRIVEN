"""Quick verification script to check indexed embeddings."""

from pathlib import Path
import numpy as np

from .storage import VectorStore, VectorStoreConfig


def verify_index(vector_db_path: Path):
    """Verify indexed embeddings."""
    print("=" * 60)
    print("Verifying Indexed Embeddings")
    print("=" * 60)
    print()

    # Initialize vector store
    config = VectorStoreConfig(
        persist_directory=str(vector_db_path),
        collection_name="conversation_embeddings",
        embedding_dimension=384
    )

    store = VectorStore(config)

    # Check count
    total = store.count()
    print(f"✓ Total indexed records: {total}")

    if total > 0:
        # Get a sample record
        sample_ids = [f"conv_{i}" for i in range(min(3, total))]
        records = store.get_by_ids(sample_ids)

        print(f"\n✓ Sample records retrieved: {len(records)}")
        for i, record in enumerate(records[:3]):
            print(f"\n  Record {i+1}:")
            print(f"    ID: {record.id}")
            print(f"    Embedding dimension: {len(record.embedding)}")
            print(f"    Metadata keys: {list(record.metadata.keys())}")
            if "cluster_name" in record.metadata:
                print(f"    Cluster: {record.metadata.get('cluster_name')}")
            if "keywords" in record.metadata:
                keywords = record.metadata.get('keywords', '')
                if keywords:
                    kw_list = keywords.split(',')[:3]
                    print(f"    Keywords: {', '.join(kw_list)}")

        # Test search
        print("\n✓ Testing semantic search...")
        query_embedding = np.random.rand(384)
        results = store.search(query_embedding, top_k=5)

        print(f"  Found {len(results)} results:")
        for i, result in enumerate(results[:3]):
            print(f"    {i+1}. ID: {result.id}, Score: {result.score:.3f}")

    print("\n" + "=" * 60)
    print("✅ Verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify indexed embeddings")
    parser.add_argument(
        "--vector-db",
        type=Path,
        required=True,
        help="Path to vector database directory"
    )

    args = parser.parse_args()

    verify_index(args.vector_db)
