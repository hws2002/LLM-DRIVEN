# infra — Repository Layer

This package is the **data-access (repository) layer** of GraphNode AI. It isolates all
external storage behind handler classes, so the pipelines (`add_node`, `macro`, `microscope`)
never talk to a database driver directly — they depend only on these handlers.

The handlers are instantiated **once** at server/worker startup and injected into the
pipelines as parameters (dependency injection).

---

## Structure

```
infra/
└── repositories/
    ├── graph/
    │   └── graphnode_repository.py   # GraphNodeDBHandler — unified facade
    ├── neo4j/
    │   └── handler.py                # Neo4jHandler — graph DB access
    ├── vectordb/
    │   ├── chunks_store.py           # VectorDBHandler — document chunk embeddings
    │   ├── conversation_node_store.py# conversation-node embeddings
    │   └── macro_node_store.py       # MacroNodeStore — macro-graph node embeddings
    └── mongodb/
        └── handler.py                # MongoDBHandler — conversations & notes
```

---

## Roles

| Handler | Backing store | Responsibility |
|---------|---------------|----------------|
| **`GraphNodeDBHandler`** | (facade) | Single entry point combining Neo4j + ChromaDB. Owns `store_standardized_data()` (the ingest write path) and exposes `.graph_db`, `.vector_db`, node stores. |
| **`Neo4jHandler`** | Neo4j | Nodes, edges, chunks and their relations; neighbor/subgraph queries used by GraphRAG (`get_neighbors`, `get_chunks_for_entity`, …). |
| **`VectorDBHandler` / stores** | ChromaDB | Embedding storage & similarity retrieval for chunks, conversation nodes, and macro nodes. |
| **`MongoDBHandler`** | MongoDB | Reads source conversations/notes by `node_id` (the production ingest path). |

---

## Why this layer exists

- **Separation of concerns:** business logic (pipelines) stays free of DB driver details.
- **Single write path:** all graph writes funnel through `GraphNodeDBHandler.store_standardized_data()`,
  keeping Neo4j and the vector store consistent (entities, edges, chunks, and entity↔chunk links).
- **Testability / swappability:** a pipeline can be exercised against any handler instance,
  and the underlying store can change without touching pipeline code.

See `graph/graphnode_repository.py` for the main facade and the ingest write path.
