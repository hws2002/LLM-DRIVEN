# Microscope

Pipeline package that converts a document into a graph and performs graph-based RAG
(query / synthesis / related questions).

---

## Pipeline overview

### 1. Ingest (document → graph storage)

There are two ingest entry paths; the internal pipeline is identical.

#### Path A — `MICROSCOPE_INGEST_FROM_NODE_REQUEST` (production path)
BE passes a MongoDB `node_id` → the worker fetches the conversation/note directly,
converts it to markdown, and ingests it.

```
BE → SQS (node_id, node_type, user_id, group_id)
  └─► worker: fetch message/note from MongoDB → convert to markdown
        └─► [shared ingest pipeline]
```

#### Path B — `MICROSCOPE_INGEST_REQUEST` (direct file upload)
A file is uploaded to S3 and passed via SQS.

```
client → S3 upload → SQS (s3_key, file_name, user_id, group_id)
  └─► worker: download from S3 → temp file
        └─► [shared ingest pipeline]
```

#### Shared ingest pipeline
```
Document file
  └─► Chunk split (RecursiveCharacterTextSplitter)
        └─► [batched] LLM entity/relation extraction  ← ontology schema injected
              └─► Entity name standardization (parallel LLM calls per type)
                    └─► ChromaDB (chunk + entity embeddings)
                          └─► Neo4j (nodes + edges)
```

**Entry point:** `microscope/call.py` → `call()`

**Key settings** (`shared/config.py`):
| Setting | Default | Description |
|---------|---------|-------------|
| `MICROSCOPE_CHUNK_SIZE` | 400 | Chunk size (tokens) |
| `MICROSCOPE_CHUNK_OVERLAP` | 80 | Chunk overlap |
| `MICROSCOPE_BATCH_MAX_TOKENS` | 10000 | Max input tokens per batch |
| `MICROSCOPE_LLM_MODEL` | gpt-5-mini | Extraction / standardization LLM |

---

### 2. RAG service (query → answer)

```
Question
  └─► Vector search (ChromaDB, top_k)
        └─► Entity extraction → graph expansion (Neo4j, N-hop)
              └─► Chunk merge (dedup)
                    └─► Build context → LLM answer
                          └─► (optional) fuse user's Macro profile for personalization
```

**Service functions** (`microscope/services/rag_service.py`):
| Function | Description |
|----------|-------------|
| `run_query()` | Question answering (hybrid RAG) |
| `run_synthesize()` | Topic synthesis / summary |
| `run_related_questions()` | Related-question generation |

---

## Directory structure

```
microscope/
├── call.py                          # Ingest pipeline entry point
│
├── graph_generation/
│   └── generator.py                 # Entity/relation extraction + standardization
│
├── services/
│   └── rag_service.py               # run_query / run_synthesize / run_related_questions
│
├── rag/
│   ├── retrieval_strategies.py      # Vector search + graph expansion
│   ├── context_builder.py           # Chunks → context string
│   ├── prompt_builder.py            # RAG prompt templates
│   ├── macro_context.py             # Fuse user's Macro-Graph profile (personalization)
│   └── answer_gen.py                # LLM answer generation
│
├── prompts/
│   ├── prompt_factory.py            # Prompt loader
│   ├── entity_relation_prompt.py    # Extraction prompt
│   └── standardization_prompt.py    # Standardization prompt
│
├── schema/
│   ├── ontology_schema_general.json # Ontology schema (node/edge types)
│   └── type_mapping.json            # English type name → localized name
│
├── block/                           # Block View (logical block segmentation)
│
└── utils/
    ├── document_utils.py            # Chunking
    └── io_utils.py                  # File load / JSON parsing
```

**Dependent infrastructure:**
```
infra/repositories/graph/
└── graphnode_repository.py          # GraphNodeDBHandler (Neo4j + ChromaDB)
```

---

## Ontology schema

The extraction is driven by `schema/ontology_schema_general.json`, which defines the
allowed node types, edge types, and per-field extraction requirements. A caller may pass a
`schema_name` to select a domain-specific schema file; if it is unset or missing, the
general schema is used as the default fallback.

---

## Running it

The pipeline runs through the SQS worker:

```bash
python -m server.worker --dev
```

Ingest is triggered by `MICROSCOPE_INGEST_FROM_NODE_REQUEST` (production) or
`MICROSCOPE_INGEST_REQUEST` (file upload); queries by `MICROSCOPE_QUERY_REQUEST`.
