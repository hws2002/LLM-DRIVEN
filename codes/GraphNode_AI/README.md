# GraphNode AI

> AI backend service for GraphNode, a knowledge-graph platform driven by large language models.
>
> This repository is the source-code submission for the course project **Large-Model-Driven Software Development**. It converts AI conversations and user notes into navigable knowledge graphs, and supports retrieval-augmented question answering over those graphs.

This README is a source-code guide. It explains the current module layout, the main entry points, and how each folder maps to the system described in the final report.

---

## 1. Project Scope

GraphNode AI is a backend engine without a standalone frontend UI. It runs as an HTTP/SQS service and coordinates several LLM-driven pipelines:

- **Macro Graph**: builds a global graph from all conversations and notes.
- **Microscope**: extracts fine-grained concept-relation graphs from a single document.
- **GraphRAG**: answers questions using graph context and vector retrieval.
- **add_node**: incrementally inserts new conversations or notes into an existing graph.

The central design is that the LLM is not an extra chat feature. It is the semantic structuring engine for clustering, topic naming, relation extraction, graph summaries, and contextual answers.

---

## 2. Top-Level Structure

```text
GraphNode_AI/
├── add_node/          # Incremental graph update pipeline
├── dto/               # Request and response data models
├── infra/             # Database and storage repository adapters
├── macro/             # Global Macro Graph generation pipeline
├── microscope/        # Single-document graph extraction and GraphRAG
├── server/            # HTTP server and SQS worker entry points
├── shared/            # Shared config, LLM providers, text processing, cost utilities
├── .env.example       # Secret template; real .env is not committed
├── .gitignore
├── README.md
└── requirements.txt
```

Generated files such as `__pycache__/`, local output folders, logs, model caches, and real `.env` files are intentionally excluded by `.gitignore`.

---

## 3. Core Modules

| Module | Responsibility | Main Entry |
|---|---|---|
| `server/` | Receives HTTP/SQS tasks, routes requests, reports progress | `server/worker.py`, `server/main.py` |
| `macro/` | Generates global knowledge graphs from all user conversations and notes | `macro/src/run_pipeline.py` |
| `microscope/` | Extracts concept-relation graphs from a single document and supports document-level analysis | `microscope/call.py` |
| `microscope/rag/` | Builds GraphRAG context and answer prompts | `microscope/rag/retrieval_strategies.py`, `microscope/rag/prompt_builder.py` |
| `microscope/services/` | Service-level RAG API logic | `microscope/services/rag_service.py` |
| `add_node/` | Adds new conversations or notes into an existing Macro Graph | `add_node/call.py` |
| `infra/` | Wraps Neo4j, ChromaDB, MongoDB, and combined graph repositories | `infra/repositories/` |
| `shared/` | Shared LLM provider abstraction, config, tokenizer/keyword cleanup, pricing, and logging | `shared/api_provider.py`, `shared/config.py`, `shared/text_core.py` |
| `dto/` | Typed request/response payloads for server and pipeline calls | `dto/server_dto.py`, `dto/microscope_dto.py` |

---

## 4. Detailed Directory Layout

```text
add_node/
├── call.py                         # Batch and single-item add-node pipeline
├── readme.md
├── analyze/                        # Conversation loading and Q-A parsing helpers
├── steps/                          # Pipeline steps: QA extraction, keywords, clustering, edges
├── stop_words/                     # Stopword resources
└── utils/                          # Embedding, clustering, IO, prompt, and similarity helpers

dto/
├── server_dto.py                   # HTTP/SQS request models
└── microscope_dto.py               # Microscope pipeline context models

infra/
├── README.md
└── repositories/
    ├── graph/                      # Combined graph repository facade
    ├── mongodb/                    # MongoDB access
    ├── neo4j/                      # Neo4j access
    └── vectordb/                   # ChromaDB stores for nodes and chunks

macro/
├── README.md
├── config.yaml
└── src/
    ├── run_pipeline.py             # Macro pipeline orchestration
    ├── extract_features.py         # Embeddings and keyword extraction
    ├── cluster_with_llm.py         # LLM-assisted cluster generation and assignment
    ├── build_edges.py              # Semantic edge construction
    ├── build_subclusters.py        # Subcluster construction
    ├── merge_graph.py              # Graph post-processing and merge logic
    ├── insights/                   # Graph summary, user pattern discovery, vector indexing
    └── util/                       # File loaders, graph utilities, Notion and raw file support

microscope/
├── README.md
├── SERVICE_OVERVIEW.md
├── call.py                         # Main document ingestion and extraction entry
├── block/                          # Block View segmentation, ordering, and assembly
├── graph_generation/               # Entity and relation extraction orchestration
├── prompts/                        # Prompt templates and prompt factory
├── rag/                            # Retrieval, context building, answer generation
├── schema/                         # Ontology and type mapping JSON files
├── services/                       # RAG service layer
├── tools/                          # Local graph visualization helpers
└── utils/                          # Document, IO, and LLM utility functions

server/
├── worker.py                       # SQS worker and production task router
└── main.py                         # Lightweight HTTP API server

shared/
├── api_provider.py                 # Unified LLM API wrapper
├── config.py                       # Public runtime configuration
├── env_loader.py                   # Environment loader
├── text_core.py                    # Keyword normalization and multilingual text cleanup
├── text_rules/                     # Stopwords and keyword cleanup rule files
├── cost_calculator.py              # Token and API cost utilities
├── token_usage.py
├── llm_model_aliases.json
├── llm_pricing.json
└── tools/mem_check.py              # Local memory profiling helper
```

---

## 5. Important Source Index

| Need to inspect | Open this |
|---|---|
| SQS worker routing | `server/worker.py` |
| HTTP request models | `dto/server_dto.py` |
| Macro pipeline orchestration | `macro/src/run_pipeline.py` |
| Macro feature extraction | `macro/src/extract_features.py` |
| LLM cluster assignment prompts and parsing | `macro/src/cluster_with_llm.py` |
| Graph summary and pattern discovery | `macro/src/insights/discovery/graph_summarizer.py` |
| Microscope main flow | `microscope/call.py` |
| Entity-relation prompt template | `microscope/prompts/entity_relation_prompt.py` |
| Block View segmentation | `microscope/block/segmenter.py` |
| GraphRAG retrieval | `microscope/rag/retrieval_strategies.py` |
| RAG service API | `microscope/services/rag_service.py` |
| Incremental add-node pipeline | `add_node/call.py` |
| Repository facade for graph/vector DB | `infra/repositories/graph/graphnode_repository.py` |
| LLM provider abstraction | `shared/api_provider.py` |
| Shared text normalization | `shared/text_core.py` |

---

## 6. Runtime Dependencies

| Category | Technology |
|---|---|
| Language | Python 3.11+ |
| LLM providers | OpenAI / Groq / Z.AI / OpenRouter through `shared/api_provider.py` |
| Embeddings | `sentence-transformers` |
| Keyword extraction | KeyBERT |
| Graph and clustering | NetworkX, cosine similarity, Louvain-style community detection |
| Graph database | Neo4j |
| Vector database | ChromaDB |
| Document database | MongoDB |
| Message queue | AWS SQS |

---

## 7. Running Locally

This service depends on external infrastructure: Neo4j, ChromaDB, MongoDB, AWS SQS, and LLM API keys. For source review, the most useful setup is:

```bash
pip install -r requirements.txt
cp .env.example .env
python -m server.worker --dev
```

Only secret values should be placed in `.env`. Public configuration such as model names, chunk sizes, and service defaults is kept in `shared/config.py`.

---

## 8. SQS Task Types

| `taskType` | Pipeline | Description |
|---|---|---|
| `ADD_NODE_REQUEST` | `add_node` | Incrementally add conversations or notes |
| `GRAPH_GENERATION_REQUEST` | `macro` | Generate a global Macro Graph |
| `GRAPH_SUMMARY_REQUEST` | `macro` | Generate graph summaries and user insights |
| `MICROSCOPE_INGEST_FROM_NODE_REQUEST` | `microscope` | Ingest one source node into Microscope |
| `MICROSCOPE_QUERY_REQUEST` | `microscope/rag` | Answer a graph-grounded question |
| `MICROSCOPE_SYNTHESIZE_REQUEST` | `microscope/rag` | Synthesize a topic from retrieved graph context |
| `MICROSCOPE_RELATED_QUESTIONS_REQUEST` | `microscope/rag` | Generate related follow-up questions |

---

## 9. Mapping to the Final Report

| Final report section | Code location |
|---|---|
| Macro Graph | `macro/` |
| Microscope | `microscope/call.py`, `microscope/block/`, `microscope/schema/` |
| GraphNode Agent / RAG | `microscope/rag/`, `microscope/services/rag_service.py` |
| LLM as knowledge structuring engine | Prompt modules in `macro/`, `microscope/prompts/`, and `microscope/block/prompts/` |
| Engineering implementation appendix | `server/`, `infra/`, `shared/`, `dto/` |

---

## 10. Submission Notes

- Do not commit a real `.env` file.
- Do not commit local databases, model caches, generated outputs, logs, or `__pycache__` files.
- Use `.env.example` only as a template for required secrets.
- If this folder is committed from the parent `LLM-DRIVEN` repository, remember that the parent `.gitignore` ignores `codes/`; use `git add -f codes/GraphNode_AI` or update the parent ignore rules intentionally.
