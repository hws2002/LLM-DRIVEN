# add_node

Pipeline package that processes a single conversation into one node and its edges.
It stores the conversation as an embedding in ChromaDB and connects it, via edges,
to existing nodes within the same major cluster.

---

## Pipeline overview

```
Conversation JSON
  └─► Step 1  Extract Q-A pairs
        └─► Step 2  Keyword + QA embedding extraction  (KeyBERT / sentence-transformers)
              └─► Step 3  Q-A clustering  (all_qa | HDBSCAN)
                    └─► Step 4  Conversation embedding  (length-weighted pooling)
                          └─► Step 5  Major-cluster assignment  (LLM)
                                └─► Step 6  Fetch existing nodes in cluster  (ChromaDB)
                                      └─► Step 7  Edge creation  (cosine similarity)
                                            └─► ChromaDB upsert (store node embedding)
```

**Entry point:** `add_node/call.py`

| Function | Description |
|----------|-------------|
| `run_add_node_pipeline()` | Process one conversation |
| `run_add_node_batch_pipeline()` | Process multiple conversations sequentially (earlier results feed later ones) |

---

## Step details

### Step 1 — Extract Q-A pairs
Extracts question (Q) / answer (A) pairs from the conversation JSON and saves them as
`qa_pairs_{conv_id}.json`. If no Q-A pairs exist, the pipeline exits early (`skipped: true`).

### Step 2 — Keyword + embedding extraction
Extracts keywords per Q-A pair and generates embeddings via sentence-transformers.

| Setting (`shared/config.py`) | Default | Description |
|------------------------------|---------|-------------|
| `ADDNODE_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Embedding model |
| `ADDNODE_KEYWORD_METHOD` | `keybert` | `keybert` \| `ngram` \| `langchain` |
| `ADDNODE_KEYWORD_TOP_N` | `10` | Top keywords per QA |
| `ADDNODE_NGRAM_MAX` | `3` | Max n-gram length |

### Step 3 — Q-A clustering
Behavior depends on `ADDNODE_QA_CLUSTERING_MODE`.

| Mode | Behavior |
|------|----------|
| `all_qa` (default) | Skip clustering; pick the top 5 keywords over all Q-A |
| `hdbscan` | Cluster Q-A with HDBSCAN, then pick 2 keywords per cluster |

### Step 4 — Conversation embedding pooling
Pools the Q-A pair embeddings with QA-length weighting to produce a single embedding
representing the whole conversation.

### Step 5 — Major-cluster assignment
The LLM examines the existing cluster list and the selected keywords, then either picks the
most suitable cluster or decides to create a new one.
- No existing clusters → always new
- LLM failure → fallback (rule-based assignment by keyword similarity)

Output: `{ cluster_id, is_new_cluster, confidence, reasoning }`

### Step 6 — Fetch existing nodes in cluster
Using the assigned cluster ID, fetches up to 20 nodes of the same cluster from ChromaDB.
Skipped for a new cluster.

### Step 7 — Edge creation
Computes cosine similarity between the new node and existing nodes to create edges.
If no edges are produced from the in-cluster candidates, it queries once more over all of
the user's nodes to compensate for misses caused by cluster-assignment error.

| Setting (`shared/config.py`) | Default | Description |
|------------------------------|---------|-------------|
| `ADDNODE_EDGE_SIMILARITY_THRESHOLD` | `0.6` | Minimum similarity for an edge |
| `ADDNODE_EDGE_TOP_K` | `5` | Max number of edges |
| `ADDNODE_EDGE_FETCH_TOP_K` | `20` | In-cluster candidate fetch count |
| `ADDNODE_EDGE_FALLBACK_ENABLED` | `true` | Use user-wide fallback fetch when 0 edges |
| `ADDNODE_EDGE_FALLBACK_TOP_K` | `20` | Fallback candidate fetch count |

---

## Output format

```json
{
  "nodes": [{ "id": "{user_id}_{conv_id}", "clusterId": "cluster_3", "numMessages": 10 }],
  "edges": [{ "source": "...", "target": "...", "weight": 0.82 }],
  "assignedCluster": {
    "clusterId": "cluster_3",
    "isNewCluster": false,
    "confidence": 0.9,
    "reasoning": "...",
    "name": "...",
    "themes": ["...", "..."]
  },
  "selectedKeywords": ["...", "..."],
  "outputDev": { "retrievedCandidates": 12, "similarityTop": [...] }
}
```

> Node embedding vectors are stripped from the external response (no `embedding` field).

---

## Batch processing behavior

`run_add_node_batch_pipeline()` processes conversations in order and:
- When a new cluster is created, it is **immediately added to the cluster candidates of later conversations** (in-batch sharing).
- After each conversation, its node is stored in ChromaDB so **later conversations' edge creation reflects it**.
- If one conversation fails, only that conversation is marked `skipped: true` while the rest continue.
- Uses a user-scoped tmp directory to avoid file collisions across concurrent requests.

---

## Directory structure

```
add_node/
├── call.py                          # Pipeline entry point (single + batch)
├── config.py                        # Env-var based settings loader
│
├── steps/
│   ├── build_qa_pairs.py            # Step 1: extract Q-A pairs
│   ├── extract_qa_keywords.py       # Step 2: keywords + embeddings
│   ├── cluster_qa.py                # Step 3: Q-A clustering
│   ├── pool_qa_embeddings.py        # Step 4: embedding pooling
│   ├── assign_cluster_llm.py        # Step 5: LLM cluster assignment
│   └── create_edges.py              # Step 7: edge creation
│
├── utils/
│   ├── embedding_utils.py           # Embedding utilities
│   ├── clustering_utils.py          # HDBSCAN clustering
│   ├── similarity_utils.py          # Cosine similarity
│   ├── keyword_tokenizer.py         # Tokenizer
│   ├── ngram_utils.py               # n-gram keywords
│   ├── prompt_builder.py            # Cluster-assignment prompt
│   ├── preprocess.py                # Text preprocessing
│   └── io_helpers.py                # File I/O
│
└── analyze/                         # Offline analysis helpers
    ├── loader.py
    └── parser.py
```

**Dependent infrastructure:**
```
infra/repositories/vectordb/
└── macro_node_store.py              # MacroNodeStore (ChromaDB node-embedding store)
```
