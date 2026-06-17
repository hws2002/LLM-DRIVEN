# Macro

Pipeline that analyzes conversation history to build a knowledge graph and generate
summary insights.

## Overall flow

```
Conversation JSON → embedding/keyword extraction → LLM clustering → edge creation → graph merge → (summary)
```

---

## Folder structure

```
macro/
├── src/
│   ├── run_pipeline.py          # Full-pipeline orchestration
│   ├── extract_features.py      # Step 1: embedding + keyword extraction
│   ├── cluster_with_llm.py      # Step 2: LLM-based clustering
│   ├── build_edges.py           # Step 3: similarity-based edge creation
│   ├── merge_graph.py           # Step 4: graph merge
│   ├── build_subclusters.py     # Step 5: sub-cluster creation
│   ├── util/                    # Shared utilities
│   └── insights/                # Graph summary & insights module
└── config.yaml                  # Pipeline configuration
```

---

## Running it

### Via SQS worker (recommended)

The macro pipeline runs as a `GRAPH_GENERATION_REQUEST` / `GRAPH_SUMMARY_REQUEST` task,
dispatched through the SQS worker:

```bash
python -m server.worker --dev
```

### Local direct run (debugging)

```bash
cd macro/src

# Full pipeline
python run_pipeline.py \
  --input <input.json> \
  --config ../config.yaml \
  --output-dir ../output \
  --provider openai \
  --model gpt-4o-mini

# Step by step
python extract_features.py --in <input.json> --out ../output/features.json --cfg ../config.yaml
python cluster_with_llm.py --input ../output/features.json --output ../output/clusters.json --provider openai --model gpt-4o-mini
python build_edges.py --intermediate ../output/features.json --clusters ../output/clusters.json --output ../output/edges.json
python merge_graph.py --features ../output/features.json --clusters ../output/clusters.json --edges ../output/edges.json --output ../output/graph.json
```

---

## SQS message types

| taskType | Description |
|----------|-------------|
| `GRAPH_GENERATION_REQUEST` | Graph generation request |
| `GRAPH_GENERATION_RESULT` | Graph generation result |
| `GRAPH_SUMMARY_REQUEST` | Summary generation request |
| `GRAPH_SUMMARY_RESULT` | Summary generation result |

---

## More detail

- Per-step pipeline details: `src/README.md`
