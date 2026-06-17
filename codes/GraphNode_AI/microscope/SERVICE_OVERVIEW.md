# Microscope RAG — Service Overview

This document briefly describes what the main query/synthesis services of `microscope` do,
and how they work. The implementations live in `microscope/services/rag_service.py`.

## 1) Hybrid Query Service — `run_query()`

What it does:
- Answers a user's question using vector search **plus** graph expansion.
- Compared to plain similar-document retrieval, it attaches context expanded to related
  concepts, producing more grounded answers.

How it works:
- First, find chunks semantically close to the question in ChromaDB.
- Extract the entities linked to those initial chunks.
- In Neo4j, expand to neighbor entities by hops (1-hop, 2-hop, …).
- Gather the chunks linked to those neighbor entities.
- Deduplicate chunks into a single context and pass it to the LLM to generate an answer.
- Optionally, the user's **Macro-Graph profile** (interests, learning style, patterns) is
  fused into the prompt for a personalized answer.

---

## 2) Synthesize Service — `run_synthesize()`

What it does:
- Summarizes the key content for a given topic.
- Focused on "organizing a topic" rather than question answering.

How it works:
- Like Hybrid Query, collects topic-related chunks via vector search + graph expansion.
- The LLM generates a summary from the collected context.
- Optionally RAG can be turned off (no context) for a general-knowledge summary.

---

## 3) Related Questions Service — `run_related_questions()`

What it does:
- Generates follow-up questions worth asking next from the current question.
- Useful for exploratory learning/research flows that auto-suggest the next question.

How it works:
- Find initial chunks related to the question via vector search.
- Extract entities from the chunks and broaden them via graph hop expansion.
- The LLM generates a follow-up question list from the original question + expanded entity set.

---

## Common concepts

- `top_k`: number of chunks to retrieve in the initial vector search
- `hop_depth`: depth of graph neighbor expansion
- `group_id`: data-scope filter (when empty, search can span all of a user's groups)
- `no_rag`: turn off context retrieval/expansion and use the question only

In essence, these services combine **vector similarity search** with **graph relation
traversal** to provide more structured context than plain retrieval.
