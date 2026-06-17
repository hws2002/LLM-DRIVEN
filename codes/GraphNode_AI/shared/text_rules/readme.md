# Text Normalization Rules (`shared/text_rules/`)

This directory holds the **text normalization and stopword rules** shared by GraphNode's
preprocessing pipelines (`Macro`, `add_node`). The keyword-extraction behavior of the
whole system can be updated by editing the text files (.txt) here — no Python changes needed.

## File structure and roles

### 1. Project domain noise
* **`base_stopwords.txt`**
  * **Description:** Words that are grammatically meaningful but, given our service
    (conversation/QA-based knowledge graph), act as "noise" that hinders graph connections.
  * **Includes:** conversational fillers, QA-related meta words, etc.

### 2. General stopwords (per language)
* **`stopwords_ko.txt`** (Korean), **`stopwords_en.txt`** (English), **`stopwords_zh.txt`** (Chinese)
  * **Description:** General-purpose stopwords with no substantive grammatical meaning in each language.
  * **Includes:** conjunctions, prepositions, articles, particles, etc.
  * **Examples (en):** `the`, `a`, `is`, `of`, `please`, `just`
  * **Examples (zh):** `的`, `了`, `是`, `和`

### 3. Tech tokens & VIPs
* **`force_keep_tokens.txt`**
  * **Description:** VIP keywords that contain special characters or spaces and could be
    destroyed by a normal tokenizer, but **must be preserved** in technical conversations.
  * **Examples:** `node.js`, `c++`, `x->0`, `a/b`, `chatgpt`, `gpt-4`

### 4. Korean-specific NLP rules
* **`foreign_ko_suffixes.txt`** — Korean particles/endings that attach right after a
  foreign-language or tech token and contaminate it (e.g. "chatgpttext" → extract "chatgpt").
* **`korean_endings.txt`** — Korean verb/adjective inflection endings with no keyword value,
  whose tails should be trimmed (e.g. `text`, `text`).
* **`korean_strip_only.json`** — Exception cases that normal rules cannot cleanly strip and
  that require an explicit 1:1 mapping (e.g. `text` → `text`, `text` → `text`).

---

## Maintenance guide (how to update)

1. A Korean suffix is not being stripped → add it to `foreign_ko_suffixes.txt`.
2. A Chinese particle (e.g. `吗`) is over-captured as a keyword → add it to `stopwords_zh.txt`.
3. A filler word pollutes nodes → add it to `base_stopwords.txt`.
4. A tech token (e.g. `next.js`) keeps splitting into `next` + `js` → add it to `force_keep_tokens.txt`.

**⚠️ Note:** Adding too many per-language stopwords (`stopwords_*.txt`) can remove keywords
that carry core meaning, so prefer managing domain-specific words in `base_stopwords.txt`.
