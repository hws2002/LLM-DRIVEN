"""Block ordering / dependency analysis prompts."""

SYSTEM = """\
You are an expert at analyzing knowledge dependencies and designing optimal learning paths.

Given a set of knowledge blocks, your task:
1. Identify conceptual prerequisites between blocks
2. Suggest one or more optimal learning paths (consciousness flow)

Edge types:
- PREREQUISITE_OF : Block A MUST be understood before Block B (hard dependency)
- FOLLOWS         : Block B naturally continues from Block A (soft, sequential dependency)
- CONTRASTS       : A and B present contrasting perspectives on the same topic
- ELABORATES      : Block B provides deeper detail on concepts introduced in Block A
- PARALLEL        : No meaningful dependency; can be studied in any order

IMPORTANT rules:
- The recommended paths do NOT need to follow the original document/conversation order
- Reorder blocks based on conceptual dependencies for optimal understanding
- Multiple valid paths are allowed (DAG structure, not strictly linear)
- Only include edges where a meaningful relationship actually exists — do not enumerate all pairs
- Prefer PREREQUISITE_OF and FOLLOWS for clear dependencies; use PARALLEL sparingly
- Output a single JSON object with no markdown fences or extra text

Output schema:
{
  "edges": [
    {
      "source": "block_001",
      "target": "block_002",
      "type": "PREREQUISITE_OF",
      "description": "<why this relationship exists — 1 sentence>",
      "confidence": 0.9
    }
  ],
  "recommended_paths": [
    ["block_001", "block_003", "block_002"]
  ],
  "ordering_rationale": "<1–2 sentences explaining the recommended ordering>"
}\
"""

USER_TEMPLATE = """\
Analyze the knowledge dependencies between these blocks and suggest optimal learning paths.

{block_summaries}\
"""


def build_block_summary(blocks) -> str:
    lines = []
    for b in blocks:
        concepts = ", ".join(b.key_concepts) if b.key_concepts else "—"
        lines.append(
            f'[{b.block_id}] "{b.title}"\n'
            f"Summary: {b.summary}\n"
            f"Key concepts: {concepts}"
        )
    return "\n\n".join(lines)


def build_ordering_prompt(blocks) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for block ordering."""
    return SYSTEM, USER_TEMPLATE.format(block_summaries=build_block_summary(blocks))
