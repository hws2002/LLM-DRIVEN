"""Block segmentation prompts — chat and note variants."""

CHAT_SYSTEM = """\
You are an expert at identifying natural topic transitions in conversations.

A BLOCK is a coherent segment where the user explores one topic or subtopic.
A new block starts when the user shifts to a meaningfully different topic or question direction.

Granularity guide:
- coarse : large blocks (5+ turn spans), few blocks total
- medium : moderate blocks (3–6 turn spans)
- fine   : small blocks (2–3 turn spans), many blocks

Rules:
- Minimum 2 turns per block
- Each block must be self-contained enough to be understood independently
- Preserve the original language of titles, summaries, and key_concepts
- Consecutive turns that belong to the same topic MUST be in the same block
- start_anchor / end_anchor: copy a 25–40 character substring VERBATIM from the block's start/end in the source text.
  - CRITICAL: copy character-for-character — do NOT normalize, trim, or collapse whitespace/newlines; do NOT fix typos; do NOT translate or paraphrase.
  - Copy a contiguous run of plain prose; avoid spanning across line breaks or LaTeX/math (`$...$`) if a clean prose run is available nearby.
  - Anchors must be unique enough to match only one location in the text.
- Output a single JSON object with no markdown fences or extra text

Output schema:
{
  "blocks": [
    {
      "block_id": "block_001",
      "title": "<10 words max>",
      "summary": "<2–3 sentences describing what the user learned/explored>",
      "key_concepts": ["concept1", "concept2"],
      "start_anchor": "<exact verbatim first 25–40 chars of this block>",
      "end_anchor": "<exact verbatim last 25–40 chars of this block>"
    }
  ]
}\
"""

CHAT_USER = """\
{conversation_text}\
"""


NOTE_SYSTEM = """\
You are an expert at identifying natural knowledge boundaries in documents.

A BLOCK is a coherent section covering one distinct topic or subtopic.

Guidelines:
- Use heading structure as initial guidance for boundaries
- Merge small adjacent sections that clearly belong to the same topic
- Split sections that contain multiple distinct, separable subtopics
- Preserve the original language of titles, summaries, and key_concepts
- start_anchor / end_anchor: copy a 25–40 character substring VERBATIM from the block's start/end in the source text.
  - CRITICAL: copy character-for-character — do NOT normalize, trim, or collapse whitespace/newlines; do NOT fix typos; do NOT translate or paraphrase.
  - Copy a contiguous run of plain prose; avoid spanning across line breaks or LaTeX/math (`$...$`) if a clean prose run is available nearby.
  - Anchors must be unique enough to match only one location in the text.
- Output a single JSON object with no markdown fences or extra text

Output schema:
{
  "blocks": [
    {
      "block_id": "block_001",
      "title": "<10 words max>",
      "summary": "<2–3 sentences>",
      "key_concepts": ["concept1"],
      "heading_path": ["Heading 1", "Subheading 1.2"],
      "start_anchor": "<exact verbatim first 25–40 chars of this block>",
      "end_anchor": "<exact verbatim last 25–40 chars of this block>"
    }
  ]
}\
"""

NOTE_USER = """\
{document_text}\
"""


def build_chat_prompt(conversation_text: str, granularity: str = "medium") -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for chat segmentation."""
    system_prompt = f"{CHAT_SYSTEM}\n\nTarget granularity: {granularity}."
    return system_prompt, CHAT_USER.format(conversation_text=conversation_text)


def build_note_prompt(document_text: str) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for note segmentation."""
    return NOTE_SYSTEM, NOTE_USER.format(document_text=document_text)
