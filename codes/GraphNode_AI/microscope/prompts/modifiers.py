"""English documentation."""

from typing import Optional

# ---------------------------------------------------------------------------
# English comment.
# ---------------------------------------------------------------------------

_MODIFIERS: dict[str, str] = {

    # English comment.
    "default": "",

    # English comment.
    "concept_map": (
        "Your goal is to build a clear conceptual map of the material.\n"
        "- Prioritize extracting key terms, definitions, and abstract concepts as nodes.\n"
        "- Emphasize hierarchical and definitional relationships: "
        "'defines', 'part_of', 'prerequisite_of', 'example_of', 'equivalent_to'.\n"
        "- Every important term mentioned in the text should become a node, "
        "even if only briefly introduced.\n"
        "- Avoid over-extracting trivial examples; keep examples as leaf nodes."
    ),

    # English comment.
    "exam_review": (
        "Your goal is to extract a study graph for exam preparation.\n"
        "- Focus on facts, definitions, theorems, and formulas that are likely to be tested.\n"
        "- Strongly prioritize dependency and prerequisite relationships: "
        "'requires', 'prerequisite_of', 'proves', 'derives_from', 'leads_to'.\n"
        "- Identify which concepts must be understood before others "
        "to reveal a clear study order.\n"
        "- Include contrasting pairs ('contrasts_with') where they clarify common confusions.\n"
        "- Skip motivational context and examples unless they directly clarify a tested concept."
    ),

    # English comment.
    "paper_reading": (
        "Your goal is to extract the logical and argumentative structure of an academic text.\n"
        "- Identify the core Problem or research question, the proposed Method or Solution, "
        "and the Results or Claims as primary nodes.\n"
        "- Prioritize causal and argumentative relationships: "
        "'solves', 'causes', 'improves', 'contrasts_with', 'leads_to', 'applied_in'.\n"
        "- Extract the evidence or assumptions underlying each claim.\n"
        "- Note limitations or open problems explicitly mentioned in the text.\n"
        "- Do not over-extract background detail unrelated to the paper's core contribution."
    ),

    # English comment.
    "lecture_review": (
        "Your goal is to reconstruct the flow and key takeaways of a lecture or class note.\n"
        "- Extract the main concepts introduced in order, preserving the pedagogical sequence.\n"
        "- Prioritize 'leads_to', 'defines', 'example_of', 'uses', and 'causes' relationships "
        "to reflect how the instructor built up ideas step by step.\n"
        "- Include concrete examples and analogies as nodes connected to their parent concept.\n"
        "- Highlight any emphasis signals in the text "
        "(e.g., 'important', 'key point', 'remember') by assigning higher confidence.\n"
        "- Preserve the informal structure of notes; do not over-formalize casual explanations."
    ),
}

# ---------------------------------------------------------------------------
# English comment.
# ---------------------------------------------------------------------------

VALID_MODIFIERS = list(_MODIFIERS.keys())


def get_modifier_prompt(modifier_name: Optional[str]) -> str:
    """English documentation."""
    if not modifier_name:
        return _MODIFIERS["default"]
    result = _MODIFIERS.get(modifier_name)
    if result is None:
        import logging
        logging.getLogger("modifiers").warning(
            "Unknown modifier '%s', falling back to 'default'.", modifier_name
        )
        return _MODIFIERS["default"]
    return result
