"""Phase 2: Entity standardization prompts."""

ENTITY_RESOLUTION_SYSTEM_PROMPT = """
You are an expert in entity resolution and knowledge representation.
Your task is to standardize entity names from a knowledge graph to ensure consistency.
"""


def get_entity_resolution_user_prompt(entity_list):
    return f"""
Below is a list of entity names extracted from a knowledge graph.
Some may refer to the same real-world entities but with different wording.

Please identify groups of entities that refer to the **exact same concept** expressed differently (e.g. abbreviations, translations, typos, alternate spellings).
Return your answer as a JSON object where the keys are the standardized names and the values are arrays of all variant names that should map to that standard name.
Only include entities that have multiple variants or need standardization.

Important rules:
- Do NOT merge entities that are merely related or belong to the same category. They must denote the exact same thing.
- Do NOT abstract specific instances into their parent category.
- Prefer the most specific and complete name as the standard form.
- **Language**: Use the same language as the majority of the entity names in the list. If most names are in Korean, the standardized name MUST be in Korean. Do NOT translate names into another language.

Entity list:
{entity_list}

Format your response as valid JSON like this:
{{
  "standardized name 1": ["variant 1", "variant 2"],
  "standardized name 2": ["variant 3", "variant 4", "variant 5"]
}}

"""
