from pathlib import Path
from typing import Dict
from .entity_relation_prompt import get_system_prompt_with_schema_dict, USER_WANTED_PROMPT
from .standardization_prompt import ENTITY_RESOLUTION_SYSTEM_PROMPT, get_entity_resolution_user_prompt
from .modifiers import get_modifier_prompt

class PromptFactory:
    def __init__(self, modifier: str | None = None, schema: Dict | None = None):
        self.PROMPT_DIR = Path(__file__).parent
        self.schema = schema
        user_wanted = get_modifier_prompt(modifier) if modifier else USER_WANTED_PROMPT
        self.PROMPT_DICT = {
            "extraction_system": lambda: get_system_prompt_with_schema_dict(self.schema),
            "extraction_user_wanted": user_wanted,
            "standardization_system": ENTITY_RESOLUTION_SYSTEM_PROMPT,
            "standardization_user" : get_entity_resolution_user_prompt
        }

    def get_prompt(self, key: str, *args):
        value = self.PROMPT_DICT[key]
        return value(*args) if callable(value) else value