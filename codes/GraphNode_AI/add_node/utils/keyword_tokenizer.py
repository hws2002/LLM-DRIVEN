"""Compatibility wrapper for the shared multilingual tokenizer.

Historically, add_node owned its own tokenizer implementation. The shared text
core now provides the canonical tokenizer so that macro and add_node extract
keywords under the same normalization rules.
"""

from __future__ import annotations

from typing import List

from shared.text_core import shared_multilingual_tokenize


def multi_lang_tokenize(text: str) -> List[str]:
    return shared_multilingual_tokenize(text)
