from __future__ import annotations

import shared.config as cfg
from shared.text_core import preprocess_text_for_pipeline


def preprocess_content(text: str) -> str:
    """English documentation."""
    text_core_profile = getattr(cfg, "TEXT_CORE_PROFILE_ADDNODE", "balanced")
    return preprocess_text_for_pipeline(
        text,
        lower=True,
        strip_code=True,
        strip_urls=True,
        strip_html=True,
        strip_citations=True,
        strip_punct=False,
        strip_inline_code=True,
        strip_emoji=True,
        segment_cjk=False,
        profile=text_core_profile,
    )
