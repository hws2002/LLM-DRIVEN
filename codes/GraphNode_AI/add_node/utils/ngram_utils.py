"""English documentation."""

from typing import List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from .keyword_tokenizer import multi_lang_tokenize


def build_candidates_for_text(text: str, ngram_max: int, max_candidates: int) -> List[str]:
    """English documentation."""
    if not text:
        return []

    vectorizer = CountVectorizer(
        analyzer="word",
        tokenizer=multi_lang_tokenize,
        token_pattern=None,
        ngram_range=(1, ngram_max),
        min_df=1,
    )
    X = vectorizer.fit_transform([text])
    vocab = np.array(vectorizer.get_feature_names_out())
    counts = X.toarray()[0]

    order = np.argsort(-counts)
    if max_candidates > 0:
        order = order[:max_candidates]

    return vocab[order].tolist()
