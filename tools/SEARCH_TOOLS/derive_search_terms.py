"""
tools/SEARCH_TOOLS/derive_search_terms.py

Lightweight text-splitting utility for generating search terms from
reasoning or quote text by splitting on punctuation and keeping
non-trivial phrases.
"""

import re

def derive_search_terms(text: str, max_terms: int = 10) -> list[str]:
    """
    Derive lightweight transcript search terms from some text.

    Strategy:
    - split on punctuation
    - keep non-trivial phrases
    - dedupe
    """
    if not text or not text.strip():
        return []

    chunks = re.split(r"[.;,\n]+", text)
    candidates = []

    for chunk in chunks:
        text = chunk.strip()
        if len(text) >= 6:
            candidates.append(text)

    seen = set()
    result = []
    for c in candidates:
        if c.lower() not in seen:
            seen.add(c.lower())
            result.append(c)
        if len(result) >= max_terms:
            break

    return result