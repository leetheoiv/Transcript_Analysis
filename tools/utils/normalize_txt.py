"""
tools/utils/normalize_txt.py

Text normalization and tokenization helpers for filename scoring and search ranking.
"""

import regex as re

def _normalize_text(text: str) -> str:
    """Normalize text to lowercase alphanumeric tokens separated by spaces.

    Args:
        text: Raw input text.

    Returns:
        Normalized string with only lowercase letters, digits, and spaces.
    """
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

def _tokenize(text: str) -> list[str]:
    """Tokenize text into a list of normalized word tokens.

    Args:
        text: Raw input text.

    Returns:
        List of non-empty token strings.
    """
    return [t for t in _normalize_text(text).split() if t]