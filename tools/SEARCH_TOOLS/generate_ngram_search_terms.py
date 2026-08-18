"""
tools/SEARCH_TOOLS/generate_ngram_search_terms.py

Generates ranked n-gram search terms from text using sklearn's CountVectorizer.
Useful for deriving candidate search queries from reasoning or claim text.
"""

from sklearn.feature_extraction.text import CountVectorizer

def generate_ngram_search_terms(
    text: str,
    ngram_range: tuple[int, int] = (1, 3),
    top_k: int = 15,
    stop_words: str | None = "english",
) -> list[str]:
    """Generate ranked n-gram search terms from text using sklearn's CountVectorizer.

    Extracts unigrams through trigrams (by default), removes stop words, and
    returns the top-k terms sorted by phrase length (longer phrases first).

    Args:
        text: Input text to extract n-grams from.
        ngram_range: Tuple of (min_n, max_n) for n-gram extraction.
        top_k: Maximum number of terms to return.
        stop_words: Stop word configuration for CountVectorizer (e.g., 'english' or None).

    Returns:
        List of up to top_k n-gram strings, sorted by descending phrase length.
    """
    if not text or not text.strip():
        return []

    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        stop_words=stop_words,
    )

    X = vectorizer.fit_transform([text])
    terms = vectorizer.get_feature_names_out().tolist()

    # Optional sort by length or phrase size
    terms = sorted(terms, key=lambda x: (-len(x.split()), -len(x)))

    return terms[:top_k]