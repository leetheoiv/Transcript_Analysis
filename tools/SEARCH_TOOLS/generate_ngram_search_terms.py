"""
tools/SEARCH_TOOLS/generate_ngram_search_terms.py

Generates ranked n-gram search terms from text using sklearn's CountVectorizer.
Useful for deriving candidate search queries from reasoning or claim text.

Supports multilingual stop word removal (English + Spanish) for transcripts
that may appear in either language.
"""

# Common Spanish stop words (articles, prepositions, conjunctions, pronouns)
_SPANISH_STOP_WORDS = [
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por",
    "un", "para", "con", "no", "una", "su", "al", "lo", "como", "mas", "pero",
    "sus", "le", "ya", "o", "este", "si", "porque", "esta", "entre", "cuando",
    "muy", "sin", "sobre", "tambien", "me", "hasta", "hay", "donde", "quien",
    "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra",
    "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mi", "antes", "algunos",
    "que", "unos", "yo", "otro", "otras", "otra", "el", "tanto", "esa", "estos",
    "mucho", "quienes", "nada", "muchos", "cual", "poco", "ella", "estar", "estas",
    "algunas", "algo", "nosotros", "mi", "mis", "tu", "te", "ti", "tu", "tus",
    "ellas", "nosotras", "vosotros", "vosotras", "os", "mio", "mia", "mios",
    "mias", "tuyo", "tuya", "tuyos", "tuyas", "suyo", "suya", "suyos", "suyas",
    "nuestro", "nuestra", "nuestros", "nuestras", "vuestro", "vuestra", "vuestros",
    "vuestras", "esos", "esas", "estoy", "estas", "esta", "estamos", "estais",
    "estan", "este", "estes", "estemos", "esteis", "esten", "estare", "estaras",
    "estara", "estaremos", "estareis", "estaran", "estaria", "estarias",
    "estariamos", "estariais", "estarian", "estaba", "estabas", "estabamos",
    "estabais", "estaban", "estuve", "estuviste", "estuvo", "estuvimos",
    "estuvisteis", "estuvieron", "he", "has", "ha", "hemos", "habeis", "han",
    "haya", "hayas", "hayamos", "hayais", "hayan", "ser", "soy", "eres", "es",
    "somos", "sois", "son", "sea", "seas", "seamos", "seais", "sean", "fue",
    "fuiste", "fuimos", "fuisteis", "fueron", "seria", "serias", "seriamos",
    "seriais", "serian", "era", "eras", "eramos", "erais", "eran", "tener",
    "tengo", "tienes", "tiene", "tenemos", "teneis", "tienen", "tenga", "tengas",
    "tengamos", "tengais", "tengan", "tuve", "tuviste", "tuvo", "tuvimos",
    "tuvisteis", "tuvieron", "tendria", "tendrias", "tendriamos", "tendriais",
    "tendrian", "tenia", "tenias", "teniamos", "teniais", "tenian",
    "haber", "hacer", "hago", "hace", "hacemos", "hacen", "ir", "voy", "vas",
    "va", "vamos", "vais", "van", "poder", "puedo", "puede", "podemos", "pueden",
    "decir", "digo", "dice", "decimos", "dicen", "dar", "doy", "da", "damos", "dan",
    "ver", "veo", "ve", "vemos", "ven", "bueno", "buena", "bien", "asi", "aqui",
    "ahora", "entonces", "tambien", "mas", "menos", "mucho", "poco", "muy",
    "si", "no", "tal", "vez", "pues", "ok", "aja", "mhm", "um", "uh",
]


def _get_bilingual_stop_words() -> list[str]:
    """Return a combined English + Spanish stop word list for transcript processing.

    Returns:
        Deduplicated list of stop words from both languages plus common
        filler words found in call transcripts.
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    combined = set(ENGLISH_STOP_WORDS) | set(_SPANISH_STOP_WORDS)
    return list(combined)


def generate_ngram_search_terms(
    text: str,
    ngram_range: tuple[int, int] = (1, 3),
    top_k: int = 15,
    stop_words: str | list[str] | None = "english",
) -> list[str]:
    """Generate ranked n-gram search terms from text using sklearn's CountVectorizer.

    Extracts unigrams through trigrams (by default), removes stop words, and
    returns the top-k terms sorted by phrase length (longer phrases first).

    Args:
        text: Input text to extract n-grams from.
        ngram_range: Tuple of (min_n, max_n) for n-gram extraction.
        top_k: Maximum number of terms to return.
        stop_words: Stop word configuration:
            - "english": English stop words only
            - "bilingual": English + Spanish stop words (recommended for mixed transcripts)
            - list[str]: Custom stop word list
            - None: No stop words removed

    Returns:
        List of up to top_k n-gram strings, sorted by descending phrase length.
    """
    if not text or not text.strip():
        return []

    from sklearn.feature_extraction.text import CountVectorizer

    if stop_words == "bilingual":
        effective_stop_words = _get_bilingual_stop_words()
    else:
        effective_stop_words = stop_words

    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        stop_words=effective_stop_words,
    )

    try:
        X = vectorizer.fit_transform([text])
    except ValueError:
        # All words are stop words or text is empty after processing
        return []

    terms = vectorizer.get_feature_names_out().tolist()

    # Sort by phrase length (longer phrases first, then by string length)
    terms = sorted(terms, key=lambda x: (-len(x.split()), -len(x)))

    return terms[:top_k]
