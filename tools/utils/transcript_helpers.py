import re
from difflib import SequenceMatcher

# Common English stopwords plus filler words frequent in call transcripts.
# Dropped from queries before token-overlap matching so scoring focuses on
# content words (e.g. "Cox", "merger", "customer") rather than glue words.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "to", "in",
    "on", "at", "for", "with", "as", "by", "is", "am", "are", "was", "were",
    "be", "been", "being", "do", "does", "did", "have", "has", "had", "i",
    "you", "he", "she", "it", "we", "they", "me", "my", "your", "our", "their",
    "this", "that", "these", "those", "there", "here", "just", "already",
    "yeah", "okay", "ok", "um", "uh", "like", "well", "right", "gonna", "wanna",
    "about", "from", "into", "would", "could", "should", "will", "can", "get",
    "got", "im", "ive", "id", "youre", "dont", "thats",
})

# Word-splitting pattern shared by the query tokenizer and line tokenizer.
_WORD_RE = re.compile(r"[a-z0-9]+")


def _content_tokens(text: str) -> list[str]:
    """Lowercase a string and return its significant (non-stopword) word tokens.

    Args:
        text: Arbitrary text (a search query or a transcript line).

    Returns:
        List of lowercase word tokens with stopwords removed. Single-character
        tokens are dropped as noise.
    """
    tokens = _WORD_RE.findall(text.lower())
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


def _get_lines(transcript: str) -> list[str]:
    """
    Split a transcript into individual line-like units for searching.

    This helper normalizes transcript text into a list of non-empty lines.

    Behavior:
    - If the transcript already contains newline characters, it is split on
      line boundaries using splitlines().
    - If the transcript is a single block of text with speaker labels such as
      "Agent:" and "Caller:", it is split before each speaker turn.
    - Leading and trailing whitespace is stripped from each returned line.
    - Empty lines are removed.

    Args:
        transcript:
            Full transcript text as a single string.

    Returns:
        A list of cleaned transcript lines suitable for downstream searching
        and chunk extraction.

    Example:
        >>> _get_lines("Agent: Hello\\nCaller: Hi there")
        ['Agent: Hello', 'Caller: Hi there']

        >>> _get_lines("Agent: Hello Caller: Hi there Agent: How can I help?")
        ['Agent: Hello', 'Caller: Hi there', 'Agent: How can I help?']

    Notes:
        - This function assumes speaker labels like "Agent:" and "Caller:" when
          no newline characters are present.
        - If other speaker labels are possible, the regex may need to be expanded.
    """
    if "\n" in transcript:
        return [line.strip() for line in transcript.splitlines() if line.strip()]
    chunks = re.split(r'(?=(?:Agent|Caller):)', transcript)
    return [c.strip() for c in chunks if c.strip()]


def _search_lines(
    lines: list[str],
    pattern: str,
    context_lines: int,
    min_overlap: float = 0.6,
) -> tuple[list[str], list[int]]:
    """
    Search transcript lines for a pattern and return matching lines with context.

    Uses a two-pass strategy so paraphrased queries still find relevant lines:

    1. Literal pass (precise): case-insensitive substring match on the whole
       query, exactly as before. Exact phrases and single keywords match here.

    2. Token-overlap pass (recall): when the literal pass finds nothing, the
       query is reduced to its content words (stopwords dropped) and a line is
       considered a match if it contains at least ``min_overlap`` fraction of
       those content words. This lets a query like
       "I am already a Cox customer" match a line such as
       "yeah I've been a Cox customer for years" via the shared words
       {cox, customer}, even though the full phrase never appears verbatim.

    For each matched line, ``context_lines`` surrounding lines are included.

    Args:
        lines:
            List of transcript lines, typically produced by _get_lines().
        pattern:
            String pattern to search for. If None or empty after stripping,
            no search is performed and empty results are returned.
        context_lines:
            Number of lines of context to include before and after each match.
        min_overlap:
            Fraction (0.0-1.0) of the query's content words that must appear in
            a line for the token-overlap pass to count it as a match. Higher is
            stricter. A single-content-word query (e.g. "Cox") matches on that
            one word being present. Set to a value > 1.0 to disable the pass.

    Returns:
        A tuple of:
        - list[str]: the matched transcript lines plus surrounding context lines
        - list[int]: the zero-based indices of those lines in the original
          transcript line list

    Example:
        >>> lines = [
        ...     "Agent: Hello",
        ...     "Caller: My internet is down",
        ...     "Agent: Let me check that",
        ...     "Agent: I can send a new modem",
        ... ]
        >>> _search_lines(lines, "internet is down", context_lines=1)
        (
            [
                "Agent: Hello",
                "Caller: My internet is down",
                "Agent: Let me check that"
            ],
            [0, 1, 2]
        )

    Notes:
        - Matching is case-insensitive throughout.
        - The literal pass runs first and, on any hit, short-circuits the
          overlap pass so precise matches are never diluted.
        - Returned indices are zero-based; callers can convert to one-based.
        - Overlapping context windows are automatically deduplicated.
    """
    if pattern is None:
        return [], []

    pattern = str(pattern).strip()
    if not pattern:
        return [], []

    def _expand(match_line_indices: set[int]) -> tuple[list[str], list[int]]:
        expanded = set()
        for i in match_line_indices:
            for j in range(max(0, i - context_lines), min(len(lines), i + context_lines + 1)):
                expanded.add(j)
        sorted_indices = sorted(expanded)
        return [lines[i] for i in sorted_indices], sorted_indices

    # --- Pass 1: literal substring / word-boundary match -------------------
    try:
        if re.fullmatch(r"\w+", pattern):
            compiled = re.compile(rf"\b{re.escape(pattern)}\b", re.IGNORECASE)
        else:
            compiled = re.compile(re.escape(pattern), re.IGNORECASE)
    except re.error:
        compiled = re.compile(re.escape(pattern), re.IGNORECASE)

    literal_hits = {i for i, line in enumerate(lines) if compiled.search(line)}
    if literal_hits:
        return _expand(literal_hits)

    # --- Pass 2: content-word overlap match (recall fallback) --------------
    query_tokens = _content_tokens(pattern)
    if not query_tokens:
        return [], []

    query_set = set(query_tokens)
    # Require at least this many shared content words. Ceil-like behavior via
    # rounding so, e.g., 0.6 * 2 tokens -> need 2, 0.6 * 3 -> need 2.
    needed = max(1, int(round(min_overlap * len(query_set))))

    overlap_hits = set()
    for i, line in enumerate(lines):
        line_set = set(_content_tokens(line))
        if not line_set:
            continue
        if len(query_set & line_set) >= needed:
            overlap_hits.add(i)

    if overlap_hits:
        return _expand(overlap_hits)

    # --- Pass 3: FUZZY content-word overlap (corruption-tolerant) ----------
    # Transcripts in this pipeline can be corrupted (e.g. the letter "s" is
    # dropped: "customer" -> "cu tomer", "spectrum" -> "pectrum"). The exact
    # token-overlap pass then fails because the query token ("customer") never
    # equals the mangled transcript token. This pass counts a query token as
    # present when it is SIMILAR ENOUGH to some token on the line, so
    # quote-derived search terms still locate the right lines in dirty text.
    fuzzy_hits = _fuzzy_overlap_hits(lines, query_set, needed)
    if fuzzy_hits:
        return _expand(fuzzy_hits)

    return [], []


def _token_is_similar(query_token: str, line_token: str, threshold: float = 0.82) -> bool:
    """Return True if two tokens are close enough to be the same word.

    Tolerant of the common corruption modes in this pipeline (dropped/replaced
    single characters, e.g. dropped "s"). Uses a length-guard for speed and
    SequenceMatcher for the actual ratio.
    """
    if query_token == line_token:
        return True
    # Cheap length filter: corruption rarely changes length by more than ~2.
    if abs(len(query_token) - len(line_token)) > 2:
        return False
    # Substring containment catches "customer" vs "cu tomer" fragments after
    # tokenization splits (e.g. matching the longer fragment "tomer" is weak,
    # so require a reasonably long shared token).
    if len(query_token) >= 4 and (
        query_token in line_token or line_token in query_token
    ):
        return True
    return SequenceMatcher(None, query_token, line_token).ratio() >= threshold


def _line_match_candidates(line: str) -> list[str]:
    """Build candidate tokens for a line, including adjacent-token joins.

    The dominant corruption mode here is a dropped/space-replaced "s", which
    splits ONE word into TWO short fragments ("customer" -> "cu tomer",
    "existing" -> "exi ting"). Individual fragments are too short/noisy to
    match, so we also emit the concatenation of each adjacent token pair (and
    the raw tokens). This reconstructs the original word closely enough for a
    fuzzy comparison ("cu"+"tomer" -> "cutomer" ~= "customer").

    All tokens here are raw word tokens (NOT stopword-filtered) so short
    fragments like "cu"/"exi" survive to be rejoined.
    """
    raw = _WORD_RE.findall(line.lower())
    candidates: list[str] = list(raw)
    for a, b in zip(raw, raw[1:]):
        candidates.append(a + b)
    return candidates


def _fuzzy_overlap_hits(lines: list[str], query_set: set[str], needed: int) -> set[int]:
    """Find line indices where enough query tokens FUZZY-match a line token.

    Matches each query content word against both the line's word tokens and the
    joins of adjacent tokens (to undo the s-drop word-splitting corruption).

    Args:
        lines: transcript lines.
        query_set: set of query content tokens.
        needed: minimum number of query tokens that must (fuzzily) appear.

    Returns:
        Set of matching line indices.
    """
    hits: set[int] = set()
    # Only bother fuzzing query tokens long enough to be discriminative.
    fuzzable_query = [t for t in query_set if len(t) >= 4]
    if not fuzzable_query:
        return hits

    for i, line in enumerate(lines):
        candidates = _line_match_candidates(line)
        if not candidates:
            continue
        matched = 0
        for qt in fuzzable_query:
            if any(_token_is_similar(qt, lt) for lt in candidates):
                matched += 1
                if matched >= needed:
                    hits.add(i)
                    break
    return hits