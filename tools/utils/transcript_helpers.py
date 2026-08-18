import re

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


def _search_lines(lines: list[str], pattern: str, context_lines: int) -> tuple[list[str], list[int]]:
    """
    Search transcript lines for a pattern and return matching lines with context.

    This helper searches a list of transcript lines for a given pattern using
    case-insensitive matching. For each matched line, it also includes a
    configurable number of surrounding context lines before and after the match.

    Args:
        lines:
            List of transcript lines, typically produced by _get_lines().

        pattern:
            String pattern to search for. If None or empty after stripping,
            no search is performed and empty results are returned.

        context_lines:
            Number of lines of context to include before and after each match.

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

        >>> _search_lines(lines, "new modem", context_lines=0)
        (
            ["Agent: I can send a new modem"],
            [3]
        )

    Notes:
        - Matching is case-insensitive.
        - The function escapes the input pattern before compiling it as regex,
          so it behaves like a literal substring search in normal cases.
        - If regex compilation of the escaped pattern fails, it falls back to
          compiling the raw pattern.
        - Returned indices are zero-based; callers can convert them to one-based
          display values if needed.
        - Overlapping context windows are automatically deduplicated.
    """
    if pattern is None:
        return [], []

    pattern = str(pattern).strip()
    if not pattern:
        return [], []

    try:
        if re.fullmatch(r"\w+", pattern):
            compiled = re.compile(rf"\b{re.escape(pattern)}\b", re.IGNORECASE)
        else:
            compiled = re.compile(re.escape(pattern), re.IGNORECASE)
    except re.error:
        compiled = re.compile(re.escape(pattern), re.IGNORECASE)

    matched_indices = set()

    for i, line in enumerate(lines):
        if compiled.search(line):
            for j in range(max(0, i - context_lines), min(len(lines), i + context_lines + 1)):
                matched_indices.add(j)

    sorted_indices = sorted(matched_indices)
    return [lines[i] for i in sorted_indices], sorted_indices