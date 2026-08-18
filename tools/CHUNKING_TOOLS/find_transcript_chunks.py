"""
tools/CHUNKING_TOOLS/find_transcript_chunks.py

Functions for extracting and formatting transcript chunks matching search terms.
Supports both individual per-term chunks and merged multi-term chunks.
"""

from dataclasses import dataclass
from ..utils.transcript_helpers import _get_lines,_search_lines

@dataclass
class TranscriptChunk:
    """A chunk of transcript lines matched by a search term.

    Attributes:
        search_term: The term (or combined terms) that produced these matches.
        lines: List of transcript line strings in the chunk.
        line_numbers: Corresponding one-based line numbers.
    """

    search_term: str
    lines: list[str]
    line_numbers: list[int]


def format_transcript_chunks(
    chunks: list[TranscriptChunk],
    include_search_term_label: bool = True,
) -> str:
    """
    Format transcript chunks into a readable line-numbered text block.

    Each chunk is rendered using the exact transcript line numbers stored in the
    TranscriptChunk object.

    Args:
        chunks:
            List of TranscriptChunk objects to format.

        include_search_term_label:
            If True, include a "Search term: ..." header before each chunk.
            If False, only output the line-numbered transcript lines.

    Returns:
        A formatted string containing all chunks. If the input list is empty,
        returns an empty string.

    Example:
        >>> print(format_transcript_chunks(chunks, include_search_term_label=True))
        Search term: internet is down
        [line 14] Caller: My internet is down
        [line 15] Agent: Let me check that for you

        Search term: new modem
        [line 27] Agent: I can get you a new modem for free
        [line 28] Caller: That would be great

        >>> print(format_transcript_chunks(chunks, include_search_term_label=False))
        [line 14] Caller: My internet is down
        [line 15] Agent: Let me check that for you

        [line 27] Agent: I can get you a new modem for free
        [line 28] Caller: That would be great
    """
    if not chunks:
        return ""

    formatted_blocks = []

    for chunk in chunks:
        block = []

        if include_search_term_label:
            block.append(f"Search term: {chunk.search_term}")

        for line_num, line in zip(chunk.line_numbers, chunk.lines):
            block.append(f"[line {line_num}] {line}")

        formatted_blocks.append("\n".join(block))

    return "\n\n".join(formatted_blocks)

def find_transcript_chunks(
    transcript: str,
    search_terms: str | list[str],
    context_lines: int = 2,
    formatted: bool = False,
    include_search_term_label: bool = True,
) -> list[TranscriptChunk] | str:
    """
    Find transcript chunks for one or more search terms.

    This function searches a transcript for each provided term and returns a
    separate TranscriptChunk for each term that produces at least one match.
    Each chunk includes the matched line(s) plus surrounding context lines.

    Args:
        transcript:
            Full transcript text as a single string.

        search_terms:
            A single string or list of strings to search for in the transcript.
            Each term is searched independently.

        context_lines:
            Number of transcript lines to include before and after each match.

        formatted:
            If False, return a list of TranscriptChunk objects.
            If True, return a formatted string produced by format_transcript_chunks().

        include_search_term_label:
            Used only when formatted=True. If True, include a "Search term: ..."
            label before each chunk.

    Returns:
        If formatted=False:
            list[TranscriptChunk]

        If formatted=True:
            str

    Example:
        >>> find_transcript_chunks(
        ...     transcript,
        ...     ["internet is down", "new modem"],
        ...     context_lines=1,
        ...     formatted=True,
        ... )
        'Search term: internet is down\\n[line 1] ...'
    """
    lines = _get_lines(transcript)

    if isinstance(search_terms, str):
        search_terms = [search_terms]

    chunks = []

    for term in search_terms:
        if not isinstance(term, str):
            continue

        term = term.strip()
        if not term:
            continue

        _, matched_indices = _search_lines(lines, term, context_lines)

        if not matched_indices:
            continue

        sorted_idx = sorted(set(matched_indices))
        chunk = TranscriptChunk(
            search_term=term,
            lines=[lines[i] for i in sorted_idx],
            line_numbers=[i + 1 for i in sorted_idx],
        )
        chunks.append(chunk)

    if formatted:
        return format_transcript_chunks(
            chunks,
            include_search_term_label=include_search_term_label,
        )

    return chunks

def find_transcript_chunk_merged(
    transcript: str,
    search_terms: str | list[str],
    context_lines: int = 2,
    formatted: bool = False,
    include_search_term_label: bool = True,
) -> TranscriptChunk | str:
    """
    Find one merged transcript chunk from one or more search terms.

    This function searches a transcript for all provided search terms, combines
    all matched line indices into a single deduplicated set, preserves original
    transcript line numbers, and returns either a TranscriptChunk or a formatted
    string.

    Args:
        transcript:
            Full transcript text as a single string.

        search_terms:
            A single string or list of strings to search for in the transcript.

        context_lines:
            Number of transcript lines to include before and after each match.

        formatted:
            If False, return a merged TranscriptChunk.
            If True, return a formatted string produced by format_transcript_chunks().

        include_search_term_label:
            Used only when formatted=True. If True, include a "Search term: ..."
            label before the merged chunk.

    Returns:
        If formatted=False:
            TranscriptChunk

        If formatted=True:
            str

    Example:
        >>> find_transcript_chunk_merged(
        ...     transcript,
        ...     ["internet is down", "new modem"],
        ...     context_lines=1,
        ...     formatted=True,
        ... )
        'Search term: internet is down | new modem\\n[line 1] ...'

    Example formatted output:
        Search term: internet is down | new modem
        [line 1] Agent: Hello
        [line 2] Caller: My internet is down
        [line 3] Agent: I can get you a new modem for free
        [line 4] Caller: That would be great
    """
    lines = _get_lines(transcript)

    if isinstance(search_terms, str):
        search_terms = [search_terms]

    matched_indices = set()
    clean_terms = []

    for term in search_terms:
        if not isinstance(term, str):
            continue

        term = term.strip()
        if not term:
            continue

        clean_terms.append(term)
        _, idx = _search_lines(lines, term, context_lines)
        matched_indices.update(idx)

    sorted_idx = sorted(matched_indices)

    chunk = TranscriptChunk(
        search_term=" | ".join(clean_terms),
        lines=[lines[i] for i in sorted_idx],
        line_numbers=[i + 1 for i in sorted_idx],
    )

    if formatted:
        return format_transcript_chunks(
            [chunk],
            include_search_term_label=include_search_term_label,
        )

    return chunk