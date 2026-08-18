"""
tools/search_transcript.py

Provides a transcript search tool for use in TranscriptExtractionAgent.

The LLM calls search_transcript(query, context_lines) to pull relevant lines
from the transcript on demand, avoiding the need to put the full transcript
in the prompt unless fallback is needed.
"""

import json

from pydantic import BaseModel, Field

from tools.utils.transcript_helpers import _get_lines, _search_lines
from tools.utils.build_tool import Tool


class SearchTranscriptInput(BaseModel):
    """Input schema for the search_transcript tool.

    Attributes:
        queries: List of search terms to find in the transcript.
        context_lines: Number of surrounding lines to return around matches.
    """

    queries: list[str] = Field(
        ...,
        description="List of search terms to find relevant lines in the transcript"
    )
    context_lines: int = Field(
        3,
        description="Number of lines of context to return around each match"
    )


class SearchTranscriptQueryResult(BaseModel):
    """Result for a single search query against the transcript.

    Attributes:
        query: The search term used.
        found: Whether any matches were found.
        line_numbers: One-based line numbers of matched lines.
        matches: Formatted match strings with line numbers.
    """

    query: str
    found: bool
    line_numbers: list[int]
    matches: list[str]


class SearchTranscriptResult(BaseModel):
    """Aggregate result from searching a transcript with multiple queries.

    Attributes:
        found_any: Whether any query found matches.
        queries: List of queries that were executed.
        results: Per-query result objects.
        transcript_for_prompt: Either matched chunks or full transcript for LLM injection.
        mode: 'matched_chunks' if evidence was found, 'full_transcript_fallback' otherwise.
    """

    found_any: bool
    queries: list[str]
    results: list[SearchTranscriptQueryResult]
    transcript_for_prompt: str
    mode: str


def dedupe_preserve_order(items):
    """Remove duplicate items from a list while preserving original order.

    Args:
        items: Iterable of hashable items.

    Returns:
        List with duplicates removed, maintaining first-occurrence order.
    """
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def search_transcript(
    transcript: str,
    queries: list[str],
    context_lines: int = 3,
    max_search_terms: int = 5,
) -> str:
    """
    Search a transcript for one or more query terms and return matching lines.
    If matches are found, transcript_for_prompt contains deduped matching chunks.
    If no matches are found, transcript_for_prompt contains the full transcript.
    """
    lines = _get_lines(transcript)
    queries = [q.strip() for q in queries if str(q).strip()][:max_search_terms]

    results = []
    found_any = False
    collected_matches = []

    for query in queries:
        matched_lines, matched_indices = _search_lines(lines, query, context_lines)

        if matched_lines:
            found = True
            found_any = True
            line_numbers = [i + 1 for i in matched_indices]
            matches = [
                f"[line {line_no}] {line}"
                for line_no, line in zip(line_numbers, matched_lines)
            ]
            collected_matches.extend(matches)
        else:
            found = False
            line_numbers = []
            matches = []

        results.append({
            "query": query,
            "found": found,
            "line_numbers": line_numbers,
            "matches": matches,
        })

    if found_any:
        transcript_for_prompt = "\n".join(dedupe_preserve_order(collected_matches))
        mode = "matched_chunks"
    else:
        transcript_for_prompt = transcript
        mode = "full_transcript_fallback"

    return json.dumps({
        "found_any": found_any,
        "queries": queries,
        "results": results,
        "transcript_for_prompt": transcript_for_prompt,
        "mode": mode,
    })


def make_search_transcript_tool(transcript: str, max_search_terms: int = 5) -> Tool:
    """
    Create a Tool object bound to a specific transcript.
    """
    def _tool_search_transcript(queries: list[str], context_lines: int = 3) -> str:
        return search_transcript(
            transcript=transcript,
            queries=queries,
            context_lines=context_lines,
            max_search_terms=max_search_terms,
        )

    return Tool(
        name="search_transcript",
        description=(
            "Search the transcript for one or more relevant terms and return "
            "matching transcript lines with line numbers and surrounding context. "
            "Also returns transcript_for_prompt containing either matched chunks "
            "or the full transcript if nothing matches."
        ),
        input_model=SearchTranscriptInput,
        func=_tool_search_transcript,
    )