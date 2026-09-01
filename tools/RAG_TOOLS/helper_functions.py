

"""
tools/rag_tools.py

Lightweight file-search / RAG helpers for LLM tools.
Searches .md, .csv, .txt files in a folder using plain-text queries
or regex patterns and returns matched lines with context.

Features:
- multi-query search
- optional regex mode
- deduplication
- top-N file limiting
- grouped-by-query structured results
"""

import json
import re
from pathlib import Path
from pydantic import BaseModel, Field
from tools.utils.build_tool import Tool
from tools.utils.normalize_txt import _normalize_text,_tokenize
from pathlib import Path


# This file lives at <project_root>/tools/RAG_TOOLS/, so the project root is
# three levels up (RAG_TOOLS -> tools -> project_root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAG_FOLDER = PROJECT_ROOT / "knowledge_base"

SUPPORTED_EXTENSIONS = {".md", ".csv", ".txt"}




def configure_rag_folder(folder: str):
    """Override the default RAG folder path used for document searches.

    Args:
        folder: New folder path to use as the default RAG folder.
    """
    global DEFAULT_RAG_FOLDER
    DEFAULT_RAG_FOLDER = folder


class RAGMatch(BaseModel):
    """A single search match result within a document.

    Attributes:
        file: Path to the file containing the match.
        line_number: One-based line number of the match.
        quote: The matched line text (stripped).
        context: Surrounding lines included for context.
        query: The search query that produced this match.
    """

    file: str
    line_number: int
    quote: str
    context: str
    query: str


class DocumentSearchInput(BaseModel):
    """Input parameters for the search_documents tool.

    Attributes:
        queries: List of search terms to look for.
        context_lines: Number of surrounding lines to include.
        regex_mode: If True, treat queries as regex patterns.
    """

    queries: list[str] = Field(
        ...,
        description="List of search terms to look for in the documents"
    )
    context_lines: int = Field(
        3,
        description="Number of surrounding lines to include above and below each match"
    )
    regex_mode: bool = Field(
        False,
        description="If true, treat queries as regex patterns. Otherwise search as escaped plain text."
    )


SEARCH_DOCUMENTS_TOOL = {
    "type": "function",
    "name": "search_documents",
    "description": (
        "Search the configured reference documents for relevant lines using one or more "
        "search terms. Returns grouped results by query."
    ),
    "parameters": DocumentSearchInput.model_json_schema(),
    "strict": True,
}


def list_files(folder_or_file, extensions=None) -> list[str]:
    """List all files in a folder (or return a single file) matching supported extensions.

    Args:
        folder_or_file: Path to a file or directory.
        extensions: Optional set of extensions to filter by (defaults to SUPPORTED_EXTENSIONS).

    Returns:
        List of file path strings matching the extensions.
    """
    exts = {e.lower() for e in (extensions or SUPPORTED_EXTENSIONS)}
    path = Path(folder_or_file)

    if path.is_file():
        return [str(path)] if path.suffix.lower() in exts else []

    if path.is_dir():
        return [str(p) for p in path.rglob("*") if p.suffix.lower() in exts]

    return []


def _compile_queries(queries: list[str], regex_mode: bool) -> list[tuple[str, re.Pattern]]:
    """Compile search queries into regex patterns.

    Args:
        queries: List of raw query strings.
        regex_mode: If True, use queries as-is; if False, escape them for literal matching.

    Returns:
        List of (original_query, compiled_pattern) tuples.
    """
    compiled = []

    for q in queries:
        q = str(q).strip()
        if not q:
            continue

        try:
            pattern = re.compile(q if regex_mode else re.escape(q), re.IGNORECASE)
            compiled.append((q, pattern))
        except re.error:
            # skip invalid regex if regex_mode=True
            continue

    return compiled


def _dedupe_matches(matches: list[RAGMatch]) -> list[RAGMatch]:
    """
    Deduplicate matches by (query, file, line_number, quote).
    Keeps first occurrence.
    """
    seen = set()
    deduped = []

    for m in matches:
        key = (m.query, m.file, m.line_number, m.quote)
        if key not in seen:
            seen.add(key)
            deduped.append(m)

    return deduped


def search_files(
    files: list[str],
    queries: list[str],
    context_lines: int = 3,
    regex_mode: bool = False,
    max_matches_per_query: int = 20,
    deduplicate: bool = True,
) -> dict:
    """
    Search files for queries and return grouped-by-query structured results.

    Returns a dict:
    {
      "found_any": bool,
      "queries": [...],
      "results_by_query": [
        {
          "query": "...",
          "found": bool,
          "matches": [...]
        }
      ]
    }
    """
    compiled = _compile_queries(queries, regex_mode=regex_mode)

    results_by_query = []
    found_any = False

    for query, pattern in compiled:
        matches = []

        for file_path in files:
            try:
                lines = Path(file_path).read_text(encoding="utf-8").splitlines()
            except Exception:
                continue

            for i, line in enumerate(lines):
                if pattern.search(line):
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    context = "\n".join(lines[start:end])

                    matches.append(RAGMatch(
                        file=file_path,
                        line_number=i + 1,
                        quote=line.strip(),
                        context=context,
                        query=query,
                    ))

                    if len(matches) >= max_matches_per_query:
                        break

            if len(matches) >= max_matches_per_query:
                break

        if deduplicate:
            matches = _dedupe_matches(matches)

        if matches:
            found_any = True

        results_by_query.append({
            "query": query,
            "found": len(matches) > 0,
            "match_count": len(matches),
            "matches": [m.model_dump() for m in matches],
        })

    return {
        "found_any": found_any,
        "queries": [q for q, _ in compiled],
        "results_by_query": results_by_query,
    }


def _score_file_for_queries(file_path: str, queries: list[str]) -> int:
    """
    Score a file path based on filename/path overlap with queries.
    Higher score = more likely relevant.
    """
    path_obj = Path(file_path)
    filename_text = _normalize_text(path_obj.stem)
    full_path_text = _normalize_text(str(path_obj))

    score = 0

    for query in queries:
        q = _normalize_text(query)
        if not q:
            continue

        q_tokens = _tokenize(q)

        # stronger score for exact substring in filename stem
        if q in filename_text:
            score += 10

        # slightly weaker score for exact substring in full path
        if q in full_path_text:
            score += 5

        # token overlap scoring
        filename_tokens = set(_tokenize(filename_text))
        path_tokens = set(_tokenize(full_path_text))

        overlap_filename = len(set(q_tokens) & filename_tokens)
        overlap_path = len(set(q_tokens) & path_tokens)

        score += overlap_filename * 3
        score += overlap_path * 1

    return score

def rank_files_by_query_match(files: list[str], queries: list[str], max_files: int | None = None) -> list[str]:
    """
    Rank files by filename/path similarity to the queries.
    Returns the top candidate files.
    """
    scored = [(file_path, _score_file_for_queries(file_path, queries)) for file_path in files]

    # sort by descending score, then stable secondary sort by file path
    scored.sort(key=lambda x: (-x[1], x[0]))

    ranked_files = [file_path for file_path, score in scored]

    if max_files is not None:
        ranked_files = ranked_files[:max_files]

    return ranked_files

