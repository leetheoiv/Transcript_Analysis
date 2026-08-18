"""
tools/RAG_TOOLS/search_documents.py

High-level document search function that combines file ranking and content
search from the helper_functions module.
"""

from .helper_functions import *


def search_documents(
    queries: list[str],
    context_lines: int = 3,
    regex_mode: bool = False,
    folder=None,
):
    """Search reference documents in a folder for matching lines.

    Ranks candidate files by filename similarity to queries, then searches
    the top candidates for matches with surrounding context.

    Args:
        queries: List of search query strings.
        context_lines: Number of lines of context around each match.
        regex_mode: If True, treat queries as regex patterns.
        folder: Folder to search (defaults to DEFAULT_RAG_FOLDER).

    Returns:
        JSON string with search results, file counts, and candidate file list.
    """
    folder = Path(folder) if folder else DEFAULT_RAG_FOLDER
    files = list_files(folder)

    candidate_files = rank_files_by_query_match(files, queries, max_files=25)

    payload = search_files(
        files=candidate_files,
        queries=queries,
        context_lines=context_lines,
        regex_mode=regex_mode,
        max_matches_per_query=10,
        deduplicate=True,
    )

    payload["searched_file_count"] = len(candidate_files)
    payload["candidate_files"] = candidate_files
    payload["folder"] = str(folder)

    return json.dumps(payload)