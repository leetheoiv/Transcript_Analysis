"""
utils/load_prompt_md.py

Parse a saved prompt markdown file (the `*_prompt_v{N}.md` files written by
Orchestrator._save_prompt_to_file) back into its parts so it can be fed straight
into the "bring your own prompt" path of the Orchestrator.

Saved format produced by the orchestrator:

    # {project} — Prompt (v{N})

    ## System Prompt

    ```
    <system prompt text>
    ```

    ## User Prompt

    ```
    <user prompt text>
    ```

    ## Output Format

    ```json
    <output format>
    ```

Usage:
    from utils.load_prompt_md import load_prompt_md

    parsed = load_prompt_md("JITS/cox_merger/ai_results/ai_results_prompt_v1.md")
    orchestrator = Orchestrator(
        df=df,
        output_dir=..., project_name=...,
        TranscriptExtractionAgent=..., JudgeAgent=...,
        prebuilt_system_prompt=parsed["system_prompt"],
        prebuilt_user_prompt=parsed["user_prompt"],
        prebuilt_output_format=parsed["output_format"],
    )
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


# Matches a "## <Section Title>" header followed by a fenced code block. The
# opening fence may carry a language hint (e.g. ```json). Capture group 1 is the
# section title; group 2 is the raw fenced content.
_SECTION_RE = re.compile(
    r"^\#\#\s*(?P<title>.+?)\s*$\n+```[^\n]*\n(?P<body>.*?)\n```",
    re.MULTILINE | re.DOTALL,
)


def _parse_output_format(raw: str) -> dict:
    """Best-effort parse of the Output Format block into a dict.

    The orchestrator writes this block with `f"{dict}"`, which yields Python
    dict-repr (single quotes), not strict JSON. Try JSON first, then fall back
    to a safe Python-literal eval. Returns {} when the block is empty/unparseable.
    """
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        value = ast.literal_eval(text)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


# Repository root: this file lives at <root>/utils/load_prompt_md.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_prompt_path(path: str | Path) -> Path:
    """Resolve a prompt path robustly, tolerating relative paths and a redundant
    leading repo-name segment.

    Tries, in order:
      1. The path exactly as given (absolute, or relative to the CWD).
      2. The path resolved against the repository root.
      3. The path with a leading repo-name segment stripped, resolved against
         the repository root — handles paths like
         "Transcript_Analysis_Automation/JITS/..." while the repo root already
         IS .../Transcript_Analysis_Automation (which would otherwise double up).

    Raises FileNotFoundError (listing what was tried) if none exist.
    """
    given = Path(path)
    candidates: list[Path] = [given]

    if not given.is_absolute():
        candidates.append((_PROJECT_ROOT / given))

        parts = given.parts
        if parts and parts[0] == _PROJECT_ROOT.name:
            stripped = parts[1:]
            if stripped:
                candidates.append(_PROJECT_ROOT.joinpath(*stripped))

    for c in candidates:
        if c.exists():
            return c

    tried = "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Prompt markdown file not found. Tried:\n  - {tried}"
    )


def load_prompt_md(path: str | Path) -> dict:
    """Load a saved prompt markdown file and split it into its parts.

    Args:
        path: Path to a `*_prompt_v{N}.md` file (relative paths are resolved
            against the current working directory).

    Returns:
        Dict with keys:
          - "system_prompt": str
          - "user_prompt": str
          - "output_format": dict (empty dict if the section is absent/empty)

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If neither a System Prompt nor a User Prompt section is found
            (i.e. the file isn't in the expected format).
    """
    p = _resolve_prompt_path(path)
    text = p.read_text(encoding="utf-8")

    # Map lowercased section title -> raw fenced body.
    sections: dict[str, str] = {}
    for m in _SECTION_RE.finditer(text):
        sections[m.group("title").strip().lower()] = m.group("body")

    system_prompt = sections.get("system prompt", "").strip()
    user_prompt = sections.get("user prompt", "").strip()
    output_format = _parse_output_format(sections.get("output format", ""))

    if not system_prompt and not user_prompt:
        raise ValueError(
            f"Could not find a 'System Prompt' or 'User Prompt' section in {p}. "
            "Expected the format written by Orchestrator._save_prompt_to_file "
            "(## System Prompt / ## User Prompt fenced code blocks)."
        )

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "output_format": output_format,
    }
