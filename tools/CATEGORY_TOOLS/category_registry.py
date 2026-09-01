"""
tools/CATEGORY_TOOLS/category_registry.py

Category Registry Tool for controlled vocabulary management during extraction.

Problem:
    When an LLM extracts call reason categories from transcripts with free range,
    it produces inconsistent labels for the same underlying reason (e.g., "billing issue",
    "billing problem", "bill dispute" all mean the same thing). This makes downstream
    reporting unreliable.

Solution:
    A CSV-backed registry scoped by extraction field, with a two-level hierarchy:
      - Field: the extraction field the category belongs to (e.g. "primary_issue_category",
        "sub_issue_category"). Lookups and registrations are filtered by field so the
        model only sees categories relevant to the field it is currently assigning.
      - Broad Category: high-level grouping (e.g., "Billing", "Technical Support")
      - Sub-Category: specific reason within the broad category (e.g., "Payment Dispute",
        "Charge Inquiry")

    Each level has its own definition. The agent queries the registry before assigning
    categories to ensure consistency.

    Tools:
      - lookup_category: fuzzy-matches a proposed sub-category against existing entries.
        Returns the matching broad category + sub-category + definitions if found.
      - register_category: adds a new broad category / sub-category pair with definitions
        when no existing entry matches.

The CSV file is created on first use and grows as new categories are discovered.
Thread-safe via file-level locking for batch processing.

Usage:
    from tools.CATEGORY_TOOLS.category_registry import make_category_registry_tools

    tools = make_category_registry_tools(registry_path="output/call_reasons.csv")
    # Returns (lookup_tool, register_tool) — register both with the agent.
"""

import csv
import json
import threading
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import BaseModel, Field

from tools.utils.build_tool import Tool


# ---------------------------------------------------------------------------
# Pydantic input models
# ---------------------------------------------------------------------------


class LookupCategoryInput(BaseModel):
    """Input schema for the lookup_category tool.

    Attributes:
        field: The extraction field the category is being assigned to.
        proposed_category: The sub-category name the LLM wants to assign.
        threshold: Minimum similarity score (0-1) to consider a match.
    """

    field: str = Field(
        ...,
        description=(
            "The name of the extraction field you are currently assigning a category to "
            "(e.g. 'primary_issue_category', 'sub_issue_category'). The tool only searches "
            "categories that were registered for this same field, so categories from unrelated "
            "fields are never returned."
        ),
    )
    proposed_category: str = Field(
        ...,
        description=(
            "The sub-category name you want to assign to this transcript. "
            "The tool will check if a matching sub-category already exists in the registry "
            "for this field and return its broad category parent."
        ),
    )
    threshold: float = Field(
        0.75,
        description=(
            "Similarity threshold (0.0 to 1.0) for fuzzy matching. "
            "Lower values match more loosely. Default 0.75 works well for category names."
        ),
    )


class RegisterCategoryInput(BaseModel):
    """Input schema for the register_category tool.

    Attributes:
        field: The extraction field the category belongs to.
        broad_category: High-level category grouping.
        broad_category_definition: Definition of the broad category.
        sub_category: Specific reason within the broad category.
        sub_category_definition: Definition of the sub-category.
    """

    field: str = Field(
        ...,
        description=(
            "The name of the extraction field this category belongs to "
            "(e.g. 'primary_issue_category', 'sub_issue_category'). The category is stored "
            "under this field so future lookups for the same field stay consistent and "
            "categories from unrelated fields are kept separate."
        ),
    )
    broad_category: str = Field(
        ...,
        description=(
            "The high-level category grouping. Use concise, title-case labels "
            "(e.g., 'Billing', 'Technical Support', 'Account Management'). "
            "If this broad category already exists in the registry, reuse the exact name."
        ),
    )
    broad_category_definition: str = Field(
        ...,
        description=(
            "A clear 1-2 sentence definition of what this broad category covers. "
            "Describes the general domain of issues grouped under this label."
        ),
    )
    sub_category: str = Field(
        ...,
        description=(
            "The specific sub-category within the broad category. Use concise, title-case "
            "labels (e.g., 'Payment Dispute', 'Signal Loss', 'Password Reset')."
        ),
    )
    sub_category_definition: str = Field(
        ...,
        description=(
            "A clear 1-2 sentence definition of what this sub-category covers. "
            "This helps future lookups determine if a new transcript matches this sub-category."
        ),
    )


# ---------------------------------------------------------------------------
# Registry class (handles CSV I/O and matching)
# ---------------------------------------------------------------------------


class CategoryRegistry:
    """CSV-backed category registry with two-level hierarchy, fuzzy matching,
    and thread-safe writes.

    The registry file has columns:
        field, broad_category, broad_category_definition, sub_category,
        sub_category_definition, created_at
    """

    COLUMNS = [
        "field",
        "broad_category",
        "broad_category_definition",
        "sub_category",
        "sub_category_definition",
        "created_at",
    ]

    def __init__(self, registry_path: str):
        """Initialize the registry bound to a specific CSV file.

        Args:
            registry_path: Path to the CSV file. Created if it doesn't exist.
        """
        self.path = Path(registry_path)
        self._lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self):
        """Create the registry CSV with headers if it doesn't already exist.

        If a file already exists but uses an older schema (missing the ``field``
        column), it is migrated in place: existing rows are preserved with an
        empty ``field`` value and the header is upgraded to the current schema.
        This keeps old registries readable and prevents column misalignment when
        new rows are appended.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                writer.writeheader()
            return

        # File exists — check whether its header matches the current schema.
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                existing_header = next(reader)
            except StopIteration:
                existing_header = []

        if existing_header == self.COLUMNS:
            return  # Already current schema — nothing to do.

        # Migrate: re-read as dicts against the OLD header, then rewrite with the
        # current schema. Missing columns (e.g. "field") are filled with "".
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            old_rows = list(csv.DictReader(f))

        migrated_rows = []
        for old in old_rows:
            migrated_rows.append({col: old.get(col, "") or "" for col in self.COLUMNS})

        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()
            writer.writerows(migrated_rows)

    def _read_all(self) -> list[dict]:
        """Read all rows from the registry CSV.

        Returns:
            List of dicts with keys matching COLUMNS.
        """
        with self._lock:
            with open(self.path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                return list(reader)

    # Tokens that carry no discriminating meaning for category names. Stripping
    # these lets "No Questions" and "No Merger Questions" collapse together, and
    # "Merger Questions" match "Merger Related Questions".
    _STOPWORDS = {
        "the", "a", "an", "of", "and", "or", "to", "for", "in", "on", "with",
        "about", "related", "regarding", "any", "some", "issue", "issues",
        "claim", "field", "category", "categories", "general", "other",
    }

    def _tokenize(self, s: str) -> set[str]:
        """Lowercase, strip punctuation, split into meaningful tokens."""
        cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s.lower())
        return {
            tok
            for tok in cleaned.split()
            if tok and tok not in self._STOPWORDS
        }

    def _token_set_ratio(self, a: str, b: str) -> float:
        """Jaccard overlap of meaningful tokens (order- and filler-insensitive).

        This is what lets "No Questions" and "No Merger Questions" be recognized
        as the same category: they share the token {questions} (and 'no'), while
        'merger' is the only difference. Returns 0-1.
        """
        ta, tb = self._tokenize(a), self._tokenize(b)
        if not ta and not tb:
            return 1.0
        if not ta or not tb:
            return 0.0
        intersection = len(ta & tb)
        union = len(ta | tb)
        return intersection / union if union else 0.0

    def _similarity(self, a: str, b: str) -> float:
        """Compute normalized similarity between two category strings.

        Category names are short, so a single distinguishing content word (e.g.
        "Contract" vs "No") should PREVENT a merge even when the raw strings
        share many characters. We therefore gate on meaningful-token overlap:

          - If the two token sets are equal (ignoring filler/stopwords), treat
            them as the same category (1.0) — this collapses "Merger Questions"
            / "Merger Related Questions" and word-order variants.
          - Otherwise, when one token set is a subset of the other (the only
            differences are extra filler-ish words), blend the character ratio
            with the token overlap. This catches "No Questions" vs
            "No Merger Questions" (subset: {questions} ⊂ {merger, questions}...
            plus shared 'no') while still requiring real overlap.
          - When each side has its own distinct content word (contract vs no),
            the token overlap caps the score so incidental character overlap
            cannot force a false merge.

        Returns:
            Float between 0.0 and 1.0.
        """
        a_norm, b_norm = a.lower().strip(), b.lower().strip()
        if a_norm == b_norm:
            return 1.0

        char_ratio = SequenceMatcher(None, a_norm, b_norm).ratio()

        ta, tb = self._tokenize(a), self._tokenize(b)
        if not ta or not tb:
            # No meaningful tokens on one side — fall back to character ratio.
            return char_ratio

        if ta == tb:
            return 1.0

        token_ratio = self._token_set_ratio(a, b)

        # Subset relationship: one is the other plus extra words. These are very
        # likely the same category with added qualifiers — let the character
        # ratio contribute so near-duplicates clear the threshold.
        if ta <= tb or tb <= ta:
            return max(token_ratio, 0.5 * char_ratio + 0.5 * token_ratio)

        # Neither is a subset. Before capping at token overlap, check whether the
        # NON-shared tokens are just typos of each other (e.g. "signal"/"signl").
        # If every extra token on the smaller side has a close fuzzy match among
        # the extra tokens on the other side, treat the pair as the same and let
        # the character ratio through — this preserves typo tolerance.
        only_a, only_b = ta - tb, tb - ta
        smaller, larger = (only_a, only_b) if len(only_a) <= len(only_b) else (only_b, only_a)
        if smaller and all(
            any(SequenceMatcher(None, x, y).ratio() >= 0.8 for y in larger)
            for x in smaller
        ):
            return char_ratio

        # Each side has a distinct content word: cap at the token overlap so
        # incidental character similarity cannot trigger a false merge.
        return token_ratio

    def lookup(
        self, field: str, proposed_category: str, threshold: float = 0.7
    ) -> dict:
        """Search the registry for an existing sub-category matching the proposal.

        Only considers categories registered under the same `field`, then matches
        against the sub_category column. Returns the full hierarchy (broad category
        + sub-category) with definitions when found.

        Args:
            field: The extraction field to scope the search to.
            proposed_category: Sub-category name to look up.
            threshold: Minimum similarity to consider a match.

        Returns:
            Dict with keys: found, match (dict or None), suggestions (list),
            field, proposed_category, registry_size.
        """
        all_rows = self._read_all()

        # Scope to categories belonging to the requested field only
        rows = [
            row
            for row in all_rows
            if row.get("field", "").lower().strip() == field.lower().strip()
        ]

        if not rows:
            return {
                "found": False,
                "match": None,
                "suggestions": [],
                "field": field,
                "proposed_category": proposed_category,
                "registry_size": 0,
            }

        scored = []
        for row in rows:
            score = self._similarity(proposed_category, row["sub_category"])
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, best_row = scored[0]

        if best_score >= threshold:
            return {
                "found": True,
                "match": {
                    "field": best_row["field"],
                    "broad_category": best_row["broad_category"],
                    "broad_category_definition": best_row["broad_category_definition"],
                    "sub_category": best_row["sub_category"],
                    "sub_category_definition": best_row["sub_category_definition"],
                    "similarity_score": round(best_score, 3),
                },
                "suggestions": [],
                "field": field,
                "proposed_category": proposed_category,
                "registry_size": len(rows),
            }

        # Return top suggestions even if below threshold (helps the LLM decide)
        suggestions = [
            {
                "field": row["field"],
                "broad_category": row["broad_category"],
                "sub_category": row["sub_category"],
                "sub_category_definition": row["sub_category_definition"],
                "similarity_score": round(score, 3),
            }
            for score, row in scored[:5]
            if score >= 0.4
        ]

        return {
            "found": False,
            "match": None,
            "suggestions": suggestions,
            "field": field,
            "proposed_category": proposed_category,
            "registry_size": len(rows),
        }

    def register(
        self,
        field: str,
        broad_category: str,
        broad_category_definition: str,
        sub_category: str,
        sub_category_definition: str,
        dedup_threshold: float = 0.7,
    ) -> dict:
        """Add a new sub-category to the registry under a broad category for a field.

        Duplicate and broad-category checks are scoped to the same `field`, so the
        same category name may exist independently under different fields. Thread-safe.

        Deduplication (fuzzy):
          - If a near-identical SUB-category already exists for this field
            (similarity >= dedup_threshold), the new one is NOT added and the
            existing entry is returned. This collapses variants like
            "No Questions" / "No Merger Questions".
          - The BROAD category is normalized to the closest existing broad
            category for this field (similarity >= dedup_threshold), so variants
            like "Merger Questions" / "Merger Related Questions" converge to one
            label and its original definition is reused.

        Args:
            field: The extraction field this category belongs to.
            broad_category: High-level category name.
            broad_category_definition: Definition of the broad category.
            sub_category: Specific sub-category name.
            sub_category_definition: Definition of the sub-category.
            dedup_threshold: Similarity (0-1) at/above which two names are treated
                as the same category.

        Returns:
            Dict with registration result including the full hierarchy.
        """
        rows = self._read_all()

        field_key = field.lower().strip()

        # Only compare against rows belonging to the same field
        field_rows = [
            row for row in rows if row.get("field", "").lower().strip() == field_key
        ]

        # Check for a near-duplicate sub-category within this field (fuzzy).
        best_dup = None
        best_dup_score = 0.0
        for row in field_rows:
            score = self._similarity(sub_category, row["sub_category"])
            if score > best_dup_score:
                best_dup_score = score
                best_dup = row

        if best_dup is not None and best_dup_score >= dedup_threshold:
            return {
                "registered": False,
                "reason": (
                    "A near-identical sub-category already exists for this field"
                    if best_dup_score < 1.0
                    else "Sub-category already exists for this field"
                ),
                "similarity_score": round(best_dup_score, 3),
                "existing_entry": {
                    "field": best_dup["field"],
                    "broad_category": best_dup["broad_category"],
                    "broad_category_definition": best_dup["broad_category_definition"],
                    "sub_category": best_dup["sub_category"],
                    "sub_category_definition": best_dup["sub_category_definition"],
                },
                "registry_size": len(rows),
            }

        # Normalize the broad category to the closest existing one for this field.
        # Prefer an exact (case-insensitive) match; otherwise snap to the best
        # fuzzy match above threshold so broad-category variants converge.
        existing_broad_def = broad_category_definition
        exact_broad = next(
            (
                row for row in field_rows
                if row["broad_category"].lower().strip() == broad_category.lower().strip()
            ),
            None,
        )
        if exact_broad is not None:
            existing_broad_def = exact_broad["broad_category_definition"]
            broad_category = exact_broad["broad_category"]  # preserve original casing
        else:
            best_broad = None
            best_broad_score = 0.0
            for row in field_rows:
                score = self._similarity(broad_category, row["broad_category"])
                if score > best_broad_score:
                    best_broad_score = score
                    best_broad = row
            if best_broad is not None and best_broad_score >= dedup_threshold:
                existing_broad_def = best_broad["broad_category_definition"]
                broad_category = best_broad["broad_category"]  # snap to canonical label

        timestamp = datetime.now(timezone.utc).isoformat()
        new_row = {
            "field": field.strip(),
            "broad_category": broad_category.strip(),
            "broad_category_definition": existing_broad_def.strip(),
            "sub_category": sub_category.strip(),
            "sub_category_definition": sub_category_definition.strip(),
            "created_at": timestamp,
        }

        with self._lock:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                writer.writerow(new_row)

        return {
            "registered": True,
            "field": new_row["field"],
            "broad_category": new_row["broad_category"],
            "broad_category_definition": new_row["broad_category_definition"],
            "sub_category": new_row["sub_category"],
            "sub_category_definition": new_row["sub_category_definition"],
            "created_at": timestamp,
            "registry_size": len(rows) + 1,
        }


# ---------------------------------------------------------------------------
# Tool factory (follows make_search_transcript_tool pattern)
# ---------------------------------------------------------------------------


def make_category_registry_tools(
    registry_path: str,
) -> tuple[Tool, Tool]:
    """Create lookup and register Tool objects bound to a specific registry CSV.

    This follows the same factory pattern as make_search_transcript_tool — the
    file path is closed over so the tools can be passed to any agent.

    Args:
        registry_path: Path to the category registry CSV file.
            Created automatically if it doesn't exist.

    Returns:
        Tuple of (lookup_tool, register_tool) ready for agent registration.

    Example:
        lookup_tool, register_tool = make_category_registry_tools("output/categories.csv")
        agent.register_tool(lookup_tool)
        agent.register_tool(register_tool)
    """
    registry = CategoryRegistry(registry_path)

    def _lookup_category(
        field: str, proposed_category: str, threshold: float = 0.75
    ) -> str:
        result = registry.lookup(field, proposed_category, threshold)
        return json.dumps(result, indent=2)

    def _register_category(
        field: str,
        broad_category: str,
        broad_category_definition: str,
        sub_category: str,
        sub_category_definition: str,
    ) -> str:
        result = registry.register(
            field=field,
            broad_category=broad_category,
            broad_category_definition=broad_category_definition,
            sub_category=sub_category,
            sub_category_definition=sub_category_definition,
        )
        return json.dumps(result, indent=2)

    lookup_tool = Tool(
        name="lookup_category",
        description=(
            "Check if a sub-category already exists in the registry for a specific field. "
            "Pass the field you are assigning a category to (e.g. 'primary_issue_category') "
            "and the sub-category name you want to assign. The tool only searches categories "
            "registered under that same field and returns the matching broad category + "
            "sub-category with definitions if one exists above the similarity threshold. "
            "Use the matched names if found. If not found, register a new one with "
            "register_category."
        ),
        input_model=LookupCategoryInput,
        func=_lookup_category,
    )

    register_tool = Tool(
        name="register_category",
        description=(
            "Register a new sub-category under a broad category for a specific field. "
            "Only call this AFTER lookup_category confirms no existing match for that field. "
            "Provide the field the category belongs to (e.g. 'primary_issue_category'), a "
            "broad category (high-level grouping), and a specific sub-category, each with a "
            "clear definition. The category is stored under the given field so it stays "
            "separate from categories for other fields. If the broad category already exists "
            "for this field, reuse its exact name — the tool preserves its original definition."
        ),
        input_model=RegisterCategoryInput,
        func=_register_category,
    )

    return lookup_tool, register_tool
