"""
tools/CATEGORY_TOOLS/category_registry.py

Category Registry Tool for controlled vocabulary management during extraction.

Problem:
    When an LLM extracts call reason categories from transcripts with free range,
    it produces inconsistent labels for the same underlying reason (e.g., "billing issue",
    "billing problem", "bill dispute" all mean the same thing). This makes downstream
    reporting unreliable.

Solution:
    A CSV-backed registry with a two-level hierarchy:
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
        proposed_category: The sub-category name the LLM wants to assign.
        threshold: Minimum similarity score (0-1) to consider a match.
    """

    proposed_category: str = Field(
        ...,
        description=(
            "The sub-category name you want to assign to this transcript. "
            "The tool will check if a matching sub-category already exists in the registry "
            "and return its broad category parent."
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
        broad_category: High-level category grouping.
        broad_category_definition: Definition of the broad category.
        sub_category: Specific reason within the broad category.
        sub_category_definition: Definition of the sub-category.
    """

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
        broad_category, broad_category_definition, sub_category,
        sub_category_definition, created_at
    """

    COLUMNS = [
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
        """Create the registry CSV with headers if it doesn't already exist."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                writer.writeheader()

    def _read_all(self) -> list[dict]:
        """Read all rows from the registry CSV.

        Returns:
            List of dicts with keys matching COLUMNS.
        """
        with self._lock:
            with open(self.path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                return list(reader)

    def _similarity(self, a: str, b: str) -> float:
        """Compute normalized similarity between two strings.

        Uses SequenceMatcher for fuzzy comparison after lowercasing/stripping.

        Args:
            a: First string.
            b: Second string.

        Returns:
            Float between 0.0 and 1.0.
        """
        return SequenceMatcher(
            None, a.lower().strip(), b.lower().strip()
        ).ratio()

    def lookup(self, proposed_category: str, threshold: float = 0.75) -> dict:
        """Search the registry for an existing sub-category matching the proposal.

        Matches against the sub_category column. Returns the full hierarchy
        (broad category + sub-category) with definitions when found.

        Args:
            proposed_category: Sub-category name to look up.
            threshold: Minimum similarity to consider a match.

        Returns:
            Dict with keys: found, match (dict or None), suggestions (list),
            proposed_category, registry_size.
        """
        rows = self._read_all()

        if not rows:
            return {
                "found": False,
                "match": None,
                "suggestions": [],
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
                    "broad_category": best_row["broad_category"],
                    "broad_category_definition": best_row["broad_category_definition"],
                    "sub_category": best_row["sub_category"],
                    "sub_category_definition": best_row["sub_category_definition"],
                    "similarity_score": round(best_score, 3),
                },
                "suggestions": [],
                "proposed_category": proposed_category,
                "registry_size": len(rows),
            }

        # Return top suggestions even if below threshold (helps the LLM decide)
        suggestions = [
            {
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
            "proposed_category": proposed_category,
            "registry_size": len(rows),
        }

    def register(
        self,
        broad_category: str,
        broad_category_definition: str,
        sub_category: str,
        sub_category_definition: str,
    ) -> dict:
        """Add a new sub-category to the registry under a broad category.

        Checks for exact sub-category duplicates before writing. Thread-safe.
        If the broad category already exists, its definition is preserved from
        the first entry (consistency).

        Args:
            broad_category: High-level category name.
            broad_category_definition: Definition of the broad category.
            sub_category: Specific sub-category name.
            sub_category_definition: Definition of the sub-category.

        Returns:
            Dict with registration result including the full hierarchy.
        """
        rows = self._read_all()

        # Check for exact sub-category duplicate (case-insensitive)
        for row in rows:
            if row["sub_category"].lower().strip() == sub_category.lower().strip():
                return {
                    "registered": False,
                    "reason": "Sub-category already exists",
                    "existing_entry": {
                        "broad_category": row["broad_category"],
                        "broad_category_definition": row["broad_category_definition"],
                        "sub_category": row["sub_category"],
                        "sub_category_definition": row["sub_category_definition"],
                    },
                    "registry_size": len(rows),
                }

        # If broad category already exists, reuse its definition for consistency
        existing_broad_def = broad_category_definition
        for row in rows:
            if row["broad_category"].lower().strip() == broad_category.lower().strip():
                existing_broad_def = row["broad_category_definition"]
                broad_category = row["broad_category"]  # preserve original casing
                break

        timestamp = datetime.now(timezone.utc).isoformat()
        new_row = {
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

    def _lookup_category(proposed_category: str, threshold: float = 0.75) -> str:
        result = registry.lookup(proposed_category, threshold)
        return json.dumps(result, indent=2)

    def _register_category(
        broad_category: str,
        broad_category_definition: str,
        sub_category: str,
        sub_category_definition: str,
    ) -> str:
        result = registry.register(
            broad_category=broad_category,
            broad_category_definition=broad_category_definition,
            sub_category=sub_category,
            sub_category_definition=sub_category_definition,
        )
        return json.dumps(result, indent=2)

    lookup_tool = Tool(
        name="lookup_category",
        description=(
            "Check if a call reason sub-category already exists in the registry. "
            "Pass the sub-category name you want to assign and the tool returns the "
            "matching broad category + sub-category with definitions if one exists "
            "above the similarity threshold. Use the matched names if found. "
            "If not found, register a new one with register_category."
        ),
        input_model=LookupCategoryInput,
        func=_lookup_category,
    )

    register_tool = Tool(
        name="register_category",
        description=(
            "Register a new call reason sub-category under a broad category in the registry. "
            "Only call this AFTER lookup_category confirms no existing match. "
            "Provide both a broad category (high-level grouping) and a specific sub-category, "
            "each with a clear definition. If the broad category already exists, reuse its "
            "exact name — the tool will preserve its original definition automatically."
        ),
        input_model=RegisterCategoryInput,
        func=_register_category,
    )

    return lookup_tool, register_tool
