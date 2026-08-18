"""
tools/EVAL_TOOLS/aggregate_judge_results.py

Aggregates a JudgeAgent results DataFrame into a structured JudgeAggregation
following a 5-step pipeline:

  Step A: Add derived columns (evidence_score, is_failure, is_severe_failure)
  Step B: Build field summary — top-10 worst fields by hallucination, groundedness, severe failures
  Step C: Build error summary — per error_type with counts, affected rows/fields, samples, avg severity
  Step D: Extract representative examples — max 20 diverse failure cases
  Step E: Create prompt lessons — keep_doing, stop_doing, prompt_changes

Input DataFrame columns (per field, from JudgeAgent.flatten_judge_results):
  - {field}_GROUNDED (bool)
  - {field}_HALLUCINATED (bool)
  - {field}_EVIDENCE_STRENGTH (str: none|low|medium|high)
  - {field}_EVIDENCE_FOUND (bool)
  - {field}_ERROR_TYPE (str|None)
  - {field}_JUDGE_EXPLANATION (str)
  - {field}_PROMPT_ADJUSTMENT_SUGGESTION (str|None)
  - {field}_EVIDENCE_CHUNK (str)
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from data_models.judge_aggregation import (
    ErrorSummary,
    FailureExample,
    FieldRanking,
    FieldSummary,
    JudgeAggregation,
    PromptLessons,
)


# Column suffix constants (aligned with JudgeAgent.flatten_judge_results)
_GROUNDED = "_GROUNDED"
_HALLUCINATED = "_HALLUCINATED"
_EVIDENCE_STRENGTH = "_EVIDENCE_STRENGTH"
_EVIDENCE_FOUND = "_EVIDENCE_FOUND"
_ERROR_TYPE = "_ERROR_TYPE"
_EXPLANATION = "_JUDGE_EXPLANATION"
_SUGGESTION = "_PROMPT_ADJUSTMENT_SUGGESTION"
_EVIDENCE_CHUNK = "_EVIDENCE_CHUNK"

# Evidence strength to numeric mapping
STRENGTH_MAP = {"none": 0, "low": 1, "medium": 2, "high": 3}


def _discover_fields(df: pd.DataFrame) -> list[str]:
    """Identify unique field prefixes from _GROUNDED columns."""
    return [
        col[: -len(_GROUNDED)]
        for col in df.columns
        if col.endswith(_GROUNDED)
    ]


# ---------------------------------------------------------------------------
# Step A: Derive columns — normalizes to a long-form DataFrame for analysis
# ---------------------------------------------------------------------------

def _build_long_df(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    """Melt the wide judge DataFrame into a long-form table with one row per field-judgment.

    Adds derived columns:
      - evidence_score: numeric 0-3
      - is_failure: not grounded OR hallucinated
      - is_severe_failure: hallucinated AND evidence_score == 0

    Args:
        df: Wide-form judge results DataFrame.
        fields: List of field prefixes discovered from columns.

    Returns:
        Long-form DataFrame with columns: row_idx, field_name, grounded,
        hallucinated, evidence_strength, evidence_score, evidence_found,
        error_type, explanation, suggestion, evidence_chunk, claim_value,
        is_failure, is_severe_failure.
    """
    records: list[dict[str, Any]] = []

    # Use DataFrame index as row identifier; fall back to positional index
    row_id_col = None
    for candidate in ("AGENTRECORDINGSESSIONID", "row_id", "id"):
        if candidate in df.columns:
            row_id_col = candidate
            break

    for idx, row in df.iterrows():
        row_id = str(row[row_id_col]) if row_id_col else str(idx)

        for field in fields:
            grounded_val = row.get(f"{field}{_GROUNDED}")
            hallucinated_val = row.get(f"{field}{_HALLUCINATED}")

            # Skip fields not judged for this row
            if pd.isna(grounded_val):
                continue

            grounded = bool(grounded_val)
            hallucinated = bool(hallucinated_val) if not pd.isna(hallucinated_val) else False

            strength_raw = str(row.get(f"{field}{_EVIDENCE_STRENGTH}", "none")).lower().strip()
            evidence_score = STRENGTH_MAP.get(strength_raw, 0)

            is_failure = (not grounded) or hallucinated
            is_severe_failure = hallucinated and (evidence_score == 0)

            records.append({
                "row_id": row_id,
                "field_name": field,
                "claim_value": row.get(field, None),
                "grounded": grounded,
                "hallucinated": hallucinated,
                "evidence_strength": strength_raw,
                "evidence_score": evidence_score,
                "evidence_found": bool(row.get(f"{field}{_EVIDENCE_FOUND}", False)),
                "error_type": row.get(f"{field}{_ERROR_TYPE}", None),
                "explanation": str(row.get(f"{field}{_EXPLANATION}", "") or ""),
                "suggestion": row.get(f"{field}{_SUGGESTION}", None),
                "evidence_chunk": str(row.get(f"{field}{_EVIDENCE_CHUNK}", "") or ""),
                "is_failure": is_failure,
                "is_severe_failure": is_severe_failure,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step B: Field Summary
# ---------------------------------------------------------------------------

def _build_field_summary(long_df: pd.DataFrame, top_n: int = 10) -> FieldSummary:
    """Compute top-10 worst fields by hallucination rate, ungrounded rate, severe failures."""

    field_stats = long_df.groupby("field_name").agg(
        total_judged=("grounded", "count"),
        hallucinated_count=("hallucinated", "sum"),
        ungrounded_count=("grounded", lambda s: (~s).sum()),
        severe_count=("is_severe_failure", "sum"),
    ).reset_index()

    field_stats["hallucination_rate"] = (
        field_stats["hallucinated_count"] / field_stats["total_judged"]
    )
    field_stats["ungrounded_rate"] = (
        field_stats["ungrounded_count"] / field_stats["total_judged"]
    )

    # Top by hallucination rate
    by_hallucination = field_stats.nlargest(top_n, "hallucination_rate")
    worst_hallucination = [
        FieldRanking(
            field_name=r["field_name"],
            rate=round(r["hallucination_rate"], 4),
            count=int(r["hallucinated_count"]),
            total_judged=int(r["total_judged"]),
        )
        for _, r in by_hallucination.iterrows()
        if r["hallucination_rate"] > 0
    ]

    # Top by ungrounded rate
    by_ungrounded = field_stats.nlargest(top_n, "ungrounded_rate")
    worst_ungrounded = [
        FieldRanking(
            field_name=r["field_name"],
            rate=round(r["ungrounded_rate"], 4),
            count=int(r["ungrounded_count"]),
            total_judged=int(r["total_judged"]),
        )
        for _, r in by_ungrounded.iterrows()
        if r["ungrounded_rate"] > 0
    ]

    # Top by severe failures (absolute count)
    by_severe = field_stats.nlargest(top_n, "severe_count")
    worst_severe = [
        FieldRanking(
            field_name=r["field_name"],
            rate=round(r["severe_count"] / r["total_judged"], 4) if r["total_judged"] else 0.0,
            count=int(r["severe_count"]),
            total_judged=int(r["total_judged"]),
        )
        for _, r in by_severe.iterrows()
        if r["severe_count"] > 0
    ]

    return FieldSummary(
        worst_by_hallucination_rate=worst_hallucination,
        worst_by_ungrounded_rate=worst_ungrounded,
        worst_by_severe_failures=worst_severe,
    )


# ---------------------------------------------------------------------------
# Step C: Error Summary
# ---------------------------------------------------------------------------

def _build_error_summaries(long_df: pd.DataFrame, max_samples: int = 3) -> list[ErrorSummary]:
    """Build per-error_type aggregation with counts, fields, examples, avg severity."""

    # Filter to rows with a non-null, non-empty error_type
    error_df = long_df[
        long_df["error_type"].notna() &
        (long_df["error_type"].astype(str).str.strip() != "")
    ].copy()

    if error_df.empty:
        return []

    summaries: list[ErrorSummary] = []

    for error_type, group in error_df.groupby("error_type"):
        total_count = len(group)
        unique_rows = group["row_id"].nunique()
        avg_evidence_score = group["evidence_score"].mean()

        # Top fields affected by this error type
        field_counts = group["field_name"].value_counts()
        top_fields = field_counts.head(5).index.tolist()

        # Sample explanations (unique, non-empty)
        explanations = (
            group["explanation"]
            .loc[group["explanation"].str.strip() != ""]
            .drop_duplicates()
            .head(max_samples)
            .tolist()
        )

        # Sample evidence chunks (unique, non-empty)
        chunks = (
            group["evidence_chunk"]
            .loc[group["evidence_chunk"].str.strip() != ""]
            .drop_duplicates()
            .head(max_samples)
            .tolist()
        )

        summaries.append(ErrorSummary(
            error_type=str(error_type),
            total_count=total_count,
            unique_rows_affected=unique_rows,
            top_fields=top_fields,
            sample_explanations=[s[:300] for s in explanations],
            sample_evidence_chunks=[s[:300] for s in chunks],
            average_evidence_score=round(avg_evidence_score, 2),
        ))

    # Sort by total_count descending
    summaries.sort(key=lambda s: s.total_count, reverse=True)
    return summaries


# ---------------------------------------------------------------------------
# Step D: Representative Examples
# ---------------------------------------------------------------------------

def _select_representative_examples(
    long_df: pd.DataFrame,
    examples_per_error_type: int = 3,
    top_error_types: int = 5,
    max_total: int = 20,
) -> list[FailureExample]:
    """Select diverse, interpretable failure examples.

    Selection strategy:
      - Pick top N error types by frequency
      - For each, select up to `examples_per_error_type` examples prioritizing:
        severe failures first, then shortest explanations (most interpretable)
      - Ensure field diversity within each error type
      - Cap at max_total

    Args:
        long_df: Long-form DataFrame with failure rows.
        examples_per_error_type: Max examples per error type.
        top_error_types: Number of error types to sample from.
        max_total: Hard cap on total examples.

    Returns:
        List of FailureExample instances.
    """
    failures = long_df[long_df["is_failure"]].copy()

    if failures.empty:
        return []

    # Filter to rows with an error_type for structured selection
    typed_failures = failures[
        failures["error_type"].notna() &
        (failures["error_type"].astype(str).str.strip() != "")
    ]

    # If no typed failures, fall back to severe failures
    if typed_failures.empty:
        typed_failures = failures

    # Identify top error types
    error_counts = typed_failures["error_type"].value_counts()
    top_types = error_counts.head(top_error_types).index.tolist()

    examples: list[FailureExample] = []
    seen_keys: set[str] = set()  # (row_id, field_name) to avoid duplicates

    for error_type in top_types:
        if len(examples) >= max_total:
            break

        group = typed_failures[typed_failures["error_type"] == error_type]

        # Prioritize severe failures, then sort by explanation length (shorter = more interpretable)
        group = group.sort_values(
            by=["is_severe_failure", "evidence_score"],
            ascending=[False, True],
        )

        # Ensure field diversity — pick from different fields
        fields_picked: set[str] = set()
        count = 0

        for _, row in group.iterrows():
            if count >= examples_per_error_type:
                break
            if len(examples) >= max_total:
                break

            key = f"{row['row_id']}_{row['field_name']}"
            if key in seen_keys:
                continue

            # Prefer diverse fields
            if row["field_name"] in fields_picked and count > 1:
                continue

            seen_keys.add(key)
            fields_picked.add(row["field_name"])
            count += 1

            suggestion = row.get("suggestion")
            if pd.isna(suggestion) or str(suggestion).strip() == "":
                suggestion = None

            examples.append(FailureExample(
                row_id=str(row["row_id"]),
                field_name=row["field_name"],
                claim_value=row["claim_value"],
                error_type=str(row["error_type"]) if not pd.isna(row["error_type"]) else None,
                evidence_strength=row["evidence_strength"],
                explanation=str(row["explanation"])[:300],
                evidence_chunk=str(row["evidence_chunk"])[:300],
                prompt_adjustment_suggestion=str(suggestion) if suggestion else None,
            ))

    # If we haven't hit max_total, fill with remaining severe failures from any type
    if len(examples) < max_total:
        remaining_severe = failures[failures["is_severe_failure"]].sort_values(
            "evidence_score", ascending=True
        )
        for _, row in remaining_severe.iterrows():
            if len(examples) >= max_total:
                break

            key = f"{row['row_id']}_{row['field_name']}"
            if key in seen_keys:
                continue

            seen_keys.add(key)
            suggestion = row.get("suggestion")
            if pd.isna(suggestion) or str(suggestion).strip() == "":
                suggestion = None

            examples.append(FailureExample(
                row_id=str(row["row_id"]),
                field_name=row["field_name"],
                claim_value=row["claim_value"],
                error_type=str(row["error_type"]) if not pd.isna(row.get("error_type")) else None,
                evidence_strength=row["evidence_strength"],
                explanation=str(row["explanation"])[:300],
                evidence_chunk=str(row["evidence_chunk"])[:300],
                prompt_adjustment_suggestion=str(suggestion) if suggestion else None,
            ))

    return examples


# ---------------------------------------------------------------------------
# Step E: Prompt Lessons
# ---------------------------------------------------------------------------

def _derive_prompt_lessons(
    long_df: pd.DataFrame,
    error_summaries: list[ErrorSummary],
    field_summary: FieldSummary,
) -> PromptLessons:
    """Derive actionable keep_doing / stop_doing / prompt_changes from the data.

    Uses patterns in the data to generate heuristic-based lessons:
    - keep_doing: fields/patterns with high grounding rates
    - stop_doing: common error patterns that indicate prompt flaws
    - prompt_changes: deduplicated prompt_adjustment_suggestions + inferred rules

    Args:
        long_df: Long-form DataFrame with all judgments.
        error_summaries: Pre-computed error summaries.
        field_summary: Pre-computed field rankings.

    Returns:
        PromptLessons instance.
    """
    keep_doing: list[str] = []
    stop_doing: list[str] = []
    prompt_changes: list[str] = []

    # --- keep_doing: identify well-performing patterns ---
    field_stats = long_df.groupby("field_name").agg(
        total=("grounded", "count"),
        grounded_count=("grounded", "sum"),
    ).reset_index()
    field_stats["grounded_rate"] = field_stats["grounded_count"] / field_stats["total"]

    high_performers = field_stats[field_stats["grounded_rate"] >= 0.9]
    if not high_performers.empty:
        top_fields = high_performers.nlargest(3, "grounded_rate")["field_name"].tolist()
        keep_doing.append(
            f"Fields with high grounding rates ({', '.join(top_fields)}) — "
            "the extraction approach for these fields is working well."
        )

    # Check if fields with quotes have higher grounding
    if "evidence_found" in long_df.columns:
        evidence_found_rate = long_df["evidence_found"].mean()
        if evidence_found_rate > 0.7:
            keep_doing.append(
                "Evidence retrieval is succeeding for most fields — "
                "continue requiring supporting quotes."
            )

    # --- stop_doing: identify anti-patterns from error types ---
    error_type_lessons = {
        "unsupported_claim": "Do not extract claims that lack explicit textual support in the transcript.",
        "overgeneralization": "Do not generalize or infer broader conclusions from specific statements.",
        "missing_from_context": "Do not extract values when the relevant information is absent from the transcript.",
        "ambiguous_evidence": "Do not mark a field as present when the supporting evidence is ambiguous or indirect.",
        "wrong_extraction": "Do not extract values from unrelated parts of the transcript.",
    }

    for es in error_summaries[:5]:
        lesson = error_type_lessons.get(es.error_type)
        if lesson:
            stop_doing.append(f"{lesson} (seen {es.total_count}x across fields: {', '.join(es.top_fields[:3])})")

    # Generic stop_doing for high hallucination
    total_judgments = len(long_df)
    total_hallucinated = long_df["hallucinated"].sum()
    if total_judgments > 0 and (total_hallucinated / total_judgments) > 0.2:
        stop_doing.append(
            "Do not infer values from related but non-explicit text. "
            "If no direct evidence exists, return null."
        )

    # --- prompt_changes: collect and deduplicate suggestions ---
    suggestions = long_df["suggestion"].dropna()
    suggestions = suggestions[suggestions.astype(str).str.strip() != ""]

    if not suggestions.empty:
        # Deduplicate (case-insensitive) and rank by frequency
        counter: Counter[str] = Counter(suggestions.astype(str).str.strip().str.lower())
        casing_map: dict[str, str] = {}
        for s in suggestions:
            key = str(s).strip().lower()
            if key not in casing_map:
                casing_map[key] = str(s).strip()

        for key, _ in counter.most_common(10):
            prompt_changes.append(casing_map[key])

    # Add inferred prompt changes from patterns
    severe_rate = long_df["is_severe_failure"].mean() if len(long_df) > 0 else 0
    if severe_rate > 0.1:
        prompt_changes.append(
            "Return null when no explicit support exists in the transcript "
            "rather than inferring a value."
        )

    # Deduplicate final lists
    keep_doing = _deduplicate(keep_doing)
    stop_doing = _deduplicate(stop_doing)
    prompt_changes = _deduplicate(prompt_changes)

    return PromptLessons(
        keep_doing=keep_doing,
        stop_doing=stop_doing,
        prompt_changes=prompt_changes,
    )


def _deduplicate(items: list[str]) -> list[str]:
    """Remove near-duplicate strings (case-insensitive)."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.lower().strip()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def aggregate_judge_results(
    judge_df: pd.DataFrame,
    top_n_fields: int = 10,
    examples_per_error_type: int = 3,
    top_error_types: int = 5,
    max_examples: int = 20,
) -> JudgeAggregation:
    """Aggregate a JudgeAgent results DataFrame into a structured summary.

    Implements the full 5-step pipeline:
      A. Derive columns (evidence_score, is_failure, is_severe_failure)
      B. Build field summary (top-10 worst by hallucination/groundedness/severe)
      C. Build error summary (per error_type breakdown)
      D. Extract representative examples (max 20 diverse failures)
      E. Create prompt lessons (keep_doing, stop_doing, prompt_changes)

    Args:
        judge_df: Wide-form DataFrame with flattened judge columns
                  (from JudgeAgent.flatten_judge_results).
        top_n_fields: Number of worst fields to include per ranking dimension.
        examples_per_error_type: Max examples per error type in Step D.
        top_error_types: Number of error types to sample from in Step D.
        max_examples: Hard cap on total representative examples.

    Returns:
        JudgeAggregation instance ready for serialization or passing to
        PromptGeneratorAgent via to_revision_brief_text().
    """
    if judge_df is None or judge_df.empty:
        return JudgeAggregation()

    fields = _discover_fields(judge_df)

    if not fields:
        return JudgeAggregation(total_rows=len(judge_df))

    # Step A: Build long-form DataFrame with derived columns
    long_df = _build_long_df(judge_df, fields)

    if long_df.empty:
        return JudgeAggregation(total_rows=len(judge_df))

    # Overall metrics
    total_rows = len(judge_df)
    total_judgments = len(long_df)
    total_grounded = int(long_df["grounded"].sum())
    total_hallucinated = int(long_df["hallucinated"].sum())
    total_failures = int(long_df["is_failure"].sum())
    total_severe = int(long_df["is_severe_failure"].sum())

    overall_grounded_rate = total_grounded / total_judgments if total_judgments else 0.0
    overall_hallucination_rate = total_hallucinated / total_judgments if total_judgments else 0.0
    overall_failure_rate = total_failures / total_judgments if total_judgments else 0.0

    # Step B: Field summary
    field_summary = _build_field_summary(long_df, top_n=top_n_fields)

    # Step C: Error summaries
    error_summaries = _build_error_summaries(long_df)

    # Step D: Representative examples
    representative_examples = _select_representative_examples(
        long_df,
        examples_per_error_type=examples_per_error_type,
        top_error_types=top_error_types,
        max_total=max_examples,
    )

    # Step E: Prompt lessons
    prompt_lessons = _derive_prompt_lessons(long_df, error_summaries, field_summary)

    return JudgeAggregation(
        total_rows=total_rows,
        total_field_judgments=total_judgments,
        total_failures=total_failures,
        total_severe_failures=total_severe,
        overall_grounded_rate=round(overall_grounded_rate, 4),
        overall_hallucination_rate=round(overall_hallucination_rate, 4),
        overall_failure_rate=round(overall_failure_rate, 4),
        field_summary=field_summary,
        error_summaries=error_summaries,
        representative_examples=representative_examples,
        prompt_lessons=prompt_lessons,
    )
