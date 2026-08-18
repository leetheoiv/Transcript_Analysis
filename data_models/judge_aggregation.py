"""
data_models/judge_aggregation.py

Structured aggregation of JudgeAgent results following a 5-step pipeline:

Step A: Derived columns (evidence_score, is_failure, is_severe_failure)
Step B: Field summary — top-10 worst fields by hallucination, groundedness, severe failures
Step C: Error summary — per error_type breakdown with counts, affected fields, examples
Step D: Representative examples — max 20 diverse, interpretable failure cases
Step E: Prompt lessons — keep_doing, stop_doing, prompt_changes

The output is designed to be passed directly to the PromptGeneratorAgent
as a revision_brief to drive targeted prompt improvement.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Step B: Field Summary
# ---------------------------------------------------------------------------

class FieldRanking(BaseModel):
    """A single field's ranking entry with its rate/count."""
    field_name: str
    rate: float = Field(0.0, ge=0.0, le=1.0)
    count: int = 0
    total_judged: int = 0


class FieldSummary(BaseModel):
    """Top-10 worst fields across three dimensions."""
    worst_by_hallucination_rate: list[FieldRanking] = Field(default_factory=list)
    worst_by_ungrounded_rate: list[FieldRanking] = Field(default_factory=list)
    worst_by_severe_failures: list[FieldRanking] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Step C: Error Summary
# ---------------------------------------------------------------------------

class ErrorSummary(BaseModel):
    """Aggregated breakdown for a single error_type."""
    error_type: str
    total_count: int = 0
    unique_rows_affected: int = 0
    top_fields: list[str] = Field(default_factory=list)
    sample_explanations: list[str] = Field(default_factory=list)
    sample_evidence_chunks: list[str] = Field(default_factory=list)
    average_evidence_score: float = 0.0


# ---------------------------------------------------------------------------
# Step D: Representative Examples
# ---------------------------------------------------------------------------

class FailureExample(BaseModel):
    """A single representative failure case for the prompt generator."""
    row_id: str = ""
    field_name: str
    claim_value: Any = None
    error_type: Optional[str] = None
    evidence_strength: str = "none"
    explanation: str = ""
    evidence_chunk: str = ""
    prompt_adjustment_suggestion: Optional[str] = None


# ---------------------------------------------------------------------------
# Step E: Prompt Lessons
# ---------------------------------------------------------------------------

class PromptLessons(BaseModel):
    """Actionable directives derived from the aggregated judge results."""
    keep_doing: list[str] = Field(default_factory=list)
    stop_doing: list[str] = Field(default_factory=list)
    prompt_changes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level aggregation
# ---------------------------------------------------------------------------

class JudgeAggregation(BaseModel):
    """
    Complete aggregation of JudgeAgent results following the 5-step pipeline.

    Designed to be serialized and passed as revision_brief to the
    PromptGeneratorAgent so it can address every identified issue.
    """
    id: UUID = Field(default_factory=uuid4)

    # Overall metrics
    total_rows: int = 0
    total_field_judgments: int = 0
    total_failures: int = 0
    total_severe_failures: int = 0
    overall_grounded_rate: float = Field(0.0, ge=0.0, le=1.0)
    overall_hallucination_rate: float = Field(0.0, ge=0.0, le=1.0)
    overall_failure_rate: float = Field(0.0, ge=0.0, le=1.0)

    # Step B
    field_summary: FieldSummary = Field(default_factory=FieldSummary)

    # Step C
    error_summaries: list[ErrorSummary] = Field(default_factory=list)

    # Step D
    representative_examples: list[FailureExample] = Field(default_factory=list)

    # Step E
    prompt_lessons: PromptLessons = Field(default_factory=PromptLessons)

    created_at: datetime = Field(default_factory=_utcnow)

    def to_revision_brief_text(self) -> str:
        """Render the aggregation as a structured text block for revision_brief.

        Returns:
            A formatted string suitable for passing to
            PromptGeneratorAgent.run(revision_brief=...).
        """
        lines: list[str] = []

        # --- Header ---
        lines.append("## Judge Evaluation Summary")
        lines.append("")
        lines.append(f"- Total rows evaluated: {self.total_rows}")
        lines.append(f"- Total field judgments: {self.total_field_judgments}")
        lines.append(f"- Overall grounded rate: {self.overall_grounded_rate:.1%}")
        lines.append(f"- Overall hallucination rate: {self.overall_hallucination_rate:.1%}")
        lines.append(f"- Overall failure rate: {self.overall_failure_rate:.1%}")
        lines.append(f"- Severe failures (hallucinated + no evidence): {self.total_severe_failures}")
        lines.append("")

        # --- Step B: Field Summary ---
        lines.append("### Worst Performing Fields")
        lines.append("")

        if self.field_summary.worst_by_hallucination_rate:
            lines.append("**By hallucination rate:**")
            for fr in self.field_summary.worst_by_hallucination_rate:
                lines.append(f"  - {fr.field_name}: {fr.rate:.1%} ({fr.count}/{fr.total_judged})")
            lines.append("")

        if self.field_summary.worst_by_ungrounded_rate:
            lines.append("**By ungrounded rate:**")
            for fr in self.field_summary.worst_by_ungrounded_rate:
                lines.append(f"  - {fr.field_name}: {fr.rate:.1%} ({fr.count}/{fr.total_judged})")
            lines.append("")

        if self.field_summary.worst_by_severe_failures:
            lines.append("**By severe failures:**")
            for fr in self.field_summary.worst_by_severe_failures:
                lines.append(f"  - {fr.field_name}: {fr.count} severe failures ({fr.total_judged} judged)")
            lines.append("")

        # --- Step C: Error Summary ---
        if self.error_summaries:
            lines.append("### Error Type Breakdown")
            lines.append("")
            for es in self.error_summaries:
                lines.append(f"**{es.error_type}** — {es.total_count} occurrences, "
                             f"{es.unique_rows_affected} unique rows, "
                             f"avg evidence score: {es.average_evidence_score:.1f}/3")
                if es.top_fields:
                    lines.append(f"  Fields most affected: {', '.join(es.top_fields)}")
                if es.sample_explanations:
                    lines.append("  Example explanations:")
                    for exp in es.sample_explanations[:3]:
                        lines.append(f"    - \"{exp}\"")
                lines.append("")

        # --- Step D: Representative Examples ---
        if self.representative_examples:
            lines.append("### Representative Failure Examples")
            lines.append("")
            for i, ex in enumerate(self.representative_examples, 1):
                lines.append(f"**Example {i}:** field={ex.field_name}, "
                             f"error_type={ex.error_type or 'unknown'}, "
                             f"evidence_strength={ex.evidence_strength}")
                lines.append(f"  Claim: {ex.claim_value}")
                lines.append(f"  Explanation: {ex.explanation}")
                if ex.evidence_chunk:
                    # Truncate long evidence chunks for readability
                    chunk = ex.evidence_chunk[:200] + "..." if len(ex.evidence_chunk) > 200 else ex.evidence_chunk
                    lines.append(f"  Evidence: {chunk}")
                if ex.prompt_adjustment_suggestion:
                    lines.append(f"  Suggestion: {ex.prompt_adjustment_suggestion}")
                lines.append("")

        # --- Step E: Prompt Lessons ---
        if any([self.prompt_lessons.keep_doing,
                self.prompt_lessons.stop_doing,
                self.prompt_lessons.prompt_changes]):
            lines.append("### Prompt Lessons")
            lines.append("")

            if self.prompt_lessons.keep_doing:
                lines.append("**Keep doing:**")
                for item in self.prompt_lessons.keep_doing:
                    lines.append(f"  - {item}")
                lines.append("")

            if self.prompt_lessons.stop_doing:
                lines.append("**Stop doing:**")
                for item in self.prompt_lessons.stop_doing:
                    lines.append(f"  - {item}")
                lines.append("")

            if self.prompt_lessons.prompt_changes:
                lines.append("**Prompt changes to make:**")
                for item in self.prompt_lessons.prompt_changes:
                    lines.append(f"  - {item}")
                lines.append("")

        return "\n".join(lines)
