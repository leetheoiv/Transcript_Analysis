from pydantic import BaseModel,Field
from typing import Literal
from datetime import datetime



class SemanticQuality(BaseModel):
    correctness_rate: float = Field(0.0, ge=0.0, le=1.0, description="Proportion of evaluated outputs judged correct.")
    consistency_rate: float = Field(0.0, ge=0.0, le=1.0, description="Proportion of evaluated outputs judged consistent.")
    hallucination_rate: float = Field(0.0, ge=0.0, le=1.0, description="Proportion of evaluated outputs judged hallucinated.")

    correctness_count: int = Field(0, ge=0)
    consistency_count: int = Field(0, ge=0)
    hallucination_count: int = Field(0, ge=0)

    total_samples: int = Field(0,ge=0,description="Total number of samples that were evaluated")
    total_evaluated: int = Field(0, ge=0, description="Total number of evaluated outputs.")

    # Retrieval-failure tracking (Option C). When the judge could not retrieve
    # usable evidence, that judgment is NOT the extractor's fault and is excluded
    # from the correctness denominator. These fields surface how much of the data
    # was unjudgeable so the signal is visible rather than silently dropped.
    retrieval_failure_rate: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Proportion of judgments the judge could not retrieve evidence for (excluded from correctness).",
    )
    retrieval_failure_count: int = Field(0, ge=0)
    correctness_source: str = Field(
        "judge",
        description="How correctness_rate was computed: 'judge' (grounding) or 'gold' (ground-truth labels).",
    )


class ConsistencyQuality(BaseModel):
    consistency_rate: float = Field(0.0, ge=0.0, le=1.0, description="Proportion of evaluated outputs that were consistent across repeated runs.")
    consistency_count: int = Field(0, ge=0, description="Number of session-field comparisons that were consistent.")
    total_evaluated: int = Field(0, ge=0, description="Total number of session-field comparisons evaluated for consistency.")




class EvaluationResult(BaseModel):
    date_time: datetime = Field(...)
    semantic_quality: SemanticQuality = Field(default_factory=SemanticQuality)
    consistency_quality: ConsistencyQuality = Field(default_factory=ConsistencyQuality)

    per_field_correctness_rate: dict[str, float] = Field(default_factory=dict)
    per_field_hallucination_rate: dict[str, float] = Field(default_factory=dict)
    per_field_consistency_rate: dict[str, float] = Field(default_factory=dict)