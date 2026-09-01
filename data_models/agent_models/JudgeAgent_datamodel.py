from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class JudgeFieldInput(BaseModel):
    task_prompt:str = Field(...,description='...entire original extraction task prompt...')
    field_name: str = Field(..., description="Name of the extracted field being judged")
    claim_value: Any = Field(..., description="Original extracted value for the field")
    quote: Optional[str] = Field(default="", description="Direct quote from the extractor, if available")
    ai_reasoning: Optional[str] = Field(default="", description="Reasoning from the extractor, if available")


class JudgeFieldResult(BaseModel):
    field_name: str
    claim_value: Any
    grounded: bool
    evidence_strength: Literal["none", "low", "medium", "high"]
    evidence_found: bool
    # Distinguishes WHY a claim is ungrounded so retrieval failures are not
    # blamed on the extractor:
    #   - supported: evidence retrieved and it supports the claim
    #   - contradicted: evidence retrieved and it contradicts the claim
    #     (a genuine extraction error / hallucination)
    #   - absent_from_transcript: evidence retrieved for the surrounding context
    #     but the claimed fact is genuinely not present (genuine extraction error)
    #   - retrieval_failure: no usable evidence could be retrieved, so grounding
    #     cannot be judged either way (NOT the extractor's fault)
    retrieval_status: Literal[
        "supported",
        "contradicted",
        "absent_from_transcript",
        "retrieval_failure",
    ] = "retrieval_failure"
    evidence_chunk: str = ""
    hallucinated: bool = False
    search_terms_used: list[str] = []
    explanation: str = ""
    error_type: Optional[Literal[
    "unsupported_claim",
    "wrong_extraction",
    "missing_from_context",
    "overgeneralization",
    "ambiguous_evidence",
    "bad_search_terms",
    "formatting_only",
    "not_hallucinated_but_unverifiable",
    "other"
        ]] = None
    prompt_adjustment_suggestion: Optional[str] = Field(default=None, description="Suggestions to improve the prompt so it avoids mislabelling")
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_total_tokens: int = 0

    @field_validator("error_type", mode="before")
    @classmethod
    def _coerce_empty_error_type(cls, v):
        if v == "" or v is None:
            return None
        return v


class GroundingJudgeResponse(BaseModel):
    results: list[JudgeFieldResult]


class GroundingJudgeLLMResponse(BaseModel):
    grounded: bool
    hallucinated: bool
    evidence_strength: Literal["none", "low", "medium", "high"]
    # Given ONLY the retrieved evidence, how does the claim relate to it?
    #   - supported: the evidence supports the claim
    #   - contradicted: the evidence directly contradicts the claim
    #   - absent: the evidence is relevant context but does not contain the
    #     claimed fact (the claim appears genuinely absent from the transcript)
    #   - no_evidence: no usable evidence was provided to evaluate against
    # This lets the harness separate genuine extraction errors (contradicted /
    # absent) from cases the judge simply could not retrieve evidence for
    # (no_evidence), which should not be counted against extraction correctness.
    claim_presence: Literal["supported", "contradicted", "absent", "no_evidence"] = "no_evidence"
    explanation: str = ""
    error_type: Optional[Literal[
        "unsupported_claim",
        "wrong_extraction",
        "missing_from_context",
        "overgeneralization",
        "ambiguous_evidence",
        "bad_search_terms",
        "formatting_only",
        "not_hallucinated_but_unverifiable",
        "other"
    ]] = None
    prompt_adjustment_suggestion: Optional[str] = None

    @field_validator("error_type", mode="before")
    @classmethod
    def _coerce_empty_error_type(cls, v):
        if v == "" or v is None:
            return None
        return v
