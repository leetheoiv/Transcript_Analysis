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
