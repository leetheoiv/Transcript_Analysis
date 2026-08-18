from pydantic import BaseModel, Field
from typing import Any, Optional


class OrchestratorResult(BaseModel):
    generated_system_prompt: Optional[str] = None
    generated_user_prompt: Optional[str] = None
    generated_schema: Optional[Any] = None

    extraction_results_count: int = 0
    judge_results_count: int = 0

    evaluation_summary: dict = Field(default_factory=dict)
    final_status: str = "UNKNOWN"