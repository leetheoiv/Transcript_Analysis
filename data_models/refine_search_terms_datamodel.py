from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

class RefinedSearchTermsResponse(BaseModel):
    refined_terms: list[str] = Field(
        default_factory=list,
        description="Short transcript-likely search phrases derived from quote and/or reasoning."
    )