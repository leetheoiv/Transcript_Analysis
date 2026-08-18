from pydantic import BaseModel, Field
from typing import Optional

class HITLDecision(BaseModel):
    decision: str = Field(..., description="Human decision code, e.g. A, R, C")
    decision_label: str = Field(..., description="Human-readable label for the decision")
    notes: Optional[str] = Field(default="", description="Optional reviewer notes")
    item_reviewed: Optional[str] = Field(default="", description="String representation of the reviewed item")