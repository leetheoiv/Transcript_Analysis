"""
models/feedback.py

Defines the three feedback / review data models:
- JudgeFeedback: per-row output from the JudgeAgent — a 0–1 score, free-text
  reasoning, and a list of typed issue flags.
- UserFeedback: per-row comment entered by a human analyst during the sample
  review step.
- RevisionBrief: aggregates all JudgeFeedback and UserFeedback from a rejected
  sample run into a single object that is handed back to the
  PromptGeneratorAgent to drive the next revision cycle.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JudgeFeedback(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_spec_id: UUID
    row_id: str
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    flags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


class UserFeedback(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_spec_id: UUID
    row_id: str
    comment: str
    score_override: Optional[float] = Field(None, ge=0.0, le=1.0)
    is_correct: bool = False
    created_at: datetime = Field(default_factory=_utcnow)


class RevisionBrief(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_spec_id: UUID                           # the rejected RunSpec version this brief is for
    judge_feedback: list[JudgeFeedback]
    user_feedback: list[UserFeedback]
    summary: Optional[str] = None               # optional human-readable summary of issues
    created_at: datetime = Field(default_factory=_utcnow)
