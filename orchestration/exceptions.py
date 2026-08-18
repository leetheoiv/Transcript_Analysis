"""
orchestration/exceptions.py

Custom exception hierarchy for the Orchestrator pipeline.

All orchestrator exceptions inherit from OrchestratorError so callers
can catch broadly or narrowly as needed.
"""


class OrchestratorError(Exception):
    """Base exception for all orchestrator pipeline errors."""

    def __init__(self, message: str, step: str | None = None, context: dict | None = None):
        self.step = step
        self.context = context or {}
        detail = f"[{step}] {message}" if step else message
        super().__init__(detail)


class ConfigurationError(OrchestratorError):
    """Raised when the orchestrator is misconfigured (missing agents, bad inputs)."""
    pass


class PromptGenerationError(OrchestratorError):
    """Raised when prompt generation fails."""

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message, step="PromptGeneration", context=context)


class SchemaGenerationError(OrchestratorError):
    """Raised when schema generation fails."""

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message, step="SchemaGeneration", context=context)


class ExtractionError(OrchestratorError):
    """Raised when transcript extraction fails."""

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message, step="Extraction", context=context)


class JudgingError(OrchestratorError):
    """Raised when the judge agent fails."""

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message, step="Judging", context=context)


class EvaluationError(OrchestratorError):
    """Raised when evaluation fails."""

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message, step="Evaluation", context=context)


class WorkflowCancelledError(OrchestratorError):
    """Raised when a human reviewer cancels the workflow."""

    def __init__(self, step: str, notes: str | None = None):
        context = {"reviewer_notes": notes} if notes else {}
        super().__init__(f"Workflow cancelled by reviewer.", step=step, context=context)


class ReviewRetryError(OrchestratorError):
    """Raised when a human reviewer requests a retry with feedback."""

    def __init__(self, step: str, notes: str | None = None):
        context = {"reviewer_notes": notes} if notes else {}
        super().__init__(
            "Reviewer requested retry.",
            step=step,
            context=context,
        )
