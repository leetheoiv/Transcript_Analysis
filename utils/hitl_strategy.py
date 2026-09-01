"""
utils/hitl_strategy.py

Pluggable Human-in-the-Loop strategies.

THE PROBLEM:
────────────
Your Orchestrator needs human decisions at certain points (approve prompt,
approve schema). The original HITL() function uses input() — great for
notebooks/scripts, but impossible when the pipeline runs behind an API.

THE SOLUTION — Strategy Pattern:
────────────────────────────────
We define a common interface (Protocol) and two implementations:
  1. TerminalHITL — wraps your existing input()-based HITL function
  2. AsyncHITL   — pauses the thread, waits for an HTTP decision

The Orchestrator doesn't know or care which one it's using. It just calls
`self.hitl_strategy.request_decision(...)` and gets back a HITLDecision.

WHAT'S A PROTOCOL?
──────────────────
Think of it as a "shape contract." Any class that has a `request_decision`
method with the right signature satisfies the protocol — no inheritance needed.
It's Python's version of a TypeScript interface or a Go interface.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Protocol

from data_models.HITL_datamodel import HITLDecision

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The Protocol (interface) — any HITL strategy must look like this
# ---------------------------------------------------------------------------

class HITLStrategy(Protocol):
    """
    Protocol that defines what a HITL strategy must implement.

    Any class with a `request_decision` method matching this signature
    is a valid HITLStrategy — no need to inherit from this class.
    """

    def request_decision(
        self,
        item_for_review: Any,
        prompt: str,
        actions: dict[str, str],
        allow_notes: bool = True,
        default: str | None = None,
    ) -> HITLDecision:
        """
        Request a human decision. Blocks until a decision is available.

        Args:
            item_for_review: The artifact to review (dict, string, etc.)
            prompt: Message describing what's being reviewed.
            actions: Map of decision codes to labels, e.g. {"A": "Approve", ...}
            allow_notes: Whether the reviewer can attach notes.
            default: Default decision code if none provided.

        Returns:
            HITLDecision with the reviewer's choice and optional notes.
        """
        ...


# ---------------------------------------------------------------------------
# Implementation 1: TerminalHITL — wraps existing input()-based behavior
# ---------------------------------------------------------------------------

class TerminalHITL:
    """
    HITL strategy that uses terminal input().

    This is your original behavior — just wrapped in the strategy interface.
    Use this when running from a script, notebook, or Streamlit app where
    someone is sitting at the terminal.
    """

    def request_decision(
        self,
        item_for_review: Any,
        prompt: str,
        actions: dict[str, str],
        allow_notes: bool = True,
        default: str | None = None,
    ) -> HITLDecision:
        # Delegate to the existing HITL function
        from utils.human_in_the_loop import HITL

        return HITL(
            item_for_review=item_for_review,
            prompt=prompt,
            actions=actions,
            allow_notes=allow_notes,
            default=default,
        )


# ---------------------------------------------------------------------------
# Implementation 2: AsyncHITL — pauses the pipeline for an API decision
# ---------------------------------------------------------------------------

class AsyncHITL:
    """
    HITL strategy for API-driven workflows.

    HOW IT WORKS:
    ─────────────
    1. Pipeline thread calls request_decision() — this BLOCKS the thread.
    2. The review item is published to `pending_review` (readable by the API).
    3. A threading.Event is used as a "gate" — the thread sleeps on .wait().
    4. When the API receives a decision, it calls submit_decision() which
       stores the decision and opens the gate (.set()).
    5. The pipeline thread wakes up, reads the decision, and continues.

    This is a classic "producer-consumer" pattern using threading primitives.

    THREAD SAFETY:
    The _lock protects shared state. The Event handles cross-thread signaling.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._event = threading.Event()

        # Public state (read by the API to serve GET /runs/{id}/review)
        self.pending_review: dict | None = None
        self.pending_step: str | None = None
        self.pending_actions: dict[str, str] | None = None

        # Private state (the decision once submitted)
        self._decision: HITLDecision | None = None
        self._cancelled = False

    def request_decision(
        self,
        item_for_review: Any,
        prompt: str,
        actions: dict[str, str],
        allow_notes: bool = True,
        default: str | None = None,
    ) -> HITLDecision:
        """
        Pause the pipeline and wait for an external decision.

        This method BLOCKS until submit_decision() is called from another
        thread (typically the API request handler).
        """
        # Reset the gate (closed = thread will block on .wait())
        self._event.clear()
        self._decision = None
        self._cancelled = False

        # Publish the review item so the API can serve it
        with self._lock:
            self.pending_review = {
                "item_for_review": self._serialize_review_item(item_for_review),
                "prompt": prompt,
                "actions": actions,
                "allow_notes": allow_notes,
                "default": default,
            }
            self.pending_step = prompt  # Used as a step label
            self.pending_actions = actions

        logger.info("AsyncHITL: paused for review — %s", prompt)

        # BLOCK here until submit_decision() or cancel() is called
        self._event.wait()

        # Clear the published review
        with self._lock:
            self.pending_review = None
            self.pending_step = None
            self.pending_actions = None

        # Check for cancellation
        if self._cancelled:
            # Use importlib to avoid circular import through orchestration/__init__.py
            import importlib
            exc_module = importlib.import_module("orchestration.exceptions")
            raise exc_module.WorkflowCancelledError(
                step="HITL", notes="Cancelled via API while awaiting review"
            )

        # Return the decision
        decision = self._decision
        self._decision = None

        logger.info("AsyncHITL: resumed with decision=%s", decision.decision if decision else "None")
        return decision

    def submit_decision(self, decision: HITLDecision) -> None:
        """
        Submit a decision from the API side, unblocking the pipeline thread.

        Called by RunManager when POST /runs/{id}/decisions is received.
        """
        self._decision = decision
        self._event.set()  # Open the gate → pipeline thread wakes up

    def cancel(self) -> None:
        """Cancel the pending review, causing the pipeline to raise WorkflowCancelledError."""
        self._cancelled = True
        self._event.set()  # Wake up the thread so it can raise

    def has_pending_review(self) -> bool:
        """Check if there's a review waiting for a decision."""
        with self._lock:
            return self.pending_review is not None

    def get_pending_review(self) -> dict | None:
        """Get the current pending review item (thread-safe)."""
        with self._lock:
            return self.pending_review.copy() if self.pending_review else None

    @staticmethod
    def _serialize_review_item(item: Any) -> Any:
        """
        Make the review item JSON-serializable.

        The Orchestrator passes dicts, strings, Pydantic model *instances*
        (e.g. prompt artifacts) AND Pydantic model *classes* (the generated
        schema is a class, not an instance). All are handled here.
        """
        if isinstance(item, dict):
            # Recursively serialize each value the same way
            return {k: AsyncHITL._serialize_review_item(v) for k, v in item.items()}

        if isinstance(item, (list, tuple)):
            return [AsyncHITL._serialize_review_item(v) for v in item]

        if isinstance(item, (str, int, float, bool, type(None))):
            return item

        # A Pydantic model CLASS (e.g. the generated schema). Calling
        # model_dump() on a class fails with "missing 'self'", so use the
        # class-level JSON schema instead.
        if isinstance(item, type):
            if hasattr(item, "model_json_schema"):
                try:
                    return item.model_json_schema()
                except Exception:  # noqa: BLE001
                    return str(item)
            return str(item)

        # A Pydantic model INSTANCE.
        if hasattr(item, "model_dump") and not isinstance(item, type):
            try:
                return item.model_dump()
            except Exception:  # noqa: BLE001
                return str(item)

        return str(item)
