"""
orchestration/orchestrator.py

Orchestrator pipeline for transcript analysis.

Coordinates prompt generation, schema generation, extraction, judging,
and evaluation — with structured error handling and logging at each step.
"""

import copy
import logging
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed


from agents.prompt_generator_agent import PromptGeneratorAgent
from agents.schema_generator_agent import SchemaGeneratorAgent
from agents.ExtractionAgent import TranscriptExtractionAgent
from agents.JudgeAgent import JudgeAgent
import pandas as pd
from utils.hitl_strategy import HITLStrategy, TerminalHITL
from utils.bootstrap_extraction import ExtractionBootstrapEvaluator
from tools.utils.load_schema_model import load_model_from_path
from data_models.prompt_datamodel import PromptManagement
from utils.save_file import save_file

from tools.EVAL_TOOLS.aggregate_judge_results import aggregate_judge_results
from tools.CATEGORY_TOOLS.category_registry import make_category_registry_tools
from data_models.judge_aggregation import JudgeAggregation

from orchestration.exceptions import (
    ConfigurationError,
    PromptGenerationError,
    SchemaGenerationError,
    ExtractionError,
    JudgingError,
    EvaluationError,
    WorkflowCancelledError,
    ReviewRetryError,
)

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    End-to-end pipeline orchestrator for transcript analysis.

    Runs through: prompt generation -> schema generation -> extraction
    -> judging -> evaluation, with optional human-in-the-loop reviews.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        prompt: str,
        knowledge_base_output_location: str | None = None,
        PromptAgent: PromptGeneratorAgent | None = None,
        SchemaAgent: SchemaGeneratorAgent | None = None,
        TranscriptExtractionAgent: TranscriptExtractionAgent | None = None,
        JudgeAgent: JudgeAgent | None = None,
        output_dir: str | None = None,
        project_name:str = None,
        hitl_strategy: HITLStrategy | None = None,
        use_category_registry: bool = False,
        transcript_column_name: str = "TRANSCRIPT",
        progress_callback=None,
        gold_labels: "pd.DataFrame | dict | None" = None,
        correctness_source: str = "judge",
    ):
        """Initialize the orchestrator with data, agents, and output configuration.

        Args:
            df: DataFrame containing transcripts and session identifiers.
            prompt: Plain-language analyst question describing what to extract.
            knowledge_base_output_location: Optional path to save knowledge base artifacts.
            PromptAgent: Optional PromptGeneratorAgent for prompt creation.
            SchemaAgent: Optional SchemaGeneratorAgent for Pydantic model generation.
            TranscriptExtractionAgent: Optional extraction agent for structured data extraction.
            JudgeAgent: Optional grounding judge agent for quality evaluation.
            output_dir: Base directory for all pipeline output files.
            project_name: Project identifier used in filenames and folder structure.
            hitl_strategy: Pluggable HITL strategy. Defaults to TerminalHITL (input()-based).
                Use AsyncHITL when running behind the API for HTTP-driven reviews.
            use_category_registry: If True, creates a category registry CSV in the output
                directory and provides lookup_category/register_category tools to the
                extraction agent. This ensures consistent categorization across transcripts.
            gold_labels: Optional ground-truth labels for measuring correctness
                INDEPENDENTLY of the judge (Option A). Accepts either:
                  - a DataFrame with a session-id column (AGENTRECORDINGSESSIONID)
                    plus one column per judged field (named exactly like the
                    extraction field, e.g. 'existing_customer_claim_field'), or
                  - a dict mapping session_id -> {field_name: expected_value}.
                When provided and correctness_source='gold', correctness is
                computed by comparing extraction output to these labels rather
                than the judge's grounding verdict.
            correctness_source: Which correctness signal drives PASS/FAIL and the
                revision loop: 'judge' (default, judge grounding) or 'gold'
                (ground-truth labels). 'gold' requires gold_labels.

        Raises:
            ConfigurationError: If df is not a valid non-empty DataFrame or prompt is empty.
        """
        # --- Input validation ---
        if df is None or not isinstance(df, pd.DataFrame):
            raise ConfigurationError(
                "A valid pandas DataFrame is required.",
                step="Init",
                context={"received_type": type(df).__name__},
            )

        if df.empty:
            raise ConfigurationError(
                "DataFrame is empty — nothing to process.",
                step="Init",
            )

        if not prompt or not isinstance(prompt, str):
            raise ConfigurationError(
                "A non-empty prompt string is required.",
                step="Init",
                context={"received_type": type(prompt).__name__},
            )
        self.df = df
        self.user_prompt = prompt
        self.project_name = project_name

        # Resolve the project output folder robustly, regardless of the current
        # working directory or redundant leading path segments.
        #
        # Design: `output_dir` is the PARENT; `project_name` is a subfolder
        # created inside it (output_dir/project_name).
        #
        # Two failure modes are handled here:
        #   1. RELATIVE output_dir — resolved against the repository root (this
        #      file lives at <root>/orchestration/orchestrator.py), NOT the
        #      process CWD, so results always land in a predictable place.
        #   2. output_dir already POINTS AT the project folder (its last segment
        #      equals project_name) — used as-is instead of nesting a second
        #      time (avoids output_dir/project_name/project_name).
        from pathlib import Path as _Path

        _project_root = _Path(__file__).resolve().parent.parent  # repo root
        _base = _Path(output_dir)

        if not _base.is_absolute():
            # Drop a redundant leading segment that repeats the repo folder name
            # (e.g. output_dir="Transcript_Analysis_Automation/JITS/..." while
            # the repo root IS .../Transcript_Analysis_Automation). Without this
            # the join would double the repo name.
            parts = _base.parts
            if parts and parts[0] == _project_root.name:
                parts = parts[1:]
            _base = _project_root.joinpath(*parts) if parts else _project_root

        if _base.name == project_name:
            # Caller pointed us straight at the project folder — use it.
            _project_folder = _base
        else:
            _project_folder = _base / project_name

        _project_folder = _project_folder.resolve()
        _project_folder.mkdir(parents=True, exist_ok=True)
        self.output_dir = str(_project_folder)

        self.generated_system_prompt: str | None = None
        self.generated_user_prompt: str | None = None
        self.generated_schema: Any = None
        self.generated_prompt_output_format: str | None = None

        self.knowledge_base_output_location = knowledge_base_output_location

        self.prompt_agent = PromptAgent
        self.schema_agent = SchemaAgent
        self.extraction_agent = TranscriptExtractionAgent
        self.judge_agent = JudgeAgent
        self.hitl_strategy = hitl_strategy or TerminalHITL()

        # Gold-label correctness (Option A): independent ground truth.
        self.correctness_source = (correctness_source or "judge").lower().strip()
        if self.correctness_source not in ("judge", "gold"):
            raise ConfigurationError(
                f"correctness_source must be 'judge' or 'gold', got '{correctness_source}'.",
                step="Init",
            )
        self.gold_labels = self._normalize_gold_labels(gold_labels)
        if self.correctness_source == "gold" and not self.gold_labels:
            raise ConfigurationError(
                "correctness_source='gold' requires non-empty gold_labels.",
                step="Init",
            )

        # Optional progress reporter: callable(phase_key: str, detail: str = "").
        # Used to surface which pipeline phase is running (e.g. in a GUI).
        self._progress_callback = progress_callback
        self.current_phase: str = "pending"
        

        self.prompt_result = None
        self.schema_result = None
        self.schema_path: str | None = None
        self.extraction_result = None
        self.judge_result = None
        self.judge_aggregation: JudgeAggregation | None = None
        self.evaluation_result = None
        self.repeated_runs_df = None
        self.final_status: str = "UNKNOWN"
        self.revision_stopped_reason: str | None = None
        # The evaluator uppercases all column names, so normalize the transcript
        # column name to uppercase for consistent matching/lookups downstream.
        self.transcript_column_name = (transcript_column_name or "TRANSCRIPT").upper()
        self.prompt_management: PromptManagement | None = None
        
        # Category registry for controlled vocabulary during extraction
        self.use_category_registry = use_category_registry
        self.category_registry_path: str | None = None
        self.category_tools: tuple | None = None
        
        if use_category_registry:
            self.category_registry_path = fr'{self.output_dir}\{self.project_name}_categories.csv'
            self.category_tools = make_category_registry_tools(self.category_registry_path)
            logger.info(
                "Category registry enabled — file: %s", self.category_registry_path
            )
   

        

        logger.info(
            "Orchestrator initialized — %d rows, prompt_agent=%s, schema_agent=%s, "
            "extraction_agent=%s, judge_agent=%s",
            len(df),
            type(PromptAgent).__name__ if PromptAgent else "None",
            type(SchemaAgent).__name__ if SchemaAgent else "None",
            type(TranscriptExtractionAgent).__name__ if TranscriptExtractionAgent else "None",
            type(JudgeAgent).__name__ if JudgeAgent else "None",
        )

    # ------------------------------------------------------------------
    # Progress reporting
    # ------------------------------------------------------------------

    def _report_phase(self, phase: str, detail: str = "") -> None:
        """Record the current pipeline phase and notify any progress callback.

        `phase` is a short machine-friendly key (e.g. 'prompt_generation').
        Never lets a callback error break the pipeline.
        """
        self.current_phase = phase
        logger.info("PHASE: %s%s", phase, f" — {detail}" if detail else "")
        if self._progress_callback is not None:
            try:
                self._progress_callback(phase, detail)
            except Exception as exc:  # noqa: BLE001
                logger.warning("progress_callback raised (ignored): %s", exc)

    # ------------------------------------------------------------------
    # Human-in-the-loop helpers
    # ------------------------------------------------------------------

    def _hitl(
        self,
        item_for_review,
        prompt: str,
        actions: dict[str, str],
        allow_notes: bool = True,
        default: str | None = None,
    ):
        """Pause workflow and request human approval/input via the pluggable strategy.

        When running from a script/notebook, this uses TerminalHITL (blocks on input()).
        When running behind the API, this uses AsyncHITL (blocks on threading.Event,
        resumes when the API receives a decision).
        """
        return self.hitl_strategy.request_decision(
            item_for_review=item_for_review,
            prompt=prompt,
            actions=actions,
            allow_notes=allow_notes,
            default=default,
        )

    def _review_prompt(self):
        """Present generated prompt artifacts to the human reviewer for approval.

        Returns:
            HITL decision object with decision ('A', 'R', or 'C') and optional notes.
        """
        review_item = {
            "generated_system_prompt": self.generated_system_prompt,
            "generated_user_prompt": self.generated_user_prompt,
            "generated_output_format": self.generated_prompt_output_format,
        }

        return self._hitl(
            item_for_review=review_item,
            prompt="Review generated prompt(s).",
            actions={"A": "Approve", "R": "Retry", "C": "Cancel"},
            allow_notes=True,
            default="A",
        )

    def _review_schema(self):
        """Present the generated schema to the human reviewer for approval.

        Returns:
            HITL decision object with decision ('A', 'R', or 'C') and optional notes.
        """
        return self._hitl(
            item_for_review=self.generated_schema,
            prompt="Review generated schema.",
            actions={"A": "Approve", "R": "Retry", "C": "Cancel"},
            allow_notes=True,
            default="A",
        )

    def _handle_review_decision(self, decision, step_name: str):
        """Interpret a HITL decision and raise appropriate exceptions for cancel/retry.

        Returns silently on approval.

        Args:
            decision: HITL decision object with .decision and .notes attributes.
            step_name: Name of the pipeline step for exception context.

        Raises:
            WorkflowCancelledError: If decision is 'C' (cancel).
            ReviewRetryError: If decision is 'R' (retry).
        """
        if decision.decision == "C":
            raise WorkflowCancelledError(step=step_name, notes=decision.notes)
        if decision.decision == "R":
            raise ReviewRetryError(step=step_name, notes=decision.notes)

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _save_prompt_to_file(self):
        """Save the current generated prompts to a markdown file in the output directory."""
        version = self.prompt_management.version if self.prompt_management else 1
        filename = f"{self.project_name}_prompt_v{version}"

        content = f"# {self.project_name} — Prompt (v{version})\n\n"
        content += "## System Prompt\n\n"
        content += f"```\n{self.generated_system_prompt or ''}\n```\n\n"
        content += "## User Prompt\n\n"
        content += f"```\n{self.generated_user_prompt or ''}\n```\n\n"

        if self.generated_prompt_output_format:
            content += "## Output Format\n\n"
            content += f"```json\n{self.generated_prompt_output_format}\n```\n"

        path = save_file(
            content=content,
            filename=filename,
            output_dir=self.output_dir,
            extension=".md",
            overwrite=True,
        )

        # Update saved location on prompt_management
        if self.prompt_management:
            self.prompt_management.saved_location_of_prompt = str(path)

        logger.info("Prompt saved to %s", path)

    def _generate_prompt(self):
        """Generate prompt artifacts using the prompt agent, or fall back to user prompt."""
        logger.info("Step: Prompt Generation — starting")

        if self.prompt_agent is None:
            logger.info("No prompt agent configured; using raw user prompt.")
            self.generated_user_prompt = self.user_prompt
            return self.generated_user_prompt

        try:
            prompt_result = self.prompt_agent.run(self.user_prompt, force_generate=True)
        except Exception as exc:
            raise PromptGenerationError(
                f"Prompt agent failed: {exc}",
                context={"original_error": str(exc), "user_prompt_preview": self.user_prompt[:200]},
            ) from exc

        if prompt_result is None:
            raise PromptGenerationError(
                "Prompt agent returned None — expected a result with system_prompt, "
                "user_prompt, and output_format attributes."
            )

        self.prompt_result = prompt_result

        # Validate expected attributes exist on the result
        for attr in ("system_prompt", "user_prompt", "output_format"):
            if not hasattr(prompt_result, attr):
                raise PromptGenerationError(
                    f"Prompt result is missing required attribute '{attr}'.",
                    context={"result_type": type(prompt_result).__name__},
                )

        self.generated_system_prompt = prompt_result.system_prompt
        self.generated_user_prompt = prompt_result.user_prompt
        self.generated_prompt_output_format = prompt_result.output_format

        # Create PromptManagement record with prompt metadata
        self.prompt_management = PromptManagement(
            prompt_title=self.project_name or "untitled",
            inital_user_prompt_request=self.user_prompt,
            generated_system_prompt=prompt_result.system_prompt,
            generated_user_prompt=prompt_result.user_prompt,
            output_format=prompt_result.output_format,
            metadata_fields=getattr(prompt_result, "metadata_fields", []),
            saved_location_of_prompt=getattr(prompt_result, "saved_location_of_prompt", None),
            model=self.prompt_agent.model if self.prompt_agent else "",
            temperature=float(self.prompt_agent.temperature) if self.prompt_agent else None,
        )

        logger.info("Prompt generation complete.")
        logger.debug("System prompt length: %d chars", len(self.generated_system_prompt or ""))
        logger.debug("User prompt length: %d chars", len(self.generated_user_prompt or ""))


        return prompt_result

    def _regenerate_prompt(self, feedback: str | None = None, revision_brief: str | None = None):
        """
        Re-run prompt generation incorporating revision feedback.

        Supports two modes:
        - feedback (human notes): passed as user_input to the agent, which uses
          conversation history to understand what to revise.
        - revision_brief (structured judge aggregation): passed via the
          revision_brief parameter so the prompt template injects it as
          "A previous version of this prompt was rejected. Here is the feedback: ..."

        When revision_brief is provided, the original user prompt is re-sent as
        user_input so the agent regenerates the extraction prompt (not a prompt
        about the feedback itself).
        """
        logger.info("Step: Prompt Re-generation — starting")

        if self.prompt_agent is None:
            raise PromptGenerationError(
                "Cannot retry prompt generation without a prompt agent."
            )

        if revision_brief:
            # Structured judge feedback — re-generate the original extraction prompt
            # with the revision brief injected into the template
            logger.info("Using revision_brief (%d chars) for prompt revision.", len(revision_brief))
            try:
                prompt_result = self.prompt_agent.run(
                    self.user_prompt,
                    revision_brief=revision_brief,
                    force_generate=True,
                )
            except Exception as exc:
                raise PromptGenerationError(
                    f"Prompt agent failed during revision: {exc}",
                    context={"original_error": str(exc)},
                ) from exc
        else:
            # Human feedback — pass directly as conversational input
            revision_input = feedback or "Please revise the prompt based on reviewer feedback."
            logger.info("Using human feedback for prompt revision: %s", revision_input[:100])
            try:
                prompt_result = self.prompt_agent.run(revision_input, force_generate=True)
            except Exception as exc:
                raise PromptGenerationError(
                    f"Prompt agent failed during retry: {exc}",
                    context={"original_error": str(exc), "feedback": feedback},
                ) from exc

        if prompt_result is None or isinstance(prompt_result, str):
            raise PromptGenerationError(
                "Prompt agent returned a non-structured response during retry.",
                context={"response_preview": str(prompt_result)[:200] if prompt_result else "None"},
            )

        self.prompt_result = prompt_result
        self.generated_system_prompt = prompt_result.system_prompt
        self.generated_user_prompt = prompt_result.user_prompt
        self.generated_prompt_output_format = prompt_result.output_format

        # Update PromptManagement record
        if self.prompt_management:
            from datetime import datetime, timezone
            self.prompt_management.generated_system_prompt = prompt_result.system_prompt
            self.prompt_management.generated_user_prompt = prompt_result.user_prompt
            self.prompt_management.output_format = prompt_result.output_format
            self.prompt_management.metadata_fields = getattr(prompt_result, "metadata_fields", [])
            self.prompt_management.saved_location_of_prompt = getattr(prompt_result, "saved_location_of_prompt", None)
            self.prompt_management.version += 1
            self.prompt_management.updated_at = datetime.now(timezone.utc)

        logger.info("Prompt re-generation complete (version %d).",
                    self.prompt_management.version if self.prompt_management else -1)


        return prompt_result

    def _generate_schema(self):
        """Generate schema using the schema agent."""
        logger.info("Step: Schema Generation — starting")

        if self.schema_agent is None:
            logger.info("No schema agent configured; skipping schema generation.")
            return None

        if not self.generated_prompt_output_format and not self.generated_user_prompt:
            raise SchemaGenerationError(
                "Cannot generate schema without a prompt output format or user prompt. "
                "Run prompt generation first."
            )

        try:
            schema_result, schema_path = self.schema_agent.run(
                output_format=self.generated_prompt_output_format,
                user_input=self.generated_user_prompt,
                output_dir=self.output_dir,
            )
        except Exception as exc:
            raise SchemaGenerationError(
                f"Schema agent failed: {exc}",
                context={"original_error": str(exc)},
            ) from exc

        if schema_path is None:
            raise SchemaGenerationError(
                "Schema agent returned a None path — cannot load model."
            )

        self.schema_path = schema_path
        self.schema_result = schema_result

        try:
            self.generated_schema = load_model_from_path(schema_path, schema_result.model_name)
        except Exception as exc:
            raise SchemaGenerationError(
                f"Failed to load schema model from path '{schema_path}': {exc}",
                context={"schema_path": schema_path, "original_error": str(exc)},
            ) from exc

        logger.info("Schema generation complete — loaded from %s", schema_path)

        # Update PromptManagement with schema metadata
        if self.prompt_management:
            from datetime import datetime, timezone
            self.prompt_management.schema_definition = (
                self.generated_schema.model_json_schema()
                if hasattr(self.generated_schema, "model_json_schema")
                else {}
            )
            self.prompt_management.schema_model_name = schema_result.model_name
            self.prompt_management.schema_path = str(schema_path)
            self.prompt_management.updated_at = datetime.now(timezone.utc)

        return schema_result

    def _regenerate_schema(self, feedback: str | None = None):
        """
        Re-run schema generation using human feedback.

        Includes the previously generated schema code in the input so the agent
        knows what to revise.
        """
        logger.info("Step: Schema Re-generation — starting (feedback: %s)", feedback or "<none>")

        if self.schema_agent is None:
            raise SchemaGenerationError(
                "Cannot retry schema generation without a schema agent."
            )

        if not self.generated_prompt_output_format and not self.generated_user_prompt:
            raise SchemaGenerationError(
                "Cannot regenerate schema without a prompt output format or user prompt."
            )

        # Build revision input with prior schema + feedback
        revision_input = self.generated_user_prompt or ""
        if self.schema_result and hasattr(self.schema_result, "code"):
            revision_input += (
                f"\n\n[PREVIOUSLY GENERATED SCHEMA]:\n```python\n{self.schema_result.code}\n```"
            )
        if feedback:
            revision_input += f"\n\n[REVIEWER FEEDBACK — apply these changes]: {feedback}"

        try:
            schema_result, schema_path = self.schema_agent.run(
                output_format=self.generated_prompt_output_format,
                user_input=revision_input,
                output_dir=self.output_dir,
            )
        except Exception as exc:
            raise SchemaGenerationError(
                f"Schema agent failed during retry: {exc}",
                context={"original_error": str(exc), "feedback": feedback},
            ) from exc

        if schema_path is None:
            raise SchemaGenerationError(
                "Schema agent returned a None path during retry — cannot load model."
            )

        self.schema_path = schema_path
        self.schema_result = schema_result

        try:
            self.generated_schema = load_model_from_path(schema_path, schema_result.model_name)
        except Exception as exc:
            raise SchemaGenerationError(
                f"Failed to load schema model from path '{schema_path}': {exc}",
                context={"schema_path": schema_path, "original_error": str(exc)},
            ) from exc

        # Update PromptManagement with new schema metadata
        if self.prompt_management:
            from datetime import datetime, timezone
            self.prompt_management.schema_definition = (
                self.generated_schema.model_json_schema()
                if hasattr(self.generated_schema, "model_json_schema")
                else {}
            )
            self.prompt_management.schema_model_name = schema_result.model_name
            self.prompt_management.schema_path = str(schema_path)
            self.prompt_management.updated_at = datetime.now(timezone.utc)

        logger.info("Schema re-generation complete — loaded from %s", schema_path)
        return schema_result

    def _run_judging(self, **judge_kwargs):
        """Run judge agent on extraction output."""
        logger.info("Step: Judging — starting")


        if self.judge_agent is None:
            logger.info("No judge agent configured; skipping judging.")
            return None

        if self.extraction_result is None:
            raise JudgingError(
                "Cannot run judging without extraction results. "
                "Run extraction before judging."
            )
        
        field_groups = {}

        for col in self.extraction_result.columns:
            # Group columns by their claim-field prefix. The naming convention is:
            #   {prefix}_claim_field            -> the claim value (the judged field)
            #   {prefix}_claim_field_quote      -> extractor quote for that claim
            #   {prefix}_claim_field_reasoning  -> extractor reasoning for that claim
            # We normalize every related column back to the SAME "{prefix}_claim_field"
            # key so the quote/reasoning columns group with their claim rather than
            # producing a separate, claim-less group with an empty field_name.
            if col.endswith('_claim_field_quote'):
                base = col[: -len('_quote')]  # -> {prefix}_claim_field
                field_groups.setdefault(base, {})['quote_column'] = col
            elif col.endswith('_claim_field_reasoning'):
                base = col[: -len('_reasoning')]  # -> {prefix}_claim_field
                field_groups.setdefault(base, {})['reasoning_column'] = col
            elif col.endswith('_claim_field'):
                field_groups.setdefault(col, {})['claim_column'] = col

        judge_config = []
        for prefix, group in field_groups.items():
            claim_column = group.get('claim_column')

            # Skip phantom groups that have quote/reasoning columns but no actual
            # claim column. Judging a field with no claim value produces junk
            # (empty field_name columns) and pollutes the aggregation metrics.
            if not claim_column:
                logger.warning(
                    "Skipping judge field group '%s' — no claim column found "
                    "(quote/reasoning present without a matching *_claim_field).",
                    prefix,
                )
                continue

            judge_config.append({
                'field_name': claim_column,
                'claim_column': claim_column,
                'quote_column': group.get('quote_column', ''),
                'reasoning_column': group.get('reasoning_column', ''),
                'task_prompt': self.generated_system_prompt + '\n' + self.generated_user_prompt
            })

        file_name = fr'{self.project_name }_judge_results.csv'
        judge_results_file_path =  fr'{self.output_dir}\{file_name}'

        # Merge transcript column from original df into extraction results
        # so the judge has access to the actual transcript text for evidence retrieval.
        judge_df = self.extraction_result.copy()
        if self.transcript_column_name not in judge_df.columns:
            # self.transcript_column_name is uppercased; look it up against an
            # uppercased view of the source df so casing differences don't break.
            source_df = self.df.rename(columns=str.upper)
            transcript_lookup = source_df.set_index("AGENTRECORDINGSESSIONID")[self.transcript_column_name]
            judge_df[self.transcript_column_name] = (
                judge_df["AGENTRECORDINGSESSIONID"].map(transcript_lookup)
            )

        try:
            judge_result = self.judge_agent.process_batch(
                rows=judge_df.to_dict('records'),
                transcript_column_name=self.transcript_column_name,
                judge_config=judge_config,
                output_file=judge_results_file_path,
                **judge_kwargs,
            )
        except Exception as exc:
            raise JudgingError(
                f"Judge agent failed during process_batch: {exc}",
                context={"original_error": str(exc)},
            ) from exc

        if judge_result is None:
            raise JudgingError("Judge agent returned None — expected results.")

        self.judge_result = pd.DataFrame(judge_result)
        logger.info("Judging complete.")
        return judge_result

    def _aggregate_judge_feedback(self) -> JudgeAggregation | None:
        """Aggregate judge results into a structured summary for prompt revision.

        Processes self.judge_result (DataFrame) through the aggregation utility
        and stores the result in self.judge_aggregation.

        Returns:
            JudgeAggregation instance, or None if no judge results are available.
        """
        if self.judge_result is None or self.judge_result.empty:
            logger.info("No judge results to aggregate.")
            return None

        self.judge_aggregation = aggregate_judge_results(self.judge_result)

        logger.info(
            "Judge aggregation complete — %d judgments, overall grounded=%.1f%%, "
            "hallucination=%.1f%%, failure_rate=%.1f%%, %d prompt changes suggested.",
            self.judge_aggregation.total_field_judgments,
            self.judge_aggregation.overall_grounded_rate * 100,
            self.judge_aggregation.overall_hallucination_rate * 100,
            self.judge_aggregation.overall_failure_rate * 100,
            len(self.judge_aggregation.prompt_lessons.prompt_changes),
        )

        return self.judge_aggregation

    def _init_evaluator(
        self,
        model: str = "gpt-4.1-mini",
        temperature: str = "0.1",
    ):
        """Initialize the ExtractionBootstrapEvaluator with shared config."""
        if self.extraction_agent is None:
            raise ConfigurationError(
                "Extraction agent is required but was not provided.",
                step="BootstrapExtraction",
            )

        if self.generated_user_prompt is None:
            raise ExtractionError(
                "No user prompt available. Run prompt generation before extraction."
            )

        # Columns are uppercased for the evaluator rows, so compare
        # case-insensitively against the source DataFrame.
        upper_columns = {str(c).upper() for c in self.df.columns}
        if self.transcript_column_name not in upper_columns:
            raise ExtractionError(
                f"Column '{self.transcript_column_name}' not found in DataFrame.",
                context={"available_columns": list(self.df.columns)},
            )

        # Pass category tools as extra_tools so each per-row agent gets them
        agent_init_kwargs = {}
        if self.category_tools:
            agent_init_kwargs["extra_tools"] = list(self.category_tools)

        self.evaluator = ExtractionBootstrapEvaluator(
            rows=self.df.rename(columns=str.upper).iloc[:, :].to_dict("records"),
            agent_cls=TranscriptExtractionAgent,
            model=model,
            temperature=temperature,
            transcript_column_name=self.transcript_column_name,
            **agent_init_kwargs,
        )

    def _run_first_extraction(
        self,
        consistency_sample_size: int | None = None,
        consistency_sample_with_replacement: bool = False,
        random_seed: int | None = None,
        **process_batch_kwargs,
    ) -> pd.DataFrame:
        """
        Run only the first extraction pass and store as extraction_result.

        This completes quickly so that the judge can begin while remaining
        consistency runs continue in a separate thread.
        """
        logger.info("Step: First Extraction Run — starting (%d rows)", len(self.df))

        try:
            first_run_df = self.evaluator._run_repeated_extractions(
                n_runs=1,
                consistency_sample_size=consistency_sample_size,
                consistency_sample_with_replacement=consistency_sample_with_replacement,
                random_seed=random_seed,
                prompt_template=self.generated_user_prompt,
                response_format=self.generated_schema,
                system_prompt=self.generated_system_prompt,
                **process_batch_kwargs,
            )
        except Exception as exc:
            raise ExtractionError(
                f"First extraction run failed: {exc}",
                context={"original_error": str(exc)},
            ) from exc

        # Store as extraction_result (used by judging)
        first_run = first_run_df.copy()
        first_run_clean = first_run.drop(columns=["RUN_ID", "CONSISTENCY_SAMPLE_SIZE"], errors="ignore")
        self.extraction_result = first_run_clean.reset_index(drop=True)

        # Save extraction results to file
        file_name = f"{self.project_name}_extraction_results.csv"
        self.extraction_results_path = f"{self.output_dir}\\{file_name}"
        self.extraction_result.to_csv(self.extraction_results_path, index=False)

        logger.info(
            "First extraction run complete — %d rows stored as extraction_result.",
            len(self.extraction_result),
        )
        return first_run_df

    def _run_remaining_extractions(
        self,
        n_remaining_runs: int,
        first_run_df: pd.DataFrame,
        consistency_sample_size: int | None = None,
        consistency_sample_with_replacement: bool = False,
        random_seed: int | None = None,
        **process_batch_kwargs,
    ) -> pd.DataFrame:
        """
        Run extraction passes 2..N and combine with the first run into
        self.repeated_runs_df. Designed to run in a background thread
        concurrently with judging.
        """
        if n_remaining_runs <= 0:
            self.repeated_runs_df = first_run_df
            return first_run_df

        logger.info(
            "Step: Remaining Extraction Runs — starting %d additional runs",
            n_remaining_runs,
        )

        try:
            remaining_df = self.evaluator._run_repeated_extractions(
                n_runs=n_remaining_runs,
                consistency_sample_size=consistency_sample_size,
                consistency_sample_with_replacement=consistency_sample_with_replacement,
                random_seed=random_seed,
                prompt_template=self.generated_user_prompt,
                response_format=self.generated_schema,
                system_prompt=self.generated_system_prompt,
                **process_batch_kwargs,
            )
        except Exception as exc:
            raise ExtractionError(
                f"Remaining extraction runs failed: {exc}",
                context={"original_error": str(exc), "n_remaining_runs": n_remaining_runs},
            ) from exc

        # Renumber remaining runs to start at 2
        remaining_df = remaining_df.copy()
        remaining_df["RUN_ID"] = remaining_df["RUN_ID"] + 1

        # Combine with first run
        self.repeated_runs_df = pd.concat(
            [first_run_df, remaining_df], ignore_index=True
        )

        logger.info(
            "Remaining extraction runs complete — %d total results across %d runs.",
            len(self.repeated_runs_df), n_remaining_runs + 1,
        )
        return self.repeated_runs_df

    def _run_bootstrap_extraction(
        self,
        n_runs: int = 3,
        model: str = "gpt-4.1-mini",
        temperature: str = "0.1",
        consistency_sample_size: int | None = None,
        consistency_sample_with_replacement: bool = False,
        random_seed: int | None = None,
        **process_batch_kwargs,
    ):
        """
        Run the ExtractionBootstrapEvaluator for N runs (sequential fallback).

        The first run's results become self.extraction_result (used by the judge).
        All runs are stored in self.repeated_runs_df for consistency evaluation.
        """
        logger.info(
            "Step: Bootstrap Extraction — starting (n_runs=%d, %d rows)",
            n_runs, len(self.df),
        )

        self._init_evaluator(model=model, temperature=temperature)

        try:
            repeated_runs_df = self.evaluator._run_repeated_extractions(
                n_runs=n_runs,
                consistency_sample_size=consistency_sample_size,
                consistency_sample_with_replacement=consistency_sample_with_replacement,
                random_seed=random_seed,
                prompt_template=self.generated_user_prompt,
                response_format=self.generated_schema,
                system_prompt=self.generated_system_prompt,
                **process_batch_kwargs,
            )
        except Exception as exc:
            raise ExtractionError(
                f"Bootstrap extraction failed: {exc}",
                context={"original_error": str(exc), "n_runs": n_runs},
            ) from exc

        self.repeated_runs_df = repeated_runs_df

        # Use the first run as the "initial" extraction result for judging
        first_run = repeated_runs_df[repeated_runs_df["RUN_ID"] == 1].copy()
        first_run = first_run.drop(columns=["RUN_ID", "CONSISTENCY_SAMPLE_SIZE"], errors="ignore")
        self.extraction_result = first_run.reset_index(drop=True)

        # Save extraction results to file
        file_name = f"{self.project_name}_extraction_results.csv"
        self.extraction_results_path = f"{self.output_dir}\\{file_name}"
        self.extraction_result.to_csv(self.extraction_results_path, index=False)

        logger.info(
            "Bootstrap extraction complete — %d total results across %d runs, "
            "first run (%d rows) stored as extraction_result.",
            len(repeated_runs_df), n_runs, len(self.extraction_result),
        )
        return self.extraction_result

    # ------------------------------------------------------------------
    # Gold-label correctness (Option A)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_gold_value(value) -> str:
        """Normalize a claim/label value for robust equality comparison.

        Lowercases, strips, and renders lists in a canonical order so that
        ["A","B"] and ["B","A"] compare equal. None/NaN become "".
        """
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        if isinstance(value, (list, tuple, set)):
            parts = sorted(str(v).strip().lower() for v in value if str(v).strip())
            return "|".join(parts)
        return str(value).strip().lower()

    def _normalize_gold_labels(self, gold_labels) -> dict:
        """Coerce gold_labels (DataFrame or dict) into a canonical dict:

            { session_id(str) -> { field_name(str) -> expected_value } }

        Returns an empty dict when no labels are supplied.
        """
        if gold_labels is None:
            return {}

        session_col = "AGENTRECORDINGSESSIONID"

        # DataFrame form: one row per session, one column per field.
        if isinstance(gold_labels, pd.DataFrame):
            if gold_labels.empty:
                return {}
            if session_col not in gold_labels.columns:
                raise ConfigurationError(
                    f"gold_labels DataFrame must contain a '{session_col}' column.",
                    step="Init",
                )
            normalized: dict = {}
            for _, row in gold_labels.iterrows():
                sid = str(row[session_col])
                field_map = {
                    col: row[col]
                    for col in gold_labels.columns
                    if col != session_col
                }
                normalized[sid] = field_map
            return normalized

        # Dict form: session_id -> {field: value}
        if isinstance(gold_labels, dict):
            return {str(sid): dict(fields) for sid, fields in gold_labels.items()}

        raise ConfigurationError(
            "gold_labels must be a pandas DataFrame or a dict "
            "(session_id -> {field: expected_value}).",
            step="Init",
        )

    def _evaluate_gold_correctness(self) -> tuple[float, dict, int, int]:
        """Compute correctness by comparing extraction output to gold labels.

        This is INDEPENDENT of the judge. For every (session, field) pair that
        exists in both the extraction result and the gold labels, the extracted
        claim value is compared to the expected label using normalized equality.

        Returns:
            (overall_correctness_rate, per_field_correctness_rate,
             correct_count, total_compared)
        """
        session_col = "AGENTRECORDINGSESSIONID"

        if (
            self.extraction_result is None
            or self.extraction_result.empty
            or not self.gold_labels
            or session_col not in self.extraction_result.columns
        ):
            logger.warning(
                "Gold correctness requested but extraction_result or gold_labels "
                "are missing/incompatible — returning 0 correctness."
            )
            return 0.0, {}, 0, 0

        # Judged fields are the extraction columns ending in _claim_field
        field_names = [
            c for c in self.extraction_result.columns if c.endswith("_claim_field")
        ]

        per_field_correct: dict[str, int] = {f: 0 for f in field_names}
        per_field_total: dict[str, int] = {f: 0 for f in field_names}

        for _, row in self.extraction_result.iterrows():
            sid = str(row.get(session_col))
            gold_for_session = self.gold_labels.get(sid)
            if not gold_for_session:
                continue

            for field in field_names:
                if field not in gold_for_session:
                    continue
                expected = self._normalize_gold_value(gold_for_session[field])
                actual = self._normalize_gold_value(row.get(field))
                per_field_total[field] += 1
                if expected == actual:
                    per_field_correct[field] += 1

        per_field_rate = {
            f: (per_field_correct[f] / per_field_total[f])
            if per_field_total[f] > 0 else 0.0
            for f in field_names
        }

        total_correct = sum(per_field_correct.values())
        total_compared = sum(per_field_total.values())
        overall_rate = total_correct / total_compared if total_compared > 0 else 0.0

        logger.info(
            "Gold correctness: %.2f%% (%d/%d comparisons across %d fields).",
            overall_rate * 100, total_correct, total_compared, len(field_names),
        )
        return overall_rate, per_field_rate, total_correct, total_compared

    def _finalize_evaluation(self):
        """
        Combine consistency metrics (from bootstrap extraction) with semantic
        quality metrics (from judging) into a final EvaluationResult.
        """
        logger.info("Step: Finalize Evaluation — combining consistency + semantic quality")

        from data_models.prompt_eval_datamodel import EvaluationResult, ConsistencyQuality, SemanticQuality
        from datetime import datetime

        # Compute consistency from repeated runs
        consistency_quality = ConsistencyQuality()
        per_field_consistency_rate = {}

        if self.repeated_runs_df is not None and not self.repeated_runs_df.empty:
            # Derive eval fields from extraction result columns with _claim_field suffix
            self.eval_fields = [
                col for col in self.repeated_runs_df.columns
                if col.endswith("_claim_field")
            ]
            consistency_quality, per_field_consistency_rate = (
                self.evaluator._evaluate_consistency(
                    df=self.repeated_runs_df,
                    fields= self.eval_fields  if  self.eval_fields  else None,
                )
            )

        # Compute semantic quality from judge results
        semantic_quality = SemanticQuality(
            consistency_rate=consistency_quality.consistency_rate,
            consistency_count=consistency_quality.consistency_count,
            total_samples=len(self.judge_result) if self.judge_result is not None and not self.judge_result.empty else len(self.df),
            total_evaluated=consistency_quality.total_evaluated,
        )
        per_field_correctness_rate = {}
        per_field_hallucination_rate = {}

        if self.judge_result is not None and not self.judge_result.empty:
            semantic_eval, per_field_correctness_rate, per_field_hallucination_rate = (
                self.evaluator._evaluate_semantic_quality(self.judge_result)
            )
            semantic_eval.consistency_rate = consistency_quality.consistency_rate
            semantic_eval.consistency_count = consistency_quality.consistency_count
            semantic_quality = semantic_eval

        # Option A: when configured, replace the judge-derived correctness with
        # gold-label correctness. Hallucination and retrieval-failure metrics from
        # the judge are preserved (they still describe judge behavior), but the
        # correctness that drives PASS/FAIL now comes from ground truth.
        if self.correctness_source == "gold" and self.gold_labels:
            gold_rate, gold_per_field, gold_correct, gold_total = (
                self._evaluate_gold_correctness()
            )
            semantic_quality.correctness_rate = gold_rate
            semantic_quality.correctness_count = gold_correct
            semantic_quality.total_evaluated = gold_total
            semantic_quality.correctness_source = "gold"
            per_field_correctness_rate = gold_per_field
            logger.info(
                "Correctness source = GOLD labels (overall %.2f%%).",
                gold_rate * 100,
            )
        else:
            semantic_quality.correctness_source = "judge"

        evaluation_result = EvaluationResult(
            date_time=datetime.now(),
            semantic_quality=semantic_quality,
            consistency_quality=consistency_quality,
            per_field_correctness_rate=per_field_correctness_rate,
            per_field_hallucination_rate=per_field_hallucination_rate,
            per_field_consistency_rate=per_field_consistency_rate,
        )

        self.evaluation_result = evaluation_result

        # Update PromptManagement with evaluation scores
        if self.prompt_management:
            from datetime import datetime, timezone
            self.prompt_management.evaluation_scores = {
                "correctness_rate": evaluation_result.semantic_quality.correctness_rate,
                "consistency_rate": evaluation_result.semantic_quality.consistency_rate,
                "hallucination_rate": evaluation_result.semantic_quality.hallucination_rate,
                "total_samples": evaluation_result.semantic_quality.total_samples,
                "total_evaluated": evaluation_result.semantic_quality.total_evaluated,
                "per_field_correctness_rate": evaluation_result.per_field_correctness_rate,
                "per_field_hallucination_rate": evaluation_result.per_field_hallucination_rate,
                "per_field_consistency_rate": evaluation_result.per_field_consistency_rate,
            }
            self.prompt_management.updated_at = datetime.now(timezone.utc)

        # Write evaluation summary to file
        file_name = f"{self.project_name}_evaluation_summary_results.csv"
        evaluation_results_file_path = f"{self.output_dir}\\{file_name}"
        self.evaluator.write_results_to_csv(
            evaluation_result=evaluation_result,
            repeated_runs_df=self.repeated_runs_df,
            summary_output_file=evaluation_results_file_path,
        )

        logger.info("Evaluation finalized.")
        return evaluation_result

    def _assess_pass_fail(
        self,
        min_correctness_rate: float | None = None,
        min_consistency_rate: float | None = None,
        max_hallucination_rate: float | None = None,
    ) -> str:
        """Assess whether the workflow passes based on evaluation thresholds."""
        if self.evaluation_result is None:
            self.final_status = "UNKNOWN"
            logger.warning("No evaluation result available — final status is UNKNOWN.")
            return self.final_status

        semantic = self.evaluation_result.semantic_quality

        failures: list[str] = []

        if min_correctness_rate is not None and semantic.correctness_rate < min_correctness_rate:
            failures.append(
                f"correctness_rate ({semantic.correctness_rate:.2%}) < threshold ({min_correctness_rate:.2%})"
            )

        if min_consistency_rate is not None and semantic.consistency_rate < min_consistency_rate:
            failures.append(
                f"consistency_rate ({semantic.consistency_rate:.2%}) < threshold ({min_consistency_rate:.2%})"
            )

        if max_hallucination_rate is not None and semantic.hallucination_rate > max_hallucination_rate:
            failures.append(
                f"hallucination_rate ({semantic.hallucination_rate:.2%}) > threshold ({max_hallucination_rate:.2%})"
            )

        if failures:
            self.final_status = "FAIL"
            self._failure_reasons = failures
            logger.warning("Assessment FAILED:\n  - %s", "\n  - ".join(failures))
        else:
            self.final_status = "PASS"
            self._failure_reasons = []
            logger.info("Assessment PASSED.")

        return self.final_status

    # ------------------------------------------------------------------
    # Revision-loop quality tracking (best-prompt selection + rollback)
    # ------------------------------------------------------------------

    def _retrieval_failure_rate(self) -> float:
        """Return the current judge retrieval-failure rate (0-1), or 0 if unknown.

        A high value means most ungrounded/failed judgments were cases where the
        judge could not retrieve usable evidence — which points to a data or
        evidence-retrieval problem, NOT a prompt problem. The revision loop uses
        this to avoid burning cycles rewriting a prompt that isn't the cause.
        """
        if self.evaluation_result is None:
            return 0.0
        return float(
            getattr(self.evaluation_result.semantic_quality, "retrieval_failure_rate", 0.0)
            or 0.0
        )

    def _score_current_evaluation(
        self,
        max_hallucination_rate: float | None = None,
    ) -> float:
        """Compute a single scalar quality score for the current evaluation.

        Used by the revision loop to compare prompt versions and keep the best
        one. Higher is better. The score rewards correctness and consistency and
        penalizes hallucination, so a revision that trades correctness for a
        marginal hallucination gain does not get falsely rewarded.

        Returns:
            A float in roughly [0, 1]. Returns -1.0 when no evaluation exists so
            any real result is preferred over "no result".
        """
        if self.evaluation_result is None:
            return -1.0

        semantic = self.evaluation_result.semantic_quality
        consistency = self.evaluation_result.consistency_quality

        correctness = float(getattr(semantic, "correctness_rate", 0.0) or 0.0)
        consistency_rate = float(getattr(consistency, "consistency_rate", 0.0) or 0.0)
        hallucination = float(getattr(semantic, "hallucination_rate", 0.0) or 0.0)

        # Correctness is the primary objective; consistency is secondary;
        # hallucination is a penalty (weighted heavily since it is the most
        # damaging failure mode). Weights are intentionally simple and
        # transparent so the selection behavior is easy to reason about.
        score = (0.70 * correctness) + (0.30 * consistency_rate) - (0.50 * hallucination)
        return score

    def _snapshot_prompt_state(self) -> dict:
        """Capture the prompt/schema state that drives extraction.

        Returns a dict that can be passed to _restore_prompt_state() to roll
        back to this exact prompt version.
        """
        return {
            "prompt_result": self.prompt_result,
            "prompt_management": copy.deepcopy(self.prompt_management)
            if self.prompt_management is not None
            else None,
            "generated_system_prompt": self.generated_system_prompt,
            "generated_user_prompt": self.generated_user_prompt,
            "generated_prompt_output_format": self.generated_prompt_output_format,
            "generated_schema": self.generated_schema,
            "evaluation_result": self.evaluation_result,
            "final_status": self.final_status,
            "failure_reasons": list(getattr(self, "_failure_reasons", []) or []),
        }

    def _restore_prompt_state(self, snapshot: dict) -> None:
        """Restore prompt/schema state captured by _snapshot_prompt_state()."""
        self.prompt_result = snapshot["prompt_result"]
        self.prompt_management = snapshot["prompt_management"]
        self.generated_system_prompt = snapshot["generated_system_prompt"]
        self.generated_user_prompt = snapshot["generated_user_prompt"]
        self.generated_prompt_output_format = snapshot["generated_prompt_output_format"]
        self.generated_schema = snapshot["generated_schema"]
        self.evaluation_result = snapshot["evaluation_result"]
        self.final_status = snapshot["final_status"]
        self._failure_reasons = list(snapshot["failure_reasons"])

    def _build_revision_brief(
        self,
        aggregation: JudgeAggregation | None,
        min_correctness_rate: float | None = None,
        min_consistency_rate: float | None = None,
        max_hallucination_rate: float | None = None,
    ) -> str:
        """Build a comprehensive revision brief combining threshold failures,
        consistency scores, and judge aggregation feedback.

        This gives the prompt generator full context on WHY the pipeline failed
        and WHERE the issues are — both correctness and consistency.

        Args:
            aggregation: Judge aggregation results (may be None if judging was skipped).
            min_correctness_rate: The correctness threshold that was set.
            min_consistency_rate: The consistency threshold that was set.
            max_hallucination_rate: The hallucination threshold that was set.

        Returns:
            Formatted revision brief text for the prompt generator.
        """
        lines: list[str] = []

        # --- Section 1: Pipeline Status & Threshold Failures ---
        lines.append("## Pipeline Assessment: FAILED")
        lines.append("")
        lines.append("The extraction prompt did not meet quality thresholds. "
                     "Below is a summary of what failed and why.")
        lines.append("")

        if hasattr(self, "_failure_reasons") and self._failure_reasons:
            lines.append("### Threshold Violations")
            lines.append("")
            for reason in self._failure_reasons:
                lines.append(f"- **FAILED:** {reason}")
            lines.append("")

        # --- Section 2: Overall Scores ---
        if self.evaluation_result:
            semantic = self.evaluation_result.semantic_quality
            consistency = self.evaluation_result.consistency_quality

            lines.append("### Current Performance Scores")
            lines.append("")
            lines.append(f"| Metric | Score | Threshold | Status |")
            lines.append(f"|--------|-------|-----------|--------|")

            # Correctness
            corr_status = "PASS" if (min_correctness_rate is None or semantic.correctness_rate >= min_correctness_rate) else "FAIL"
            lines.append(
                f"| Correctness Rate | {semantic.correctness_rate:.1%} | "
                f"{min_correctness_rate:.1%} | {corr_status} |"
                if min_correctness_rate is not None else
                f"| Correctness Rate | {semantic.correctness_rate:.1%} | — | — |"
            )

            # Consistency
            cons_status = "PASS" if (min_consistency_rate is None or consistency.consistency_rate >= min_consistency_rate) else "FAIL"
            lines.append(
                f"| Consistency Rate | {consistency.consistency_rate:.1%} | "
                f"{min_consistency_rate:.1%} | {cons_status} |"
                if min_consistency_rate is not None else
                f"| Consistency Rate | {consistency.consistency_rate:.1%} | — | — |"
            )

            # Hallucination
            hall_status = "PASS" if (max_hallucination_rate is None or semantic.hallucination_rate <= max_hallucination_rate) else "FAIL"
            lines.append(
                f"| Hallucination Rate | {semantic.hallucination_rate:.1%} | "
                f"≤{max_hallucination_rate:.1%} | {hall_status} |"
                if max_hallucination_rate is not None else
                f"| Hallucination Rate | {semantic.hallucination_rate:.1%} | — | — |"
            )
            lines.append("")

            # --- Section 3: Per-Field Consistency Breakdown ---
            per_field_consistency = self.evaluation_result.per_field_consistency_rate
            if per_field_consistency:
                lines.append("### Per-Field Consistency Rates")
                lines.append("")
                lines.append("Fields with low consistency produce different answers across "
                             "repeated extractions of the same transcript. The prompt needs "
                             "to be more specific/constrained for these fields.")
                lines.append("")

                # Sort worst first
                sorted_fields = sorted(per_field_consistency.items(), key=lambda x: x[1])
                low_consistency_fields = []
                for field_name, rate in sorted_fields:
                    is_low = rate < (min_consistency_rate or 0.8)
                    flag = " ⚠️ LOW" if is_low else ""
                    lines.append(f"- {field_name}: {rate:.1%}{flag}")
                    if is_low:
                        low_consistency_fields.append(field_name)
                lines.append("")

                # Add actionable consistency remediation guidance
                if low_consistency_fields:
                    lines.append("### Consistency Remediation (CRITICAL)")
                    lines.append("")
                    lines.append("The following fields are inconsistent — the model produces "
                                 "different answers on the same transcript across runs. "
                                 "You MUST apply these strategies to the revised prompt "
                                 "for each low-consistency field:")
                    lines.append("")
                    lines.append("**Strategies to improve consistency:**")
                    lines.append("")
                    lines.append("1. **Add explicit decision rules with IF/THEN logic.** "
                                 "For each field, define unambiguous criteria: "
                                 "\"If the customer mentions X AND Y, classify as Z.\" "
                                 "Do not leave classification to interpretation.")
                    lines.append("")
                    lines.append("2. **Add priority/tiebreaker rules.** "
                                 "When multiple valid answers could apply, specify which takes precedence: "
                                 "\"If multiple issues are discussed, select the PRIMARY issue — "
                                 "the one that triggered the call or consumed the most discussion time.\"")
                    lines.append("")
                    lines.append("3. **Constrain the output space.** "
                                 "Where possible, provide an explicit list of allowed values or "
                                 "categories instead of free-text. Use phrasing like: "
                                 "\"Choose one of the following: [A, B, C, D, Other].\"")
                    lines.append("")
                    lines.append("4. **Add a default/fallback rule.** "
                                 "Specify what to output when evidence is ambiguous or absent: "
                                 "\"If the transcript does not clearly indicate X, output 'Unknown'.\"")
                    lines.append("")
                    lines.append("5. **Remove subjective language.** "
                                 "Replace vague instructions like 'determine the general sentiment' "
                                 "with concrete criteria: 'Classify as Positive if the customer "
                                 "explicitly expresses satisfaction or thanks the agent; Negative if "
                                 "they express frustration, anger, or threaten to cancel; Neutral otherwise.'")
                    lines.append("")
                    lines.append(f"**Fields requiring these fixes:** {', '.join(low_consistency_fields)}")
                    lines.append("")

            # --- Section 4: Per-Field Correctness Breakdown ---
            per_field_correctness = self.evaluation_result.per_field_correctness_rate
            if per_field_correctness:
                lines.append("### Per-Field Correctness Rates")
                lines.append("")
                lines.append("Fields with low correctness are producing answers that don't "
                             "match the transcript evidence. The prompt instructions for "
                             "these fields may be ambiguous or under-specified.")
                lines.append("")

                sorted_fields = sorted(per_field_correctness.items(), key=lambda x: x[1])
                for field_name, rate in sorted_fields:
                    flag = " ⚠️ LOW" if rate < (min_correctness_rate or 0.8) else ""
                    lines.append(f"- {field_name}: {rate:.1%}{flag}")
                lines.append("")

        # --- Section 5: Judge Aggregation Detail ---
        if aggregation:
            lines.append("---")
            lines.append("")
            lines.append(aggregation.to_revision_brief_text())

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        review_prompt: bool = True,
        review_schema: bool = True,
        run_judging: bool = True,
        run_evaluation: bool = True,
        n_runs: int = 3,
        parallel_judging: bool = True,
        min_correctness_rate: float | None = None,
        min_consistency_rate: float | None = None,
        max_hallucination_rate: float | None = None,
        max_revision_cycles: int = 3,
        extraction_kwargs: dict | None = None,
        judge_kwargs: dict | None = None,
        template_params: dict = {},
        retrieval_failure_abort_threshold: float | None = 0.2,
    ) -> dict:
        """
        Execute the full orchestration pipeline.

        When parallel_judging=True (default) and n_runs > 1:
            - Run 1 extraction completes first
            - Judging on run 1 results and remaining extraction runs (2..N)
              execute concurrently in separate threads
            - This significantly reduces total wall-clock time

        When parallel_judging=False:
            - Falls back to fully sequential execution (original behavior)

        Automated prompt revision loop:
            After evaluation, if the pipeline FAILs the quality thresholds and
            a prompt agent is available, the orchestrator will:
              1. Aggregate judge feedback into a structured revision brief
              2. Feed the brief to the prompt generator to produce a revised prompt
              3. Re-run extraction → judging → evaluation
              4. Repeat up to max_revision_cycles times or until PASS

        Args:
            max_revision_cycles: Maximum number of automated prompt revision
                cycles to attempt before giving up. Set to 0 to disable.
            retrieval_failure_abort_threshold: If the judge's retrieval-failure
                rate is at/above this fraction (0-1), the revision loop is
                skipped or stopped early, because the failures are driven by the
                judge being unable to retrieve evidence (a data/retrieval
                problem) rather than by a poor prompt. Set to None to disable
                this circuit-breaker. Default 0.2. When triggered, the reason is
                returned as 'revision_stopped_reason'.

        Raises:
            ConfigurationError: If required agents or data are missing.
            PromptGenerationError: If prompt generation fails.
            SchemaGenerationError: If schema generation fails.
            ExtractionError: If extraction fails.
            JudgingError: If the judge agent fails.
            EvaluationError: If evaluation fails.
            WorkflowCancelledError: If a reviewer cancels.

        Returns:
            dict with keys: prompt_result, prompt_management, schema_result,
            extraction_result, judge_result, judge_aggregation, evaluation_result,
            repeated_runs_df, final_status, generated_system_prompt,
            generated_user_prompt, generated_schema, revision_cycles_used.
        """
        extraction_kwargs = extraction_kwargs or {}
        judge_kwargs = judge_kwargs or {}

        # Determine effective number of extraction runs
        effective_n_runs = n_runs if run_evaluation else 1

        logger.info("=" * 60)
        logger.info("ORCHESTRATOR RUN STARTED")
        logger.info("=" * 60)

        # Step 1: Generate prompt (with retry loop on human feedback)
        self._report_phase("prompt_generation")
        self._generate_prompt()

        if review_prompt:
            while True:
                self._report_phase("prompt_review", "Waiting for your review of the generated prompt")
                decision = self._review_prompt()
                if decision.decision == "C":
                    raise WorkflowCancelledError(step="PromptReview", notes=decision.notes)
                if decision.decision == "A":
                    break
                # Retry — feed human notes back to the agent as revision guidance
                logger.info("Prompt review: retry requested. Notes: %s", decision.notes)
                self._report_phase("prompt_generation", "Regenerating prompt from your feedback")
                self._regenerate_prompt(feedback=decision.notes)
        # Save prompt to .md file
        self._save_prompt_to_file()

        # Step 2: Generate schema (with retry loop on human feedback)
        self._report_phase("schema_generation")
        self._generate_schema()

        if review_schema and self.generated_schema is not None:
            while True:
                self._report_phase("schema_review", "Waiting for your review of the generated schema")
                decision = self._review_schema()
                if decision.decision == "C":
                    raise WorkflowCancelledError(step="SchemaReview", notes=decision.notes)
                if decision.decision == "A":
                    break
                # Retry — feed human notes back to the schema agent
                logger.info("Schema review: retry requested. Notes: %s", decision.notes)
                self._report_phase("schema_generation", "Regenerating schema from your feedback")
                self._regenerate_schema(feedback=decision.notes)

        # Step 3 + 4: Extraction and Judging
        # Use concurrent mode when we have multiple runs AND judging is enabled
        use_concurrent = parallel_judging and run_judging and effective_n_runs > 1

        # Pre-compute batch kwargs (extraction_kwargs minus evaluator init params)
        batch_kwargs = {
            k: v for k, v in extraction_kwargs.items()
            if k not in ("model", "temperature")
        }

        if use_concurrent:
            # --- CONCURRENT MODE ---
            # Run 1 extraction first, then overlap remaining runs with judging
            self._report_phase(
                "extraction_and_judging",
                f"Extracting ({effective_n_runs} runs) and judging in parallel",
            )
            logger.info(
                "Using concurrent mode: judging will run in parallel with "
                "extraction runs 2..%d", effective_n_runs
            )

            self._init_evaluator(**{
                k: v for k, v in extraction_kwargs.items()
                if k in ("model", "temperature")
            })

            # Run first extraction synchronously
            first_run_df = self._run_first_extraction(**batch_kwargs)

            # Now run judging and remaining extractions concurrently
            extraction_error = None
            judging_error = None

            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit remaining extractions (runs 2..N)
                remaining_future = executor.submit(
                    self._run_remaining_extractions,
                    n_remaining_runs=effective_n_runs - 1,
                    first_run_df=first_run_df,
                    **batch_kwargs,
                )

                # Submit judging (operates on self.extraction_result from run 1)
                judging_future = executor.submit(
                    self._run_judging,
                    **judge_kwargs,
                )

                # Collect results, preserving exceptions for re-raise
                for future in as_completed([remaining_future, judging_future]):
                    try:
                        future.result()
                    except (ExtractionError, JudgingError) as exc:
                        if isinstance(exc, ExtractionError):
                            extraction_error = exc
                        else:
                            judging_error = exc
                    except Exception as exc:
                        if future == remaining_future:
                            extraction_error = ExtractionError(
                                f"Remaining extraction runs failed: {exc}",
                                context={"original_error": str(exc)},
                            )
                        else:
                            judging_error = JudgingError(
                                f"Judge agent failed: {exc}",
                                context={"original_error": str(exc)},
                            )

            # Re-raise errors (extraction errors take priority)
            if extraction_error:
                raise extraction_error
            if judging_error:
                raise judging_error

            logger.info(
                "Concurrent extraction + judging complete. "
                "Repeated runs: %d, Judge results: %d rows.",
                len(self.repeated_runs_df) if self.repeated_runs_df is not None else 0,
                len(self.judge_result) if self.judge_result is not None else 0,
            )

        else:
            # --- SEQUENTIAL MODE (original behavior) ---
            self._report_phase("extraction", f"Running extraction ({effective_n_runs} run(s))")
            self._run_bootstrap_extraction(
                n_runs=effective_n_runs,
                **extraction_kwargs,
            )

            if run_judging:
                self._report_phase("judging", "Judging extraction results")
                self._run_judging(**judge_kwargs)

        # Step 5: Finalize evaluation (combine consistency + semantic quality)
        revision_cycles_used = 0

        if run_evaluation:
            self._report_phase("evaluation", "Computing consistency + quality metrics")
            self._finalize_evaluation()

            # Step 6: Threshold assessment
            self._assess_pass_fail(
                min_correctness_rate=min_correctness_rate,
                min_consistency_rate=min_consistency_rate,
                max_hallucination_rate=max_hallucination_rate,
            )

            # Step 7: Automated revision loop — re-run until PASS or max cycles
            revision_cycles_used = 0
            can_revise = (
                self.prompt_agent is not None
                and run_judging
                and max_revision_cycles > 0
            )

            # Best-prompt tracking: remember the best-scoring prompt version seen
            # so far so a revision that REGRESSES quality never becomes the final
            # output. Without this, the loop keeps whatever the last cycle
            # produced — which can be worse than where it started.
            best_score = self._score_current_evaluation(
                max_hallucination_rate=max_hallucination_rate
            )
            best_snapshot = self._snapshot_prompt_state()
            best_cycle = 0
            logger.info(
                "Revision baseline (pre-revision) score: %.4f (status=%s)",
                best_score, self.final_status,
            )

            # Option 4: retrieval-health circuit-breaker. If most failures are
            # driven by the judge being unable to retrieve evidence (a data /
            # retrieval problem), revising the prompt cannot help — so we refuse
            # to enter/continue the loop and flag the real cause instead.
            self.revision_stopped_reason: str | None = None

            if (
                can_revise
                and retrieval_failure_abort_threshold is not None
                and self._retrieval_failure_rate() >= retrieval_failure_abort_threshold
            ):
                rate = self._retrieval_failure_rate()
                self.revision_stopped_reason = (
                    f"Retrieval-failure rate {rate:.1%} >= "
                    f"{retrieval_failure_abort_threshold:.1%}. Most ungrounded "
                    "judgments are cases where the judge could not retrieve "
                    "evidence, which indicates a DATA / EVIDENCE-RETRIEVAL problem "
                    "(e.g. corrupted transcripts), not a prompt problem. Prompt "
                    "revision was skipped — fix transcript/evidence quality first."
                )
                logger.warning(
                    "REVISION LOOP SKIPPED — %s", self.revision_stopped_reason
                )
                can_revise = False

            while self.final_status == "FAIL" and can_revise and revision_cycles_used < max_revision_cycles:
                revision_cycles_used += 1
                self._report_phase(
                    "revision",
                    f"Auto-revising prompt (cycle {revision_cycles_used}/{max_revision_cycles})",
                )
                logger.info(
                    "=" * 40 + " REVISION CYCLE %d/%d " + "=" * 40,
                    revision_cycles_used, max_revision_cycles,
                )

                # 7a: Aggregate judge feedback
                aggregation = self._aggregate_judge_feedback()
                if aggregation is None:
                    logger.warning("No judge feedback to aggregate — stopping revision loop.")
                    break

                # 7b: Build comprehensive revision brief with threshold failures,
                # consistency scores, and judge feedback — then feed to prompt generator
                revision_brief_text = self._build_revision_brief(
                    aggregation=aggregation,
                    min_correctness_rate=min_correctness_rate,
                    min_consistency_rate=min_consistency_rate,
                    max_hallucination_rate=max_hallucination_rate,
                )

                # Save the revision brief to .md for traceability
                brief_filename = f"{self.project_name}_revision_brief_cycle_{revision_cycles_used}"
                save_file(
                    content=revision_brief_text,
                    filename=brief_filename,
                    output_dir=self.output_dir,
                    extension=".md",
                    overwrite=True,
                )
                logger.info("Revision brief saved to %s/%s.md", self.output_dir, brief_filename)

                logger.info("Feeding comprehensive revision brief to prompt generator.")
                self._regenerate_prompt(revision_brief=revision_brief_text)
                self._save_prompt_to_file()
                logger.info(
                    "Prompt revised to version %d.",
                    self.prompt_management.version if self.prompt_management else -1,
                )

                # 7c: Re-run extraction with the revised prompt
                if use_concurrent:
                    # Re-init evaluator for the new run
                    self._init_evaluator(**{
                        k: v for k, v in extraction_kwargs.items()
                        if k in ("model", "temperature")
                    })
                    first_run_df = self._run_first_extraction(**batch_kwargs)

                    extraction_error = None
                    judging_error = None

                    with ThreadPoolExecutor(max_workers=2) as executor:
                        remaining_future = executor.submit(
                            self._run_remaining_extractions,
                            n_remaining_runs=effective_n_runs - 1,
                            first_run_df=first_run_df,
                            **batch_kwargs,
                        )
                        judging_future = executor.submit(
                            self._run_judging,
                            **judge_kwargs,
                        )

                        for future in as_completed([remaining_future, judging_future]):
                            try:
                                future.result()
                            except (ExtractionError, JudgingError) as exc:
                                if isinstance(exc, ExtractionError):
                                    extraction_error = exc
                                else:
                                    judging_error = exc
                            except Exception as exc:
                                if future == remaining_future:
                                    extraction_error = ExtractionError(
                                        f"Remaining extraction runs failed (cycle {revision_cycles_used}): {exc}",
                                        context={"original_error": str(exc)},
                                    )
                                else:
                                    judging_error = JudgingError(
                                        f"Judge agent failed (cycle {revision_cycles_used}): {exc}",
                                        context={"original_error": str(exc)},
                                    )

                    if extraction_error:
                        raise extraction_error
                    if judging_error:
                        raise judging_error
                else:
                    # Sequential mode
                    self._run_bootstrap_extraction(
                        n_runs=effective_n_runs,
                        **extraction_kwargs,
                    )
                    self._run_judging(**judge_kwargs)

                # 7d: Re-evaluate and re-assess
                self._finalize_evaluation()
                self._assess_pass_fail(
                    min_correctness_rate=min_correctness_rate,
                    min_consistency_rate=min_consistency_rate,
                    max_hallucination_rate=max_hallucination_rate,
                )

                # 7d-guard: retrieval-health circuit-breaker (Option 4). If this
                # cycle still FAILs and the failures are dominated by retrieval
                # misses, stop — further prompt revisions cannot fix a data /
                # evidence-retrieval problem.
                if (
                    self.final_status == "FAIL"
                    and retrieval_failure_abort_threshold is not None
                    and self._retrieval_failure_rate() >= retrieval_failure_abort_threshold
                ):
                    rate = self._retrieval_failure_rate()
                    self.revision_stopped_reason = (
                        f"Retrieval-failure rate {rate:.1%} >= "
                        f"{retrieval_failure_abort_threshold:.1%} after cycle "
                        f"{revision_cycles_used}. Remaining failures are "
                        "retrieval-driven (likely corrupted transcripts / weak "
                        "evidence), not prompt-driven — stopping the revision loop."
                    )
                    logger.warning(
                        "REVISION LOOP STOPPED EARLY — %s", self.revision_stopped_reason
                    )
                    # Still let this cycle compete for "best" below before breaking.
                    cycle_score = self._score_current_evaluation(
                        max_hallucination_rate=max_hallucination_rate
                    )
                    if cycle_score > best_score:
                        best_score = cycle_score
                        best_snapshot = self._snapshot_prompt_state()
                        best_cycle = revision_cycles_used
                    break

                # 7e: Compare this cycle's quality against the best seen so far.
                # If it PASSED, it is definitively the one we want. Otherwise we
                # only promote it to "best" when it actually improved the score,
                # so a regressing revision never becomes the final output.
                cycle_score = self._score_current_evaluation(
                    max_hallucination_rate=max_hallucination_rate
                )
                logger.info(
                    "Revision cycle %d score: %.4f (best so far: %.4f from cycle %d)",
                    revision_cycles_used, cycle_score, best_score, best_cycle,
                )

                if self.final_status == "PASS":
                    best_score = cycle_score
                    best_snapshot = self._snapshot_prompt_state()
                    best_cycle = revision_cycles_used
                    logger.info(
                        "Pipeline PASSED after %d revision cycle(s).",
                        revision_cycles_used,
                    )
                    break

                if cycle_score > best_score:
                    logger.info(
                        "Cycle %d improved quality (%.4f > %.4f) — keeping as new best.",
                        revision_cycles_used, cycle_score, best_score,
                    )
                    best_score = cycle_score
                    best_snapshot = self._snapshot_prompt_state()
                    best_cycle = revision_cycles_used
                else:
                    logger.warning(
                        "Cycle %d did NOT improve quality (%.4f <= %.4f) — "
                        "this revision will not be kept as the final prompt.",
                        revision_cycles_used, cycle_score, best_score,
                    )

            # After the loop: if the best result did not come from the most
            # recent cycle, roll back to the best-scoring prompt version so the
            # harness never leaves a regressed prompt as the final output.
            if best_cycle != revision_cycles_used:
                logger.info(
                    "Restoring best prompt version (from cycle %d, score %.4f) "
                    "over the last revision (cycle %d).",
                    best_cycle, best_score, revision_cycles_used,
                )
                self._restore_prompt_state(best_snapshot)
                self._save_prompt_to_file()

            if self.final_status == "FAIL" and revision_cycles_used >= max_revision_cycles:
                logger.warning(
                    "Pipeline still FAILING after %d revision cycles — "
                    "max_revision_cycles exhausted. Final prompt is the "
                    "best-scoring version seen (cycle %d, score %.4f).",
                    revision_cycles_used, best_cycle, best_score,
                )

        self._report_phase("complete", f"Final status: {self.final_status}")
        logger.info("=" * 60)
        logger.info("ORCHESTRATOR RUN COMPLETE — final_status=%s", self.final_status)
        logger.info("=" * 60)

        return {
            "prompt_result": self.prompt_result,
            "prompt_management": self.prompt_management,
            "schema_result": self.schema_result,
            "extraction_result": self.extraction_result,
            "judge_result": self.judge_result,
            "judge_aggregation": self.judge_aggregation,
            "evaluation_result": self.evaluation_result,
            "repeated_runs_df": self.repeated_runs_df,
            "final_status": self.final_status,
            "generated_system_prompt": self.generated_system_prompt,
            "generated_user_prompt": self.generated_user_prompt,
            "generated_schema": self.generated_schema,
            "revision_cycles_used": revision_cycles_used,
            "revision_stopped_reason": getattr(self, "revision_stopped_reason", None),
            "retrieval_failure_rate": self._retrieval_failure_rate(),
        }
