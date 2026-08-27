"""
agents/grounding_judge_agent.py

GroundingJudgeAgent

Evaluates whether extracted field-level claims are grounded in transcript evidence.

For each field:
- attempts to retrieve evidence from the transcript using a direct quote if available
- otherwise derives search terms from ai_reasoning and searches the transcript
- returns whether evidence was found
- judges whether the evidence supports the original claim
- assigns evidence strength
"""

from typing import Any, Optional, Literal
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from agents.BaseAgent import BaseToolAgent
from tools.CHUNKING_TOOLS.find_transcript_chunks import find_transcript_chunk_merged
from data_models.agent_models.JudgeAgent_datamodel import JudgeFieldInput,JudgeFieldResult,GroundingJudgeResponse,GroundingJudgeLLMResponse
from tools.SEARCH_TOOLS.derive_search_terms import derive_search_terms
from tools.SEARCH_TOOLS.generate_ngram_search_terms import generate_ngram_search_terms
from prompts.agent_prompts.judge_PROMPT import *
from prompts.general_prompts.refine_search_terms_PROMPT import REFINE_TERMS_SYSTEM_PROMPT,REFINE_TERMS_USER_PROMPT
from data_models.refine_search_terms_datamodel import RefinedSearchTermsResponse
import csv
import os
import math

class JudgeAgent(BaseToolAgent):
    """Agent that evaluates whether extracted field-level claims are grounded in transcript evidence.

    For each extracted field, the agent attempts to retrieve supporting evidence
    from the transcript (via direct quote or LLM-generated search terms), then
    asks the LLM to judge whether the evidence supports the original claim.
    """

    def __init__(
        self,
        api_key: str = None,
        assistant_id: str = None,
        model: str = "gpt-4.1-mini",
        temperature: str = "0.1",
        extra_tools=None,
        enable_tool_logging: bool = True,
    ):
        """Initialize the judge agent.

        Args:
            api_key: API key for the LLM service.
            assistant_id: Optional assistant identifier.
            model: Model name for inference.
            temperature: Sampling temperature as string.
            extra_tools: Additional Tool objects to register.
            enable_tool_logging: Whether to enable debug logging.
        """
        super().__init__(
            model=model,
            api_key=api_key,
            assistant_id=assistant_id,
            temperature=temperature,
            extra_tools=extra_tools,
            enable_tool_logging=enable_tool_logging,
        )

        self.api_key = api_key
        self.model = model
        self.assistant_id = assistant_id
        self.temperature = float(temperature)

    def build_judge_field_inputs(
        self,
        row: dict,
        judge_config: list[dict],
        skip_none_claims: bool = True,
        require_task_prompt: bool = False,
    ) -> list[JudgeFieldInput]:
        """Build JudgeFieldInput objects from a data row and judge configuration.

        Reads claim values, quotes, reasoning, and task prompts from the row
        based on the column mappings defined in judge_config.

        Args:
            row: Dict representing a single data row with extraction results.
            judge_config: List of config dicts specifying field_name, claim_column,
                quote_column, reasoning_column, and task_prompt mappings.
            skip_none_claims: If True, skip fields where the claim value is None/NaN.
            require_task_prompt: If True, raise ValueError when task_prompt is empty.

        Returns:
            List of JudgeFieldInput objects ready for judging.

        Raises:
            ValueError: If require_task_prompt is True and a field has no task_prompt.
        """
        field_inputs = []

        for cfg in judge_config:
            field_name = cfg["field_name"]
            claim_column = cfg["claim_column"]
            quote_column = cfg.get("quote_column")
            reasoning_column = cfg.get("reasoning_column")
            task_prompt_column = cfg.get("task_prompt_column")
            static_task_prompt = cfg.get("task_prompt", "")

            claim_value = row.get(claim_column)

            is_missing_claim = (
                claim_value is None
                or (isinstance(claim_value, float) and math.isnan(claim_value))
            )

            if skip_none_claims and is_missing_claim:
                continue

            task_prompt = (
                row.get(task_prompt_column, "")
                if task_prompt_column
                else static_task_prompt
            ) or ""

            if require_task_prompt and not str(task_prompt).strip():
                raise ValueError(f"Missing task_prompt for field {field_name}")

            quote_value = row.get(quote_column, "") if quote_column else ""
            reasoning_value = row.get(reasoning_column, "") if reasoning_column else ""

            if isinstance(quote_value, float) and math.isnan(quote_value):
                quote_value = ""
            if isinstance(reasoning_value, float) and math.isnan(reasoning_value):
                reasoning_value = ""

            field_inputs.append(
                JudgeFieldInput(
                    field_name=field_name,
                    claim_value=claim_value,
                    quote=str(quote_value) if quote_value is not None else "",
                    ai_reasoning=str(reasoning_value) if reasoning_value is not None else "",
                    task_prompt=str(task_prompt),
                )
            )

        return field_inputs
            


    def flatten_judge_results(self,results: list[JudgeFieldResult]) -> dict:
        """Flatten a list of JudgeFieldResult objects into a single flat dict.

        Each field result is expanded with prefixed keys for grounded, hallucinated,
        evidence_strength, evidence_found, evidence_chunk, search_terms_used,
        judge_explanation, and LLM token usage.

        Args:
            results: List of JudgeFieldResult objects.

        Returns:
            Flat dict with prefixed keys for each field's judgment results.
        """
        output = {}

        for result in results:
            prefix = result.field_name
            output[prefix] = result.claim_value
            output[f"{prefix}_GROUNDED"] = result.grounded
            output[f"{prefix}_HALLUCINATED"] = result.hallucinated
            output[f"{prefix}_EVIDENCE_STRENGTH"] = result.evidence_strength
            output[f"{prefix}_EVIDENCE_FOUND"] = result.evidence_found
            output[f"{prefix}_EVIDENCE_CHUNK"] = result.evidence_chunk
            output[f"{prefix}_SEARCH_TERMS_USED"] = " | ".join(result.search_terms_used) if result.search_terms_used else ""
            output[f"{prefix}_JUDGE_EXPLANATION"] = result.explanation
            output[f"{prefix}_ERROR_TYPE"] = result.error_type
            output[f"{prefix}_PROMPT_ADJUSTMENT_SUGGESTION"] = result.prompt_adjustment_suggestion
            output[f"{prefix}_LLM_INPUT_TOKENS"] = result.llm_input_tokens
            output[f"{prefix}_LLM_OUTPUT_TOKENS"] = result.llm_output_tokens
            output[f"{prefix}_LLM_TOTAL_TOKENS"] = result.llm_total_tokens
        return output
    

    def _setup_tools(self):
        """JudgeAgent does not use tools; initialize empty tool list."""
        self.tools = []
    
    def _make_row_agent(self):
        """Create a fresh JudgeAgent instance for per-row processing.

        Returns:
            A new JudgeAgent with the same configuration as self.
        """
        return JudgeAgent(
            api_key=self.api_key,
            assistant_id=getattr(self, "assistant_id", None),
            model=self.model,
            temperature=self.temperature,
            enable_tool_logging=self.enable_tool_logging,
        )

    def _generate_search_terms_with_llm(
        self,
        field_input: JudgeFieldInput,
        system_prompt: str = REFINE_TERMS_SYSTEM_PROMPT,
        user_prompt_template: str = REFINE_TERMS_USER_PROMPT,
        max_words:int = 3
    ) -> tuple[list[str], dict]:
        """
        Use the LLM to generate transcript-likely search terms from the task prompt,
        field metadata, extractor quote, and extractor reasoning.
        """
        prompt_params = {
            "task_prompt": field_input.task_prompt or "",
            "field_name": field_input.field_name,
            "claim_value": field_input.claim_value,
            "quote": field_input.quote or "",
            "ai_reasoning": field_input.ai_reasoning or "",
        }

        system_prompt_params = {
            'MAX_WORDS':max_words
        }

        user_content = self._render_user_prompt(user_prompt_template, prompt_params)
        system_prompt = self._render_user_prompt(system_prompt, system_prompt_params)

        messages = self._build_messages(system_prompt, user_content)

        result = self.chat(
            messages=messages,
            temperature=self.temperature,
            use_history=False,
        )

        usage = self.extract_usage_tokens(result)
        raw_text = self._extract_chat_text(result)

        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1].rsplit("\n```", 1)[0].strip()

        parsed = self._safe_json_parse(raw_text)
        validated = RefinedSearchTermsResponse.model_validate(parsed)

        terms = []
        seen = set()

        for term in validated.refined_terms:
            t = str(term).strip()
            key = t.lower()
            if t and key not in seen:
                seen.add(key)
                terms.append(t)

        return terms,usage
    

    def _retrieve_evidence(
        self,
        transcript: str,
        field_input: JudgeFieldInput,
        context_lines: int = 2,
    ) -> tuple[bool, str, list[str], dict]:
        """
        Retrieve candidate transcript evidence for a field.

        Retrieval order:
        1. Try direct quote search if quote is provided (splits long quotes into
           sentence fragments for better matching across line boundaries).
        2. Extract n-gram fragments from the quote as search terms (no LLM call needed).
        3. Generate refined search terms with the LLM using quote/reasoning/task prompt.
        4. Search transcript using refined terms.
        5. Return whether evidence was found, the evidence chunk, and search terms used.
        """
        quote = (field_input.quote or "").strip()
        evidence_chunk = ""
        zero_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        # 1. Direct quote search — split into sentence fragments for better matching
        if quote:
            # First try the full quote
            evidence_chunk = find_transcript_chunk_merged(
                transcript,
                quote,
                context_lines=context_lines,
                formatted=True,
                include_search_term_label=False,
            )
            if evidence_chunk and evidence_chunk.strip():
                return True, evidence_chunk, [], zero_usage

            # If full quote fails, split into sentence-level fragments and search each
            import re
            fragments = [
                frag.strip()
                for frag in re.split(r'[.!?;]+', quote)
                if len(frag.strip()) > 10
            ]
            if fragments:
                evidence_chunk = find_transcript_chunk_merged(
                    transcript,
                    fragments,
                    context_lines=context_lines,
                    formatted=True,
                    include_search_term_label=True,
                )
                if evidence_chunk and evidence_chunk.strip():
                    return True, evidence_chunk, fragments, zero_usage

        # 2. Extract n-gram phrases from the quote directly (no LLM call)
        if quote:
            ngram_terms = generate_ngram_search_terms(
                quote,
                ngram_range=(2, 4),
                top_k=8,
                stop_words="bilingual",
            )
            if ngram_terms:
                evidence_chunk = find_transcript_chunk_merged(
                    transcript,
                    ngram_terms,
                    context_lines=context_lines,
                    formatted=True,
                    include_search_term_label=True,
                )
                if evidence_chunk and evidence_chunk.strip():
                    return True, evidence_chunk, ngram_terms, zero_usage

        # 3. Generate refined search terms with LLM
        refined_terms, term_usage = self._generate_search_terms_with_llm(field_input)

        # 4. Search with refined terms
        if refined_terms:
            evidence_chunk = find_transcript_chunk_merged(
                transcript,
                refined_terms,
                context_lines=context_lines,
                formatted=True,
                include_search_term_label=True,
            )
            if evidence_chunk and evidence_chunk.strip():
                return True, evidence_chunk, refined_terms, term_usage

            no_result_chunk = f"Search term: {' | '.join(refined_terms)} [YIELDED NO RESULTS]"
            return False, no_result_chunk, refined_terms, term_usage

        return False, "[NO SEARCH TERMS GENERATED]", [], zero_usage    

    def _judge_one_field(
        self,
        transcript: str,
        field_input: JudgeFieldInput,
        context_lines: int = 2,
        system_prompt: str = SYSTEM_PROMPT,
        user_prompt_template: str = USER_PROMPT,
        extra_template_params: dict | None = None,
    ) -> JudgeFieldResult:
        """
        Retrieve transcript evidence for a single field, then ask the LLM to judge
        whether the evidence supports the claim.
        """
        evidence_found, evidence_chunk, search_terms_used,term_usage = self._retrieve_evidence(
            transcript=transcript,
            field_input=field_input,
            context_lines=context_lines,
        )

        # Build field-level params — these are the core placeholders in the judge prompt
        prompt_params = {
            "task_prompt": field_input.task_prompt or "",
            "field_name": field_input.field_name,
            "claim_value": field_input.claim_value,
            "quote": field_input.quote or "",
            "ai_reasoning": field_input.ai_reasoning or "",
            "evidence_chunk": evidence_chunk if evidence_chunk else "[No relevant evidence found.]",
        }

        # Merge any extra template params (e.g. row-level data), but field-level
        # params take precedence so they can never be overwritten
        if extra_template_params:
            merged_params = {**extra_template_params, **prompt_params}
        else:
            merged_params = prompt_params

        user_content = self._render_user_prompt(user_prompt_template, merged_params)
        messages = self._build_messages(system_prompt, user_content)

        result = self.chat(
            messages=messages,
            temperature=float(self.temperature),
            use_history=False,
        )

        judge_usage = self.extract_usage_tokens(result)
        raw_text = self._extract_chat_text(result)

        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1].rsplit("\n```", 1)[0].strip()

        parsed = self._safe_json_parse(raw_text)

        
        validated = GroundingJudgeLLMResponse.model_validate(parsed)

        total_input_tokens = term_usage["input_tokens"] + judge_usage["input_tokens"]
        total_output_tokens = term_usage["output_tokens"] + judge_usage["output_tokens"]
        total_tokens = term_usage["total_tokens"] + judge_usage["total_tokens"]

        hallucinated = validated.hallucinated
        if validated.grounded:
            hallucinated = False

        return JudgeFieldResult(
            field_name=field_input.field_name,
            claim_value=field_input.claim_value,
            grounded=validated.grounded,
            hallucinated=hallucinated,
            evidence_strength=validated.evidence_strength,
            evidence_found=evidence_found,
            evidence_chunk=evidence_chunk,
            search_terms_used=search_terms_used,
            explanation=validated.explanation,
            error_type=validated.error_type,
            prompt_adjustment_suggestion=validated.prompt_adjustment_suggestion,
            llm_input_tokens=total_input_tokens,
            llm_output_tokens=total_output_tokens,
            llm_total_tokens=total_tokens
        )

    def run(
        self,
        transcript: str,
        field_inputs: list[JudgeFieldInput],
        context_lines: int = 2,
        system_prompt: str = SYSTEM_PROMPT,
        user_prompt_template: str = USER_PROMPT,
        template_params = None
    ) -> list[JudgeFieldResult]:
        """
        Judge grounding for one or more extracted fields against a transcript.

        For each field:
        - attempts quote-based retrieval first
        - falls back to reasoning-derived transcript search
        - evaluates whether the retrieved evidence supports the claim

        Returns:
            A list of per-field JudgeFieldResult objects.
        """
        results = []

        for field_input in field_inputs:
            result = self._judge_one_field(
                transcript=transcript,
                field_input=field_input,
                context_lines=context_lines,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                extra_template_params=template_params,
            )
            results.append(result)
        
        return results

    def process_batch(
        self,
        transcript_column_name: str,
        rows: list[dict],
        judge_config: list[dict],
        session_id_column: str = "AGENTRECORDINGSESSIONID",
        max_workers: int = 3,
        output_file: str = None,
        include_columns: list[str] = None,
        max_retries: int = 3,
        context_lines: int = 2,
        template_params = None
    ) -> list[dict]:
        """Process a batch of rows through the judge agent in parallel.

        For each row, builds field inputs from judge_config, runs grounding
        judgments, and collects results. Supports incremental CSV output and
        deduplication by session ID.

        Args:
            transcript_column_name: Column containing transcript text.
            rows: List of dict rows to judge.
            judge_config: List of field configuration dicts for building inputs.
            session_id_column: Column name for the unique session identifier.
            max_workers: Number of parallel workers.
            output_file: Optional CSV path to append results as they complete.
            include_columns: Optional extra columns to copy into output.
            max_retries: Retry count per row on failure.
            context_lines: Number of context lines for evidence retrieval.
            template_params: Optional template parameters for prompt rendering.

        Returns:
            List of result dicts, one per processed row.
        """
        def _process(row):
            sessionid = row.get(session_id_column)
            print("Judging Sessioon ID:", sessionid)

            for attempt in range(1, max_retries + 1):
                try:
                    row_agent = self._make_row_agent()
                    transcript = row.get(transcript_column_name, "")

                    if template_params:
                        params = (
                            template_params(row)
                            if callable(template_params)
                            else {k: row.get(v) for k, v in template_params.items()}
                        )
                    else:
                        params = None

                    field_inputs = self.build_judge_field_inputs(row, judge_config)
                    print("FIELD_INPUTS:", field_inputs)
                    
                    judge_results = row_agent.run(
                        transcript=transcript,
                        field_inputs=field_inputs,
                        context_lines=context_lines,
                        template_params = params
                    )

                    answer = {
                        session_id_column: sessionid,
                        "PROCESS_STATUS": "TRUE",
                    }

                    if include_columns:
                        answer.update({col: row.get(col) for col in include_columns})

                    flattened_results = self.flatten_judge_results(judge_results)
                    print('Flattened Results:',flattened_results)
                    answer.update(flattened_results)
    

                    return answer

                except Exception as e:
                    print(f"Attempt {attempt}/{max_retries} failed for {sessionid}: {e}")
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)

            return {
                session_id_column: sessionid,
                "PROCESS_STATUS": "FALSE",
            }

        results = []
        writer = None
        file_exists = os.path.exists(output_file) if output_file else False
        f = open(output_file, "a", newline="", encoding="utf-8") if output_file else None
        seen_agentrecordingsessionid = set()

        try:
            if max_workers == 1:
                for i, row in enumerate(rows):
                    result = _process(row)
                    sessionid = result.get(session_id_column)

                    if sessionid in seen_agentrecordingsessionid:
                        print(f"Skipping duplicate result for SessionID {sessionid}")
                        continue

                    seen_agentrecordingsessionid.add(sessionid)
                    results.append(result)

                    print(
                        f"Completed {i + 1}/{len(rows)}: "
                        f"{session_id_column} {result.get(session_id_column)} | "
                        f"Status: {result.get('PROCESS_STATUS')}"
                    )

                    if f:
                        if writer is None:
                            writer = csv.DictWriter(f, fieldnames=result.keys())
                            if not file_exists:
                                writer.writeheader()
                        writer.writerow(result)
                        f.flush()

            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(_process, row): row for row in rows}

                    for i, future in enumerate(as_completed(futures)):
                        result = future.result()
                        sessionid = result.get(session_id_column)

                        if sessionid in seen_agentrecordingsessionid:
                            print(f"Skipping duplicate result for SessionID {sessionid}")
                            continue

                        seen_agentrecordingsessionid.add(sessionid)
                        results.append(result)

                        print(
                            f"Completed {i + 1}/{len(rows)}: "
                            f"{session_id_column} {result.get(session_id_column)} | "
                            f"Status: {result.get('PROCESS_STATUS')}"
                        )

                        if f:
                            if writer is None:
                                writer = csv.DictWriter(f, fieldnames=result.keys())
                                if not file_exists:
                                    writer.writeheader()
                            writer.writerow(result)
                            f.flush()

        finally:
            if f:
                f.close()

        return results