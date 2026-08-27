"""
agents/transcript_extraction_agent.py

TranscriptExtractionAgent

Extracts structured data from a transcript using one of two strategies:

1. Inline mode for shorter transcripts
   - The full transcript is injected directly into the prompt as {{TRANSCRIPT}}.

2. Tool mode for longer transcripts
   - The prompt initially receives a placeholder instead of the full transcript.
   - The agent uses the search_transcript tool to locate relevant transcript sections.
   - If relevant matches are found, the matched transcript chunk(s) are injected
     into {{TRANSCRIPT}} for final extraction.
   - If no relevant matches are found after retry attempts, the agent falls back
     to injecting the full transcript into {{TRANSCRIPT}}.

The agent validates the model response against a provided Pydantic response model.

Retrieval-only evidence means evidence is populated only when the search tool successfully returns relevant 
transcript chunks. If retrieval fails, the agent may still fallback to the full transcript for extraction, 
but the evidence field remains blank.

Returns:
    (validated_model, search_terms, found_any, matched_evidence)

Where:
    validated_model:
        A validated Pydantic model instance, or None if validation fails.

    search_terms:
        A deduplicated list of transcript search queries attempted during tool mode.

    found_any:
        True if the search_transcript tool found at least one matching transcript
        chunk during tool mode; otherwise False.

    matched_evidence:
        The transcript chunk text returned by the search tool when matches were
        found, or an empty string if no matched chunk evidence was used.
"""
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from tools.utils.build_tool import Tool
from jinja2 import Template
from pydantic import BaseModel
import pandas as pd

from agents.BaseAgent import BaseToolAgent
from tools.SEARCH_TOOLS.search_transcript import make_search_transcript_tool, dedupe_preserve_order,SearchTranscriptInput

from tools.CHUNKING_TOOLS.find_transcript_chunks import find_transcript_chunk_merged


class TranscriptExtractionAgent(BaseToolAgent):
    """Agent that extracts structured data from transcripts using inline or tool-based strategies.

    For short transcripts (below token_threshold), the full transcript is injected
    directly into the prompt. For longer transcripts, a search_transcript tool is
    used to locate relevant sections before extraction.
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
        """Initialize the transcript extraction agent.

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
        self._setup_tools()
        self.api_key = api_key
        self.model = model
        self.assistant_id = assistant_id
        self.temperature = temperature
        self._captured_search_terms = []

    def _setup_tools(self):
        """Initialize tool list. Tools are registered dynamically per-run in tool mode."""
        pass

    def _empty_usage(self) -> dict:
        """Return a zeroed-out token usage dict.

        Returns:
            Dict with input_tokens, output_tokens, and total_tokens all set to 0.
        """
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    def _add_usage(self, total: dict, usage: dict) -> dict:
        """Accumulate token usage from a new call into a running total.

        Args:
            total: Running total usage dict.
            usage: New usage dict to add.

        Returns:
            Updated total usage dict.
        """
        return {
            "input_tokens": total.get("input_tokens", 0) + usage.get("input_tokens", 0),
            "output_tokens": total.get("output_tokens", 0) + usage.get("output_tokens", 0),
            "total_tokens": total.get("total_tokens", 0) + usage.get("total_tokens", 0),
        }

    def unregister_tool(self, tool_name: str):
        """Remove a tool from the registered tools list by name.

        Args:
            tool_name: Name of the tool to remove.
        """
        self.tools = [t for t in self.tools if t.name != tool_name]
        self._log(f"Unregistered tool: {tool_name}")

    def _render_user_prompt(self, user_prompt_template: str, template_params: dict) -> str:
        """Render a Jinja2 user prompt template with provided parameters.

        Args:
            user_prompt_template: Jinja2 template string.
            template_params: Variables to inject into the template.

        Returns:
            Rendered prompt string.
        """
        return Template(user_prompt_template).render(**template_params)

    def _build_messages(self, system_prompt: str, user_content: str) -> list[dict]:
        """Build a system + user message pair for the LLM.

        Args:
            system_prompt: System instruction text.
            user_content: Rendered user prompt.

        Returns:
            List of message dicts.
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _reset_search_terms(self):
        """Clear the captured search terms list for a new extraction run."""
        self._captured_search_terms = []

    def _record_search_terms(self, queries):
        """Record search queries attempted during tool mode.

        Args:
            queries: A single query string or list of query strings.
        """
        if isinstance(queries, str):
            queries = [queries]
        if not queries:
            return

        clean = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        self._captured_search_terms.extend(clean)

    def _execute_tool(self, tool_name, args):
        """Execute a tool and capture search terms from search_transcript calls.

        Overrides BaseToolAgent._execute_tool to intercept search queries for
        tracking which terms were attempted during tool mode.

        Args:
            tool_name: Name of the tool to execute.
            args: Dict of arguments to pass to the tool.

        Returns:
            Tool result as a string.
        """
        if tool_name == "search_transcript":
            self._record_search_terms(args.get("queries", []))

        result = super()._execute_tool(tool_name, args)

        try:
            parsed = json.loads(result) if isinstance(result, str) else result
            if tool_name == "search_transcript" and isinstance(parsed, dict):
                self._record_search_terms(parsed.get("queries", []))
        except Exception:
            pass

        return result


    def _run_inline_mode(
        self,
        transcript: str,
        system_prompt: str,
        user_prompt_template: str,
        template_params: dict,
    ) ->  tuple[str, dict]:
        """Extract from a transcript by injecting the full text directly into the prompt.

        Used for shorter transcripts that fit within token limits.
        When category tools are registered, uses the agent loop to handle
        multi-step tool calls (lookup → register → extract).

        Args:
            transcript: Full transcript text.
            system_prompt: System instruction for the LLM.
            user_prompt_template: Jinja2 template for the user message.
            template_params: Template variables (TRANSCRIPT is injected automatically).

        Returns:
            Tuple of (raw LLM response text, token usage dict).
        """
        params = dict(template_params or {})
        params["TRANSCRIPT"] = transcript

        user_content = self._render_user_prompt(user_prompt_template, params)
        messages = self._build_messages(system_prompt, user_content)

        # If category tools (or any tools) are registered, use the full agent loop
        # so multi-step tool calls (lookup → register → final answer) resolve properly.
        if self.tools:
            result = self.run_agent(
                messages=messages,
                tools=self._tool_schemas(),
                available_functions=self._available_functions(),
                temperature=float(self.temperature),
                max_iterations=5,
            )
            usage = self._empty_usage()
            raw_text = result.get("response", "") if isinstance(result, dict) else str(result)
            return raw_text.strip(), usage

        result = self.chat_with_tools(
            messages=messages,
            temperature=self.temperature,
            use_history=False,
        )
        usage = self.extract_usage_tokens(result)
        return self._extract_chat_text(result), usage
    
    def _run_tool_mode(
        self,
        transcript: str,
        system_prompt: str,
        user_prompt_template: str,
        temperature: float,
        max_iterations: int,
        template_params: dict,
        search_retry_limit: int = 2,
        max_search_terms: int = 5,
    ) -> tuple[str, list[str], bool, str, dict]:
        """Extract from a transcript using the search_transcript tool for evidence retrieval.

        The LLM uses a search tool to locate relevant transcript sections before
        performing extraction. Falls back to full transcript injection if searches
        fail after retry attempts.

        Args:
            transcript: Full transcript text.
            system_prompt: System instruction for the LLM.
            user_prompt_template: Jinja2 template for the user message.
            temperature: Sampling temperature.
            max_iterations: Maximum tool-calling loop iterations.
            template_params: Template variables.
            search_retry_limit: Number of times to retry failed searches.
            max_search_terms: Maximum search terms per tool call.

        Returns:
            Tuple of (raw_text, search_terms, found_any, matched_evidence, usage_totals).
        """
        search_terms: list[str] = []

        search_tool = make_search_transcript_tool(transcript, max_search_terms)
        found_any_match = False
        transcript_for_prompt = ""
        usage_totals = self._empty_usage()

        params = dict(template_params or {})
        params["TRANSCRIPT"] = (
            "[Transcript not included directly. "
            "Use the search_transcript tool to inspect relevant parts of it.]"
        )
        user_content = self._render_user_prompt(user_prompt_template, params)
        messages = self._build_messages(system_prompt, user_content)

        failed_search_rounds = 0
        self.register_tool(search_tool)

        try:
            for _ in range(max_iterations):
                self._log(f"Tool mode iteration {_+1}/{max_iterations}")
                result = self.chat_with_tools(
                    messages=messages,
                    temperature=temperature,
                    use_history=False,
                )
                usage_totals = self._add_usage(usage_totals, self.extract_usage_tokens(result))

                tool_calls = self._get_tool_calls(result)
                self._log(f"Iteration {_+1}: tool_calls = {tool_calls}")

                if not tool_calls:
                    failed_search_rounds += 1

                    if failed_search_rounds <= search_retry_limit:
                        self._reset_search_terms()
                        self._log(
                            f"No tool call made after failed search. "
                            f"Retrying search round {failed_search_rounds}/{search_retry_limit}."
                        )
                        messages.append({
                            "role": "user",
                            "content": (
                                """The previous search did not find relevant transcript evidence. 
                                Do not answer yet. Call the search_transcript tool again with different, more specific search
                                terms that are likely to appear verbatim in the transcript."""
                            ),
                        })
                        continue
    
                    self._log("No tool call made in tool mode. Falling back to full transcript.")
                    final_params = dict(template_params or {})
                    final_params["TRANSCRIPT"] = transcript

                    final_user_content = self._render_user_prompt(
                        user_prompt_template,
                        final_params,
                    )
                    final_messages = self._build_messages(system_prompt, final_user_content)

                    final_result = self.chat_with_tools(
                        messages=final_messages,
                        temperature=temperature,
                        use_history=False,
                    )
                    usage_totals = self._add_usage(usage_totals, self.extract_usage_tokens(final_result))
                    raw_text = self._extract_chat_text(final_result)
                    final_terms = dedupe_preserve_order(self._captured_search_terms)
                    
                    return raw_text, final_terms, found_any_match, transcript_for_prompt, usage_totals

                tc = tool_calls[0]
                args = json.loads(tc["arguments"])
                tool_name = tc["name"]
                tool_call_id = tc.get("call_id")

                
                
                tool_result = self._execute_tool(tool_name, args)
                parsed_tool_result = json.loads(tool_result)
                

                executed_queries = parsed_tool_result.get("queries", [])
                if isinstance(executed_queries, str):
                    executed_queries = [executed_queries]

                executed_queries = [
                    q.strip() for q in executed_queries
                    if isinstance(q, str) and q.strip()
                ]

                search_terms.extend(executed_queries)
                self._log(f"Accumulated search_terms from tool result: {search_terms}")

                found_any = bool(parsed_tool_result.get("found_any"))
                transcript_for_prompt = parsed_tool_result.get("transcript_for_prompt", transcript)

                if found_any:
                    found_any_match = True
                    final_params = dict(template_params or {})
                    final_params["TRANSCRIPT"] = transcript_for_prompt

                    final_user_content = self._render_user_prompt(
                        user_prompt_template,
                        final_params,
                    )
                    final_messages = self._build_messages(system_prompt, final_user_content)

                    final_result = self.chat_with_tools(
                        messages=final_messages,
                        temperature=temperature,
                        use_history=False,
                    )
                    raw_text = self._extract_chat_text(final_result)

                    search_terms = dedupe_preserve_order(search_terms)
                    self._log(f"Final deduped search_terms: {search_terms}")
                    return raw_text, search_terms, found_any_match, transcript_for_prompt, usage_totals

                failed_search_rounds += 1

                if failed_search_rounds <= search_retry_limit:
                    assistant_msg = {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": tc["arguments"],
                                },
                            }
                        ],
                    }
                    messages.append(assistant_msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": tool_result,
                    })
                    messages.append({
                        "role": "user",
                        "content": "The previous search did not find relevant transcript chunks. Try again with different, more targeted search terms."
                    })
                    continue

                self._log("Search retry limit reached. Falling back to full transcript.")

                final_params = dict(template_params or {})
                final_params["TRANSCRIPT"] = transcript

                final_user_content = self._render_user_prompt(
                    user_prompt_template,
                    final_params,
                )
                final_messages = self._build_messages(system_prompt, final_user_content)

                final_result = self.chat_with_tools(
                    messages=final_messages,
                    temperature=temperature,
                    use_history=False,
                )
                raw_text = self._extract_chat_text(final_result)

                search_terms = dedupe_preserve_order(self._captured_search_terms)
                self._log(f"Final deduped search_terms: {search_terms}")
                return raw_text, search_terms,found_any_match,transcript_for_prompt,usage_totals

            self._log("Max iterations reached. Falling back to full transcript.")

            final_params = dict(template_params or {})
            final_params["TRANSCRIPT"] = transcript

            final_user_content = self._render_user_prompt(
                user_prompt_template,
                final_params,
            )
            final_messages = self._build_messages(system_prompt, final_user_content)

            final_result = self.chat_with_tools(
                messages=final_messages,
                temperature=temperature,
                use_history=False,
            )
            raw_text = self._extract_chat_text(final_result)

            search_terms = dedupe_preserve_order(self._captured_search_terms)
            return raw_text, search_terms ,found_any,transcript_for_prompt

        finally:
            self.unregister_tool(search_tool.name)


    def run(
        self,
        transcript: str,
        system_prompt: str,
        user_prompt_template: str,
        response_format: type[BaseModel],
        token_threshold: int = 500,
        temperature: float = 0.2,
        max_iterations: int = 10,
        template_params: dict = None,
        search_retry_limit: int = 2,
        max_search_terms: int = 10,
        retrieval_only_evidence: bool = True
    ):
        """Execute transcript extraction with automatic mode selection.

        Chooses inline mode for short transcripts and tool mode for longer ones.
        Validates the LLM response against the provided Pydantic model.

        Args:
            transcript: Full transcript text to extract from.
            system_prompt: System instruction for the LLM.
            user_prompt_template: Jinja2 template for the user message.
            response_format: Pydantic BaseModel class for response validation.
            token_threshold: Token count above which tool mode is used.
            temperature: Sampling temperature.
            max_iterations: Max tool-calling loop iterations (tool mode only).
            template_params: Additional template variables.
            search_retry_limit: Retry count for failed searches in tool mode.
            max_search_terms: Max search terms per tool call.
            retrieval_only_evidence: If True, only use retrieved chunks as evidence.

        Returns:
            Tuple of (validated_model, search_terms, found_any, matched_evidence, usage_totals).
            validated_model is None if validation fails.
        """
        params = dict(template_params or {})
        use_tool_mode = self.count_tokens(transcript) > token_threshold
        self._reset_search_terms()

        # If category registry tools are registered, augment the system prompt
        # so the LLM knows to use them for any categorical/classification fields.
        effective_system_prompt = system_prompt
        has_category_tools = any(
            t.name in ("lookup_category", "register_category") for t in self.tools
        )
        if has_category_tools:
            category_instruction = (
                "\n\n[CATEGORY REGISTRY INSTRUCTIONS]\n"
                "You have access to a category registry via the lookup_category and "
                "register_category tools. The registry uses a two-level hierarchy: "
                "Broad Category (high-level grouping) → Sub-Category (specific reason).\n\n"
                "For ANY field that requires assigning a category, classification, or reason label:\n"
                "1. FIRST call lookup_category with your proposed sub-category name.\n"
                "2. If a match is found (found=true), use the EXACT broad_category and "
                "sub_category names returned — do not invent your own variant.\n"
                "3. If no match is found (found=false), call register_category with:\n"
                "   - broad_category: high-level grouping (e.g., 'Billing', 'Technical Support')\n"
                "   - broad_category_definition: what this broad category covers\n"
                "   - sub_category: specific reason (e.g., 'Payment Dispute', 'Signal Loss')\n"
                "   - sub_category_definition: what this sub-category specifically covers\n"
                "4. Then use the registered broad_category and sub_category in your extraction output.\n"
                "This ensures consistent categorization across all transcripts."
            )
            effective_system_prompt = (system_prompt or "") + category_instruction

        if not use_tool_mode:
            raw_text,usage_totals = self._run_inline_mode(
                transcript=transcript,
                system_prompt=effective_system_prompt,
                user_prompt_template=user_prompt_template,
                template_params=params,
            )
            search_terms = []
            found_any = False
            matched_evidence =""
        else:
            raw_text, search_terms,found_any,matched_evidence,usage_totals = self._run_tool_mode(
                transcript=transcript,
                system_prompt=effective_system_prompt,
                user_prompt_template=user_prompt_template,
                temperature=temperature,
                max_iterations=max_iterations,
                template_params=params,
                search_retry_limit=search_retry_limit,
                max_search_terms=max_search_terms,
            )

        if not raw_text:
            return None, search_terms, False,"",usage_totals

        raw_text = raw_text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1].rsplit("\n```", 1)[0].strip()

        try:
            validated = response_format.model_validate(self._safe_json_parse(raw_text))
            return validated, search_terms,found_any,matched_evidence,usage_totals
        
        except Exception as e:
            self._log(f"Validation failed: {e}")
            return None, search_terms,False,"",usage_totals
        

    def _resolve_response_format(self, row, response_format):
        """Resolve the response format for a given row.

        Supports static Pydantic model classes, callables that return a model
        class, and dict-based column/mapping configurations.

        Args:
            row: Dict representing a single data row.
            response_format: A Pydantic BaseModel class, callable, or config dict.

        Returns:
            Resolved Pydantic BaseModel class.

        Raises:
            ValueError: If the response_format configuration is invalid.
        """
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            return response_format

        if callable(response_format):
            return response_format(row)

        if isinstance(response_format, dict) and "column" in response_format and "mapping" in response_format:
            column = response_format["column"]
            mapping = response_format["mapping"]
            key = row.get(column)

            if key not in mapping:
                raise ValueError(f"No response format found for {column}={key!r}")

            return mapping[key]

        if isinstance(response_format, dict):
            raise ValueError(
                "If response_format is a dict, it must either be a direct mapping with "
                "response_format_column provided separately, or a config dict with "
                "'column' and 'mapping' keys."
            )

        return response_format

    def _make_row_agent(self):
        """Create a fresh TranscriptExtractionAgent instance for per-row processing.

        Propagates extra_tools so that shared tools (e.g., category registry)
        are available to each per-row agent instance.

        Returns:
            A new TranscriptExtractionAgent with the same configuration as self.
        """        
        return TranscriptExtractionAgent(
            api_key=self.api_key,
            assistant_id=getattr(self, "assistant_id", None),
            model=self.model,
            temperature=self.temperature,
            extra_tools=self.tools if self.tools else None,
            enable_tool_logging=self.enable_tool_logging,
        )

    def process_batch(
        self,
        transcript_column_name: str,
        rows: list[dict],
        prompt_template: str,
        response_format,
        template_params,
        system_prompt: str = None,
        max_workers: int = 3,
        verbose: bool = False,
        output_file: str = None,
        include_columns: list[str] = None,
        token_threshold: int = 500,
        max_retries: int = 3,
        retrieval_only_evidence:bool =True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Process a batch of transcript rows in parallel using per-row agent instances.

        Args:
            transcript_column_name: Column containing transcript text
            rows: List of dict rows
            prompt_template: Jinja2 template string
            response_format: Pydantic model class, callable, or mapping config
            template_params: Dict mapping template vars to row columns, or callable(row)->dict
            system_prompt: Optional system prompt
            max_workers: Number of parallel workers
            verbose: Reserved for compatibility
            output_file: Optional CSV path to append results as they complete
            include_columns: Optional extra columns to copy into output
            token_threshold: Transcript token threshold above which tool mode is used
            max_retries: Retry count per row
            retrieval_only_evidence=True
                Restrict output evidence to transcript chunks that were successfully 
                retrieved via the search_transcript tool. If retrieval does not return relevant matches, 
                the evidence field is left blank even if the main extraction falls back to the full transcript.
            retrieval_only_evidence=False
                Prefer retrieved transcript chunks when available, 
                but allow fallback evidence generation when retrieval does not produce relevant matches.
                That wording is important because it clarifies:
                extraction fallback can still happen
                evidence can remain blank intentionally
                evidence policy is separate from inference policy
                            **kwargs: Passed through to run()

        Returns:
            List of result dicts
        """
        def _process(row):
            
            sessionid = row.get("AGENTRECORDINGSESSIONID")
            print("Processing Session ID:", sessionid)

            for attempt in range(1, max_retries + 1):
                search_terms = []

                try:
                    row_agent = self._make_row_agent()

                    params = (
                        template_params(row)
                        if callable(template_params)
                        else {k: row.get(v) for k, v in template_params.items()}
                    )

                    transcript = row.get(transcript_column_name)

                    resolved_response_format = self._resolve_response_format(
                        row=row,
                        response_format=response_format,
                    )

                    result, search_terms,found_any,matched_evidence,usage_totals = row_agent.run(
                        transcript=transcript,
                        system_prompt=system_prompt or "",
                        user_prompt_template=prompt_template,
                        response_format=resolved_response_format,
                        token_threshold=token_threshold,
                        template_params=params,
                        retrieval_only_evidence=retrieval_only_evidence,
                        **kwargs,
                    )

                    

                    if result is None:
                        raise ValueError("extraction returned None")

                    answer = {
                        "AGENTRECORDINGSESSIONID": sessionid,
                        "PROCESS_STATUS": "TRUE"
                    }

                    if include_columns:
                        answer.update({col: row.get(col) for col in include_columns})

                    model_data = result.model_dump()
                    flat_data = self.flatten_dict(model_data)
                    answer.update(flat_data)

                    
                    if found_any and matched_evidence:
                        evidence_chunk = matched_evidence

                    elif not retrieval_only_evidence and search_terms:
                        evidence_chunk = find_transcript_chunk_merged(
                            transcript,
                            search_terms,
                            context_lines=3,
                            formatted=True,
                            include_search_term_label=True,
                        )
                    else:
                        evidence_chunk = ""

                    answer["RETRIEVAL_FOUND_ANY"] = found_any
                    answer["USED_MATCHED_EVIDENCE"] = bool(found_any and matched_evidence)
                    answer["TRANSCRIPT_TOKENS"] = self.count_tokens(transcript)
                    answer["EVIDENCE_CHUNK"] = evidence_chunk
                    answer["CHUNK_TOKENS"] = self.count_tokens(evidence_chunk) if evidence_chunk else 0
                    answer["ATTEMPTED_SEARCH_TERMS"] = " | ".join(search_terms) if search_terms else ""
                    answer["LLM_INPUT_TOKENS"] = usage_totals.get("input_tokens", 0)
                    answer["LLM_OUTPUT_TOKENS"] = usage_totals.get("output_tokens", 0)
                    answer["LLM_TOTAL_TOKENS"] = usage_totals.get("total_tokens", 0)

                    

                    return answer

                except Exception as e:
                    print(f"Attempt {attempt}/{max_retries} failed for UCID {sessionid}: {e}")
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)

            return {
                "AGENTRECORDINGSESSIONID": sessionid,
                "PROCESS_STATUS": "FALSE",
                "ATTEMPTED_SEARCH_TERMS": " | ".join(search_terms) if search_terms else "",
                "TRANSCRIPT_TOKENS": "",
                "CHUNK_TOKENS": "",
                "TRANSCRIPT_TOKENS": "",
                "CHUNK_TOKENS": "",
                "LLM_INPUT_TOKENS": 0,
                "LLM_OUTPUT_TOKENS": 0,
                "LLM_TOTAL_TOKENS": 0,
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
                    sessionid = result.get("AGENTRECORDINGSESSIONID")

                    if sessionid in seen_agentrecordingsessionid:
                        print(f"Skipping duplicate result for SessionID {sessionid}")
                        continue

                    seen_agentrecordingsessionid.add(sessionid)
                    results.append(result)

                    print(
                        f"Completed {i + 1}/{len(rows)}: "
                        f"AGENTRECORDINGSESSIONID {result.get('AGENTRECORDINGSESSIONID')} | "
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
                        sessionid = result.get("AGENTRECORDINGSESSIONID")

                        if sessionid in seen_agentrecordingsessionid:
                            print(f"Skipping duplicate result for SessionID {sessionid}")
                            continue

                        seen_agentrecordingsessionid.add(sessionid)
                        results.append(result)

                        print(
                            f"Completed {i + 1}/{len(rows)}: "
                            f"AGENTRECORDINGSESSIONID {result.get('AGENTRECORDINGSESSIONID')} | "
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
        df = pd.DataFrame(results)

        if 'UCID' in df.columns:
            df['UCID'] = df['UCID'].astype(str)

        return df

    