import re
import csv
import json
import tomllib
import traceback
from io import StringIO, BytesIO
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
from concurrent.futures import ThreadPoolExecutor, as_completed
from jinja2 import Template
import requests
from pydantic import BaseModel, Field
import streamlit as st
import os
import sys
import time
import random
import tiktoken
import datetime as dt
from pydantic import ValidationError





class SpectrumClient:
    """SpectrumGPT API client for sending prompts and handling responses."""

    def __init__(self, url: str = None, api_key: str = None, assistant_id: str = None,model='gpt-4.1-mini',temperature:str='0.4'):
        self.model = model
        self.messages = [] # stores short-term memory message history
        self.tool_defs = []
        self.available_functions = {}

        if url and api_key and assistant_id:
            self.url = url
            self.headers = {"api-key": api_key, "assistant-id": assistant_id}
            return

        secrets_path = Path(__file__).parent / "config" / "secrets.toml"
        if not secrets_path.exists():
            raise FileNotFoundError("secrets.toml not found")

        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)

        assistant_id_map = {
            "gpt-4.1-mini": {
                "0.4": secrets["GPT_4_1_TEMP_04_MINI_ASSISTANT_ID"],
                "0.1": secrets["GPT_4_1_TEMP_01_MINI_ASSISTANT_ID"],
            },
            "gpt-5-mini": secrets["GPT_5_MINI_ASSISTANT_ID"],
        }

        if model not in assistant_id_map:
            raise ValueError(f"Unsupported model: {model!r}. Choose from {list(assistant_id_map)}")

        entry = assistant_id_map[model]
        if isinstance(entry, dict):
            temp_key = str(temperature)
            if temp_key not in entry:
                raise ValueError(f"Unsupported temperature {temperature!r} for model {model!r}.")
            assistant_id = entry[temp_key]
        else:
            assistant_id = entry

        self.url = secrets["url"]
        self.headers = {
            "api-key": secrets["API_KEY"],
            "assistant-id": assistant_id,
        }

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        # Coerce non-string input (e.g. pandas NaN floats, None) to a safe string.
        # tiktoken.encode on a float raises "argument of type 'float' is not iterable".
        if not isinstance(text, str):
            if text is None:
                text = ""
            else:
                try:
                    import math
                    if isinstance(text, float) and math.isnan(text):
                        text = ""
                    else:
                        text = str(text)
                except Exception:
                    text = str(text)
        enc = tiktoken.encoding_for_model(self.model)
        return len(enc.encode(text))

    def extract_usage_tokens(self, result: dict) -> dict:
        """
        Safely extract token usage information from a raw API response.
        """
        if not isinstance(result, dict):
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

        usage = result.get("usage") or {}

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens")

        if total_tokens is None:
            total_tokens = input_tokens + output_tokens

        return {
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "total_tokens": int(total_tokens or 0),
        }

    def register_tool(self, schema: dict, name: str, func):
        self.tool_defs.append(schema)
        self.available_functions[name] = func

    def _build_payload(self, messages: list, temperature: float, tools: list = None,
                   tool_choice: str = "auto", response_format=None) -> dict:
        """Assemble the request payload."""
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        payload = {
            "model": self.model,
            "input": prompt,
            "messages": messages,
            "temperature": temperature,
            "top_p": 1,
        }

        if tools:
            strict_tools = []

            for tool in tools:
                tool_copy = dict(tool)
                if "parameters" in tool_copy and isinstance(tool_copy["parameters"], dict):
                    tool_copy["parameters"] = self._make_strict_tool_schema(tool_copy["parameters"])
                strict_tools.append(tool_copy)

            payload["tools"] = strict_tools
            payload["tool_choice"] = tool_choice

        if response_format:
            raw_schema = response_format.model_json_schema()
            strict_schema = self._make_strict_schema(raw_schema)

            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "strict": True,
                    "schema": strict_schema
                }
            }

        return payload

    def _safe_json_parse(self, text: str) -> dict:

        """Parse JSON, repairing invalid escapes and malformed output from the model.

        Strategy (most reliable first):
          1. Strict json.loads.
          2. Repair invalid backslash escapes, then json.loads.
          3. json_repair for structurally broken JSON.
          4. Only as a last resort, when the text is clearly NOT a JSON
             object/array, fall back to a crude line-by-line key:value parse.

        The line-by-line fallback is deliberately gated behind the structured
        parsers. Running it first would mangle valid-but-slightly-broken JSON
        (e.g. a single-line ``{...}`` blob) into a garbage single-key dict by
        splitting on the first colon.
        """
        if not isinstance(text, str):
            # Already parsed upstream (dict/list) — hand it straight back.
            return text

        stripped = text.strip()

        # 1. Strict parse.
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # 2. Repair invalid escapes (e.g. \' produced by str() of a Python dict)
        #    and drop stray backslashes that aren't valid JSON escapes.
        fixed = re.sub(r"\\'", "'", stripped)
        fixed = re.sub(r'\\([^"\\\/bfnrtu])', r'\1', fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # 3. Structural repair for JSON-shaped payloads.
        looks_like_json = stripped.startswith(("{", "["))
        if looks_like_json:
            repaired = self._try_json_repair(fixed)
            if isinstance(repaired, (dict, list)):
                return repaired

        # 4. Last-resort line-by-line parse — only for clearly non-JSON text.
        #    Never applied to text that starts with { or [ so we don't corrupt
        #    a mangled-but-recoverable JSON object into a single bogus key.
        if not looks_like_json:
            result = {}
            lines = [l.strip() for l in stripped.splitlines() if l.strip()]
            for i, line in enumerate(lines):
                if line.endswith(':'):
                    key = line[:-1].strip()
                    value = lines[i + 1].strip() if i + 1 < len(lines) else ""
                    if key and value:
                        result[key] = value
                elif ':' in line:
                    key, _, value = line.partition(':')
                    key, value = key.strip(), value.strip()
                    if key and value:
                        result[key] = value
            if result:
                return result

        # 5. Absolute fallback: let json_repair try even on non-JSON-looking
        #    text so callers still get a best-effort dict instead of a crash.
        repaired = self._try_json_repair(fixed)
        if repaired is not None:
            return repaired

        # json_repair unavailable and nothing else worked. Raise a clear error
        # rather than silently returning garbage that fails downstream validation.
        raise ValueError(
            "Could not parse model output as JSON and json_repair is unavailable. "
            "Install it with: pip install json-repair"
        )

    def _try_json_repair(self, text: str):
        """Attempt structural JSON repair, tolerating a missing json_repair package.

        Returns the parsed object on success, or None if repair is unavailable
        or fails. Never raises.
        """
        try:
            from json_repair import repair_json
        except ImportError:
            return None
        try:
            return json.loads(repair_json(text))
        except Exception:
            return None

    def flatten_dict(self,d, parent_key="", sep="__"):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, json.dumps(v, ensure_ascii=False)))
            else:
                items.append((new_key, v))
        return dict(items)

    def _make_strict_schema(self, schema: dict) -> dict:
        if not isinstance(schema, dict):
            return schema

        if "anyOf" in schema:
            non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
            if len(non_null) == 1:
                schema.clear()
                schema.update(non_null[0])

        if schema.get("type") == "object":
            schema["additionalProperties"] = False
            props = schema.get("properties", {})
            schema["required"] = list(props.keys())
            for prop in props.values():
                self._make_strict_schema(prop)

        elif schema.get("type") == "array":
            items = schema.get("items")
            if items:
                self._make_strict_schema(items)

        for key in ("$defs", "definitions"):
            if key in schema and isinstance(schema[key], dict):
                for sub in schema[key].values():
                    self._make_strict_schema(sub)

        return schema

    def _make_strict_tool_schema(self, schema: dict) -> dict:
        if not isinstance(schema, dict):
            return schema

        schema = dict(schema)

        if "anyOf" in schema:
            non_null = [
                s for s in schema["anyOf"]
                if not (isinstance(s, dict) and s.get("type") == "null")
            ]
            if len(non_null) == 1:
                schema.clear()
                schema.update(non_null[0])

        if schema.get("type") == "object":
            schema["additionalProperties"] = False

            props = schema.get("properties", {})
            for key, prop in props.items():
                props[key] = self._make_strict_tool_schema(prop)

            for key in ("$defs", "definitions"):
                if key in schema and isinstance(schema[key], dict):
                    for def_key, sub in schema[key].items():
                        schema[key][def_key] = self._make_strict_tool_schema(sub)

        elif schema.get("type") == "array":
            items = schema.get("items")
            if items:
                schema["items"] = self._make_strict_tool_schema(items)

        return schema

    def _post(self, payload: dict, max_retries: int = 6) -> dict:
        """Send a single request with retry/backoff for rate limits and transient errors."""
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    self.url,
                    headers=self.headers,
                    json=payload,
                    timeout=180
                )

                if response.status_code == 429:
                    print(f"429 body: {response.text}")
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        sleep_time = float(retry_after)
                    else:
                        try:
                            
                            reset_str = json.loads(response.text)["error"]["message"].split("Limit resets at: ")[1].strip()
                            reset_time = dt.datetime.strptime(reset_str, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=dt.timezone.utc)
                            sleep_time = max((reset_time - dt.datetime.now(dt.timezone.utc)).total_seconds() + 2, 1)
                        except Exception as parse_err:
                            print(f"Reset time parse failed: {parse_err}")
                            sleep_time = min(2 ** attempt, 60) + random.uniform(0, 1)
                    print(f"429 received. Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                    continue

                if response.status_code in (500, 502, 503, 504):
                    sleep_time = min(2 ** attempt, 60) + random.uniform(0, 1)
                    print(f"Server error {response.status_code}. Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                    continue

                if not response.ok:
                    print(f"Error {response.status_code}: {response.text}")

                response.raise_for_status()
                return response.json()

            except requests.RequestException as e:
                last_error = e
                if attempt == max_retries:
                    break

                sleep_time = min(2 ** attempt, 60) + random.uniform(0, 1)
                print(f"Request failed ({e}). Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)

        raise last_error

    def _extract_text(self, result: dict) -> str:
        """Pull the assistant's text out of a Responses API result."""
        output = result.get("output", [])
        for block in output:
            if block.get("type") == "message":
                for content_block in block.get("content", []):
                    if content_block.get("type") in ("text", "output_text"):
                        return content_block.get("text", "")
            if block.get("type") in ("text", "output_text"):
                return block.get("text", "")
        return ""

    def _get_tool_calls(self, result: dict) -> list:
        """Return all function_call blocks from a Responses API result."""
        return [
            block for block in result.get("output", [])
            if block.get("type") == "function_call"
        ]

    # -----------------------------------------------------------------------
    # Core multi-step agent loop
    # -----------------------------------------------------------------------

    def run_agent(
        self,
        messages: list,
        tools: list,
        available_functions: dict,
        temperature: float = 0.2,
        response_format=None,
        max_iterations: int = 10,
    ) -> dict:
        """
        Run the full agent loop until the model stops calling tools.

        The loop:
          1. POST current messages to the API.
          2. If the response contains function_call blocks → execute each tool,
             append results to messages, go to 1.
          3. If the response is a plain text message → return it.

        Args:
            messages:            Conversation so far (list of role/content dicts).
            tools:               OpenAI-style tool schema list.
            available_functions: Dict mapping tool name → callable.
            temperature:         Sampling temperature.
            response_format:     Optional Pydantic model for structured output.
            max_iterations:      Safety cap on the number of API round-trips.

        Returns:
            {
                "response":      str   — final assistant text,
                "tool_calls":    list  — log of every tool call made,
                "iterations":    int   — number of API round-trips used,
            }
        """
        # Work on a copy so callers keep their original message list clean
        working_messages = list(messages)
        tool_call_log = []
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            print(f"\n[Agent] Iteration {iteration}")

            # --- Call the API -----------------------------------------------
            payload = self._build_payload(
                working_messages, temperature, tools, "auto", response_format
            )
            result = self._post(payload)

            # --- Check for tool calls ---------------------------------------
            tool_calls = self._get_tool_calls(result)

            if not tool_calls:
                # No tool calls → model is done, extract final text
                final_text = self._extract_text(result)
                print(f"[Agent] Done after {iteration} iteration(s).")
                return {
                    "response": final_text,
                    "tool_calls": tool_call_log,
                    "iterations": iteration,
                }

            # --- Execute every tool call in this turn -----------------------
            for tool_call in tool_calls:
                function_name = tool_call["name"]
                function_args = json.loads(tool_call["arguments"])
                call_id       = tool_call["call_id"]

                print(f"[Agent] Calling tool: {function_name}({function_args})")

                # Execute
                fn = available_functions.get(function_name)
                if fn is None:
                    tool_result = json.dumps({
                        "success": False,
                        "error": f"Unknown tool: '{function_name}'"
                    })
                else:
                    raw = fn(**function_args)
                    tool_result = (
                        raw if isinstance(raw, str) else json.dumps(raw)
                    )

                print(f"[Agent] Tool result: {tool_result[:200]}")

                # Log it
                tool_call_log.append({
                    "tool": function_name,
                    "args": function_args,
                    "result": tool_result,
                })

                # Feed result back into the conversation
                working_messages.append({
                    "role": "assistant",
                    "content": f"Called {function_name}",
                })
                working_messages.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": call_id,
                    "name": function_name,
                })

        # Safety cap hit
        print(f"[Agent] WARNING: max_iterations ({max_iterations}) reached.")
        return {
            "response": "Agent reached maximum iteration limit without a final answer.",
            "tool_calls": tool_call_log,
            "iterations": iteration,
        }

    # -----------------------------------------------------------------------
    # Original chat() — unchanged for backward compatibility
    # -----------------------------------------------------------------------

    def chat(self, messages: list = None, prompt: str = None, temperature: float = 0.1,
             tools: list = None, tool_choice: str = "auto",
             available_functions: dict = None, response_format=None,
             use_history: bool = False) -> str:
        """Send a prompt to SpectrumGPT API and return the generated text response.

        Args:
            messages: List of message dicts with 'role' and 'content' keys (OpenAI format)
            prompt: Simple string prompt (legacy support)
            temperature: Controls randomness (0.0-1.0, lower = more deterministic)
            tools: List of tool definitions for function calling
            tool_choice: 'auto' or 'required' to control tool usage
            available_functions: Dict mapping function names to callable functions

        Returns:
            Generated text response from the AI model
        """
         # Normalize prompt into messages
        if messages is None and prompt is not None:
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(messages, dict):
            messages = [messages]
        elif isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Ensure self.messages is a list
        if not isinstance(self.messages, list):
            self.messages = []
        # -----------------------------------------------------------------------
        # Convert messages to a payload for the LLM
        # -----------------------------------------------------------------------
        if use_history:
            if messages:
                self.messages.extend(messages)
            messages = self.messages

        # Always build payload here
        payload = self._build_payload(
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format
        )

        # Outputs the information sent to the LLM for more visibility
        # print("=== OUTGOING PAYLOAD ===")
        # print(json.dumps(payload, indent=2, default=str))
        
        result = self._post(payload)
        

        # -----------------------------------------------------------------------
        # If the response contains a function call, execute it and return the result
        # -----------------------------------------------------------------------
        if tools is None:
            tools = getattr(self, "tools", None)

        if available_functions is None:
            available_functions = getattr(self, "available_functions", None)
                    
        if available_functions and result['output'][0].get('type') == 'function_call':
            tool_call = result['output'][0]
            function_name = tool_call['name']
            function_args = json.loads(tool_call['arguments'])

            function_to_call = available_functions[function_name]
            function_result = function_to_call(**function_args)



            try:
                parsed_result = json.loads(function_result) if isinstance(function_result, str) else function_result
            except (json.JSONDecodeError, TypeError):
                parsed_result = function_result

            # Prints the parsed function call results
            # print("FUNCTION RESULT PARSED:", parsed_result)

            messages.append({'role': 'assistant', 'content': f"Called {function_name}"})
            messages.append({'role': 'tool', 'content': function_result if isinstance(function_result, str) else json.dumps(function_result),
                             'tool_call_id': tool_call['call_id'], 'name': function_name})

            final_response = self.chat(messages=messages, tools=tools, temperature=temperature,
                                       available_functions=available_functions,
                                       response_format=response_format)



            if isinstance(final_response, dict) and 'output' in final_response:
                ai_text = final_response['output'][0]['content'][0]['text']
            else:
                ai_text = final_response

            if response_format:
                text = ai_text.strip()
                if text.startswith('```'):
                    text = text.split('\n', 1)[1].rsplit('\n```', 1)[0]
                try:
                    json_data = json.loads(text)
                    return response_format(**json_data)
                except:
                    pass

            return {
                'response': ai_text,
                'function_result': parsed_result,
                'function_name': function_name
            }

        return result
    