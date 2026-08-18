"""
agents/base_tool_agent.py

BaseToolAgent

Reusable base class for tool-enabled agents built on SpectrumClient.

Features:
- Anthropic-style tool registration via _setup_tools()
- Centralized _execute_tool() dispatcher
- Pydantic validation of tool inputs
- Automatic conversion of Tool objects into API tool schemas
- Compatible with SpectrumClient.chat(... tools=..., available_functions=...)

Subclass responsibilities:
- set agent-specific prompts/state
- optionally override _setup_tools()
- implement run()
"""

"""
agents/base_tool_agent.py

BaseToolAgent

Reusable base class for tool-enabled agents built on SpectrumClient.

Features:
- Anthropic-style tool registration via _setup_tools()
- Supports extra_tools passed at initialization
- Centralized _execute_tool() dispatcher
- Pydantic validation of tool inputs
- Automatic conversion of Tool objects into API tool schemas
- Built-in logging for tool registration, execution, and chat-with-tools calls
"""

from typing import Dict, List, Any, Optional
from pydantic import ValidationError

from spectrum_client import SpectrumClient
from tools.utils.build_tool import Tool
import json
from jinja2 import Template

class BaseToolAgent(SpectrumClient):
    """Reusable base class for tool-enabled agents built on SpectrumClient.

    Provides centralized tool registration, Pydantic-validated tool dispatch,
    automatic conversion of Tool objects into OpenAI-compatible schemas, and
    built-in logging for tool execution and chat calls.

    Subclasses should override ``_setup_tools()`` to register built-in tools
    and implement ``run()`` for their specific workflow.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        assistant_id: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        temperature: str = "0.4",
        extra_tools: Optional[List[Tool]] = None,
        enable_tool_logging: bool = True,
    ):
        """Initialize the base tool agent.

        Args:
            api_key: API key for the underlying LLM service.
            assistant_id: Optional assistant identifier for the service.
            model: Model name to use for inference.
            temperature: Sampling temperature (passed as string).
            extra_tools: Additional Tool objects to register at init time.
            enable_tool_logging: Whether to print debug log messages.
        """
        self.enable_tool_logging = enable_tool_logging

        super().__init__(
            model=model,
            api_key=api_key,
            assistant_id=assistant_id,
            temperature=temperature,
        )

        self.tools: List[Tool] = []
        self._setup_tools()

        if extra_tools:
            for tool in extra_tools:
                self.register_tool(tool)

        self._log(f"Initialized with {len(self.tools)} tool(s)")

    def _setup_tools(self):
        """Override in subclasses to register built-in agent tools.

        Default implementation initializes an empty tool list.
        """
        self.tools = []

    def _render_user_prompt(self, template_str: str, params: dict) -> str:
        """Render a Jinja2 template string with the given parameters.

        Args:
            template_str: Jinja2 template string.
            params: Dictionary of template variables.

        Returns:
            Rendered string.
        """
        return Template(template_str).render(**params)

    def _build_messages(self, system_prompt: str, user_content: str) -> list[dict]:
        """Build a standard system + user message list for the LLM.

        Args:
            system_prompt: System-level instruction text.
            user_content: User-level prompt content.

        Returns:
            List of message dicts ready for the chat API.
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _log(self, message: str):
        """Print a debug log message prefixed with the agent class name.

        Args:
            message: Log message to print.
        """
        if self.enable_tool_logging:
            print(f"[{self.__class__.__name__}] {message}")

    def register_tool(self, tool: Tool):
        """Register a single Tool instance for use in chat_with_tools calls.

        Args:
            tool: Tool object to register.
        """
        self.tools.append(tool)
        self._log(f"Registered tool: {tool.name}")

    def register_tools(self, tools: List[Tool]):
        """Register multiple Tool instances at once.

        Args:
            tools: List of Tool objects to register.
        """
        for tool in tools:
            self.register_tool(tool)

    def _tool_schemas(self) -> list[dict]:
        """Convert all registered tools to OpenAI-compatible function schemas.

        Returns:
            List of tool schema dicts.
        """
        schemas = [tool.to_openai_schema() for tool in self.tools]
        self._log(f"Prepared {len(schemas)} tool schema(s)")
        return schemas

    def _available_functions(self) -> dict:
        """Build a name-to-callable mapping for all registered tools.

        Each callable validates input via Pydantic and dispatches to _execute_tool.

        Returns:
            Dict mapping tool name to its runner function.
        """
        funcs = {}

        for tool in self.tools:
            def make_runner(tool_name):
                return lambda **kwargs: self._execute_tool(tool_name, kwargs)

            funcs[tool.name] = make_runner(tool.name)

        self._log(f"Prepared {len(funcs)} available function(s)")
        return funcs

    def _get_tool(self, tool_name: str) -> Tool | None:
        """Look up a registered tool by name.

        Args:
            tool_name: Name of the tool to find.

        Returns:
            The matching Tool object, or None if not found.
        """
        return next((t for t in self.tools if t.name == tool_name), None)

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Validate input and execute a registered tool by name.

        Args:
            tool_name: Name of the tool to execute.
            tool_input: Raw keyword arguments for the tool.

        Returns:
            Tool result as a string, or an error message string on failure.
        """
        self._log(f"Executing tool: {tool_name}")
        self._log(f"Tool input: {tool_input}")

        try:
            tool = self._get_tool(tool_name)
            if not tool:
                msg = f"Unknown tool: {tool_name}"
                self._log(msg)
                return msg

            validated = tool.input_model.model_validate(tool_input)
            self._log(f"Validated input for tool: {tool_name}")

            result = tool.func(**validated.model_dump())

            preview = str(result)
            if len(preview) > 500:
                preview = preview[:500] + "...[truncated]"

            self._log(f"Tool result from {tool_name}: {preview}")
            return result

        except ValidationError as e:
            msg = f"Invalid input for {tool_name}: {e}"
            self._log(msg)
            return msg

        except Exception as e:
            msg = f"Error executing {tool_name}: {str(e)}"
            self._log(msg)
            return msg

    def _extract_chat_text(self, result) -> str:
        """Extract the assistant's text content from a chat result.

        Handles both dict-style (with 'response' key) and raw results.

        Args:
            result: Raw chat response from the LLM service.

        Returns:
            Stripped text string.
        """
        if isinstance(result, dict) and "response" in result:
            return str(result["response"]).strip()

        return self._extract_text(result).strip()

    def _extract_usage_tokens(self, result: dict) -> dict:
        """Extract token usage statistics from a chat result.

        Args:
            result: Raw chat response dict.

        Returns:
            Dict with input_tokens, output_tokens, and total_tokens.
        """
        return self.client.extract_usage_tokens(result)

    def chat_with_tools(
        self,
        messages: list,
        temperature: float = 0.6,
        use_history: bool = False,
        tool_choice: str = "auto",
        response_format=None,
    ):
        """Send messages to the LLM with registered tools available for function calling.

        Automatically includes all registered tool schemas and their executable
        functions so the model can invoke tools during the conversation.

        Args:
            messages: List of message dicts for the conversation.
            temperature: Sampling temperature for the model.
            use_history: Whether to include prior conversation history.
            tool_choice: Tool selection strategy ('auto', 'none', or a specific tool name).
            response_format: Optional response format constraint.

        Returns:
            Raw chat response from the underlying SpectrumClient.
        """
        tool_schemas = self._tool_schemas()

        # print("=== TOOL SCHEMAS ===")
        # print(json.dumps(tool_schemas, indent=2, default=str))

        available_functions = self._available_functions()

        self._log(
            f"Calling chat_with_tools with {len(messages) if messages else 0} message(s), "
            f"{len(tool_schemas)} tool(s), tool_choice={tool_choice}"
        )

        kwargs = {
            "messages": messages,
            "temperature": temperature,
            "use_history": use_history,
            "response_format": response_format,
        }

        if tool_schemas:
            kwargs["tools"] = tool_schemas
            kwargs["tool_choice"] = tool_choice

        if available_functions:
            kwargs["available_functions"] = available_functions

        return self.chat(**kwargs)