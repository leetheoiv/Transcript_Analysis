"""
agents/schema_generator_agent.py

SchemaGeneratorAgent — takes an approved PromptModel and the analyst's original question,
produces a Pydantic v2 BaseModel class definition, writes it to file, and validates it imports cleanly.

Returns a (SchemaGeneratorResult, Path) tuple.
"""



from pathlib import Path
from jinja2 import Template
from agents.BaseAgent import BaseToolAgent
from prompts.agent_prompts.schema_generator_PROMPT import SYSTEM_PROMPT, USER_PROMPT
from data_models.agent_models.SchemaAgent_datamodel import SchemaGeneratorResult
from tools.WRITE_TOOLS.code_writer import write_code


class SchemaGeneratorAgent(BaseToolAgent):
    """Agent that generates Pydantic v2 BaseModel class definitions from prompt output formats.

    Takes an approved PromptModel's output_format and the analyst's original question,
    produces valid Python code for a Pydantic model, writes it to file, and validates
    that it imports cleanly.
    """

    def __init__(
        self,
        api_key: str = None,
        assistant_id: str = None,
        model: str = "gpt-4.1-mini",
        temperature: str = "0.4",
        extra_tools=None,
        enable_tool_logging: bool = True,
    ):
        """Initialize the schema generator agent.

        Args:
            api_key: API key for the LLM service.
            assistant_id: Optional assistant identifier.
            model: Model name for inference.
            temperature: Sampling temperature as string.
            extra_tools: Additional Tool objects to register.
            enable_tool_logging: Whether to enable debug logging.
        """
        self.system_prompt = SYSTEM_PROMPT
        self.user_prompt = USER_PROMPT

        super().__init__(
            model=model,
            api_key=api_key,
            assistant_id=assistant_id,
            temperature=temperature,
            extra_tools=extra_tools,
            enable_tool_logging=enable_tool_logging,
        )

    def _setup_tools(self):
        """
        SchemaGeneratorAgent does not require tools by default,
        but inherits the tool framework for consistency and future extensibility.
        """
        self.tools = []

    def _format_user_prompt(self, output_format: dict, user_input: str) -> str:
        """Render the user prompt template with output format and user input.

        Args:
            output_format: The approved prompt output_format dict.
            user_input: The analyst's original plain-language request.

        Returns:
            Rendered user prompt string.
        """
        return Template(self.user_prompt).render(
            input=user_input,
            output_format=output_format,
        )

    def run(
        self,
        output_format: dict,
        user_input: str,
        output_dir: str,
        max_retries: int = 2,
    ) -> tuple[SchemaGeneratorResult, Path]:
        """
        Generate a Pydantic model class from the approved prompt's output_format and write it to file.

        :param output_format: Approved prompt output_format dict from PromptGeneratorAgent
        :param user_input: The analyst's original plain-language request
        :param output_dir: Directory to write the generated model file into
        :return: (SchemaGeneratorResult, Path)
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._format_user_prompt(output_format, user_input)},
        ]

        # Step 1: Get a valid SchemaGeneratorResult from the LLM
        schema_result = None

        for attempt in range(max_retries + 1):
            result = self.chat_with_tools(
                messages=messages,
                use_history=False,
            )
            text = self._extract_chat_text(result)

            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("\n```", 1)[0].strip()

            try:
                schema_result = SchemaGeneratorResult.model_validate(self._safe_json_parse(text))
                break

            except Exception as e:
                if attempt == max_retries:
                    raise

                self._log(f"Validation failed on attempt {attempt + 1}: {e}")

                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your response failed validation:\n{e}\n\n"
                        "Return valid JSON only, no markdown fences."
                    )
                })

        # Step 2: Write to file, retry with LLM fix on import error
        code = schema_result.code

        for attempt in range(max_retries + 1):
            try:
                path = write_code(code, schema_result.model_name, output_dir)
                self._log(f"Successfully wrote schema model to: {path}")

                print(f"\n{'=' * 70}")
                print(f"Generated schema: {schema_result.model_name}  ->  {path}")
                print(f"{'=' * 70}")
                print(schema_result.code)
                print(f"{'=' * 70}\n")

                return schema_result, path

            except Exception as e:
                if attempt == max_retries:
                    raise

                self._log(f"Import failed on attempt {attempt + 1}: {e}. Asking LLM to fix code.")

                messages.append({"role": "assistant", "content": code})
                messages.append({
                    "role": "user",
                    "content": (
                        f"The generated code failed to import:\n{e}\n\n"
                        "Fix the code field in your JSON and return the full corrected JSON only, no markdown fences."
                    )
                })

                result = self.chat_with_tools(
                    messages=messages,
                    temperature=self.temperature,
                    use_history=False,
                )
                text = self._extract_chat_text(result)

                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("\n```", 1)[0].strip()

                schema_result = SchemaGeneratorResult.model_validate(self._safe_json_parse(text))
                code = schema_result.code