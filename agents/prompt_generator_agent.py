"""
agents/prompt_generator_agent.py

PromptGeneratorAgent — conversational agent for building Jinja2 extraction prompt templates.

Takes a plain-language analyst question and produces a structured PromptModel containing a
system prompt, a user prompt (with {{TRANSCRIPT}} injection point and embedded JSON output
format), and an output_format dict that the SchemaGeneratorAgent uses to build the Pydantic
model for ask_transcript_questions.

Supports two interaction modes within a single open conversation:
- generate: user requests a new prompt or modification to an existing one — returns a PromptModel.
- converse: user asks a question or gives feedback without requesting a change — returns plain text.

Intent is classified automatically on each call using the last assistant message as context,
so follow-up questions ("does this include a response format?") and modification requests
("add a sentiment field") are routed correctly without the caller managing state.

Call reset() to clear conversation history before starting a new analysis question.
"""

from pathlib import Path
from jinja2 import Template

from agents.BaseAgent import BaseToolAgent
from tools.utils.build_tool import Tool
from tools.RAG_TOOLS.helper_functions import DocumentSearchInput, DEFAULT_RAG_FOLDER
from tools.RAG_TOOLS.search_documents import search_documents

from prompts.agent_prompts.prompt_generator_PROMPT import SYSTEM_PROMPT, USER_PROMPT
from prompts.agent_prompts.user_intent_prompt import INTENT_PROMPT
from data_models.agent_models.PromptGeneratorAgent_datamodel import PromptModel
from data_models.IntentModel import IntentModel


class PromptGeneratorAgent(BaseToolAgent):
    """Conversational agent for building Jinja2 extraction prompt templates.

    Takes a plain-language analyst question and produces a structured PromptModel
    containing system prompt, user prompt, and output format. Supports two
    interaction modes: 'generate' for creating/modifying prompts, and 'converse'
    for answering questions about the current prompt.
    """

    def __init__(
        self,
        api_key=None,
        assistant_id=None,
        model="gpt-4.1-mini",
        temperature="0.4",
        PROJECT_ROOT = Path(__file__).resolve().parent.parent,
        rag_folder = None
    ):
        """Initialize the prompt generator agent.

        Args:
            api_key: API key for the LLM service.
            assistant_id: Optional assistant identifier.
            model: Model name for inference.
            temperature: Sampling temperature as string.
            PROJECT_ROOT: Root path for resolving relative paths.
            rag_folder: Path to the RAG knowledge base folder (defaults to knowledge_base/).
        """
        self.system_prompt = SYSTEM_PROMPT
        self.user_prompt = USER_PROMPT
        self.model = model
        self.temperature = temperature
        

        super().__init__(
            api_key=api_key,
            assistant_id=assistant_id,
            model=model,
            temperature=temperature,
        )
        self._setup_tools()
        self.rag_folder = self._resolve_rag_folder(rag_folder)

    def _resolve_rag_folder(self, rag_folder) -> Path:
        """Resolve and validate the RAG folder path.

        Args:
            rag_folder: Optional folder path (absolute or relative to project root).

        Returns:
            Resolved absolute Path to the RAG folder.

        Raises:
            ValueError: If the folder doesn't exist or isn't a directory.
        """
        project_root = Path(__file__).resolve().parent.parent

        try:
            if rag_folder:
                candidate = Path(rag_folder)
                resolved = candidate if candidate.is_absolute() else (project_root / candidate).resolve()
            else:
                resolved = DEFAULT_RAG_FOLDER.resolve()

            if not resolved.exists():
                raise FileNotFoundError(f"RAG folder does not exist: {resolved}")

            if not resolved.is_dir():
                raise NotADirectoryError(f"RAG folder is not a directory: {resolved}")

            print(f"[PromptGeneratorAgent] Using rag_folder: {resolved}")
            return resolved

        except Exception as e:
            raise ValueError(f"Invalid rag_folder: {rag_folder!r}") from e

    def _setup_tools(self):
        """Register the search_documents tool for RAG-based document retrieval."""
        self.tools = [
            Tool(
                name="search_documents",
                description="Search the configured reference documents for relevant lines.",
                input_model=DocumentSearchInput,
                func=self._tool_search_documents,
            )
        ]

    def _tool_search_documents(self, queries, context_lines=3, regex_mode=False):
        """Execute document search against the configured RAG folder.

        Args:
            queries: List of search query strings.
            context_lines: Number of context lines around each match.
            regex_mode: Whether to interpret queries as regex patterns.

        Returns:
            JSON string with search results.

        Raises:
            AttributeError: If rag_folder is not initialized.
        """

        if not hasattr(self, "rag_folder"):
            raise AttributeError("rag_folder is not initialized on PromptGeneratorAgent")
        return search_documents(
            queries=queries,
            context_lines=context_lines,
            regex_mode=regex_mode,
            folder=self.rag_folder,
        )

    def format_user_prompt(self, user_prompt, answers=None, revision_brief=None):
        """Render the user prompt template with input, answers, and revision context.

        Args:
            user_prompt: The analyst's plain-language input.
            answers: Optional previously generated answers for context.
            revision_brief: Optional revision guidance from human feedback.

        Returns:
            Rendered user prompt string.
        """
        template = Template(self.user_prompt)
        return template.render(
            USER_INPUT=user_prompt,
            answers=answers,
            revision_brief=revision_brief,
        )

    def _classify_intent(self, user_input: str) -> str:
        """Classify user input as 'generate' or 'converse' intent.

        Uses the LLM with the intent classification prompt and the last
        assistant message as context to determine whether the user is
        requesting a prompt change or asking a question.

        Args:
            user_input: The user's latest message text.

        Returns:
            Intent string: 'generate' or 'converse'.
        """
        last_assistant = next(
            (m for m in reversed(self.messages) if m["role"] == "assistant"),
            None
        )
        context = [last_assistant] if last_assistant else []

        result = self.chat(
            messages=[
                {"role": "system", "content": INTENT_PROMPT},
                *context,
                {"role": "user", "content": user_input}
            ],
            temperature=0.0,
            use_history=False
        )

        text = self._extract_text(result).strip()
        try:
            return IntentModel.model_validate(self._safe_json_parse(text)).intent
        except Exception:
            return "generate"

    def _print_generated_prompt(self, prompt_model: PromptModel) -> None:
        """Print the generated prompt cleanly, with real newlines for readability.

        The prompt strings come back from JSON with escaped ``\\n`` sequences.
        Printing the string values directly renders them as real line breaks so
        the system/user prompts are easy to read in the console.

        Args:
            prompt_model: The validated PromptModel to display.
        """
        import json

        divider = "=" * 70

        print(f"\n{divider}")
        print("GENERATED PROMPT")
        print(divider)

        print("\n--- SYSTEM PROMPT ---\n")
        print(prompt_model.system_prompt)

        print("\n--- USER PROMPT ---\n")
        print(prompt_model.user_prompt)

        if prompt_model.metadata_fields:
            print("\n--- METADATA FIELDS ---\n")
            for field in prompt_model.metadata_fields:
                print(f"  - {field}")

        print("\n--- OUTPUT FORMAT ---\n")
        print(json.dumps(prompt_model.output_format, indent=2, ensure_ascii=False))

        if prompt_model.saved_location_of_prompt:
            print(f"\nSaved location: {prompt_model.saved_location_of_prompt}")

        print(f"{divider}\n")

    def run(self, user_input: str, answers=None, revision_brief=None, temperature=0.6, max_retries=2, force_generate=False):
        """Run the prompt generator agent with automatic intent classification.

        Routes the user input to either conversational response or structured
        prompt generation based on classified intent.

        Args:
            user_input: The analyst's input text.
            answers: Optional prior answers for context.
            revision_brief: Optional revision guidance.
            temperature: Sampling temperature for generation.
            max_retries: Number of retries on validation failure.
            force_generate: If True, bypass intent classification and always generate.

        Returns:
            For 'converse' intent: plain text response string.
            For 'generate' intent: validated PromptModel instance.

        Raises:
            Exception: If validation fails after all retries.
        """
        if not self.messages:
            self.messages.append({"role": "system", "content": self.system_prompt})

        if force_generate:
            intent = "generate"
        else:
            intent = self._classify_intent(user_input)

        if intent == "converse":
            self.messages.append({"role": "user", "content": user_input})
            result = self.chat_with_tools(
                messages=self.messages,
                temperature=temperature,
                use_history=False,
            )
            text = self._extract_chat_text(result)
            self.messages.append({"role": "assistant", "content": text})
            return text

        self.messages.append({
            "role": "user",
            "content": self.format_user_prompt(user_input, answers, revision_brief)
        })

        for attempt in range(max_retries + 1):
            result = self.chat_with_tools(
                messages=self.messages,
                temperature=temperature,
                use_history=False,
            )
            text = self._extract_chat_text(result)

            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("\n```", 1)[0].strip()

            try:
                validated = PromptModel.model_validate(self._safe_json_parse(text))
                self.messages.append({"role": "assistant", "content": text})
                self._print_generated_prompt(validated)
                return validated
            except Exception as e:
                if attempt == max_retries:
                    raise

                self.messages.append({"role": "assistant", "content": text})
                self.messages.append({
                    "role": "user",
                    "content": f"Your response failed validation:\n{e}\n\nReturn valid JSON only, no markdown fences."
                })
        


