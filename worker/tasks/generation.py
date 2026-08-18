"""
worker/tasks/generation.py

Celery tasks for prompt and schema generation.

NEW CONCEPT — ignore_result:
─────────────────────────────
  By default, Celery stores the return value in Redis (the result backend).
  For these tasks we WANT the result (we need the generated prompt/schema),
  so we leave ignore_result=False (the default).

  When would you set ignore_result=True?
  - Fire-and-forget tasks (send an email, log an event)
  - Tasks where you don't need to .get() the result later
  - Saves Redis memory on high-volume tasks

NEW CONCEPT — soft_time_limit vs time_limit:
─────────────────────────────────────────────
  - time_limit=300: hard kill after 300s (SIGKILL — no cleanup)
  - soft_time_limit=240: raises SoftTimeLimitExceeded at 240s
    Your code can catch it and do cleanup before the hard kill.

  We set soft limits on generation tasks because LLM calls can hang
  if the API is degraded — better to fail fast than block the queue.

DESIGN NOTE — Why generation tasks are simpler:
────────────────────────────────────────────────
  Unlike extraction/judging which process N rows, generation tasks
  handle a SINGLE LLM interaction (one prompt in, one structured output).
  They're short-lived and low-volume, but latency-sensitive because
  a human may be waiting for the result.
"""

from worker.celery_app import app


@app.task(
    bind=True,
    name="worker.tasks.generation.generate_prompt",
    soft_time_limit=240,
    time_limit=300,
)
def generate_prompt(
    self,
    user_input: str,
    model: str = "gpt-4.1-mini",
    temperature: str = "0.4",
    rag_folder: str | None = None,
    answers: dict | None = None,
    revision_brief: dict | None = None,
) -> dict:
    """
    Generate a prompt using the PromptGeneratorAgent.

    Args:
        user_input: The analyst's plain-language question.
        model: LLM model name.
        temperature: LLM temperature.
        rag_folder: Optional path to knowledge base folder for RAG.
        answers: Optional prior answers for context.
        revision_brief: Optional revision brief dict (from a rejected cycle).

    Returns:
        dict with:
            - "status": "success" or "error"
            - "system_prompt": generated system prompt
            - "user_prompt": generated user prompt
            - "output_format": dict for schema generation
            - "metadata_fields": list of metadata fields
    """
    from agents.prompt_generator_agent import PromptGeneratorAgent

    try:
        agent = PromptGeneratorAgent(
            model=model,
            temperature=temperature,
            rag_folder=rag_folder,
        )

        result = agent.run(
            user_input=user_input,
            answers=answers,
            revision_brief=revision_brief,
            force_generate=True,
        )

        # result is a PromptModel (Pydantic) — serialize to dict
        return {
            "status": "success",
            "system_prompt": result.system_prompt,
            "user_prompt": result.user_prompt,
            "output_format": result.output_format,
            "metadata_fields": result.metadata_fields,
            "saved_location_of_prompt": getattr(result, "saved_location_of_prompt", None),
        }

    except Exception as exc:
        if self.request.retries < 2:
            raise self.retry(exc=exc, countdown=30, max_retries=2)

        return {
            "status": "error",
            "error": str(exc),
        }


@app.task(
    bind=True,
    name="worker.tasks.generation.generate_schema",
    soft_time_limit=240,
    time_limit=300,
)
def generate_schema(
    self,
    output_format: dict,
    user_input: str,
    output_dir: str,
    model: str = "gpt-4.1-mini",
    temperature: str = "0.4",
) -> dict:
    """
    Generate a Pydantic schema using the SchemaGeneratorAgent.

    Args:
        output_format: The output_format dict from prompt generation.
        user_input: The analyst's original question (context for the schema).
        output_dir: Directory to write the generated .py model file.
        model: LLM model name.
        temperature: LLM temperature.

    Returns:
        dict with:
            - "status": "success" or "error"
            - "model_name": name of the generated Pydantic class
            - "schema_path": file path to the generated .py file
            - "code": the generated Python code
    """
    from agents.schema_generator_agent import SchemaGeneratorAgent

    try:
        agent = SchemaGeneratorAgent(
            model=model,
            temperature=temperature,
        )

        schema_result, schema_path = agent.run(
            output_format=output_format,
            user_input=user_input,
            output_dir=output_dir,
        )

        return {
            "status": "success",
            "model_name": schema_result.model_name,
            "schema_path": str(schema_path),
            "code": schema_result.code,
        }

    except Exception as exc:
        if self.request.retries < 2:
            raise self.retry(exc=exc, countdown=30, max_retries=2)

        return {
            "status": "error",
            "error": str(exc),
        }
