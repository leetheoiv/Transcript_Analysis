"""
worker/tasks/extraction.py

Celery task for batch transcript extraction.

KEY CONCEPTS:
─────────────
1. @app.task — This decorator registers the function with Celery.
   When you call run_extraction_batch.delay(...), Celery:
     a) Serializes your arguments to JSON
     b) Puts the message on the "extraction" queue (per our routing config)
     c) Returns immediately with an AsyncResult (a future/promise)

2. bind=True — Gives the task access to `self`, which is the Task instance.
   Useful for retries: self.retry(exc=exc, countdown=60)

3. Serialization constraint — Everything going IN and coming OUT must be
   JSON-serializable. That means:
     - No Pydantic models as arguments (pass dicts instead)
     - No DataFrames as return values (return list[dict] or a file path)
     - No class instances (pass config needed to reconstruct them)

4. The worker reconstructs objects from the serialized args:
     - Receives rows as list[dict]
     - Instantiates its own TranscriptExtractionAgent
     - Calls process_batch (which uses ThreadPoolExecutor internally)
     - Returns a file path or list[dict]

FLOW:
─────
  Your code                     Redis                  Worker process
  ──────────                    ─────                  ──────────────
  run_extraction_batch.delay(   →  queued in           →  worker picks up
    rows=[...],                    "extraction"           deserializes args
    prompt_template="...",         queue                  creates agent
    ...                                                  runs process_batch
  )                                                      saves CSV
  returns AsyncResult           ←  result stored  ←    returns file path
"""

from worker.celery_app import app
from agents.ExtractionAgent import TranscriptExtractionAgent


@app.task(bind=True, name="worker.tasks.extraction.run_extraction_batch")
def run_extraction_batch(
    self,
    rows: list[dict],
    prompt_template: str,
    system_prompt: str,
    response_format_path: str,
    response_format_model_name: str,
    transcript_column_name: str = "TRANSCRIPT",
    output_file: str | None = None,
    template_params: dict | None = None,
    token_threshold: int = 500,
    max_workers: int = 3,
    max_retries: int = 3,
    model: str = "gpt-4.1-mini",
    temperature: str = "0.1",
) -> dict:
    """
    Run batch extraction as a Celery task.

    Args:
        rows: List of row dicts (each row has transcript + metadata columns).
        prompt_template: Jinja2 user prompt template string.
        system_prompt: System prompt string.
        response_format_path: File path to the generated Pydantic model (.py).
        response_format_model_name: Class name inside that file (e.g. "VAOModel").
        transcript_column_name: Column name containing transcript text.
        output_file: Optional CSV path to save results.
        template_params: Dict mapping template vars to row column names.
        token_threshold: Threshold above which tool mode is used.
        max_workers: Parallel threads for extraction.
        max_retries: Retries per row.
        model: LLM model name.
        temperature: LLM temperature.

    Returns:
        dict with:
            - "status": "success" or "error"
            - "output_file": path to saved CSV (if output_file was provided)
            - "row_count": number of rows processed
            - "results": list[dict] of extraction results (if no output_file)
    """
    from tools.utils.load_schema_model import load_model_from_path

    try:
        # Reconstruct the Pydantic response model from the saved .py file
        response_format = load_model_from_path(response_format_path, response_format_model_name)

        # Create a fresh agent instance (workers are separate processes)
        agent = TranscriptExtractionAgent(
            model=model,
            temperature=temperature,
        )

        # Run the batch (this uses ThreadPoolExecutor internally)
        results_df = agent.process_batch(
            transcript_column_name=transcript_column_name,
            rows=rows,
            prompt_template=prompt_template,
            response_format=response_format,
            system_prompt=system_prompt,
            template_params=template_params or {},
            output_file=output_file,
            token_threshold=token_threshold,
            max_workers=max_workers,
            max_retries=max_retries,
        )

        # Convert DataFrame to serializable format
        results = results_df.to_dict("records")

        return {
            "status": "success",
            "output_file": output_file,
            "row_count": len(results),
            "results": results,
        }

    except Exception as exc:
        # Celery retry: re-queue this task after 60s, up to 2 retries.
        # self.request.retries tracks how many times we've retried.
        if self.request.retries < 2:
            raise self.retry(exc=exc, countdown=60, max_retries=2)

        return {
            "status": "error",
            "error": str(exc),
            "row_count": 0,
            "results": [],
        }
