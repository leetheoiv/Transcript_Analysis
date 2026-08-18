"""
worker/tasks/judging.py

Celery task for batch transcript judging (grounding evaluation).

SAME PATTERN AS EXTRACTION:
────────────────────────────
- Receives JSON-serializable args (rows as list[dict], config as list[dict])
- Reconstructs the JudgeAgent inside the worker process
- Calls process_batch (ThreadPoolExecutor internally)
- Returns results as list[dict]

NEW CONCEPT — task acks_late:
─────────────────────────────
  By default, Celery acknowledges a task as soon as the worker picks it up.
  With acks_late=True, the ack happens AFTER the task finishes.

  Why? If the worker crashes mid-task:
    - acks_late=False (default): task is lost, never retried
    - acks_late=True: task stays in the queue and another worker picks it up

  We use acks_late for judging because batches can be long-running and
  we don't want a crashed worker to silently lose a judge run.
"""

from worker.celery_app import app
from agents.JudgeAgent import JudgeAgent


@app.task(
    bind=True,
    name="worker.tasks.judging.run_judging_batch",
    acks_late=True,  # Don't ack until task completes (crash-safe)
)
def run_judging_batch(
    self,
    rows: list[dict],
    judge_config: list[dict],
    transcript_column_name: str = "TRANSCRIPT",
    session_id_column: str = "AGENTRECORDINGSESSIONID",
    output_file: str | None = None,
    template_params: dict | None = None,
    max_workers: int = 5,
    max_retries: int = 3,
    context_lines: int = 2,
    model: str = "gpt-4.1-mini",
    temperature: str = "0.1",
) -> dict:
    """
    Run batch judging as a Celery task.

    Args:
        rows: List of row dicts (extraction output — each row has claim fields,
              quotes, reasoning, and transcript).
        judge_config: List of field config dicts, each with:
            - field_name, claim_column, quote_column, reasoning_column, task_prompt
        transcript_column_name: Column containing the original transcript.
        session_id_column: Column used as the row identifier.
        output_file: Optional CSV path to save results.
        template_params: Dict mapping template vars to row column names.
        max_workers: Parallel threads for judging.
        max_retries: Retries per row.
        context_lines: Lines of surrounding context for evidence retrieval.
        model: LLM model name.
        temperature: LLM temperature.

    Returns:
        dict with:
            - "status": "success" or "error"
            - "output_file": path to saved CSV (if provided)
            - "row_count": number of rows judged
            - "results": list[dict] of judge results
    """
    try:
        # Create judge agent inside the worker process
        agent = JudgeAgent(
            model=model,
            temperature=temperature,
        )

        # Run batch judging
        results = agent.process_batch(
            transcript_column_name=transcript_column_name,
            rows=rows,
            judge_config=judge_config,
            session_id_column=session_id_column,
            output_file=output_file,
            template_params=template_params or {},
            max_workers=max_workers,
            max_retries=max_retries,
            context_lines=context_lines,
        )

        # process_batch returns list[dict] for JudgeAgent
        if not isinstance(results, list):
            results = results.to_dict("records")

        return {
            "status": "success",
            "output_file": output_file,
            "row_count": len(results),
            "results": results,
        }

    except Exception as exc:
        if self.request.retries < 2:
            raise self.retry(exc=exc, countdown=60, max_retries=2)

        return {
            "status": "error",
            "error": str(exc),
            "row_count": 0,
            "results": [],
        }
