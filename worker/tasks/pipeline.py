"""
worker/tasks/pipeline.py

Celery task that orchestrates the full pipeline by chaining subtasks.

NEW CONCEPTS — Celery Canvas (composition primitives):
──────────────────────────────────────────────────────

  Celery gives you building blocks to compose tasks together:

  1. chain(task_a.s(), task_b.s(), task_c.s())
     ─────────────────────────────────────────
     Run A → pass A's result to B → pass B's result to C.
     Like a Unix pipe: A | B | C

     Example:
       chain(generate_prompt.s(input), generate_schema.s())
       # generate_prompt runs, its return dict is passed as first arg to generate_schema

  2. group(task_a.s(), task_b.s(), task_c.s())
     ────────────────────────────────────────
     Run A, B, C in PARALLEL. Returns a GroupResult (list of results).

     Example:
       group(extract_batch.s(chunk1), extract_batch.s(chunk2))
       # Both chunks process simultaneously on different workers

  3. chord(group(...), callback_task.s())
     ────────────────────────────────────
     Run a group in parallel, then when ALL finish, run the callback
     with the collected results.

     Example:
       chord(
           group(extract_batch.s(chunk1), extract_batch.s(chunk2)),
           merge_results.s()
       )
       # Extract both chunks in parallel, then merge when both are done

  4. .s() vs .si()
     ─────────────
     .s(args)  = "signature" — a lazy task call. In a chain, the previous
                 task's result is prepended as the first argument.
     .si(args) = "immutable signature" — ignores the previous result.
                 Use when a task doesn't need input from the prior step.

DESIGN NOTE — Why a pipeline task?
────────────────────────────────────
  We COULD just call chain() from FastAPI directly. But wrapping it in its
  own task means:
    - The pipeline itself runs on a worker (doesn't block your API process)
    - You can monitor/retry the whole pipeline as a single unit
    - HITL gates can pause the pipeline (the task sleeps or polls)

  The tradeoff: HITL (human-in-the-loop) is harder in async tasks.
  For now, this pipeline assumes NO human review gates (automated mode).
  HITL is handled at the API layer before dispatching to Celery.
"""

from worker.celery_app import app
from celery import chain

from worker.tasks.generation import generate_prompt, generate_schema
from worker.tasks.extraction import run_extraction_batch
from worker.tasks.judging import run_judging_batch
from worker.tasks.evaluation import run_evaluation


@app.task(bind=True, name="worker.tasks.pipeline.run_pipeline")
def run_pipeline(
    self,
    user_input: str,
    rows: list[dict],
    output_dir: str,
    project_name: str,
    transcript_column_name: str = "TRANSCRIPT",
    model: str = "gpt-4.1-mini",
    temperature: str = "0.1",
    token_threshold: int = 500,
    max_workers: int = 5,
    n_runs: int = 1,
) -> dict:
    """
    Run the full automated pipeline (no HITL gates).

    This coordinates the steps sequentially, calling each subtask
    synchronously (.apply() or .delay().get()) so we can pass
    results between steps.

    For the HITL version, the orchestrator.run() method handles
    the human gates directly — Celery takes over only for the
    compute-heavy steps (extraction, judging, evaluation).

    Args:
        user_input: Analyst's plain-language question.
        rows: Transcript rows as list[dict].
        output_dir: Where to save outputs.
        project_name: Project identifier.
        transcript_column_name: Column with transcript text.
        model: LLM model.
        temperature: LLM temperature.
        token_threshold: Chunking threshold.
        max_workers: Parallel workers for extraction/judging.
        n_runs: Number of extraction runs for consistency evaluation.

    Returns:
        dict with final pipeline results.
    """
    try:
        # ── Step 1: Generate prompt ──────────────────────────────────────
        prompt_result = generate_prompt.apply(
            kwargs={
                "user_input": user_input,
                "model": model,
                "temperature": "0.4",
            }
        ).get()

        if prompt_result["status"] != "success":
            return {"status": "error", "step": "prompt_generation", "error": prompt_result.get("error")}

        # ── Step 2: Generate schema ──────────────────────────────────────
        schema_result = generate_schema.apply(
            kwargs={
                "output_format": prompt_result["output_format"],
                "user_input": user_input,
                "output_dir": output_dir,
                "model": model,
                "temperature": "0.4",
            }
        ).get()

        if schema_result["status"] != "success":
            return {"status": "error", "step": "schema_generation", "error": schema_result.get("error")}

        # ── Step 3: Extraction (N runs for consistency) ──────────────────
        all_extraction_results = []

        for run_idx in range(1, n_runs + 1):
            extraction_result = run_extraction_batch.apply(
                kwargs={
                    "rows": rows,
                    "prompt_template": prompt_result["user_prompt"],
                    "system_prompt": prompt_result["system_prompt"],
                    "response_format_path": schema_result["schema_path"],
                    "response_format_model_name": schema_result["model_name"],
                    "transcript_column_name": transcript_column_name,
                    "token_threshold": token_threshold,
                    "max_workers": max_workers,
                    "model": model,
                    "temperature": temperature,
                }
            ).get()

            if extraction_result["status"] != "success":
                return {"status": "error", "step": f"extraction_run_{run_idx}", "error": extraction_result.get("error")}

            # Tag each row with its run ID
            for row in extraction_result["results"]:
                row["RUN_ID"] = run_idx
            all_extraction_results.extend(extraction_result["results"])

        # Use first run for judging
        first_run_results = [r for r in all_extraction_results if r.get("RUN_ID") == 1]

        # ── Step 4: Judging ──────────────────────────────────────────────
        # Build judge config from extraction columns
        claim_fields = [
            col for col in first_run_results[0].keys()
            if col.endswith("_claim_field")
        ] if first_run_results else []

        judge_config = []
        for col in claim_fields:
            prefix = col[:-len("_claim_field")]
            judge_config.append({
                "field_name": col,
                "claim_column": col,
                "quote_column": f"{prefix}_quote",
                "reasoning_column": f"{prefix}_reasoning",
                "task_prompt": prompt_result["system_prompt"] + "\n" + prompt_result["user_prompt"],
            })

        judge_result = run_judging_batch.apply(
            kwargs={
                "rows": first_run_results,
                "judge_config": judge_config,
                "transcript_column_name": transcript_column_name,
                "max_workers": max_workers,
                "model": model,
                "temperature": "0.1",
            }
        ).get()

        if judge_result["status"] != "success":
            return {"status": "error", "step": "judging", "error": judge_result.get("error")}

        # ── Step 5: Evaluation ───────────────────────────────────────────
        eval_result = run_evaluation.apply(
            kwargs={
                "repeated_runs": all_extraction_results,
                "judge_results": judge_result["results"],
            }
        ).get()

        if eval_result["status"] != "success":
            return {"status": "error", "step": "evaluation", "error": eval_result.get("error")}

        return {
            "status": "success",
            "prompt": prompt_result,
            "schema": schema_result,
            "extraction_row_count": len(first_run_results),
            "judge_row_count": judge_result["row_count"],
            "evaluation": eval_result,
        }

    except Exception as exc:
        return {
            "status": "error",
            "step": "pipeline",
            "error": str(exc),
        }
