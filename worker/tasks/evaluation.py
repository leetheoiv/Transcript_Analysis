"""
worker/tasks/evaluation.py

Celery task for evaluation metrics (consistency + semantic quality).

This is the only CPU-bound task in the pipeline — pure pandas operations,
no LLM calls. It's fast (<5s) but isolated in its own queue so it doesn't
compete with I/O-bound workers for thread slots.
"""

from worker.celery_app import app


@app.task(bind=True, name="worker.tasks.evaluation.run_evaluation")
def run_evaluation(
    self,
    repeated_runs: list[dict],
    judge_results: list[dict] | None = None,
    session_id_column: str = "AGENTRECORDINGSESSIONID",
    run_id_column: str = "RUN_ID",
) -> dict:
    """
    Compute consistency and semantic quality metrics.

    Args:
        repeated_runs: List of row dicts from N extraction runs (with RUN_ID column).
        judge_results: Optional list of row dicts from judging (with _GROUNDED/_HALLUCINATED cols).
        session_id_column: Column identifying unique transcripts.
        run_id_column: Column identifying which run a row belongs to.

    Returns:
        dict with evaluation scores.
    """
    import pandas as pd
    from utils.bootstrap_extraction import ExtractionBootstrapEvaluator
    from data_models.prompt_eval_datamodel import EvaluationResult, ConsistencyQuality, SemanticQuality
    from datetime import datetime

    try:
        repeated_runs_df = pd.DataFrame(repeated_runs)
        judge_df = pd.DataFrame(judge_results) if judge_results else pd.DataFrame()

        # We only need the evaluator for its metric methods — no agent needed
        evaluator = ExtractionBootstrapEvaluator(
            rows=[],
            agent_cls=None,
            session_id_column=session_id_column,
        )

        # Consistency from repeated runs
        consistency_quality = ConsistencyQuality()
        per_field_consistency_rate = {}

        if not repeated_runs_df.empty:
            eval_fields = [
                col for col in repeated_runs_df.columns
                if col.endswith("_claim_field")
            ]
            consistency_quality, per_field_consistency_rate = evaluator._evaluate_consistency(
                df=repeated_runs_df,
                run_id_column=run_id_column,
                fields=eval_fields if eval_fields else None,
            )

        # Semantic quality from judge results
        semantic_quality = SemanticQuality(
            consistency_rate=consistency_quality.consistency_rate,
            consistency_count=consistency_quality.consistency_count,
            total_evaluated=consistency_quality.total_evaluated,
        )
        per_field_correctness_rate = {}
        per_field_hallucination_rate = {}

        if not judge_df.empty:
            semantic_quality, per_field_correctness_rate, per_field_hallucination_rate = (
                evaluator._evaluate_semantic_quality(judge_df)
            )
            semantic_quality.consistency_rate = consistency_quality.consistency_rate
            semantic_quality.consistency_count = consistency_quality.consistency_count

        return {
            "status": "success",
            "correctness_rate": semantic_quality.correctness_rate,
            "consistency_rate": semantic_quality.consistency_rate,
            "hallucination_rate": semantic_quality.hallucination_rate,
            "total_samples": semantic_quality.total_samples,
            "total_evaluated": semantic_quality.total_evaluated,
            "per_field_correctness_rate": per_field_correctness_rate,
            "per_field_hallucination_rate": per_field_hallucination_rate,
            "per_field_consistency_rate": per_field_consistency_rate,
        }

    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
        }
