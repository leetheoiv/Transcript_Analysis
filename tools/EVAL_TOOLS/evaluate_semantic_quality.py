"""
tools/EVAL_TOOLS/evaluate_semantic_quality.py

Computes per-field and overall semantic quality metrics (correctness rate,
hallucination rate) from judge results stored in a DataFrame.
"""

import pandas as pd

def evaluate_semantic_quality(df: pd.DataFrame) -> SemanticQualityEvaluation:
    """Evaluate semantic quality metrics from judge results.

    Computes per-field and overall correctness rates and hallucination rates
    from columns ending in '_GROUNDED' and '_HALLUCINATED'.

    Args:
        df: DataFrame containing judge results with _GROUNDED and _HALLUCINATED columns.

    Returns:
        SemanticQualityEvaluation with overall and per-field metrics.
    """
    if df.empty:
        return SemanticQualityEvaluation(
            overall=SemanticQuality(),
            per_field=[],
        )

    grounded_suffix = "_GROUNDED"
    hallucinated_suffix = "_HALLUCINATED"

    grounded_cols = [c for c in df.columns if c.endswith(grounded_suffix)]
    hallucinated_cols = [c for c in df.columns if c.endswith(hallucinated_suffix)]

    total_samples = len(df)

    per_field_metrics = []

    total_correctness_count = 0
    total_hallucination_count = 0
    total_evaluated_correctness = 0
    total_evaluated_hallucination = 0

    # Build a quick lookup for hallucination columns by field
    hallucination_map = {
        col[: -len(hallucinated_suffix)]: col
        for col in hallucinated_cols
    }

    for grounded_col in grounded_cols:
        field_name = grounded_col[: -len(grounded_suffix)]
        hallucinated_col = hallucination_map.get(field_name)

        grounded_series = df[grounded_col].dropna()
        grounded_total = len(grounded_series)
        correctness_count = int((grounded_series == True).sum())
        correctness_rate = correctness_count / grounded_total if grounded_total else 0.0

        hallucination_count = 0
        hallucination_rate = 0.0

        if hallucinated_col and hallucinated_col in df.columns:
            hallucinated_series = df[hallucinated_col].dropna()
            hallucinated_total = len(hallucinated_series)
            hallucination_count = int((hallucinated_series == True).sum())
            hallucination_rate = (
                hallucination_count / hallucinated_total if hallucinated_total else 0.0
            )
        else:
            hallucinated_total = 0

        per_field_metrics.append(
            FieldSemanticMetric(
                field_name=field_name,
                correctness_rate=correctness_rate,
                hallucination_rate=hallucination_rate,
                correctness_count=correctness_count,
                hallucination_count=hallucination_count,
                total_evaluated=grounded_total,
            )
        )

        total_correctness_count += correctness_count
        total_hallucination_count += hallucination_count
        total_evaluated_correctness += grounded_total
        total_evaluated_hallucination += hallucinated_total

    overall = SemanticQuality(
        correctness_rate=(
            total_correctness_count / total_evaluated_correctness
            if total_evaluated_correctness else 0.0
        ),
        consistency_rate=0.0,
        hallucination_rate=(
            total_hallucination_count / total_evaluated_hallucination
            if total_evaluated_hallucination else 0.0
        ),
        correctness_count=total_correctness_count,
        consistency_count=0,
        hallucination_count=total_hallucination_count,
        total_samples=total_samples,
        total_evaluated=total_evaluated_correctness,
    )

    return SemanticQualityEvaluation(
        overall=overall,
        per_field=per_field_metrics,
    )