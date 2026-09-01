
import pandas as pd
from copy import deepcopy
from data_models.prompt_eval_datamodel import *
from datetime import datetime
import random


class ExtractionBootstrapEvaluator:
    def __init__(self,rows: list[dict],
        agent_cls,
        model: str = "gpt-4.1-mini",
        temperature: str = "0.1",
        transcript_column_name: str = "TRANSCRIPT",
        session_id_column: str = "AGENTRECORDINGSESSIONID",
        **agent_init_kwargs):
        self.rows = rows
        self.agent_cls = agent_cls
        self.model = model
        self.temperature = float(temperature)
        self.transcript_column_name = transcript_column_name
        self.session_id_column = session_id_column
        self.agent_init_kwargs = agent_init_kwargs

        self.agent_factory = lambda: self.agent_cls(
                    model=self.model,
                    temperature=str(self.temperature),
                    **self.agent_init_kwargs,
                )

    def _evaluate_consistency(
        self,
        df: pd.DataFrame,
        run_id_column: str = "RUN_ID",
        fields: list[str] | None = None,
        exclude_columns: list[str] | None = None,
        exclude_suffixes: tuple[str, ...] = (
            "_QUOTE",
            "_REASONING",
            "_SEARCH_TERMS",
            "_EVIDENCE_CHUNK",
            "_JUDGE_EXPLANATION",
            "_LLM_INPUT_TOKENS",
            "_LLM_OUTPUT_TOKENS",
            "_LLM_TOTAL_TOKENS",
        ),
    ) -> tuple[ConsistencyQuality, dict[str, float]]:
        if df.empty:
            return ConsistencyQuality(), {}

        if exclude_columns is None:
            exclude_columns = [
                self.session_id_column,
                run_id_column,
                "PROCESS_STATUS",
                "TRANSCRIPT_TOKENS",
                "CHUNK_TOKENS",
                "ATTEMPTED_SEARCH_TERMS",
                "RETRIEVAL_FOUND_ANY",
                "USED_MATCHED_EVIDENCE",
                "EVIDENCE_CHUNK",
                "LLM_INPUT_TOKENS",
                "LLM_OUTPUT_TOKENS",
                "LLM_TOTAL_TOKENS",
                "RUN_MODEL",
                "RUN_TEMPERATURE",
                "RUN_TIMESTAMP",
            ]

        if fields is None:
            fields = []
            for col in df.columns:
                if col in exclude_columns:
                    continue
                exclude_suffixes = tuple(s.lower() for s in exclude_suffixes)
                if any(col.lower().endswith(suffix) for suffix in exclude_suffixes):
                    continue
                fields.append(col)

        overall_consistency_count = 0
        overall_total_evaluated = 0
        per_field_consistency_rate = {}

        

        for field in fields:
            consistency_count = 0
            total_evaluated = 0

            for _, group in df.groupby(self.session_id_column):
                values = group[field].dropna().tolist()
                if not values:
                    continue

                total_evaluated += 1
                normalized = [str(v).strip() for v in values]

                if len(set(normalized)) == 1:
                    consistency_count += 1

            per_field_consistency_rate[field] = (
                consistency_count / total_evaluated if total_evaluated else 0.0
            )

            overall_consistency_count += consistency_count
            overall_total_evaluated += total_evaluated

        consistency_quality = ConsistencyQuality(
            consistency_rate=(
                overall_consistency_count / overall_total_evaluated
                if overall_total_evaluated else 0.0
            ),
            consistency_count=overall_consistency_count,
            total_evaluated=overall_total_evaluated,
        )

        return consistency_quality, per_field_consistency_rate

    def _evaluate_semantic_quality(
            self,
            df: pd.DataFrame,
            grounded_suffix: str = "_GROUNDED",
            hallucinated_suffix: str = "_HALLUCINATED",
            retrieval_status_suffix: str = "_RETRIEVAL_STATUS",
        ) -> tuple[SemanticQuality, dict[str, float], dict[str, float]]:
            """
            Evaluate correctness and hallucination from a judge output dataframe.

            Option C: when a ``{field}_RETRIEVAL_STATUS`` column is present, any
            judgment marked ``retrieval_failure`` is EXCLUDED from the correctness
            denominator — the judge could not retrieve evidence, so it cannot be
            counted against the extractor. The proportion of such exclusions is
            surfaced as ``retrieval_failure_rate`` instead of silently dropped.

            Falls back to the original grounded-based correctness when no
            retrieval-status column exists (backward compatible with old results).

            Returns:
                (
                    semantic_quality,
                    per_field_correctness_rate,
                    per_field_hallucination_rate,
                )
            """
            if df.empty:
                return SemanticQuality(), {}, {}

            grounded_cols = [c for c in df.columns if c.endswith(grounded_suffix)]
            hallucinated_cols = [c for c in df.columns if c.endswith(hallucinated_suffix)]

            total_samples = len(df)

            total_correctness_count = 0
            total_hallucination_count = 0
            total_grounded_evaluated = 0
            total_hallucination_evaluated = 0
            total_retrieval_failures = 0
            total_judged_before_exclusion = 0

            per_field_correctness_rate = {}
            per_field_hallucination_rate = {}

            hallucination_map = {
                col[: -len(hallucinated_suffix)]: col
                for col in hallucinated_cols
            }

            for grounded_col in grounded_cols:
                field_name = grounded_col[: -len(grounded_suffix)]

                # Skip phantom/claim-less columns (empty field name). Including
                # them would count junk judgments toward correctness.
                if not field_name.strip():
                    continue

                grounded_series = df[grounded_col].dropna()
                grounded_total = len(grounded_series)

                # Option C: exclude retrieval failures from the correctness
                # denominator when the retrieval-status column is available.
                retrieval_col = f"{field_name}{retrieval_status_suffix}"
                if retrieval_col in df.columns:
                    status_series = df[retrieval_col]
                    # Align to the same rows we counted for grounding
                    status_aligned = status_series.loc[grounded_series.index]
                    is_retrieval_failure = (
                        status_aligned.astype(str).str.strip() == "retrieval_failure"
                    )
                    field_retrieval_failures = int(is_retrieval_failure.sum())

                    # Judgeable rows = grounded judgments that were NOT retrieval failures
                    judgeable_mask = ~is_retrieval_failure.values
                    judgeable_grounded = grounded_series[judgeable_mask]
                    field_judgeable_total = len(judgeable_grounded)
                    field_correct_count = int((judgeable_grounded == True).sum())
                else:
                    # Backward-compatible fallback: no retrieval-status info
                    field_retrieval_failures = 0
                    field_judgeable_total = grounded_total
                    field_correct_count = int((grounded_series == True).sum())

                correctness_rate = (
                    field_correct_count / field_judgeable_total
                    if field_judgeable_total > 0 else 0.0
                )
                per_field_correctness_rate[field_name] = correctness_rate

                total_correctness_count += field_correct_count
                total_grounded_evaluated += field_judgeable_total
                total_retrieval_failures += field_retrieval_failures
                total_judged_before_exclusion += grounded_total

                hallucinated_col = hallucination_map.get(field_name)
                if hallucinated_col and hallucinated_col in df.columns:
                    hallucinated_series = df[hallucinated_col].dropna()
                    hallucinated_total = len(hallucinated_series)
                    hallucinated_true_count = int((hallucinated_series == True).sum())

                    hallucination_rate = (
                        hallucinated_true_count / hallucinated_total
                        if hallucinated_total > 0
                        else 0.0
                    )
                    per_field_hallucination_rate[field_name] = hallucination_rate

                    total_hallucination_count += hallucinated_true_count
                    total_hallucination_evaluated += hallucinated_total
                else:
                    per_field_hallucination_rate[field_name] = 0.0

            semantic_quality = SemanticQuality(
                correctness_rate=(
                    total_correctness_count / total_grounded_evaluated
                    if total_grounded_evaluated > 0 else 0.0
                ),
                hallucination_rate=(
                    total_hallucination_count / total_hallucination_evaluated
                    if total_hallucination_evaluated > 0 else 0.0
                ),
                correctness_count=total_correctness_count,
                hallucination_count=total_hallucination_count,
                total_samples=total_samples,
                total_evaluated=total_grounded_evaluated,
                retrieval_failure_count=total_retrieval_failures,
                retrieval_failure_rate=(
                    total_retrieval_failures / total_judged_before_exclusion
                    if total_judged_before_exclusion > 0 else 0.0
                ),
                correctness_source="judge",
            )

            return semantic_quality, per_field_correctness_rate, per_field_hallucination_rate

    def flatten_evaluation_result(self, result: EvaluationResult) -> dict:

        output = {}
        # date-time report is run
        run_date = datetime.now()
        # Semantic quality
        output['process_date'] = run_date
        output["correctness_rate"] = result.semantic_quality.correctness_rate
        output["consistency_rate"] = result.semantic_quality.consistency_rate
        output["hallucination_rate"] = result.semantic_quality.hallucination_rate
        output["correctness_count"] = result.semantic_quality.correctness_count
        output["consistency_count"] = result.semantic_quality.consistency_count
        output["hallucination_count"] = result.semantic_quality.hallucination_count
        output["total_samples"] = result.semantic_quality.total_samples
        output["total_evaluated"] = result.semantic_quality.total_evaluated

        # Consistency quality
        output["consistency_quality_rate"] = result.consistency_quality.consistency_rate
        output["consistency_quality_count"] = result.consistency_quality.consistency_count
        output["consistency_quality_total_evaluated"] = result.consistency_quality.total_evaluated

        # Per-field rates
        for field, value in result.per_field_correctness_rate.items():
            output[f"{field}_correctness_rate"] = value

        for field, value in result.per_field_hallucination_rate.items():
            output[f"{field}_hallucination_rate"] = value

        for field, value in result.per_field_consistency_rate.items():
            output[f"{field}_consistency_rate"] = value

        return output
    
    def write_results_to_csv(
        self,
        evaluation_result: EvaluationResult,
        repeated_runs_df: pd.DataFrame,
        summary_output_file: str | None = None,
        repeated_runs_output_file: str | None = None,
        index: bool = False,
    ):
        """
        Write evaluation outputs to CSV.

        - summary_output_file: one-row flattened evaluation summary
        - repeated_runs_output_file: full repeated-run dataframe
        """
        if summary_output_file:
            summary_row = self.flatten_evaluation_result(evaluation_result)
            pd.DataFrame([summary_row]).to_csv(summary_output_file, index=index)

        if repeated_runs_output_file:
            repeated_runs_df.to_csv(repeated_runs_output_file, index=index)
    def _sample_rows(
        self,
        rows: list[dict],
        sample_size: int | None = None,
        with_replacement: bool = False,
        rng: random.Random | None = None,
    ) -> list[dict]:
        if rng is None:
            rng = random.Random()

        if sample_size is None or sample_size >= len(rows):
            return deepcopy(rows)

        if with_replacement:
            return [deepcopy(rng.choice(rows)) for _ in range(sample_size)]

        return deepcopy(rng.sample(rows, sample_size))
            
    def _run_repeated_extractions(
        self,
        n_runs: int,
        run_id_column: str = "RUN_ID",
        consistency_sample_size: int | None = None,
        consistency_sample_with_replacement: bool = False,
        random_seed: int | None = None,
        template_params:dict ={},
        **process_batch_kwargs,
    ) -> pd.DataFrame:
        all_results = []
        rng = random.Random(random_seed)

        # Sample once and reuse across all runs
        base_rows = self._sample_rows(
            rows=self.rows,
            sample_size=consistency_sample_size,
            with_replacement=consistency_sample_with_replacement,
            rng=rng,
        )

        actual_sample_size = len(base_rows)

        for run_idx in range(1, n_runs + 1):
            print(f"Starting extraction run {run_idx}/{n_runs}")
            agent = self.agent_factory()

            batch_results = agent.process_batch(
                transcript_column_name=self.transcript_column_name,
                template_params = template_params,
                rows=deepcopy(base_rows),
                **process_batch_kwargs,
            )

  

            batch_results = batch_results.copy()
            batch_results[run_id_column] = run_idx
            batch_results["CONSISTENCY_SAMPLE_SIZE"] = actual_sample_size

            all_results.append(batch_results)
        repeated_runs_df = pd.concat(all_results, ignore_index=True)
        return repeated_runs_df
    
    def evaluate(
        self,
        n_runs: int = 3,
        run_id_column: str = "RUN_ID",
        fields: list[str] | None = None,
        judge_df: pd.DataFrame | None = None,
        run_consistency: bool = True,
        consistency_sample_size: int | None = None,
        consistency_sample_with_replacement: bool = False,
        random_seed: int | None = None,
        template_params:dict = {},
        **process_batch_kwargs,
    ) -> tuple[EvaluationResult, pd.DataFrame]:
        repeated_runs_df = pd.DataFrame()
        consistency_quality = ConsistencyQuality()
        per_field_consistency_rate = {}

        if not run_consistency and (judge_df is None or judge_df.empty):
            raise ValueError(
                "No evaluation input provided. Set run_consistency=True or supply a non-empty judge_df."
            )

        if run_consistency:
            repeated_runs_df = self._run_repeated_extractions(
                n_runs=n_runs,
                run_id_column=run_id_column,
                consistency_sample_size=consistency_sample_size,
                consistency_sample_with_replacement=consistency_sample_with_replacement,
                random_seed=random_seed,
                template_params=template_params,
                **process_batch_kwargs,
            )

            consistency_quality, per_field_consistency_rate = self._evaluate_consistency(
                df=repeated_runs_df,
                run_id_column=run_id_column,
                fields=fields,
            )

        semantic_quality = SemanticQuality(
            consistency_rate=consistency_quality.consistency_rate,
            consistency_count=consistency_quality.consistency_count,
            total_samples=len(judge_df) if judge_df is not None and not judge_df.empty else len(self.rows),
            total_evaluated=consistency_quality.total_evaluated,
        )

        per_field_correctness_rate = {}
        per_field_hallucination_rate = {}

        if judge_df is not None and not judge_df.empty:
            semantic_eval, per_field_correctness_rate, per_field_hallucination_rate = (
                self._evaluate_semantic_quality(judge_df)
            )

            semantic_eval.consistency_rate = consistency_quality.consistency_rate
            semantic_eval.consistency_count = consistency_quality.consistency_count
            semantic_quality = semantic_eval

        evaluation_result = EvaluationResult(
            date_time=datetime.now(),
            semantic_quality=semantic_quality,
            consistency_quality=consistency_quality,
            per_field_correctness_rate=per_field_correctness_rate,
            per_field_hallucination_rate=per_field_hallucination_rate,
            per_field_consistency_rate=per_field_consistency_rate,
        )

        return evaluation_result, repeated_runs_df