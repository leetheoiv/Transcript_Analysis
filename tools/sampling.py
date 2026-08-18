"""
tools/sampling.py

Bootstrap sampling tool for creating validation sets.
Supports true bootstrap (with replacement) and stratified bootstrap.
Writes the sample to CSV and returns a DataFrame.
"""

import pandas as pd
from pathlib import Path


def bootstrap_sample(
    df: pd.DataFrame,
    n: int,
    output_path: str,
    stratify_by: list[str] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Draw a bootstrap sample (with replacement) from a DataFrame and write to CSV.

    For stratified sampling, rows are drawn proportionally per stratum, capped
    at the total number of rows in that stratum to prevent a single row dominating.

    :param df:           Source DataFrame.
    :param n:            Total number of rows to sample.
    :param output_path:  Path to write the sample CSV.
    :param stratify_by:  Optional list of column names to stratify on.
    :param random_seed:  Random seed for reproducibility.
    :return:             Sampled DataFrame with a `bootstrap_weight` column
                         indicating how many times each original row was drawn.
    """
    if stratify_by:
        frames = []
        strata = df.groupby(stratify_by)
        n_strata = len(strata)

        for _, group in strata:
            stratum_n = min(round(n / n_strata), len(group))
            frames.append(group.sample(n=stratum_n, replace=True, random_state=random_seed))

        sample = pd.concat(frames).reset_index(drop=True)
    else:
        sample = df.sample(n=n, replace=True, random_state=random_seed).reset_index(drop=True)

    # count how many times each original index appears
    weight_map = sample.index.value_counts()
    sample["bootstrap_weight"] = sample.index.map(weight_map).fillna(1).astype(int)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)

    return sample
