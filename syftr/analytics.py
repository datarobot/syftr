import typing as T

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr
from sklearn.isotonic import spearmanr


def get_rank_correlation(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    agg: str | None = "mean",
    rank_by: str | T.List[str] | None = "values_0",
) -> T.Dict[str, T.Dict[str, T.Any]]:
    """
    Calculate the rank correlation between two dataframes.
    The dataframes must have a column named 'user_attrs_flow' that is used to
    group the dataframes before calculating the correlation.
    The rank correlation is calculated using the specified method (pearson, spearman, kendall).
    """

    rank_by = [rank_by, rank_by] if isinstance(rank_by, str) else rank_by
    assert isinstance(rank_by, list), f"rank_by must be a list, got {type(rank_by)}"
    assert len(rank_by) == 2, f"rank_by must be a list of length 2, got {len(rank_by)}"

    df_tmp_a = df_a.copy()
    df_tmp_b = df_b.copy()

    df_tmp_a = df_tmp_a.sort_values("user_attrs_flow")
    df_tmp_b = df_tmp_b.sort_values("user_attrs_flow")

    df_tmp_a = df_tmp_a.groupby("user_attrs_flow").agg({rank_by[0]: agg}).reset_index()
    df_tmp_b = df_tmp_b.groupby("user_attrs_flow").agg({rank_by[1]: agg}).reset_index()

    df_tmp_a.set_index("user_attrs_flow", inplace=True, drop=True)
    df_tmp_b.set_index("user_attrs_flow", inplace=True, drop=True)

    intersection = df_tmp_a.index.intersection(df_tmp_b.index)

    if len(intersection) < 2:
        return {
            "pearson": {
                "correlation": np.nan,
                "p_value": np.nan,
                "support": len(intersection),
            },
            "spearman": {
                "correlation": np.nan,
                "p_value": np.nan,
                "support": len(intersection),
            },
            "kendall": {
                "correlation": np.nan,
                "p_value": np.nan,
                "support": len(intersection),
            },
        }

    df_tmp_a = df_tmp_a.loc[intersection]
    df_tmp_b = df_tmp_b.loc[intersection]

    assert df_tmp_a.index.equals(df_tmp_b.index)

    results: T.Dict[str, T.Dict[str, T.Any]] = {
        "pearson": {},
        "spearman": {},
        "kendall": {},
    }

    correlation, p_value = pearsonr(
        df_tmp_a[rank_by[0]].values, df_tmp_b[rank_by[1]].values
    )
    results["pearson"]["correlation"] = correlation
    results["pearson"]["p_value"] = p_value
    results["pearson"]["support"] = len(intersection)

    correlation, p_value = spearmanr(
        df_tmp_a[rank_by[0]].values, df_tmp_b[rank_by[1]].values
    )
    results["spearman"]["correlation"] = correlation
    results["spearman"]["p_value"] = p_value
    results["spearman"]["support"] = len(intersection)

    correlation, p_value = kendalltau(
        df_tmp_a[rank_by[0]].values, df_tmp_b[rank_by[1]].values
    )
    results["kendall"]["correlation"] = correlation
    results["kendall"]["p_value"] = p_value
    results["kendall"]["support"] = len(intersection)
    return results
