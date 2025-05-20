import typing as T

import numpy as np
import pandas as pd

from syftr.helpers import get_flows_from_trials

NON_NAN_COLUMNS = [
    "number",
    "values_0",
    "values_1",
    "datetime_start",
    "datetime_complete",
    "duration",
    "user_attrs_dataset",
    "user_attrs_flow",
    "user_attrs_flow_name",
    "user_attrs_is_seeding",
    "user_attrs_metric_acc_confidence",
    "user_attrs_metric_accuracy",
    "user_attrs_metric_eval_duration",
    "user_attrs_metric_eval_end",
    "user_attrs_metric_eval_start",
    "user_attrs_metric_f1_score",
    "user_attrs_metric_failed",
    "user_attrs_metric_flow_duration",
    "user_attrs_metric_flow_end",
    "user_attrs_metric_flow_start",
    "user_attrs_metric_is_pruned",
    "user_attrs_metric_max_time",
    "user_attrs_metric_mean_time",
    "user_attrs_metric_median_time",
    "user_attrs_metric_min_time",
    "user_attrs_metric_num_errors",
    "user_attrs_metric_num_success",
    "user_attrs_metric_num_total",
    "user_attrs_metric_p80_time",
    "user_attrs_metric_passing_std",
    "user_attrs_metric_obj1_confidence",
    "user_attrs_metric_obj2_confidence",
    "user_attrs_metric_obj2_value",
    "user_attrs_metric_objective_1_name",
    "user_attrs_metric_objective_2_name",
    "user_attrs_metric_run_times_p80_std",
    "user_attrs_metric_run_times_std",
    "state",
]


MANDATORY_PARAMS = ["rag_mode"]


def check_param_flow_consistency(df_trials: pd.DataFrame, param_names: T.List[str]):
    assert param_names, "No parameter names specified"
    assert not df_trials.empty, "No trials available for evaluation"
    flows = get_flows_from_trials(df_trials)
    for param_name in param_names:
        flow_values = [f[param_name] for f in flows]
        assert all(df_trials[f"params_{param_name}"].values == flow_values), (
            f"Inconsistent flow: {param_name}: {df_trials[f'params_{param_name}'].values} != {flow_values}"
        )


def check_trials_are_flawless(df_trials: pd.DataFrame):
    assert not df_trials.empty, "No trials"
    assert "user_attrs_metric_exception_class" not in df_trials.columns, (
        "There are failed trials"
    )
    assert not np.isinf(df_trials["values_1"]).any(), "Latency is inf"
    assert all(df_trials["state"] == "COMPLETE"), "Not all trials completed"
    assert all(df_trials[NON_NAN_COLUMNS].notna()), "Trial data is not complete"
    check_param_flow_consistency(df_trials=df_trials, param_names=MANDATORY_PARAMS)
