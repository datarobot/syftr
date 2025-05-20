import typing as T

import pandas as pd

from syftr.baselines import set_baselines
from syftr.configuration import cfg
from syftr.optimization import StudyRunner
from syftr.optuna_helper import get_completed_trials
from syftr.ray.utils import ray_init
from syftr.studies import StudyConfig
from syftr.tuner.qa_tuner import objective, run_flow
from tests.check_trials import check_param_flow_consistency


def run_and_test_optimization(
    study_config: StudyConfig, expected_params: T.List[str] | None = []
) -> pd.DataFrame:
    cfg.ray.local = True

    ray_init()

    set_baselines(study_config)

    optimization = StudyRunner(
        objective=objective,
        study_config=study_config,
        seeder=run_flow,
    )
    optimization.run()

    df_trials: pd.DataFrame = get_completed_trials(study_config.name)

    if expected_params:
        check_param_flow_consistency(df_trials, expected_params)

    return df_trials
