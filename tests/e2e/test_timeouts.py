from pathlib import Path

import pandas as pd

from syftr.baselines import set_baselines
from syftr.configuration import cfg
from syftr.optimization import StudyRunner
from syftr.optuna_helper import get_completed_trials
from syftr.ray.utils import ray_init
from syftr.studies import StudyConfig
from syftr.tuner.qa_tuner import objective, run_flow

PARAM_LIST = ["llm_name", "template_name", "rag_mode"]


def test_eval_pruning():
    study_config_file = Path(cfg.paths.test_studies_dir / "test-eval-pruning.yaml")
    study_config = StudyConfig.from_file(study_config_file)
    study_config.optimization.use_hf_embedding_models = False

    cfg.ray.local = True

    ray_init()

    set_baselines(study_config)

    optimization = StudyRunner(
        objective=objective,
        study_config=study_config,
        seeder=run_flow,
    )
    study = optimization.run()

    df_trials: pd.DataFrame = get_completed_trials(study)

    assert df_trials.empty, "All trials should have been pruned"
