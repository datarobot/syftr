from pathlib import Path

import optuna

from syftr.configuration import cfg
from syftr.studies import StudyConfig
from tests.check_trials import check_trials_are_flawless
from tests.e2e.optimization import run_and_test_optimization


def test_agents_with_synthetic_crag():
    study_config_file_name = "test-synthetic-crag.yaml"
    study_config_file = Path(cfg.paths.test_studies_dir / study_config_file_name)
    study_config = StudyConfig.from_file(study_config_file)
    study_config.optimization.use_hf_embedding_models = False

    df_trials = run_and_test_optimization(study_config)
    check_trials_are_flawless(df_trials)

    study = optuna.load_study(
        study_name=study_config.name,
        storage=cfg.postgres.get_optuna_storage(),
    )
    df_all = study.trials_dataframe()

    # all seeding runs have been completed
    assert len(df_trials) == len(df_all)
