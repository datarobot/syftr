from pathlib import Path

import optuna

from syftr.configuration import cfg
from syftr.optuna_helper import trial_exists
from syftr.studies import StudyConfig


def test_flow_exists():
    study_config_file_name = "test-study-src.yaml"
    study_config_file = Path(cfg.paths.test_studies_dir / study_config_file_name)
    study_config = StudyConfig.from_file(study_config_file)
    study = optuna.load_study(
        study_name=study_config.name,
        storage=cfg.postgres.get_optuna_storage(),
    )
    for trial in study.get_trials():
        assert trial_exists(study_config.name, trial.params)
