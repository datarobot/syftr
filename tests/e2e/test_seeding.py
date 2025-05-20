from pathlib import Path

import optuna
import pytest

from syftr.configuration import cfg
from syftr.studies import StudyConfig
from tests.check_trials import check_trials_are_flawless
from tests.e2e.optimization import run_and_test_optimization


def test_seeding_individual_and_variations():
    study_config_file_name = "test-seeding-individual-and-variations.yaml"
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


def test_seeding_individual_baselines_with_full_evaluation():
    study_config_file_name = "test-seeding-individual.yaml"
    study_config_file = Path(cfg.paths.test_studies_dir / study_config_file_name)
    study_config = StudyConfig.from_file(study_config_file)
    study_config.optimization.use_hf_embedding_models = False

    df_trials = run_and_test_optimization(study_config)

    study = optuna.load_study(
        study_name=study_config.name,
        storage=cfg.postgres.get_optuna_storage(),
    )
    df_all = study.trials_dataframe()

    # all seeding runs have been completed
    assert len(df_trials) == len(df_all)

    # no pruning for individual baselines
    assert all(
        df_trials["user_attrs_metric_num_total"]
        == study_config.optimization.num_eval_samples
    )


def test_seeding_with_search_space_violation():
    study_config_file_name = "test-seeding-violation.yaml"
    study_config_file = Path(cfg.paths.test_studies_dir / study_config_file_name)
    study_config = StudyConfig.from_file(study_config_file)
    study_config.optimization.use_hf_embedding_models = False

    df_trials = run_and_test_optimization(study_config)

    assert len(df_trials) == 2, "Wrong number of trials"


def test_seeding_with_search_space_violation_and_exception():
    study_config_file_name = "test-seeding-violation.yaml"
    study_config_file = Path(cfg.paths.test_studies_dir / study_config_file_name)
    study_config = StudyConfig.from_file(study_config_file)
    study_config.optimization.use_hf_embedding_models = False
    study_config.optimization.raise_on_invalid_baseline = True

    with pytest.raises(ValueError):
        run_and_test_optimization(study_config)
