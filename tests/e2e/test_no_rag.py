from pathlib import Path

from syftr.configuration import cfg
from syftr.studies import StudyConfig
from tests.check_trials import check_trials_are_flawless
from tests.e2e.optimization import run_and_test_optimization

PARAM_LIST = ["rag_mode"]


def test_no_rag_seeding():
    study_config_file = Path(cfg.paths.test_studies_dir / "test-no-rag-seeding.yaml")
    study_config = StudyConfig.from_file(study_config_file)
    study_config.optimization.use_hf_embedding_models = False

    df_trials = run_and_test_optimization(study_config, PARAM_LIST)

    check_trials_are_flawless(df_trials)
    assert len(df_trials) == 9


def test_no_rag_optimization():
    study_config_file = Path(
        cfg.paths.test_studies_dir / "test-no-rag-optimization.yaml"
    )
    study_config = StudyConfig.from_file(study_config_file)
    study_config.optimization.use_hf_embedding_models = False

    study_config.optimization.baselines = []
    study_config.optimization.num_random_trials = 3
    study_config.optimization.num_trials = 6

    df_trials = run_and_test_optimization(study_config, PARAM_LIST)

    check_trials_are_flawless(df_trials)
    assert len(df_trials) > 5
