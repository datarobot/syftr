from pathlib import Path

from syftr.configuration import cfg
from syftr.studies import StudyConfig
from tests.check_trials import check_trials_are_flawless
from tests.e2e.optimization import run_and_test_optimization


def test_seeding_and_optimization_using_lats():
    study_config_file_name = "test-lats.yaml"
    study_config_file = Path(cfg.paths.test_studies_dir / study_config_file_name)
    study_config = StudyConfig.from_file(study_config_file)
    study_config.optimization.use_hf_embedding_models = False

    df_trials = run_and_test_optimization(study_config)
    check_trials_are_flawless(df_trials)
