import pytest

from syftr.configuration import cfg
from syftr.studies import StudyConfig
from tests.check_trials import check_trials_are_flawless
from tests.e2e.optimization import run_and_test_optimization


@pytest.mark.parametrize(
    "config_file_name",
    [
        "test-tools-anthropic-haiku-35.yaml",
        # "test-tools-gemini-flash2.yaml", # Seems to be flaky with tools
        "test-tools-gpt-4o-mini.yaml",
        # "test-tools-llama-33-70B.yaml", # Seems to be flaky with tools
        "test-tools-mistral-large.yaml",
        "test-tools-o3-mini.yaml",
        "test-tools-phi-4.yaml",
    ],
)
def test_agents_with_tool_use(config_file_name):
    study_config_file = cfg.paths.test_studies_dir / config_file_name
    study_config = StudyConfig.from_file(study_config_file)
    study_config.optimization.use_hf_embedding_models = False

    df_trials = run_and_test_optimization(study_config)
    check_trials_are_flawless(df_trials)
    assert len(df_trials) == 5, f"Expected 5 trials, but got {len(df_trials)}"
