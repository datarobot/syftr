import pathlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import optuna
import pandas as pd
import pytest

from syftr import api
from syftr.configuration import cfg
from syftr.studies import StudyConfig

# Create a mock study object that contains 10 trials of actual data
MOCK_STUDY = MagicMock(spec=optuna.Study)
MOCK_STUDY.study_name = "example-dr-docs"
TRIALS_DF_PATH = pathlib.Path(__file__).parent / "mock_trials_df.pkl"
MOCK_TRIALS_DF = pd.read_pickle(TRIALS_DF_PATH)
MOCK_STUDY.trials_dataframe.return_value = MOCK_TRIALS_DF

EXAMPLE_STUDY_CONFIG_PATH = Path(cfg.paths.studies_dir / "example-dr-docs.yaml")
NON_EXISTANT_STUDY_CONFIG_PATH = Path(
    cfg.paths.test_studies_dir / "hotpot-toy-non-existent.yaml"
)


# This fixture mocks the optuna storage so that no actual database connection is needed
@pytest.fixture(autouse=True)
def mock_optuna_storage():
    with patch("syftr.configuration.Database.get_optuna_storage") as mock_storage:
        mock_storage.return_value = None
        yield


def test_init_no_path():
    study_config_path = Path(EXAMPLE_STUDY_CONFIG_PATH)
    study = api.Study(StudyConfig.from_file(study_config_path))
    assert study.study_path == study_config_path


def test_pareto_df():
    with patch("syftr.api.optuna.load_study", return_value=MOCK_STUDY):
        study = api.Study.from_file(EXAMPLE_STUDY_CONFIG_PATH)
        pareto_df = study.pareto_df
    assert pareto_df is not None
    assert isinstance(pareto_df, pd.DataFrame)
    assert not pareto_df.empty
    assert "user_attrs_flow" in pareto_df.columns
    assert "values_0" in pareto_df.columns
    assert "values_1" in pareto_df.columns


def test_flows_df_returns_dataframe():
    with patch("syftr.api.optuna.load_study", return_value=MOCK_STUDY):
        study = api.Study.from_file(EXAMPLE_STUDY_CONFIG_PATH)
        df = study.flows_df
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "user_attrs_flow" in df.columns
    assert "values_0" in df.columns
    assert "values_1" in df.columns


def test_pareto_df_study_does_not_exist():
    study = api.Study.from_file(NON_EXISTANT_STUDY_CONFIG_PATH)
    with pytest.raises(api.SyftrUserAPIError) as exc:
        study.pareto_df
    assert exc.value.args[0].startswith("Cannot find this study in the database:")


def test_pareto_flows():
    with patch("syftr.api.optuna.load_study", return_value=MOCK_STUDY):
        study = api.Study.from_file(EXAMPLE_STUDY_CONFIG_PATH)
        pareto_flows = study.pareto_flows
    assert pareto_flows
    assert all("llm_cost_mean" in flow["metrics"] for flow in pareto_flows)
    assert all("accuracy" in flow["metrics"] for flow in pareto_flows)


def test_pareto_flows_study_does_not_exist():
    study = api.Study.from_file(NON_EXISTANT_STUDY_CONFIG_PATH)
    with pytest.raises(api.SyftrUserAPIError) as exc:
        study.pareto_flows
    assert exc.value.args[0].startswith("Cannot find this study in the database")


def test_knee_point():
    with patch("syftr.api.optuna.load_study", return_value=MOCK_STUDY):
        study = api.Study.from_file(EXAMPLE_STUDY_CONFIG_PATH)
        with pytest.raises(api.SyftrUserAPIError) as exc:
            study.knee_point
    assert (
        exc.value.args[0]
        == "Not enough points in the Pareto front to find a knee point."
    )


def test_knee_point_study_does_not_exist():
    study = api.Study.from_file(NON_EXISTANT_STUDY_CONFIG_PATH)
    with pytest.raises(api.SyftrUserAPIError) as exc:
        study.knee_point
    assert exc.value.args[0].startswith("Cannot find this study in the database")


def test_status_completed():
    with patch("syftr.api.optuna.load_study", return_value=MOCK_STUDY):
        study = api.Study.from_file(EXAMPLE_STUDY_CONFIG_PATH)
        status = study.status["job_status"]
    assert status == api.SyftrStudyStatus.COMPLETED


def test_status_non_existent():
    study = api.Study.from_file(NON_EXISTANT_STUDY_CONFIG_PATH)
    assert study.status["job_status"] == api.SyftrStudyStatus.INITIALIZED


def test_get_study_non_existent_in_db():
    with pytest.raises(api.SyftrUserAPIError) as exc:
        api.Study.from_db("non_existent_study")
    assert exc.value.args[0] == "Cannot find study non_existent_study in the database."


def test_from_file_study():
    study = api.Study.from_file(EXAMPLE_STUDY_CONFIG_PATH)
    assert study
    assert isinstance(study, api.Study)
