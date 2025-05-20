import time
from pathlib import Path

import pytest

from syftr import api
from syftr.configuration import cfg
from syftr.studies import StudyConfig


def test_init_no_path():
    study_config_path = Path(cfg.paths.studies_dir / "example-dr-docs.yaml")
    study = api.Study(StudyConfig.from_file(study_config_path))
    assert study.study_path == study_config_path


def test_pareto_df():
    study = api.Study.from_file(
        cfg.paths.studies_dir / "bench14--small-models--drdocs.yaml"
    )
    pareto_df = study.pareto_df
    assert pareto_df is not None


def test_pareto_df_study_does_not_exist():
    study = api.Study.from_file(
        cfg.paths.test_studies_dir / "hotpot-toy-non-existent.yaml"
    )
    with pytest.raises(api.SyftrUserAPIError) as exc:
        study.pareto_df
    assert exc.value.args[0] == "Cannot find this study in the database."


def test_pareto_flows():
    study = api.Study.from_file(
        cfg.paths.studies_dir / "bench14--small-models--drdocs.yaml"
    )
    pareto_flows = study.pareto_flows
    assert pareto_flows
    assert all("llm_cost_mean" in flow["metrics"] for flow in pareto_flows)
    assert all("accuracy" in flow["metrics"] for flow in pareto_flows)


def test_pareto_flows_study_does_not_exist():
    study = api.Study.from_file(
        cfg.paths.test_studies_dir / "hotpot-toy-non-existent.yaml"
    )
    with pytest.raises(api.SyftrUserAPIError) as exc:
        study.pareto_flows
    assert exc.value.args[0] == "Cannot find this study in the database."


def test_knee_point():
    study = api.Study.from_file(
        cfg.paths.studies_dir / "bench14--small-models--drdocs.yaml"
    )
    knee_point = study.knee_point
    assert knee_point
    assert knee_point["metrics"]["accuracy"]
    assert knee_point["metrics"]["llm_cost_mean"]


def test_knee_point_study_does_not_exist():
    study = api.Study.from_file(
        cfg.paths.test_studies_dir / "hotpot-toy-non-existent.yaml"
    )
    with pytest.raises(api.SyftrUserAPIError) as exc:
        study.knee_point
    assert exc.value.args[0] == "Cannot find this study in the database."


def test_status_completed():
    study = api.Study.from_file(
        cfg.paths.studies_dir / "bench14--small-models--drdocs.yaml"
    )
    assert study.status["job_status"] == api.SyftrStudyStatus.COMPLETED


def test_status_non_existent():
    study = api.Study.from_file(
        cfg.paths.test_studies_dir / "hotpot-toy-non-existent.yaml"
    )
    assert study.status["job_status"] == api.SyftrStudyStatus.INITIALIZED


def test_start_stop():
    study = api.Study.from_file(cfg.paths.studies_dir / "example-dr-docs.yaml")
    study.run()
    time.sleep(45)
    assert study.status["job_status"] == api.SyftrStudyStatus.RUNNING
    study.stop()
    time.sleep(45)
    assert study.status["job_status"] == api.SyftrStudyStatus.STOPPED


def test_start_stop_resume():
    study = api.Study.from_file(cfg.paths.studies_dir / "example-dr-docs.yaml")
    study.run()
    time.sleep(45)
    assert study.status["job_status"] == api.SyftrStudyStatus.RUNNING
    study.stop()
    time.sleep(45)
    assert study.status["job_status"] == api.SyftrStudyStatus.STOPPED
    study.resume()
    time.sleep(45)
    assert study.status["job_status"] == api.SyftrStudyStatus.RUNNING


@pytest.mark.skip(reason="Can take too long.")
def test_wait_for_completion():
    study = api.Study.from_file(cfg.paths.studies_dir / "example-dr-docs.yaml")
    study.run()
    study.wait_for_completion()
    assert study.status["job_status"] == api.SyftrStudyStatus.COMPLETED


def test_wait_for_completion_timeout():
    study = api.Study.from_file(cfg.paths.studies_dir / "example-dr-docs.yaml")
    study.run()
    study.wait_for_completion(timeout=30)
    assert study.status["job_status"] == api.SyftrStudyStatus.STOPPED


def test_wait_for_completion_timeout_stream_logs():
    study = api.Study.from_file(cfg.paths.studies_dir / "example-dr-docs.yaml")
    study.run()
    time.sleep(30)
    study.wait_for_completion(timeout=30, stream_logs=True)
    time.sleep(30)
    assert study.status["job_status"] == api.SyftrStudyStatus.STOPPED


def test_get_study_non_existent_in_db():
    with pytest.raises(api.SyftrUserAPIError) as exc:
        api.Study.from_db("non_existent_study")
    assert exc.value.args[0] == "Cannot find study non_existent_study in the database."


def test_from_file_study():
    study = api.Study.from_file(cfg.paths.studies_dir / "example-dr-docs.yaml")
    assert study


def test_get_study_get_delete():
    study = api.Study.from_file(cfg.paths.studies_dir / "example-dr-docs.yaml")
    study.run()
    time.sleep(30)

    assert study.study_path
    assert study.study_config
    assert not study.remote

    study.stop()
    study.delete()

    with pytest.raises(api.SyftrUserAPIError) as exc:
        study = api.Study.from_db("example-dr-docs")
    assert exc.value.args[0] == "Cannot find study example-dr-docs in the database."


def test_stop_not_running():
    study = api.Study.from_file(cfg.paths.studies_dir / "example-dr-docs.yaml")
    with pytest.raises(api.SyftrUserAPIError) as exc:
        study.stop()
    assert exc.value.args[0] == "This study is not running. Run it first."


def test_delete_study_non_existent():
    study = api.Study.from_file("tests/studies/hotpot-toy-non-existent.yaml")
    with pytest.raises(api.SyftrUserAPIError) as exc:
        study.delete()
    assert (
        exc.value.args[0]
        == "Study hotpot-dev-toy-non-existent has no study config in the database."
    )
