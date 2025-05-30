import time
from pathlib import Path

import pytest

from syftr import api
from syftr.configuration import cfg

EXAMPLE_STUDY_CONFIG_PATH = Path(cfg.paths.studies_dir / "example-dr-docs.yaml")
NON_EXISTANT_STUDY_CONFIG_PATH = Path(
    cfg.paths.test_studies_dir / "hotpot-toy-non-existent.yaml"
)


def test_start_stop():
    study = api.Study.from_file(EXAMPLE_STUDY_CONFIG_PATH)
    study.run()
    time.sleep(45)
    assert study.status["job_status"] == api.SyftrStudyStatus.RUNNING
    study.stop()
    time.sleep(45)
    assert study.status["job_status"] == api.SyftrStudyStatus.STOPPED


def test_start_stop_resume():
    study = api.Study.from_file(EXAMPLE_STUDY_CONFIG_PATH)
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
    study = api.Study.from_file(EXAMPLE_STUDY_CONFIG_PATH)
    study.run()
    study.wait_for_completion()
    assert study.status["job_status"] == api.SyftrStudyStatus.COMPLETED


def test_wait_for_completion_timeout():
    study = api.Study.from_file(EXAMPLE_STUDY_CONFIG_PATH)
    study.run()
    study.wait_for_completion(timeout=30)
    assert study.status["job_status"] == api.SyftrStudyStatus.STOPPED


def test_wait_for_completion_timeout_stream_logs():
    study = api.Study.from_file(EXAMPLE_STUDY_CONFIG_PATH)
    study.run()
    time.sleep(30)
    study.wait_for_completion(timeout=30, stream_logs=True)
    time.sleep(30)
    assert study.status["job_status"] == api.SyftrStudyStatus.STOPPED


def test_get_study_get_delete():
    study = api.Study.from_file(EXAMPLE_STUDY_CONFIG_PATH)
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
    study = api.Study.from_file(EXAMPLE_STUDY_CONFIG_PATH)
    with pytest.raises(api.SyftrUserAPIError) as exc:
        study.stop()
    assert exc.value.args[0] == "This study is not running. Run it first."


def test_delete_study_non_existent():
    study = api.Study.from_file(NON_EXISTANT_STUDY_CONFIG_PATH)
    with pytest.raises(api.SyftrUserAPIError) as exc:
        study.delete()
    assert (
        exc.value.args[0]
        == "Study hotpot-dev-toy-non-existent has no study config in the database."
    )
