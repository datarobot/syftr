from pathlib import Path

from syftr.configuration import cfg
from syftr.helpers import get_baselines_from_trials
from syftr.optuna_helper import (
    get_completed_trials,
    get_flows_from_trials,
    get_pareto_flows,
)
from syftr.studies import StudyConfig
from syftr.tuner.qa_tuner import run
from tests.check_trials import check_trials_are_flawless
from tests.e2e.optimization import run_and_test_optimization


def test_reusing_a_study():
    study_config_file_name = "test-one-of-each.yaml"
    study_config_file = Path(cfg.paths.test_studies_dir / study_config_file_name)
    study_config = StudyConfig.from_file(study_config_file)
    study_config.optimization.use_hf_embedding_models = False

    df_trials = run_and_test_optimization(study_config)
    check_trials_are_flawless(df_trials)

    baselines = study_config.optimization.baselines
    baselines_reused = get_baselines_from_trials(df_trials)

    # doing this step by step to get simple error messages
    for b in baselines:
        assert b in baselines_reused, (
            f"Cannot reinstantiate this baseline from flow JSONs: {b}"
        )

    # rerun
    study_config.name = "test-one-of-each-rerun"
    study_config.optimization.baselines = baselines_reused
    df_trials_rerun = run_and_test_optimization(study_config)
    check_trials_are_flawless(df_trials)

    cols = df_trials.columns.str.startswith("params")
    df1 = df_trials.loc[:, cols]

    rerun_rols = df_trials_rerun.columns.str.startswith("params")
    df2 = df_trials_rerun.loc[:, rerun_rols]

    df1_sorted = (
        df1.sort_index(axis=1)
        .sort_values(by=df1.columns.tolist())
        .reset_index(drop=True)
    )
    df2_sorted = (
        df2.sort_index(axis=1)
        .sort_values(by=df2.columns.tolist())
        .reset_index(drop=True)
    )

    assert df1_sorted.equals(df2_sorted)


def test_separate_pareto_evaluation():
    study_config_file_name = "test-one-of-each-pareto.yaml"
    study_config_file = Path(cfg.paths.test_studies_dir / study_config_file_name)
    study_config = StudyConfig.from_file(study_config_file)
    study_config.optimization.use_hf_embedding_models = False

    pareto_flows_test = get_pareto_flows(study_config)

    run(
        study_config=study_config,
        skip_optimization=True,
        skip_pareto=False,
    )
    df_trials = get_completed_trials(study_config.pareto.name)
    pareto_flows_holdout = get_flows_from_trials(df_trials)

    assert sorted(pareto_flows_test, key=lambda d: sorted(d.items())) == sorted(
        pareto_flows_holdout, key=lambda d: sorted(d.items())
    ), "Pareto flows are different"


def test_separate_pareto_replace_llm_evaluation():
    """Tests the evaluation of the Pareto flows but with all LLMs replaced by a single LLM"""
    study_config_file_name = "test-one-of-each-pareto-replace-llm.yaml"
    study_config_file = Path(cfg.paths.test_studies_dir / study_config_file_name)
    study_config = StudyConfig.from_file(study_config_file)
    study_config.optimization.use_hf_embedding_models = False

    pareto_flows_test = get_pareto_flows(study_config)

    run(
        study_config=study_config,
        skip_optimization=True,
        skip_pareto=False,
    )
    df_trials = get_completed_trials(study_config.pareto.name)
    pareto_flows_holdout = get_flows_from_trials(df_trials)

    assert len(pareto_flows_test) == len(pareto_flows_holdout), (
        "Different number of Pareto flows"
    )

    # assert that all flows have the same llm name
    for flow in pareto_flows_holdout:
        llm_name = flow["response_synthesizer_llm"]
        assert llm_name == study_config.pareto.replacement_llm_name, (
            "Flow llm name does not match the replacement llm name"
        )
