from pathlib import Path

import pytest

from syftr.configuration import cfg
from syftr.optimization import initialize_from_study
from syftr.optuna_helper import get_completed_trials, get_pareto_df
from syftr.studies import StudyConfig, get_subspace


@pytest.mark.xfail(
    reason="depends on a database state, should fail when a database is empty."
)
def test_study_initialization():
    success_rate = 0.9

    src_name = "test-study-src"
    dst_name = "test-study-dst"

    src_file = Path(cfg.paths.test_studies_dir / f"{src_name}.yaml")
    src_config = StudyConfig.from_file(src_file)
    dst_config = src_config.model_copy()
    dst_config.name = dst_name
    dst_config.reuse_study = False
    dst_config.recreate_study = True
    dst_config.optimization.skip_existing = False

    df_pareto = get_pareto_df(src_config.name, success_rate)
    df_pareto.reset_index(drop=True, inplace=True)
    pareto_space = get_subspace(df_pareto, dst_config.search_space)

    dst_config.search_space = pareto_space

    initialize_from_study(
        src_config=src_config,
        dst_config=dst_config,
        src_df=df_pareto,
    )

    df_dst = get_completed_trials(dst_config.name, success_rate=success_rate)

    assert len(df_pareto) >= len(df_dst)

    cols = [
        c
        for c in df_pareto.columns
        if c.startswith("params_") or c.startswith("values_")
    ]
    assert df_pareto[cols].equals(df_pareto[cols])
