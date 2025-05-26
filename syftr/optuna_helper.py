import json
import re
import typing as T

import numpy as np
import optuna
import pandas as pd
import ray
from kneed import KneeLocator
from numpy import ceil
from optuna.storages import BaseStorage
from paretoset import paretoset
from ray._private.worker import RemoteFunction1
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from syftr.configuration import NDIGITS, cfg
from syftr.hierarchical_tpe import HierarchicalTPESampler
from syftr.logger import logger
from syftr.studies import StudyConfig


def get_study_names(
    include_regex: T.Union[str, T.List[str]],
    exclude_regex: T.Optional[T.Union[str, T.List[str]]] = None,
) -> T.List[str]:
    exclude_regex = exclude_regex or []
    include_regex = [include_regex] if isinstance(include_regex, str) else include_regex
    exclude_regex = [exclude_regex] if isinstance(exclude_regex, str) else exclude_regex
    all_study_names = optuna.get_all_study_names(
        storage=cfg.database.get_optuna_storage()
    )
    assert all_study_names, "No studies found"
    return sorted(
        [
            s
            for s in all_study_names
            if any(re.match(regex, s) for regex in include_regex)
            and not any(re.match(regex, s) for regex in exclude_regex)
        ]
    )


def without_non_search_space_params(
    params: T.Dict[str, T.Any], study_config: StudyConfig
) -> T.Dict[str, T.Any]:
    result = {}
    for k, v in params.items():
        if k not in study_config.search_space.non_search_space_params:
            result[k] = v
    return result


def trial_exists(
    study_name: str,
    params: T.Dict[str, T.Any],
    storage: str | BaseStorage | None = None,
):
    storage = storage or cfg.database.get_optuna_storage()
    logger.debug("Loading '%s' from storage: %s", study_name, storage)
    study = optuna.load_study(study_name=study_name, storage=storage)
    for trial in study.get_trials():
        if params == trial.params:
            return True
    return False


def recreate_with_completed_trials(
    study_config: StudyConfig,
    storage: str | BaseStorage | None = None,
):
    study_name = study_config.name
    storage = storage or cfg.database.get_optuna_storage()
    try:
        study: optuna.Study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        logger.warning(
            "Cannot recreate study '%s' because it does not exist", study_name
        )
        return
    completed_trials = [
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            values=t.values,
            params=t.params,
            distributions=t.distributions,
            user_attrs=t.user_attrs,
            system_attrs=t.system_attrs,
            intermediate_values=t.intermediate_values,
        )
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    logger.info("Keeping %i completed trials", len(completed_trials))
    logger.info("Deleting study '%s'", study_name)
    optuna.delete_study(study_name=study_name, storage=storage)
    logger.info("Recreating study '%s'", study_name)
    sampler = get_sampler(study_config)
    new_study = optuna.create_study(
        storage=storage,
        sampler=sampler,
        directions=["maximize", "minimize"],
        study_name=study_name,
        load_if_exists=False,
    )
    new_study.add_trials(completed_trials)
    logger.info("Recreated study '%s' successfully", study_name)


@retry(stop=stop_after_attempt(6), wait=wait_fixed(10))
def _get_completed_trials(
    study: optuna.Study,
    success_rate: float | None = None,
    ndigits: int = NDIGITS,
):
    df: pd.DataFrame = study.trials_dataframe()
    if df.empty:
        return df
    df = df[df["state"] == "COMPLETE"]
    if "user_attrs_metric_failed" in df:
        df = df[df["user_attrs_metric_failed"] == False]
    df = df.dropna(subset=["values_0", "values_1"])
    if success_rate:
        try:
            success = df["user_attrs_metric_num_success"]
            n_total = df["user_attrs_metric_num_success"].max()
            df = df[(success / n_total) >= success_rate]
        except KeyError:
            logger.warning(f"Cannot filter {study.study_name} by success rate yet")
    df = df.reset_index(drop=True)
    for col in df.select_dtypes(include=[np.float64]).columns:
        if col.startswith("params_"):
            df[col] = df[col].round(ndigits)
    return df


def get_completed_trials(
    study: T.Union[optuna.Study, str, T.List[optuna.Study], T.List[str]],
    storage: str | BaseStorage | None = None,
    success_rate: float | None = None,
    drop_inf_latency: bool | None = True,
) -> pd.DataFrame:
    study_list = study if isinstance(study, list) else [study]
    storage = storage or cfg.database.get_optuna_storage()
    tmp: T.List[optuna.Study] = []
    for s in study_list:
        if isinstance(s, str):
            tmp.append(optuna.load_study(study_name=s, storage=storage))
        else:
            assert isinstance(s, optuna.Study), "Bad study provided"
            tmp.append(s)
    studies = tmp
    dfs: T.List[pd.DataFrame] = []
    for s in tqdm(
        studies, desc="Loading trials from studies", disable=len(studies) <= 1
    ):
        df_tmp: pd.DataFrame = _get_completed_trials(
            study=s,  # type: ignore[arg-type]
            success_rate=success_rate,
        )
        if df_tmp is None:
            logger.warning("Trials dataframe should not be None")
            continue
        if not df_tmp.empty:
            df_tmp["study_name"] = s.study_name  # type: ignore[union-attr]
            dfs.append(df_tmp)
    if dfs:
        df_result = pd.concat(dfs, ignore_index=True)
        if drop_inf_latency:
            df_result = df_result[df_result["values_1"] < np.inf]
            df_result = df_result.reset_index(drop=True)
        logger.info(f"Loaded {len(df_result)} trials")
        return df_result
    return pd.DataFrame()


def get_pareto_mask(
    study: optuna.Study | str | pd.DataFrame,
    success_rate: float | None = None,
    accuracy_col: str = "values_0",
    latency_col: str = "values_1",
) -> np.ndarray:
    if isinstance(study, pd.DataFrame):
        df_trials = study.copy()
        if success_rate:
            success = df_trials["user_attrs_metric_num_success"]
            total = df_trials["user_attrs_metric_num_total"]
            df_trials = df_trials[(success / total) >= success_rate]
    else:
        assert isinstance(study, (optuna.Study, str)), "Bad study provided"
        df_trials = get_completed_trials(study, success_rate=success_rate)
    return paretoset(df_trials[[accuracy_col, latency_col]], sense=["max", "min"])


def get_pareto_df(
    study_config: StudyConfig | str, success_rate: float | None = None
) -> pd.DataFrame:
    """
    Get the pareto front of the study in form of a Pandas DataFrame.
    The DataFrame contains the trials that are on the pareto front.
    Args:
        study_config: The study config or the name of the study.
        success_rate: The success rate to filter the trials.
    """
    study_name = (
        study_config.name if isinstance(study_config, StudyConfig) else study_config
    )
    study: optuna.Study = optuna.load_study(
        study_name=study_name, storage=cfg.database.get_optuna_storage()
    )
    df_trials: pd.DataFrame = get_completed_trials(study, success_rate=success_rate)
    pareto_mask = get_pareto_mask(df_trials)
    return df_trials[pareto_mask]


def get_flows_from_trials(df: pd.DataFrame) -> T.List[T.Dict[str, T.Any]]:
    return df["user_attrs_flow"].apply(json.loads).tolist()


def get_pareto_flows(
    study_config: StudyConfig | str, success_rate: float | None = None
) -> T.List[T.Dict[str, T.Any]]:
    df_pareto = get_pareto_df(study_config, success_rate=success_rate)
    return get_flows_from_trials(df_pareto)


def get_knee_df(
    study_config: StudyConfig, success_rate: float | None = None
) -> pd.DataFrame:
    df_pareto: pd.DataFrame = get_pareto_df(study_config, success_rate=success_rate)
    if df_pareto.empty:
        return df_pareto
    knee = KneeLocator(
        df_pareto["values_1"],
        df_pareto["values_0"],
        curve="concave",
        direction="increasing",
    )
    return knee.knee


def get_knee_flow(
    study_config: StudyConfig, success_rate: float | None = None
) -> T.Dict[str, T.Any]:
    df_knee = get_knee_df(study_config, success_rate=success_rate)
    if df_knee.empty:
        return {}
    return get_flows_from_trials(df_knee)[0]


def get_failed_trials(study_names: str | T.List[str]) -> pd.DataFrame:
    study_names = study_names if isinstance(study_names, list) else [study_names]
    storage = cfg.database.get_optuna_storage()
    dfs: T.List[pd.DataFrame] = []
    for study_name in study_names:
        study: optuna.Study = optuna.load_study(study_name=study_name, storage=storage)
        df: pd.DataFrame = study.trials_dataframe()
        if df.empty:
            continue
        if "user_attrs_metric_failed" not in df.columns:
            continue
        df = df[df["user_attrs_metric_failed"] == True]
        df["study_name"] = study.study_name
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def run_flows(
    flows: T.List[T.Dict[str, T.Any]],
    study_config: StudyConfig,
    runner: RemoteFunction1[None, dict[str, T.Any], StudyConfig],
):
    n_flows = len(flows)
    n_parallel = study_config.optimization.max_concurrent_trials
    n_batches = int(ceil(n_flows / n_parallel))

    num_cpus = study_config.optimization.cpus_per_trial
    num_gpus = study_config.optimization.gpus_per_trial
    logger.info(
        "Running %i flows (max %i in parallel) using study config: %s",
        n_flows,
        n_parallel,
        study_config.name,
    )
    for i in range(n_batches):
        batch_args = flows[i * n_parallel : (i + 1) * n_parallel]
        tasks = [
            runner.options(num_cpus=num_cpus, num_gpus=num_gpus).remote(b, study_config)
            for b in batch_args
        ]
        ray.get(tasks)


def seed_study(seeder, study_config: StudyConfig):
    flows = study_config.optimization.baselines
    if study_config.optimization.shuffle_baselines:
        logger.info("Shuffling baselines")
        flows = sorted(flows, key=lambda _: np.random.random())
    run_flows(flows, study_config, seeder)


def _set_metric(trial: optuna.trial.BaseTrial, metric_name: str, score: T.Any):
    trial.set_user_attr("metric_" + metric_name, score)


def set_metrics(trial: optuna.trial.BaseTrial, metrics: T.Dict[str, float] | None):
    assert metrics, "No metrics provided"
    for metric_name, score in metrics.items():
        _set_metric(trial, metric_name, score)


def recreate_locally(study: optuna.Study):
    logger.info("Creating in-memory study from existing study")
    new_study = optuna.create_study(
        study_name=study.study_name,
        directions=study.directions,
    )
    for trial in tqdm(study.trials, desc="Copying trials"):
        new_study.add_trial(trial)
    return new_study


def get_sampler(study_config: StudyConfig) -> optuna.samplers.BaseSampler:
    if study_config.optimization.sampler == "tpe":
        return optuna.samplers.TPESampler(
            n_startup_trials=study_config.optimization.num_random_trials,
            constant_liar=True,
            multivariate=True,
        )
    elif study_config.optimization.sampler == "hierarchical":
        return HierarchicalTPESampler(
            constant_liar=True,
            n_startup_trials=study_config.optimization.num_random_trials,
        )
    else:
        raise ValueError("Invalid sampler")
