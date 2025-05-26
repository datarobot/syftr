import concurrent
import concurrent.futures
import json
import random
import typing as T

import optuna
import pandas as pd
import ray
from optuna.storages import BaseStorage
from ray import tune
from ray._private.worker import RemoteFunction1

from syftr.configuration import cfg
from syftr.logger import logger
from syftr.optuna_helper import (
    get_completed_trials,
    get_knee_flow,
    get_sampler,
    recreate_with_completed_trials,
    seed_study,
    trial_exists,
    without_non_search_space_params,
)
from syftr.studies import StudyConfig


def tune_objective(
    objective,
    study_config: StudyConfig,
    components: T.List[str],
    block_name: str | None = None,
) -> T.Dict[str, float] | None:
    study = optuna.load_study(
        study_name=study_config.name,
        storage=cfg.database.get_optuna_storage(),
    )
    trial = study.ask()
    if block_name:
        trial.set_user_attr("block_name", block_name)
    try:
        obj_1, obj_2 = objective(trial, study_config, components)
        study.tell(trial, [obj_1, obj_2])
        return {
            study_config.optimization.objective_1_name: obj_1,
            study_config.optimization.objective_2_name: obj_2,
        }
    except optuna.TrialPruned:
        return None


def user_confirm_delete(study_config: StudyConfig) -> bool:
    study_name = study_config.name

    if cfg.optuna.noconfirm:
        logger.warning("noconfirm set; going to delete %s if it exists", study_name)
        return True

    confirm = input(
        f"Are you sure you want to overwrite study {study_name} if it exists? yes/no\n>>> "
    )
    if confirm != "yes":
        logger.warning(f"Cowardly refusing to delete study {study_name}")
        return False
    return True


def save_study_config_to_db(study: optuna.Study, study_config: StudyConfig):
    attrs = study_config.model_dump(mode="json")
    logger.info("Saving study config of %s to the database", study.study_name)
    for attr, value in attrs.items():
        study.set_user_attr(attr, value)


def get_study(study_config: StudyConfig) -> optuna.Study:
    study_name = study_config.name
    storage = cfg.database.get_optuna_storage()
    if study_config.reuse_study:
        logger.info(
            "We are going to reuse study '%s' if it exists or create a new one otherwise",
            study_name,
        )
        if study_config.recreate_study:
            recreate_with_completed_trials(study_config, storage)
    elif user_confirm_delete(study_config):
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            logger.info("Study '%s' deleted", study_name)
        except KeyError:
            logger.info(
                "Study '%s' does not exist and we will create a new one",
                study_name,
            )

    sampler = get_sampler(study_config)
    study = optuna.create_study(
        study_name=study_name,  # heartbeat doesn't work when using ask and tell
        storage=storage,
        load_if_exists=study_config.reuse_study,
        directions=[
            "maximize",
            "minimize",
        ],
        sampler=sampler,
    )
    save_study_config_to_db(study, study_config)
    return study


def _get_user_attrs(row: pd.Series) -> T.Dict[str, T.Any]:
    user_attrs = {}
    for key, value in row.items():
        if key.startswith("user_attrs_"):
            user_attrs[key.replace("user_attrs_", "")] = value
    return user_attrs


def initialize_from_study(
    src_config: StudyConfig,
    dst_config: StudyConfig,
    storage: str | BaseStorage | None = None,
    success_rate: float | None = None,
    src_df: pd.DataFrame | None = None,
) -> optuna.Study:
    """
    Initialize a study with trials from another study from the same Optuna storage.
    Args:
        src_config: Source study configuration.
        dst_config: Destination study configuration.
        storage: Storage for the studies.
        success_rate: Success rate for filtering trials.
        src_df: DataFrame containing trials to initialize from. If no DataFrame is provided, all trials will be fetched from the source study.
    """
    assert success_rate is None or src_df is None, (
        "Cannot use both success_rate and src_df"
    )

    storage = storage or cfg.database.get_optuna_storage()

    src_name = src_config.name
    dst_name = dst_config.name

    logger.info("Initializing study '%s' from study '%s'", dst_name, src_name)

    study = get_study(dst_config)

    if src_df is None:
        src_df = get_completed_trials(
            src_name, storage=storage, success_rate=success_rate
        )
    for _, row in src_df.iterrows():
        if not isinstance(row["user_attrs_flow"], str):
            logger.info(
                "Skipping invalid flow in source study '%s': %s",
                src_name,
                row["user_attrs_flow"],
            )
            continue
        src_flow = json.loads(row["user_attrs_flow"])
        src_flow = without_non_search_space_params(src_flow, src_config)

        if dst_config.optimization.skip_existing and trial_exists(
            dst_name, src_flow, storage
        ):
            logger.warning(
                "Flow already exists in destination study '%s': %s",
                dst_name,
                str(src_flow),
            )
            continue

        distributions = dst_config.search_space.build_distributions(params=src_flow)

        obj1 = row["values_0"]
        obj2 = row["values_1"]

        trial = optuna.create_trial(
            values=[obj1, obj2],
            params=src_flow,
            distributions=distributions,
            user_attrs=_get_user_attrs(row),
        )

        study.add_trial(trial)
        logger.debug("Initialization added trial with params: %s", trial.params)

    return study


class StudyRunner:
    """
    Class to run the study optimization block by block.
    A block contains flow components that are optimized together while the parameter
    values for other components are fixed.
    The optimization is done using Ray Tune, which allows for parallel evaluation of trials.
    For a global optimization, there would be only one block with all components.
    """

    def __init__(
        self, study_config: StudyConfig, objective: T.Callable, seeder: RemoteFunction1
    ):
        self.study_config = study_config
        self.name = study_config.name
        self.objective = objective
        self.seeder = seeder
        self.seeder_timeout = study_config.optimization.seeder_timeout
        self.method = study_config.optimization.method
        self.blocks = study_config.optimization.blocks.copy()
        self.shuffle_blocks = study_config.optimization.shuffle_blocks
        self.num_trials = study_config.optimization.num_trials
        self.cpus_per_trial = self.study_config.optimization.cpus_per_trial
        self.gpus_per_trial = self.study_config.optimization.gpus_per_trial
        self.max_concurrent_trials = (
            self.study_config.optimization.max_concurrent_trials
        )
        self.raise_on_failed_trial = (
            self.study_config.optimization.raise_on_failed_trial
        )
        self.success_rate = self.study_config.optimization.pareto_eval_success_rate

    def run(self) -> optuna.Study:
        """
        Start the optimization process as configured.
        For every block, the components are optimized using Ray Tune.
        We run as many trails for a block as specified in the configuration.
        If after one iteration through all blocks, the maximum number of trials
        has not been reached, we start over with another round.
        This allows for a step-by-step optimization where we run only a few trials
        per block and revisit the blocks.
        Having only a single block is equivalent to a global optimization.
        """
        assert ray.is_initialized(), "Please initialize ray"
        study_config = self.study_config

        study = get_study(study_config)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(seed_study, self.seeder, study_config)
            if self.seeder_timeout is not None:
                logger.info("Seeding timeout is %i seconds", int(self.seeder_timeout))
            else:
                logger.info("Seeding without timeout")
            try:
                future.result(timeout=self.seeder_timeout)
            except concurrent.futures.TimeoutError:
                logger.warning("Seeding continues after timeout")
            logger.info("Starting ray tune optimization")

            components = []
            trials = 0
            while trials < self.num_trials:
                if self.shuffle_blocks:
                    logger.info("Shuffling optimization blocks")
                    random.shuffle(self.blocks)
                for block in self.blocks:
                    logger.info("Starting optimization block: %s", block.name)
                    if self.method == "expanding":
                        logger.info(
                            "Expanding searchable components with block: %s", block.name
                        )
                        components.extend(block.components)
                    else:
                        logger.info(
                            "Using searchable components from block: %s", block.name
                        )
                        components = block.components
                    num_trials = min(block.num_trials, self.num_trials - trials)
                    trials += num_trials
                    tune.run(
                        lambda _: tune_objective(
                            study_config=study_config,
                            objective=self.objective,
                            components=components,
                            block_name=block.name,
                        ),
                        num_samples=num_trials,
                        resources_per_trial={
                            "CPU": self.cpus_per_trial,
                            "GPU": self.gpus_per_trial,
                        },
                        max_concurrent_trials=self.max_concurrent_trials,
                        checkpoint_config=ray.train.CheckpointConfig(
                            checkpoint_frequency=0
                        ),
                        raise_on_failed_trial=self.raise_on_failed_trial,
                    )
                    if self.method == "knee":
                        flow = get_knee_flow(
                            study_config=study_config,
                            success_rate=self.success_rate,
                        )
                        defaults = {
                            key: value
                            for key, value in flow.items()
                            if key in components
                        }
                        logger.info(
                            "Knee flow defaults from block '%s': %s",
                            block.name,
                            defaults,
                        )
                        study_config.search_space.update_defaults(defaults)

                future.result()  # Wait for seeding to finish

        logger.info("Finished optimizing '%s'", self.name)
        return study
