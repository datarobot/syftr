import argparse
import json
from pathlib import Path

import pandas as pd

from syftr.configuration import cfg
from syftr.logger import logger
from syftr.optimization import StudyRunner
from syftr.optuna_helper import get_failed_trials
from syftr.ray.utils import ray_init
from syftr.studies import StudyConfig
from syftr.tuner.qa_tuner import objective, run_flow

STORAGE = cfg.database.get_optuna_storage()


def main():
    """
    Example usage: fix issues in connection with study "bad1" and then rerun with

        rerun_incomplete --remote --src bad1 --dst bad1-rerun --study-config hotpot

    This would run all incomplete flows from study "bad1" using study config file
    "studies/hotpot.yaml" but overriding the study name with "bad1-rerun".
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        help="Source study name to rerun",
    )
    parser.add_argument(
        "--dst",
        help="Destination study to rerun source study",
    )
    parser.add_argument(
        "--study-config",
        help="Study config file name to run",
    )
    parser.add_argument(
        "--remote",
        help="Use remote Ray cluster",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()

    study_config_file = Path(cfg.paths.studies_dir / f"{args.study_config}.yaml")
    study_config = StudyConfig.from_file(study_config_file)

    df_failed: pd.DataFrame = get_failed_trials(args.src)
    num_incomplete = len(df_failed)
    logger.info(f"The study '{args.src}' has {num_incomplete} failed trials")
    if num_incomplete == 0:
        logger.warning(f"Nothing to rerun in study {args.src}")

    flow_texts = list(df_failed["user_attrs_flow"].values)
    flow_texts_cleaned = [flow for flow in flow_texts if isinstance(flow, str)]
    if not flow_texts_cleaned:
        logger.error("Terminating because no flows available")
        return
    elif len(flow_texts_cleaned) < len(flow_texts):
        logger.warning("Some flows are missing")

    assert flow_texts_cleaned, ""
    flows = [json.loads(flow) for flow in flow_texts_cleaned]

    study_config_file = Path(f"studies/{args.study_config}.yaml")
    study_config = StudyConfig.from_file(study_config_file)
    study_config.name = args.dst
    study_config.optimization.baselines = flows
    study_config.optimization.num_random_trials = 0
    study_config.optimization.blocks[0].num_trials = 0

    logger.info("Active configuration is: %s", study_config)

    ray_init(force_remote=args.remote)

    logger.info("Running study: %s", study_config.name)

    optimization = StudyRunner(
        objective=objective,
        study_config=study_config,
        seeder=run_flow,
    )
    optimization.run()


if __name__ == "__main__":
    main()
