import argparse
import json
import typing as T

import optuna
import ray

from syftr.configuration import cfg
from syftr.logger import logger
from syftr.optuna_helper import get_pareto_df
from syftr.ray.utils import ray_init
from syftr.studies import StudyConfig, get_default_study_name
from syftr.tuner.qa_tuner import build_flow, eval_dataset

LLM_MAP = {
    "gpt-4o-mini": "gpt-4o-std",
    "anthropic-haiku-35": "anthropic-sonnet-35",
    "gemini-flash": "gemini-pro",
}


def replace_models(
    flow_params: T.Dict[str, T.Any],
    study_config: StudyConfig,
    new_study: optuna.Study,
    parent_number: int,
):
    old_flow_params = flow_params.copy()
    for param, value in flow_params.items():
        if (param.endswith("llm") or param.endswith("llm_name")) and value in LLM_MAP:
            logger.info(
                "Replacing %s with %s for %s in flow params",
                value,
                LLM_MAP[value],
                param,
            )
            flow_params[param] = LLM_MAP[value]

    logger.info("Building pareto flow from params: %s", flow_params)
    try:
        flow = build_flow(study_config=study_config, params=flow_params)
    except Exception:
        logger.exception("Failed to build flow from %s", flow_params)
        return
    logger.info("Evaluating updated flow and attaching it to the trial...")
    try:
        eval_results = eval_dataset(
            study_config, study_config.dataset, flow, study_config.evaluation.mode
        )
    except Exception:
        logger.exception("Failed to evaluate resulting flow %s", flow)
        return
    obj1 = eval_results[study_config.optimization.objective_1_name]
    obj2 = eval_results[study_config.optimization.objective_2_name]
    distributions = study_config.search_space.build_distributions(
        params=old_flow_params
    )
    logger.info("Saving the updated flow...")
    if "enforce_full_evaluation" in old_flow_params:
        old_flow_params.pop("enforce_full_evaluation")
    new_trial = optuna.create_trial(
        values=[obj1, obj2],
        params=old_flow_params,
        distributions=distributions,
        state=optuna.trial.TrialState.COMPLETE,
    )
    new_trial.set_user_attr("parent_number", parent_number)
    new_study.add_trial(new_trial)
    logger.info("Done replacing the models for %s", flow_params)


@ray.remote
def replace_models_remote(
    flow_params: T.Dict[str, T.Any],
    study_config: StudyConfig,
    new_study: optuna.Study,
    parent_number: int,
):
    """A wrapper for running model replacement on a Ray cluster."""
    return replace_models(flow_params, study_config, new_study, parent_number)


def run_pareto_flows_replace_models(study_path: str, remote: bool = False):
    """
    Reads Pareto flows, replaces models in them and re-evaluates resulting flow.
    """
    logger.info("Loading study config from file %s", study_path)
    study_config = StudyConfig.from_file(study_path)
    logger.info("Loading pareto flows for study %s", study_config.name)
    storage = cfg.database.get_optuna_storage()
    study = optuna.load_study(
        study_name=study_config.name,
        storage=storage,
    )
    df_pareto = get_pareto_df(study_config, success_rate=0.5)
    new_study_name = f"{study_config.name}_large_models"
    logger.info("Creating new study %s for updated trials...", new_study_name)
    po_study = optuna.create_study(
        study_name=new_study_name,
        directions=study.directions,
        storage=storage,
        load_if_exists=True,
    )
    logger.info("Running the replacement...")
    if remote:
        opts: T.List[T.Any] = []
        for _, row in df_pareto.iterrows():
            flow_params = json.loads(row["user_attrs_flow"])
            parent_number = row["number"]
            opts.append(
                replace_models_remote.remote(
                    flow_params, study_config, po_study, parent_number
                )
            )
        ray.get(opts)
    else:
        for _, row in df_pareto.iterrows():
            parent_number = row["number"]
            flow_params = json.loads(row["user_attrs_flow"])
            replace_models(flow_params, study_config, po_study, parent_number)
    logger.info("Done")


def run_opt(study_path: str, remote: bool):
    """Main workflow wrapper."""
    if remote:
        ray_init()
    run_pareto_flows_replace_models(study_path, remote=remote)


def main():
    """An argument parser."""
    parser = argparse.ArgumentParser()
    default = get_default_study_name()
    parser.add_argument(
        "--study-config",
        help=f"Path to study config yaml (default: {default})",
        default=default,
    )
    parser.add_argument(
        "--remote",
        help="Run the script on Ray",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    run_opt(
        study_path=args.study_config,
        remote=args.remote,
    )


if __name__ == "__main__":
    main()
