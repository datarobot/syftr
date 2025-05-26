import argparse
import ast
import itertools
import json
import socket
import sys
import traceback
import typing as T
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

import numpy as np
import optuna
import ray
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from ray.util import state

from syftr.baselines import set_baselines
from syftr.evaluation import eval_dataset
from syftr.flows import (
    CritiqueAgentFlow,
    Flow,
    LATSAgentFlow,
    RAGFlow,
    ReActAgentFlow,
    RetrieverFlow,
    SubQuestionRAGFlow,
)
from syftr.hf_endpoint_embeddings import HFEndpointEmbeddings
from syftr.huggingface_helper import get_embedding_model
from syftr.logger import logger
from syftr.optimization import StudyRunner, get_study
from syftr.optuna_helper import (
    get_pareto_flows,
    run_flows,
    trial_exists,
    without_non_search_space_params,
)
from syftr.ray.utils import ray_init
from syftr.retrievers.build import build_rag_retriever
from syftr.startup import prepare_worker
from syftr.studies import (
    RetrieverStudyConfig,
    StudyConfig,
    get_default_study_name,
    get_pareto_study_config,
)
from syftr.templates import get_template
from syftr.tracing import get_span_exporter, set_tracing_metrics


def _get_examples(example_retriever: BaseRetriever, query_str: str):
    retrieved_nodes: T.List[NodeWithScore] = example_retriever.retrieve(query_str)
    result_strs = []
    for n in retrieved_nodes:
        try:
            raw_dict = ast.literal_eval(n.text)
            query = raw_dict["query"]
            response = raw_dict["response"]
            result_str = dedent(
                f"""\
                Question: {query}
                Answer: {response}"""
            )
            result_strs.append(result_str)
        except SyntaxError as exc:
            logger.warning("Converting example to dictionary failed: %s", exc)
            result_strs.append(n.text)
    return "\n\n".join(result_strs)


def _get_example_retriever(params, study_config: StudyConfig, embedding_model):
    assert embedding_model, "No embedding model for dynamic few-shot prompting"
    logger.info("Building few-shot retriever")
    dataset_iter = study_config.dataset.iter_examples(partition="train")
    logger.info("Getting few-shot examples from dataset")
    if study_config.toy_mode:
        dataset_iter = itertools.islice(dataset_iter, 20)
    few_shot_nodes = []
    for pair in dataset_iter:
        line = f"{{'query': '''{pair.question}''', 'response': '''{pair.answer}'''}}"
        few_shot_nodes.append(TextNode(text=line))

    if not isinstance(embedding_model, HFEndpointEmbeddings):
        embedding_model.reset_timeouts(total_chunks=len(few_shot_nodes))
    logger.info("Building few-shot retriever index")
    few_shot_index = VectorStoreIndex(nodes=few_shot_nodes, embed_model=embedding_model)
    logger.info("Built few-shot retriever index")
    few_shot_retriever = few_shot_index.as_retriever(
        similarity_top_k=params["few_shot_top_k"], similarity_threshold=None
    )

    def get_qa_examples(query_str, **kwargs):
        return _get_examples(few_shot_retriever, query_str)

    return get_qa_examples


def evaluate(
    params: T.Dict,
    study_config: StudyConfig,
) -> T.Tuple[float, float, T.Dict[str, T.Any], str]:
    prepare_worker()
    flow_start = datetime.now(timezone.utc).timestamp()
    logger.info("Evaluating flow with config: %s", params)
    flow_json = json.dumps(params)
    if study_config.evaluation.use_tracing_metrics:
        span_exporter = get_span_exporter()
    obj1, obj2, results = _evaluate(params, study_config)
    if study_config.evaluation.use_tracing_metrics:
        set_tracing_metrics(span_exporter, results)
    results["failed"] = False
    results["flow_start"] = flow_start
    results["flow_end"] = datetime.now(timezone.utc).timestamp()
    results["flow_duration"] = float(results["flow_end"]) - float(results["flow_start"])
    logger.info("Evaluation finished. Finalizing trial report. %s", results)
    return obj1, obj2, results, flow_json


def build_flow(params: T.Dict, study_config: StudyConfig) -> Flow:
    from syftr.llm import get_llm

    response_synthesizer_llm = get_llm(params["response_synthesizer_llm"])
    enforce_full_evaluation = params.get("enforce_full_evaluation", False)

    if study_config.is_retriever_study:
        hyde_llm = (
            get_llm(params["hyde_llm_name"]) if params.get("hyde_enabled") else None
        )
        retriever, docstore = build_rag_retriever(study_config, params)
        return RetrieverFlow(
            response_synthesizer_llm=response_synthesizer_llm,
            retriever=retriever,
            docstore=docstore,
            hyde_llm=hyde_llm,
            additional_context_num_nodes=params.get("additional_context_num_nodes", 0),
            params=params,
            enforce_full_evaluation=enforce_full_evaluation,
        )

    get_qa_examples = None
    is_few_shot = study_config.search_space.is_few_shot(params)
    if is_few_shot:
        few_shot_embedding_model_name = params["few_shot_embedding_model"]
        few_shot_embedding_model, _ = get_embedding_model(
            few_shot_embedding_model_name,
            timeout_config=study_config.timeouts,
            device=study_config.optimization.embedding_device,
            use_hf_endpoint_models=study_config.optimization.use_hf_embedding_models,
        )
        get_qa_examples = _get_example_retriever(
            params, study_config, few_shot_embedding_model
        )

    do_rag = params["rag_mode"] != "no_rag"
    template_name = params["template_name"]
    template = get_template(
        template_name, with_context=do_rag, with_few_shot_prompt=is_few_shot
    )

    flow: T.Any

    if not do_rag:
        flow = Flow(
            response_synthesizer_llm=response_synthesizer_llm,
            template=template,
            get_examples=get_qa_examples,
            params=params,
            enforce_full_evaluation=enforce_full_evaluation,
        )
    else:
        hyde_llm = reranker_llm = reranker_top_k = None
        if params.get("hyde_enabled"):
            hyde_llm = get_llm(params["hyde_llm_name"])
        if params.get("reranker_enabled"):
            reranker_llm = get_llm(params["reranker_llm_name"])
            reranker_top_k = params["reranker_top_k"]
        if params.get("additional_context_enabled"):
            additional_context_num_nodes = params["additional_context_num_nodes"]
        else:
            additional_context_num_nodes = 0

        rag_retriever, rag_docstore = build_rag_retriever(study_config, params)

        match params["rag_mode"]:
            case "rag":
                flow = RAGFlow(
                    retriever=rag_retriever,
                    response_synthesizer_llm=response_synthesizer_llm,
                    docstore=rag_docstore,
                    template=template,
                    get_examples=get_qa_examples,
                    hyde_llm=hyde_llm,
                    reranker_llm=reranker_llm,
                    reranker_top_k=reranker_top_k,
                    additional_context_num_nodes=additional_context_num_nodes,
                    enforce_full_evaluation=enforce_full_evaluation,
                    params=params,
                )
            case "react_rag_agent":
                subquestion_engine_llm = get_llm(params["subquestion_engine_llm"])
                subquestion_response_synthesizer_llm = get_llm(
                    params["subquestion_response_synthesizer_llm"]
                )
                max_iterations = params["max_iterations"]
                flow = ReActAgentFlow(
                    retriever=rag_retriever,
                    response_synthesizer_llm=response_synthesizer_llm,
                    subquestion_response_synthesizer_llm=subquestion_response_synthesizer_llm,
                    subquestion_engine_llm=subquestion_engine_llm,
                    max_iterations=max_iterations,
                    docstore=rag_docstore,
                    template=template,
                    get_examples=get_qa_examples,
                    hyde_llm=hyde_llm,
                    reranker_llm=reranker_llm,
                    reranker_top_k=reranker_top_k,
                    additional_context_num_nodes=additional_context_num_nodes,
                    dataset_name=study_config.dataset.name,
                    dataset_description=study_config.dataset.description,
                    enforce_full_evaluation=enforce_full_evaluation,
                    params=params,
                )
            case "critique_rag_agent":
                subquestion_engine_llm = get_llm(params["subquestion_engine_llm"])
                subquestion_response_synthesizer_llm = get_llm(
                    params["subquestion_response_synthesizer_llm"]
                )
                critique_agent_llm = get_llm(params["critique_agent_llm"])
                reflection_agent_llm = get_llm(params["reflection_agent_llm"])
                max_iterations = params["max_iterations"]
                flow = CritiqueAgentFlow(
                    response_synthesizer_llm=response_synthesizer_llm,
                    subquestion_engine_llm=subquestion_engine_llm,
                    subquestion_response_synthesizer_llm=subquestion_response_synthesizer_llm,
                    critique_agent_llm=critique_agent_llm,
                    reflection_agent_llm=reflection_agent_llm,
                    max_iterations=max_iterations,
                    retriever=rag_retriever,
                    docstore=rag_docstore,
                    template=template,
                    get_examples=get_qa_examples,
                    hyde_llm=hyde_llm,
                    reranker_llm=reranker_llm,
                    reranker_top_k=reranker_top_k,
                    additional_context_num_nodes=additional_context_num_nodes,
                    dataset_name=study_config.dataset.name,
                    dataset_description=study_config.dataset.description,
                    enforce_full_evaluation=enforce_full_evaluation,
                    params=params,
                )
            case "sub_question_rag":
                subquestion_engine_llm = get_llm(params["subquestion_engine_llm"])
                subquestion_response_synthesizer_llm = get_llm(
                    params["subquestion_response_synthesizer_llm"]
                )
                flow = SubQuestionRAGFlow(
                    response_synthesizer_llm=response_synthesizer_llm,
                    subquestion_engine_llm=subquestion_engine_llm,
                    subquestion_response_synthesizer_llm=subquestion_response_synthesizer_llm,
                    retriever=rag_retriever,
                    docstore=rag_docstore,
                    template=template,
                    get_examples=get_qa_examples,
                    hyde_llm=hyde_llm,
                    reranker_llm=reranker_llm,
                    reranker_top_k=reranker_top_k,
                    additional_context_num_nodes=additional_context_num_nodes,
                    dataset_name=study_config.dataset.name,
                    dataset_description=study_config.dataset.description,
                    enforce_full_evaluation=enforce_full_evaluation,
                    params=params,
                )
            case "lats_rag_agent":
                flow = LATSAgentFlow(
                    retriever=rag_retriever,
                    response_synthesizer_llm=response_synthesizer_llm,
                    docstore=rag_docstore,
                    template=template,
                    get_examples=get_qa_examples,
                    hyde_llm=hyde_llm,
                    reranker_llm=reranker_llm,
                    reranker_top_k=reranker_top_k,
                    additional_context_num_nodes=additional_context_num_nodes,
                    dataset_name=study_config.dataset.name,
                    dataset_description=study_config.dataset.description,
                    num_expansions=params["lats_num_expansions"],
                    max_rollouts=params["lats_max_rollouts"],
                    enforce_full_evaluation=enforce_full_evaluation,
                    params=params,
                )
            case _:
                raise ValueError(f"Invalid rag_mode: {params['rag_mode']}")

    return flow


def _evaluate(
    params: T.Dict,
    study_config: StudyConfig,
) -> T.Tuple[float, float, T.Dict[str, float | str]]:
    flow = build_flow(params, study_config)

    results: T.Dict[str, T.Any] = eval_dataset(
        study_config=study_config,
        dataset_iter=study_config.dataset,
        flow=flow,
        evaluation_mode=study_config.evaluation.mode,
    )

    obj1 = results[study_config.optimization.objective_1_name]
    obj2 = results[study_config.optimization.objective_2_name]
    if np.isnan(obj2):
        logger.fatal(
            "%s value is NaN and the trial will crash.\nParams: %s",
            study_config.optimization.objective_2_name,
            json.dumps(params, indent=2),
        )
    return obj1, obj2, results


def objective(
    trial: optuna.Trial,
    study_config: StudyConfig,
    components: T.List[str],
) -> T.Tuple[float, float]:
    from syftr.tuner.core import set_trial

    logger.debug("Syftr objective is running with: %s", sys.executable)

    search_space = study_config.search_space
    params: dict[str, str | bool | int | float]
    for i in range(study_config.optimization.num_retries_unique_params):
        params = search_space.sample(trial, components)
        if not study_config.optimization.skip_existing:
            logger.info("Using generated parameter combination without check")
            break
        if not trial_exists(study_config.name, params):
            logger.info(
                "Found novel parameter combination after %i retries: %s",
                i,
                str(params),
            )
            break
    try:
        obj1, obj2, metrics, flow_json = evaluate(params, study_config)
    except Exception as ex:
        logger.exception("Objective had an unhandled exception: %s", ex)
        metrics = {
            "failed": True,
            "exception_message": str(ex),
            "exception_stacktrace": traceback.format_exc(),
            "exception_class": ex.__class__.__name__,
        }
        flow_json = json.dumps(params)
        raise ex
    finally:
        set_trial(
            trial=trial,
            study_config=study_config,
            params=params,
            is_seeding=False,
            metrics=metrics,
            flow_json=flow_json,
        )
        worker_state = state.get_worker(ray.get_runtime_context().get_worker_id())
        if worker_state is not None:
            trial.set_user_attr("ray_worker_pid", worker_state.pid)
        trial.set_user_attr("ray_worker_hostname", socket.gethostname())
    logger.debug("Optimizer finished trial with params: %s", trial.params)
    return obj1, obj2


@ray.remote
def run_flow(flows: T.Dict[str, T.Any], study_config: StudyConfig) -> None:
    try:
        _run_flow(flows, study_config)
    except Exception:
        logger.exception("Trial had an unhandled exception: %s", flows)


def _run_flow(flow: T.Dict[str, T.Any], study_config: StudyConfig) -> None:
    from syftr.configuration import cfg
    from syftr.tuner.core import set_trial

    logger.debug("Syftr seeder is running with: %s", sys.executable)

    study_name = study_config.name
    params = without_non_search_space_params(flow, study_config)
    search_space = study_config.search_space

    logger.info("Loading study: %s", study_name)
    study = optuna.load_study(
        study_name=study_name, storage=cfg.database.get_optuna_storage()
    )

    if study_config.optimization.skip_existing and trial_exists(
        study_name, params, cfg.database.get_optuna_storage()
    ):
        logger.warning(
            "Flow already exists in study '%s': %s",
            study_name,
            str(flow),
        )
        return

    logger.info("Seeding study '%s' with baseline: %s", study_name, str(params))
    distributions = search_space.build_distributions(params=params)
    try:
        obj1, obj2, metrics, flow_json = evaluate(flow, study_config)
        trial = optuna.create_trial(
            values=[obj1, obj2],
            params=params,
            distributions=distributions,
        )
    except Exception as ex:
        logger.exception("Seeding had an unhandled exception: %s", ex)
        trial = optuna.create_trial(
            params=params,
            distributions=distributions,
            state=optuna.trial.TrialState.FAIL,
        )
        metrics = {
            "failed": True,
            "exception_message": str(ex),
            "exception_stacktrace": traceback.format_exc(),
            "exception_class": ex.__class__.__name__,
        }
        flow_json = json.dumps(params)
        raise ex
    set_trial(trial, study_config, params, True, metrics, flow_json)
    worker_state = state.get_worker(ray.get_runtime_context().get_worker_id())
    if worker_state is not None:
        trial.set_user_attr("ray_worker_pid", worker_state.pid)
    trial.set_user_attr("ray_worker_hostname", socket.gethostname())
    study.add_trial(trial)
    logger.debug("Seeding added trial with params: %s", trial.params)


def run_pareto_flows(study_config: StudyConfig):
    assert study_config.pareto is not None, "Pareto config is not set"
    if study_config.pareto.raise_on_same_study:
        assert study_config.pareto.name != study_config.name, (
            "Pareto study is the same as the main study"
        )
    pareto_flows = get_pareto_flows(study_config)
    logger.info(
        "Evaluating %i Pareto-flows from %s using study %s",
        len(pareto_flows),
        study_config.name,
        study_config.pareto.name,
    )
    pareto_study_config = get_pareto_study_config(study_config)

    # if we want to replace llm name with a single llm name
    if study_config.pareto.replacement_llm_name:
        for flow in pareto_flows:
            study_config.replace_llm_name(flow)

    get_study(pareto_study_config)
    logger.info("Pareto study config is: %s", pareto_study_config)
    run_flows(
        flows=pareto_flows,
        study_config=pareto_study_config,
        runner=run_flow,
    )


def run(
    study_config: StudyConfig | str,
    skip_optimization: bool = False,
    skip_pareto: bool = True,
    remote: bool = False,
) -> None:
    if isinstance(study_config, str):
        study_config_file = Path(study_config)
        study_config = StudyConfig.from_file(study_config_file)
        if study_config.evaluation.mode == "retriever":
            study_config = RetrieverStudyConfig.from_file(study_config_file)
    logger.info("Active configuration is: %s", study_config)

    ray_init(remote)

    if skip_optimization:
        logger.info("Skipping optimization")
    else:
        logger.info("Running optimization using study: %s", study_config.name)
        study_config = set_baselines(study_config)
        # Kick off the study
        optimization = StudyRunner(
            objective=objective,
            study_config=study_config,
            seeder=run_flow,
        )
        optimization.run()

    # Evaluate the Pareto front as configured
    if skip_pareto:
        logger.info("Skipping Pareto evaluation")
    else:
        if study_config.pareto:
            run_pareto_flows(study_config)
        else:
            logger.warning("No Pareto config found in study config")


def main():
    parser = argparse.ArgumentParser()
    default = get_default_study_name()
    parser.add_argument(
        "--study-config",
        help=f"Path to study config yaml (default: {default})",
        default=default,
    )
    parser.add_argument(
        "--skip-optimization",
        help="Do not run the optimization if set",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--skip-pareto",
        help="Do not run the the separate Pareto evaluation",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--remote",
        help="Run the optimization remotely",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    run(
        study_config=args.study_config,
        skip_optimization=args.skip_optimization,
        skip_pareto=args.skip_pareto,
        remote=args.remote,
    )


if __name__ == "__main__":
    main()
