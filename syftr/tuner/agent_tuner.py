import argparse
import ast
import itertools
import json
import typing as T
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

import optuna
import ray
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import NodeWithScore, TextNode

from syftr.agent_flows import FlowJSONHandler, LlamaIndexReactRAGAgentFlow
from syftr.baselines import set_baselines
from syftr.evaluation import eval_dataset
from syftr.flows import Flow
from syftr.huggingface_helper import get_embedding_model
from syftr.logger import logger
from syftr.optimization import StudyRunner
from syftr.optuna_helper import trial_exists
from syftr.ray.utils import ray_init
from syftr.storage import SyftrQADataset
from syftr.studies import AgentStudyConfig, get_default_study_name
from syftr.templates import get_agent_template, get_template
from syftr.tuner.core import build_splitter


def roundrobin(*iterables):
    """Yields an item from each iterable, alternating between them.

        >>> list(roundrobin('ABC', 'D', 'EF'))
        ['A', 'D', 'E', 'B', 'F', 'C']

    This function produces the same output as :func:`interleave_longest`, but
    may perform better for some inputs (in particular when the number of
    iterables is small).

    """
    pending = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for n in nexts:
                yield n()
        except StopIteration:
            pending -= 1
            nexts = itertools.cycle(itertools.islice(nexts, pending))


def _get_examples(example_retriever: BaseRetriever, query_str: str):
    retrieved_nodes: T.List[NodeWithScore] = example_retriever.retrieve(query_str)
    result_strs = []
    for n in retrieved_nodes:
        raw_dict = ast.literal_eval(n.text)
        query = raw_dict["query"]
        response = raw_dict["response"]
        result_str = dedent(
            f"""\
            Question: {query}
            Answer: {response}"""
        )
        result_strs.append(result_str)
    return "\n\n".join(result_strs)


def _build_dense_index(
    dataset: SyftrQADataset,
    transforms,
    embedding_model: BaseEmbedding,
) -> VectorStoreIndex:
    from syftr.configuration import cfg

    pipeline = IngestionPipeline(transformations=transforms)
    nodes = pipeline.run(
        documents=list(dataset.iter_grounding_data()),
        show_progress=cfg.optuna.show_progress,
    )
    return VectorStoreIndex(
        nodes=nodes,
        embed_model=embedding_model,
        insert_batch_size=2048,
        show_progress=cfg.optuna.show_progress,
    )


def _get_example_retriever(params, study_config: AgentStudyConfig):
    if params["prompt_name"] != "few-shot":
        return None, None

    dataset_iter = roundrobin(
        *[ds.iter_grounding_data() for ds in study_config.datasets]
    )
    if study_config.toy_mode:
        dataset_iter = list(itertools.islice(dataset_iter, 20))
    few_shot_nodes = []
    for pair in dataset_iter:
        line = f"{{'query': '''{pair.question}''', 'response': '''{pair.answer}'''}}"
        few_shot_nodes.append(TextNode(text=line))
    embedding_model_name = params.get("embedding_model", None)
    embedding_model, is_onnx = get_embedding_model(embedding_model_name)
    few_shot_index = VectorStoreIndex(nodes=few_shot_nodes, embed_model=embedding_model)
    few_shot_retriever = few_shot_index.as_retriever(
        similarity_top_k=10, similarity_threshold=None
    )

    def get_qa_examples(query_str, **kwargs):
        return _get_examples(few_shot_retriever, query_str)

    return (
        get_qa_examples,
        embedding_model,
    )


def _get_kwargs(params: T.Dict):
    from syftr.llm import get_llm

    kwargs = deepcopy(params)
    kwargs.pop("retriever", None)
    kwargs.pop("embedding_model", None)
    kwargs.pop("splitter", None)
    kwargs.pop("prompt_name", None)
    kwargs.pop("rag_mode", None)
    kwargs["llm_name"] = kwargs.pop("llm")
    reranker_llm = get_llm(kwargs.pop("reranker_llm", None))
    if reranker_llm:
        kwargs["reranker_llm"] = reranker_llm
    hyde_llm = get_llm(kwargs.pop("hyde_llm", None))
    if hyde_llm:
        kwargs["hyde_llm"] = hyde_llm
    return kwargs


def evaluate(
    params: T.Dict,
    study_config: AgentStudyConfig,
) -> T.Tuple[float, float, T.Dict[str, T.Any], str]:
    flow_start = datetime.now(timezone.utc).timestamp()
    logger.info("Evaluating agent flow with config: %s", params)
    flow_json = json.dumps(params)
    num_cpus = study_config.optimization.cpus_per_trial
    num_gpus = study_config.optimization.gpus_per_trial

    future = _evaluate.options(num_cpus=num_cpus, num_gpus=num_gpus).remote(
        params, study_config
    )
    timeout_secs = study_config.timeouts.eval_timeout
    logger.info("Launched evaluation task for trial. Trial timeout is %s", timeout_secs)
    try:
        acc, lat, results, flow_json = ray.get(future, timeout=timeout_secs)
    except Exception as exc:
        ray.cancel(future, force=True)
        logger.exception("Agent flow evaluation failed. Reporting error data")
        results = {
            "failed": True,
            "exception": str(exc),
            "exception_class": exc.__class__.__name__,
        }
        raise
    results["flow_start"] = flow_start
    results["flow_end"] = datetime.now(timezone.utc).timestamp()
    results["flow_duration"] = float(results["flow_end"]) - float(results["flow_start"])
    logger.info("Agent flow evaluation finished. Finalizing trial report. %s", results)
    return acc, lat, results, flow_json


@ray.remote
def _evaluate(
    params: T.Dict,
    study_config: AgentStudyConfig,
) -> T.Tuple[float, float, T.Dict[str, float | str], str]:
    from syftr.llm import get_llm

    logger.info("Evaluating flow with config: %s", params)
    params = deepcopy(params)
    response_synthesizer_llm = get_llm(params["llm"])
    get_qa_examples = _get_example_retriever(params, study_config)
    system_template = get_agent_template(params["prompt_name"])

    # For baseline.
    flow: T.Any
    if params["rag_mode"] == "no_rag":
        template = get_template("default")
        flow = Flow(
            response_synthesizer_llm=response_synthesizer_llm,
            template=template,
            get_examples=get_qa_examples,
        )
    elif params["rag_mode"] == "rag":
        embedding_model_name = params.get("embedding_model", None)
        embedding_model, is_onnx = get_embedding_model(embedding_model_name)
        template = get_template("default", with_context=True)
        # AgentStudyConfig should be refactored
        splitter = build_splitter(study_config, params)  # type: ignore
        indexes = []
        for ds in study_config.datasets:
            index = _build_dense_index(ds.xname, [splitter], embedding_model)  # type: ignore
            indexes.append((ds.xname, ds.description, index))

        flow = LlamaIndexReactRAGAgentFlow(
            indexes=indexes,
            llm=response_synthesizer_llm,
            template=template,
            system_prompt=PromptTemplate(system_template),
            verbose=True,
            use_hyde=False,
            use_reranker=False,
        )
    else:
        raise ValueError(f"Invalid rag_mode: {params['rag_mode']}")

    flow_json = json.dumps(flow, cls=FlowJSONHandler)

    dataset_iter = roundrobin(
        *[ds.iter_grounding_data() for ds in study_config.datasets]
    )

    results: T.Dict[str, T.Any] = eval_dataset(
        study_config=study_config,
        dataset_iter=dataset_iter,
        flow=flow,
    )
    acc = results["accuracy"]
    lat = results["p80_time"]
    results["is_onnx"] = bool(is_onnx)
    return acc, lat, results, flow_json


def objective(
    trial: optuna.Trial,
    study_config: AgentStudyConfig,
) -> T.Tuple[float, float]:
    from syftr.tuner.core import set_trial

    search_space = study_config.search_space
    params: dict[str, str | bool | int | float]
    for i in range(study_config.optimization.num_retries_unique_params):
        params = {
            "prompt_name": trial.suggest_categorical(
                "prompt_name",
                search_space.prompt_names,
            ),
            "llm": trial.suggest_categorical(
                "llm",
                search_space.llms,
            ),
            "rag_mode": trial.suggest_categorical(
                "rag_mode",
                search_space.rag_modes,
            ),
            "embedding_models": trial.suggest_categorical(
                "embedding_models",
                search_space.embedding_models,
            ),
        }
        if params["rag_mode"] == "rag":
            params.update(search_space.splitter.sample(trial))

        flow_name = "LlamaIndexReactRAGAgentFlow"
        trial.set_user_attr("flow_name", flow_name)
        trial.set_user_attr(
            "datasets", ", ".join(ds.xname for ds in study_config.datasets)
        )

        if not study_config.optimization.skip_existing:
            logger.info("Using generated parameter combination without check")
            break
        if not trial_exists(trial.study.study_name, params):
            logger.info(
                "Found novel parameter combination after %i retries: %s",
                i,
                str(params),
            )
            break

    obj1, obj2, metrics, flow_json = evaluate(params, study_config)
    set_trial(trial, study_config, params, False, metrics, flow_json)  # type: ignore
    return obj1, obj2


@ray.remote
def seed_baselines(
    baseline: T.Dict[str, T.Any], study_config: AgentStudyConfig
) -> None:
    try:
        _seed_baselines(baseline, study_config)
    except Exception:
        logger.exception("Baseline had an unhandled exception. Skipping this baseline")


def _seed_baselines(
    baseline: T.Dict[str, T.Any], study_config: AgentStudyConfig
) -> None:
    from syftr.configuration import cfg
    from syftr.tuner.core import set_trial

    study_name = study_config.name
    search_space = study_config.search_space
    study = optuna.load_study(
        study_name=study_name, storage=cfg.database.get_optuna_storage()
    )
    if trial_exists(study_name, baseline, cfg.database.get_optuna_storage()):
        logger.warning(
            "Skipping evaluation of baseline which already exists in study '%s': %s",
            study_name,
            str(baseline),
        )
        return
    logger.info("Seeding study '%s' with baseline: %s", study_name, str(baseline))
    distributions: T.Dict[str, optuna.distributions.BaseDistribution] = {
        "llm": optuna.distributions.CategoricalDistribution(search_space.llms),
        "rag_mode": optuna.distributions.CategoricalDistribution(
            search_space.rag_modes
        ),
        "prompt_name": optuna.distributions.CategoricalDistribution(
            search_space.prompt_names
        ),
    }
    if baseline["rag_mode"] == "rag":
        distributions.update(**search_space.splitter.build_distributions())

    obj1, obj2, metrics, flow_json = evaluate(baseline, study_config)
    trial = optuna.create_trial(
        values=[obj1, obj2],
        params=baseline,
        distributions=distributions,
    )
    set_trial(trial, study_config, baseline, True, metrics, flow_json)  # type: ignore
    study.add_trial(trial)
    logger.debug("Seeding added trial with params: %s", trial.params)


def main():
    from syftr.configuration import cfg

    parser = argparse.ArgumentParser()
    default = get_default_study_name()
    parser.add_argument(
        "--study-config",
        help=f"Path to study config yaml (default: {default})",
        default=default,
    )
    parser.add_argument(
        "--remote",
        help="Use remote Ray cluster",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()

    study_config_file = Path(args.study_config)
    study_config = AgentStudyConfig.from_file(study_config_file)
    cfg.ray.local = False if args.remote else cfg.ray.local

    ray_init()

    set_baselines(study_config)
    logger.info("Running study: %s", study_config.name)
    optimization = StudyRunner(
        objective=objective,
        study_config=study_config,
        seeder=seed_baselines,
    )
    optimization.run()


if __name__ == "__main__":
    main()
