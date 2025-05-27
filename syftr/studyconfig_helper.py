import typing as T
from copy import deepcopy
from pathlib import Path

import yaml

from syftr.configuration import cfg
from syftr.storage import DRDocsHF, SyftrQADataset
from syftr.studies import (
    CritiqueRAGAgent,
    Evaluation,
    FewShotRetriever,
    Hyde,
    LATSRagAgent,
    OptimizationConfig,
    QueryDecomposition,
    ReactRAGAgent,
    Reranker,
    Retriever,
    SearchSpace,
    Splitter,
    StudyConfig,
    SubQuestionRAGAgent,
    TimeoutConfig,
    TopK,
    TransferLearningConfig,
)


def _build_study_name(
    dataset: SyftrQADataset, bench_num: int, prefix: str, run_name: str
) -> str:
    dataset_name = dataset.name.replace("/", "-")
    study_name = f"{prefix}{bench_num}--{run_name}--{dataset_name}"
    if hasattr(dataset, "subset") and dataset.subset:
        study_name += f"--{dataset.subset}"
    return study_name


def build_configs(
    datasets: T.List[SyftrQADataset],
    search_space: SearchSpace,
    optimization_config: OptimizationConfig,
    evaluation: Evaluation,
    bench_num: int,
    reuse_study: bool,
    recreate_study: bool,
    prefix: str,
    run_name: str,
    embedding_max_time: int,
    transfer_learning: TransferLearningConfig | None = None,
) -> T.Tuple[T.List[StudyConfig], T.List[Path]]:
    configs = []
    for dataset in datasets:
        _search_space = deepcopy(search_space)
        configs.append(
            StudyConfig(
                name=_build_study_name(dataset, bench_num, prefix, run_name),
                dataset=dataset,
                search_space=_search_space,
                optimization=optimization_config,
                # transfer_learning=transfer_learning,
                toy_mode=False,
                reuse_study=reuse_study,
                recreate_study=recreate_study,
                evaluation=evaluation,
                timeouts=TimeoutConfig(
                    embedding_timeout_active=True, embedding_max_time=embedding_max_time
                ),
            )
        )

    paths = [cfg.paths.studies_dir / f"{config.name}.yaml" for config in configs]
    for config, path in zip(configs, paths):
        with open(path, "w") as handle:
            yaml.dump(config.dict(), handle)

    paths = [path.relative_to(cfg.paths.root_dir) for path in paths]

    return configs, paths


def build_example_config(
    llms: T.List[str],
    embedding_models: T.List[str],
    num_trials: int | None = 10,
    max_concurrent_trials: int | None = 1,
    num_eval_samples: int | None = 50,
    num_random_trials: int | None = 10,
    embedding_max_time: int | None = 3600 * 8,
    datasets: T.List[SyftrQADataset] | None = None,
    optimization_config: OptimizationConfig | None = None,
):
    assert isinstance(num_trials, int), "num_trials must be an int"
    assert isinstance(max_concurrent_trials, int), (
        "max_concurrent_trials must be an int"
    )
    assert isinstance(num_eval_samples, int), "num_eval_samples must be an int"
    assert isinstance(num_random_trials, int), "num_random_trials must be an int"
    assert isinstance(embedding_max_time, int), "embedding_max_time must be an int"

    datasets = datasets or [DRDocsHF()]

    optimization_config = optimization_config or OptimizationConfig(
        method="expanding",
        shuffle_blocks=False,
        num_trials=num_trials,
        baselines_cycle_llms=False,
        shuffle_baselines=True,
        max_concurrent_trials=max_concurrent_trials,
        num_eval_samples=num_eval_samples,
        num_eval_batch=5,
        rate_limiter_max_coros=10,
        rate_limiter_period=60,
        max_trial_cost=40.0,
        cpus_per_trial=1,
        seeder_timeout=3600 * 1,  # None: wait until finished, 0: don't wait
        num_random_trials=num_random_trials,
        use_individual_baselines=False,
        use_agent_baselines=False,
        use_variations_of_baselines=False,
        use_pareto_baselines=False,
        use_pareto_pruner=True,
        use_cost_pruner=True,
        use_runtime_pruner=True,
        use_toy_baselines=False,
        sampler="tpe",
    )

    search_space = SearchSpace(
        few_shot_enabled=[False, True],
        additional_context_enabled=[False, True],
        hyde_enabled=[False, True],
        reranker_enabled=[False, True],
        splitter=Splitter(
            methods=[
                "recursive",
                "sentence",
                "token",
            ],
            chunk_min_exp=7,
            chunk_max_exp=10,
            chunk_overlap_frac_min=0.0,
            chunk_overlap_frac_max=0.5,
            chunk_overlap_frac_step=0.05,
        ),
        rag_modes=[
            "no_rag",
            "rag",
            "lats_rag_agent",
            "react_rag_agent",
            "critique_rag_agent",
            "sub_question_rag",
        ],
        template_names=[
            "default",
            "concise",
            "CoT",
            # "finance-expert",
        ],
        response_synthesizer_llms=llms,
        rag_retriever=Retriever(
            embedding_models=embedding_models,
            methods=["dense", "sparse", "hybrid"],
            top_k=TopK(kmin=1, kmax=10, log=False),
            query_decomposition=QueryDecomposition(
                llm_names=llms,
                num_queries_min=2,
                num_queries_max=5,
                num_queries_step=1,
            ),
        ),
        react_rag_agent=ReactRAGAgent(
            subquestion_engine_llms=llms,
            subquestion_response_synthesizer_llms=llms,
        ),
        sub_question_rag=SubQuestionRAGAgent(
            subquestion_engine_llms=llms,
            subquestion_response_synthesizer_llms=llms,
        ),
        critique_rag_agent=CritiqueRAGAgent(
            subquestion_engine_llms=llms,
            subquestion_response_synthesizer_llms=llms,
            critique_agent_llms=llms,
            reflection_agent_llms=llms,
        ),
        lats_rag_agent=LATSRagAgent(),
        reranker=Reranker(llms=llms),
        hyde=Hyde(llms=llms),
        few_shot_retriever=FewShotRetriever(
            embedding_models=embedding_models,
        ),
    )

    evaluation = Evaluation(
        mode="random",
        llms=llms,
        raise_on_exception=False,
    )

    configs, paths = build_configs(
        datasets=datasets,
        search_space=search_space,
        optimization_config=optimization_config,
        evaluation=evaluation,
        bench_num=1,
        reuse_study=True,
        recreate_study=True,
        prefix="example",
        run_name="small-study",
        embedding_max_time=embedding_max_time,
        transfer_learning=None,
    )
    return configs, paths
