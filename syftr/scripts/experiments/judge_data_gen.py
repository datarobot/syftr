import asyncio
import time
import typing as T

from syftr.configuration import cfg
from syftr.experiments import iter_all_job_logs
from syftr.logger import logger
from syftr.optimization import user_confirm_delete
from syftr.optuna_helper import get_pareto_flows
from syftr.ray.submit import get_client, start_study
from syftr.storage import (
    CragTask3HF,
    DRDocsHF,  # noqa
    FinanceBenchHF,
    HotPotQAHF,
    MultiHopRAGHF,
)
from syftr.studies import (
    LOCAL_EMBEDDING_MODELS,  # noqa
    LOCAL_LLMS,
    Block,
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
    SubQuestionRAGAgent,
    TopK,
)
from syftr.studyconfig_helper import build_configs

DRY_RUN = True

BENCH_NUM = 1
PREFIX = "judge-data"
RUN_NAME = "test"

NUM_TRIALS = 0  # total number of optimization trials per submission
MAX_CONCURRENT_TRIALS = 10
NUM_EVAL_SAMPLES = 50
REUSE_STUDY = True  # WARNING: if set to False, exsting studies will be deleted!
RECREATE_STUDY = (
    False  # WARNING: do not use with simultaneous runs using the same study!
)
EMBEDDING_MAX_TIME = 3600 * 8
MINUTES_BEFORE_NEXT_SUBMISSION = 2
OBJ2_NAME = "llm_cost_mean"  # "p80_time", "llm_cost_mean", "retriever_context_length"

EVAL_MODE: T.Literal["single", "random", "consensus"] = "single"
EVALUATION = Evaluation(
    mode=EVAL_MODE,
    llms=["gpt-4o-mini"],
    raise_on_exception=False,
)
BLOCKS = [
    Block(
        name="global",
        num_trials=NUM_TRIALS,
        components=[
            "rag_retriever",
            "splitter",
            "additional_context",
            "few_shot_retriever",
            "hyde",
            "critique_rag_agent",
            "lats_rag_agent",
            "react_rag_agent",
            "rag_mode",
            "reranker",
            "response_synthesizer_llm",
            "sub_question_rag",
            "template_name",
        ],
    ),
]

LLMS: T.List[str] = LOCAL_LLMS

EMBEDDING_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "thenlper/gte-large",
    "mixedbread-ai/mxbai-embed-large-v1",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "TencentBAC/Conan-embedding-v1",
    "Linq-AI-Research/Linq-Embed-Mistral",
    "Snowflake/snowflake-arctic-embed-l-v2.0",
    "BAAI/bge-multilingual-gemma2",
]

SEARCH_SPACE = SearchSpace(
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
        # "no_rag",
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
    response_synthesizer_llms=LLMS,
    rag_retriever=Retriever(
        embedding_models=EMBEDDING_MODELS,
        methods=["dense", "sparse", "hybrid"],
        top_k=TopK(kmin=1, kmax=10, log=False),
        query_decomposition=QueryDecomposition(
            llm_names=LLMS,
            num_queries_min=2,
            num_queries_max=5,
            num_queries_step=1,
        ),
    ),
    react_rag_agent=ReactRAGAgent(
        subquestion_engine_llms=LLMS,
        subquestion_response_synthesizer_llms=LLMS,
    ),
    sub_question_rag=SubQuestionRAGAgent(
        subquestion_engine_llms=LLMS,
        subquestion_response_synthesizer_llms=LLMS,
    ),
    critique_rag_agent=CritiqueRAGAgent(
        subquestion_engine_llms=LLMS,
        subquestion_response_synthesizer_llms=LLMS,
        critique_agent_llms=LLMS,
        reflection_agent_llms=LLMS,
    ),
    lats_rag_agent=LATSRagAgent(),
    reranker=Reranker(llms=LLMS),
    hyde=Hyde(llms=LLMS),
    few_shot_retriever=FewShotRetriever(
        embedding_models=EMBEDDING_MODELS,
    ),
)

BASELINE_STUDIES: T.Dict = {
    "seeding1--training--crag_hf-music--music": {
        "dataset": CragTask3HF(subset="music")
    },
    "seeding1--training--financebench_hf": {"dataset": FinanceBenchHF()},
    "seeding1--training--hotpotqa_hf-train_hard--train_hard": {
        "dataset": HotPotQAHF(subset="train_hard")
    },
    "seeding1--training--multihoprag_hf": {"dataset": MultiHopRAGHF()},
}

for study, metadata in BASELINE_STUDIES.items():
    metadata["baselines"] = []
    for flow in get_pareto_flows(study, 0.9):
        if flow not in metadata["baselines"]:
            metadata["baselines"].append(flow)


def get_optimization_parameters():
    for base_study, metadata in BASELINE_STUDIES.items():
        baselines = metadata["baselines"]
        dataset = metadata["dataset"]

        optimization_config = OptimizationConfig(
            method="expanding",
            blocks=BLOCKS,
            shuffle_blocks=False,
            num_trials=NUM_TRIALS,
            baselines=baselines,
            baselines_cycle_llms=False,
            shuffle_baselines=False,
            max_concurrent_trials=MAX_CONCURRENT_TRIALS,
            num_eval_samples=NUM_EVAL_SAMPLES,
            num_eval_batch=5,
            # rate_limiter_max_coros=30,  # control the number of concurrent evals ...
            rate_limiter_max_coros=60,  # control the number of concurrent evals ...
            rate_limiter_period=60,  # ... per given time unit
            max_trial_cost=40.0,
            cpus_per_trial=1,
            seeder_timeout=None,  # None: wait until finished, 0: don't wait
            # -----------------------------------------------
            num_random_trials=0,
            # -----------------------------------------------
            use_individual_baselines=False,
            use_agent_baselines=False,
            use_variations_of_baselines=False,
            # -----------------------------------------------
            use_pareto_baselines=False,  # required for transfer learning
            # -----------------------------------------------
            use_pareto_pruner=False,
            use_cost_pruner=True,
            use_runtime_pruner=True,
            # -----------------------------------------------
            use_toy_baselines=False,
            # -----------------------------------------------
            sampler="tpe",
            objective_2_name=OBJ2_NAME,
        )

        yield [dataset], SEARCH_SPACE, optimization_config, EVALUATION


def main():
    cfg.ray.local = False

    client = get_client()
    job_ids = []
    for (
        datasets,
        search_space,
        optimization_config,
        evaluation,
    ) in get_optimization_parameters():
        configs, paths = build_configs(
            datasets=datasets,
            search_space=search_space,
            optimization_config=optimization_config,
            evaluation=evaluation,
            bench_num=BENCH_NUM,
            reuse_study=REUSE_STUDY,
            recreate_study=RECREATE_STUDY,
            prefix=PREFIX,
            run_name=RUN_NAME,
            embedding_max_time=EMBEDDING_MAX_TIME,
            transfer_learning=None,
        )

        if DRY_RUN:
            print("Not submitting jobs because DRY_RUN is set to True")
            continue

        delete_confirmed = user_confirm_delete(configs[0])

        # launch benchmarks
        assert delete_confirmed

        for i, (config, path) in enumerate(zip(configs, paths)):
            job_id = start_study(
                client, path, config, delete_confirmed=delete_confirmed
            )
            job_ids.append(job_id)
            logger.info("Started job %s", job_id)
            # This might help the checkpointing bug
            sleep_time = 60 * MINUTES_BEFORE_NEXT_SUBMISSION
            logger.info(f"Sleeping for {sleep_time} seconds before the next submission")
            time.sleep(int(sleep_time))

    # monitor benchmarks
    log_tailers = [client.tail_job_logs(job) for job in job_ids]

    asyncio.run(iter_all_job_logs(log_tailers))


if __name__ == "__main__":
    main()
