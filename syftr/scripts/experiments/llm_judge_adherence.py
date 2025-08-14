import argparse
import asyncio
import json
import time
import typing as T

from slugify import slugify

from syftr.configuration import cfg
from syftr.experiments import iter_all_job_logs
from syftr.llm import LLM_NAMES__LOCAL_MODELS
from syftr.logger import logger
from syftr.optimization import user_confirm_delete
from syftr.optuna_helper import get_completed_flows, get_pareto_flows
from syftr.ray.submit import get_client, start_study
from syftr.storage import (  # noqa
    BrightHF,
    CragTask3HF,
    DRDocsHF,
    FinanceBenchHF,
    HotPotQAHF,
    InfiniteBenchHF,
    MultiHopRAGHF,
    PhantomWikiv050,
    SyftrQADataset,
    SyntheticCragTask3HF,
    SyntheticFinanceBenchHF,
    SyntheticHotPotQAHF,
)
from syftr.studies import (  # noqa
    LOCAL_EMBEDDING_MODELS,
    Block,
    CritiqueRAGAgent,
    Evaluation,
    FewShotRetriever,
    Hyde,
    LATSRagAgent,
    LLMConfig,
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
from syftr.studyconfig_helper import build_configs

# -------------------------------------------------------
PREFIX = "judge"  # this three parameters
BENCH_NUM = 1  # kind of version number of a type of experiment and used to name
# configurations of private experiments get a prefix 'private--' and are not put under version control
IS_PRIVATE = True
# -------------------------------------------------------
NUM_TRIALS = 100  # total number of optimization trials per submission
REUSE_STUDY = True  # WARNING: if set to False, exsting studies will be deleted!
RECREATE_STUDY = True  # if set to True, recreating an existing study without failed or running trials
EVAL_MODE: T.Literal["single", "random", "consensus"] = "single"
DRY_RUN = False  #  a dry run will not submit jobs but create the study configs
EMBEDDING_MAX_TIME = 3600 * 8
MINUTES_BEFORE_NEXT_SUBMISSION = 1
CUSTOM_BASELINES = None  # "pareto", "all", "silver", None
# -------------------------------------------------------
BASELINE_STUDIES: T.List[str] = []

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
            "response_synthesizer_llm_name",
            "sub_question_rag",
            "template_name",
        ],
    ),
    # Block(
    #     name="rag_retriever",
    #     num_trials=200,
    #     components=["rag_retriever"],
    # ),
    # Block(
    #     name="main",
    #     num_trials=NUM_TRIALS - 200,
    #     components=[
    #         "splitter",
    #         "additional_context",
    #         "few_shot_retriever",
    #         "hyde",
    #         "critique_rag_agent",
    #         "lats_rag_agent",
    #         "react_rag_agent",
    #         "rag_mode",
    #         "reranker",
    #         "response_synthesizer_llm_name",
    #         "sub_question_rag",
    #         "template_name",
    #     ],
    # ),
]

BASELINES = []
if CUSTOM_BASELINES == "pareto":
    for study in BASELINE_STUDIES:
        for flow in get_pareto_flows(study, 0.9):
            if flow not in BASELINES:
                BASELINES.append(flow)
    logger.info(f"We have {len(BASELINES)} Pareto-baselines for seeding")
elif CUSTOM_BASELINES == "all":
    for study in BASELINE_STUDIES:
        for flow in get_completed_flows(study):
            if flow not in BASELINES:
                BASELINES.append(flow)
    logger.info(f"We have {len(BASELINES)} baselines for seeding")
elif CUSTOM_BASELINES == "silver":
    BASELINES = json.load(open(cfg.paths.results_dir / "silver-bullets.json", "r"))
    logger.info(f"We have {len(BASELINES)} silver bullet baselines for seeding")
else:
    logger.info("No custom baselines provided")

# TRANSFER_LEARNING = TransferLearningConfig(
#     studies=[
#         "bench14--small-models--crag-music",
#         "bench14--small-models--drdocs",
#         "bench14--small-models--financebench",
#     ],
#     max_fronts=6,
#     max_total=37,
#     success_rate=0.9,
#     embedding_model="BAAI/bge-large-en-v1.5",
# )

LLMS: T.List[str] = LLM_NAMES__LOCAL_MODELS

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
    response_synthesizer_llm_config=LLMConfig(llm_names=LLMS),
    rag_retriever=Retriever(
        embedding_models=EMBEDDING_MODELS,
        methods=["dense", "sparse", "hybrid"],
        top_k=TopK(kmin=1, kmax=10, log=False),
        query_decomposition=QueryDecomposition(
            llm_config=LLMConfig(llm_names=LLMS),
            num_queries_min=2,
            num_queries_max=5,
            num_queries_step=1,
        ),
    ),
    react_rag_agent=ReactRAGAgent(
        subquestion_engine_llm_config=LLMConfig(llm_names=LLMS),
        subquestion_response_synthesizer_llm_config=LLMConfig(llm_names=LLMS),
    ),
    sub_question_rag=SubQuestionRAGAgent(
        subquestion_engine_llm_config=LLMConfig(llm_names=LLMS),
        subquestion_response_synthesizer_llm_config=LLMConfig(llm_names=LLMS),
    ),
    critique_rag_agent=CritiqueRAGAgent(
        subquestion_engine_llm_config=LLMConfig(llm_names=LLMS),
        subquestion_response_synthesizer_llm_config=LLMConfig(llm_names=LLMS),
        critique_agent_llm_config=LLMConfig(llm_names=LLMS),
        reflection_agent_llm_config=LLMConfig(llm_names=LLMS),
    ),
    lats_rag_agent=LATSRagAgent(),
    reranker=Reranker(llm_config=LLMConfig(llm_names=LLMS)),
    hyde=Hyde(llm_config=LLMConfig(llm_names=LLMS)),
    few_shot_retriever=FewShotRetriever(
        embedding_models=EMBEDDING_MODELS,
    ),
)

DATASETS = [
    # BrightHF(subset="biology"),
    # CragTask3HF(subset="music"),
    # CragTask3HF(subset="sports"),
    # DRDocsHF(),
    FinanceBenchHF(),
    # HotPotQAHF(subset="train_hard"),
    InfiniteBenchHF(),
    MultiHopRAGHF(),
    # PhantomWikiv050(),
    # -----------------------------------------------
    # BrightHF(subset="stackoverflow"),
    # -----------------------
    # BrightHF(subset="psychology"),
    # BrightHF(subset="earth_science"),
    # BrightHF(subset="economics"),
    # BrightHF(subset="robotics"),
    # BrightHF(subset="sustainable_living"),
    # BrightHF(subset="pony"),
]
assert DATASETS, "No datasets found. Please check the dataset list."


def get_optimization_parameters():
    optimization_config = OptimizationConfig(
        method="expanding",
        blocks=BLOCKS,
        shuffle_blocks=False,
        num_trials=NUM_TRIALS,
        baselines=BASELINES,
        baselines_cycle_llms=True,
        shuffle_baselines=True,
        max_concurrent_trials=49,
        num_eval_samples=49,
        num_eval_batch=5,
        rate_limiter_max_coros=30,  # control the number of concurrent evals ...
        rate_limiter_period=60,  # ... per given time unit
        max_trial_cost=40.0,
        cpus_per_trial=1,
        seeder_timeout=None,  # None: wait until finished, 0: don't wait
        # -----------------------------------------------
        # num_random_trials=0,
        num_random_trials=NUM_TRIALS,
        # -----------------------------------------------
        use_individual_baselines=False,
        use_agent_baselines=False,
        use_variations_of_baselines=False,
        # -----------------------------------------------
        use_pareto_baselines=False,  # required for transfer learning
        # -----------------------------------------------
        use_pareto_pruner=True,
        use_cost_pruner=True,
        use_runtime_pruner=True,
        # -----------------------------------------------
        use_toy_baselines=False,
        # -----------------------------------------------
        sampler="tpe",
    )

    for llm in LLMS:
        evaluation = Evaluation(
            mode=EVAL_MODE,
            raise_on_exception=False,
        )
        evaluation.llm_names = [llm]
        yield DATASETS, SEARCH_SPACE, optimization_config, evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--remote",
        help="Use remote Ray cluster",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--number",
        type=int,
        default=BENCH_NUM,
        help="The benchmark number used to set up configurations",
    )
    args = parser.parse_args()
    cfg.ray.local = False if args.remote else cfg.ray.local

    job_ids = []
    client = get_client()
    for (
        datasets,
        search_space,
        optimization_config,
        evaluation,
    ) in get_optimization_parameters():
        run_name = slugify(evaluation.llm_names[0])
        configs, paths = build_configs(
            datasets=datasets,
            search_space=search_space,
            optimization_config=optimization_config,
            evaluation=evaluation,
            bench_num=args.number,
            reuse_study=REUSE_STUDY,
            recreate_study=RECREATE_STUDY,
            prefix=PREFIX,
            run_name=run_name,
            embedding_max_time=EMBEDDING_MAX_TIME,
            transfer_learning=None,
            is_private=IS_PRIVATE,
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
            if i + 1 < len(configs):
                # I think this might help the checkpointing bug
                logger.info("Sleeping for 60 seconds before the next submission")
                time.sleep(int(60 * MINUTES_BEFORE_NEXT_SUBMISSION))

    # monitor benchmarks
    log_tailers = [client.tail_job_logs(job) for job in job_ids]

    asyncio.run(iter_all_job_logs(log_tailers))  # type: ignore


if __name__ == "__main__":
    main()
