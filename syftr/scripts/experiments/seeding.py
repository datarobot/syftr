import argparse
import asyncio
import json
import time
import typing as T

from syftr.configuration import cfg
from syftr.experiments import iter_all_job_logs
from syftr.helpers import get_flows_from_trials
from syftr.llm import LLM_NAMES__LOCAL_MODELS
from syftr.logger import logger
from syftr.optimization import user_confirm_delete
from syftr.optuna_helper import (
    get_completed_trials,
    get_pareto_flows,
)
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
PREFIX = "seeding"  # this three parameters
BENCH_NUM = 1  # are used to name
RUN_NAME = "testing-transfer"
# -------------------------------------------------------
NUM_TRIALS = 1000  # total number of optimization trials per submission
NUM_RANDOM_TRIALS = 0
MAX_CONCURRENT_TRIALS = 25
NUM_EVAL_SAMPLES = 50
REUSE_STUDY = True  # WARNING: if set to False, exsting studies will be deleted!
RECREATE_STUDY = (
    True  # WARNING: do not use with simultaneous runs using the same study!
)
EVAL_MODE: T.Literal["single", "random", "consensus"] = "single"
DRY_RUN = False  #  a dry run will not submit jobs but create the study configs
EMBEDDING_MAX_TIME = 3600 * 8
MINUTES_BEFORE_NEXT_SUBMISSION = 2
OBJ2_NAME = "p80_time"  # "p80_time", "llm_cost_mean", "retriever_context_length"
# -------------------------------------------------------
# To seed with silver bullets, you first create the input file using silver_bullets.ipynb notebook
CUSTOM_BASELINES = "transfer"  # valid values are: "pareto", "all", "silver", "transfer", None  # valid values are: "pareto", "all", "silver", "tansfer", None
BASELINES_BATCH_SIZE = 100  # we require batching of baselines to avoid Ray OOM issues
BASELINES_START = 0  # you can restrict the number of baselines ...
BASELINES_END = 100  # ... to start with here to avoid OOM issues
BASELINE_STUDIES: T.List[str] = [
    "seeding1--training--crag_hf-music--music",
    "seeding1--training--financebench_hf",
    "seeding1--training--hotpotqa_hf-train_hard--train_hard",
    "seeding1--training--multihoprag_hf",
]

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

TRANSFER_LEARNING = None
BASELINES = []
if CUSTOM_BASELINES == "pareto":
    for study in BASELINE_STUDIES:
        for flow in get_pareto_flows(study, 0.9):
            if flow not in BASELINES:
                BASELINES.append(flow)
    logger.info(f"We have {len(BASELINES)} Pareto-baselines for seeding")
elif CUSTOM_BASELINES == "all":
    df_trials = get_completed_trials(study=BASELINE_STUDIES)
    df_trials = df_trials.sort_values(by="number")
    flows = get_flows_from_trials(df_trials)
    BASELINES.extend(flows)
    logger.info(f"We have {len(BASELINES)} baselines for seeding")
elif CUSTOM_BASELINES == "silver":
    BASELINES = json.load(open(cfg.paths.results_dir / "silver-bullets.json", "r"))
    logger.info(f"We have {len(BASELINES)} silver bullet baselines for seeding")
elif CUSTOM_BASELINES == "transfer":
    TRANSFER_LEARNING = TransferLearningConfig(
        studies=BASELINE_STUDIES,
        max_fronts=2,
        max_total=23,
        success_rate=0.9,
        embedding_model="BAAI/bge-large-en-v1.5",
    )
else:
    logger.info("No custom baselines provided")


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

EVALUATION = Evaluation(
    mode=EVAL_MODE,
    raise_on_exception=False,
)
EVALUATION.llm_names = ["gpt-4o-mini"]

DATASETS: T.List[SyftrQADataset] = [
    # CragTask3HF(subset="music"),
    # FinanceBenchHF(),
    # HotPotQAHF(subset="train_hard"),
    # MultiHopRAGHF(),
    # -----------------------------------------------
    BrightHF(subset="biology"),
    DRDocsHF(),
    InfiniteBenchHF(),
    PhantomWikiv050(),
    # ###############################################
    # BrightHF(subset="earth_science"),
    # BrightHF(subset="economics"),
    # BrightHF(subset="pony"),
    # BrightHF(subset="psychology"),
    # BrightHF(subset="robotics"),
    # BrightHF(subset="stackoverflow"),
    # BrightHF(subset="sustainable_living"),
    # CragTask3HF(subset="sports"),
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
        max_concurrent_trials=MAX_CONCURRENT_TRIALS,
        num_eval_samples=NUM_EVAL_SAMPLES,
        num_eval_batch=5,
        rate_limiter_max_coros=30,  # control the number of concurrent evals ...
        rate_limiter_period=60,  # ... per given time unit
        max_trial_cost=40.0,
        cpus_per_trial=1,
        seeder_timeout=None,  # None: wait until finished, 0: don't wait
        # -----------------------------------------------
        num_random_trials=NUM_RANDOM_TRIALS,
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
        objective_2_name=OBJ2_NAME,
    )
    if BASELINES:
        start = BASELINES_START or 0
        end = BASELINES_END or len(BASELINES)
        for i in range(start, end, BASELINES_BATCH_SIZE):
            optimization_config = optimization_config.model_copy()
            optimization_config.baselines = BASELINES[i : i + BASELINES_BATCH_SIZE]
            yield DATASETS, SEARCH_SPACE, optimization_config, EVALUATION
    else:
        yield DATASETS, SEARCH_SPACE, optimization_config, EVALUATION


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
        configs, paths = build_configs(
            datasets=datasets,
            search_space=search_space,
            optimization_config=optimization_config,
            evaluation=evaluation,
            bench_num=args.number,
            reuse_study=REUSE_STUDY,
            recreate_study=RECREATE_STUDY,
            prefix=PREFIX,
            run_name=RUN_NAME,
            embedding_max_time=EMBEDDING_MAX_TIME,
            transfer_learning=TRANSFER_LEARNING,
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
            logger.info(
                f"Sleeping for {MINUTES_BEFORE_NEXT_SUBMISSION} minutes before the next submission"
            )
            time.sleep(int(sleep_time))

    # monitor benchmarks
    log_tailers = [client.tail_job_logs(job) for job in job_ids]

    asyncio.run(iter_all_job_logs(log_tailers))  # type: ignore


if __name__ == "__main__":
    main()
