import argparse
import asyncio
import time
import typing as T

from ray.job_submission import JobStatus

from syftr.configuration import cfg
from syftr.logger import logger
from syftr.optimization import user_confirm_delete
from syftr.optuna_helper import get_pareto_flows
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
    DEFAULT_LLMS,
    LOCAL_EMBEDDING_MODELS,
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
    StudyConfig,
    SubQuestionRAGAgent,
    TimeoutConfig,
    TopK,
    TransferLearningConfig,
)
from syftr.studyconfig_helper import build_configs

PREFIX = "rank"
BENCH_NUM = 0
NUM_TRIALS = 10
USE_PARETO_BASELINES = False
RUN_NAME = "rag-and-agents"
REUSE_STUDY = True
RECREATE_STUDY = True
EVAL_MODE: T.Literal["single", "random", "consensus"] = "random"
DRY_RUN = False  #  a dry run will not submit jobs but create the study configs
EMBEDDING_MAX_TIME = 3600 * 8

blocks = [
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
    # Block(
    #     name="rag_retriever",
    #     num_trials=100,
    #     components=["rag_retriever"],
    # ),
    # Block(
    #     name="main",
    #     num_trials=900,
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
    #         "response_synthesizer_llm",
    #         "sub_question_rag",
    #         "template_name",
    #     ],
    # ),
]


baseline_studies = [
    "rank0--rag-and-agents--financebench_hf",
    "rank1--rag-and-agents--bright_hf",
    "rank1--rag-and-agents--crag_hf-music",
    "rank1--rag-and-agents--crag_hf-sports",
    "rank1--rag-and-agents--drdocs_hf",
    "rank1--rag-and-agents--financebench_hf",
    "rank1--rag-and-agents--hotpotqa_hf-train_hard",
    "rank1--rag-and-agents--infinitebench_hf",
    "rank1--rag-and-agents--multihoprag_hf",
    "rank1--rag-and-agents--phantomwikiv050_hf",
    "rank2--rag-and-agents--bright_hf",
    "rank2--rag-and-agents--crag_hf-music",
    "rank2--rag-and-agents--crag_hf-sports",
    "rank2--rag-and-agents--drdocs_hf",
    "rank2--rag-and-agents--financebench_hf",
    "rank2--rag-and-agents--hotpotqa_hf-train_hard",
    "rank2--rag-and-agents--infinitebench_hf",
    "rank2--rag-and-agents--multihoprag_hf",
    "rank2--rag-and-agents--phantomwikiv050_hf",
]
baselines = []
if USE_PARETO_BASELINES:
    for study in baseline_studies:
        for flow in get_pareto_flows(study, 0.9):
            if flow not in baselines:
                baselines.append(flow)
    print(f"We have {len(baselines)} Pareto-baselines for seeding")

# baselines = json.load(
#     open(cfg.paths.results_dir / "silver-bullet-like-flows.json", "r")
# )

optimization_config = OptimizationConfig(
    method="expanding",
    # blocks=blocks,
    shuffle_blocks=False,
    num_trials=NUM_TRIALS,
    baselines=baselines,
    baselines_cycle_llms=False,
    shuffle_baselines=True,
    max_concurrent_trials=10,
    num_eval_samples=50,
    num_eval_batch=5,
    rate_limiter_max_coros=30,
    rate_limiter_period=60,
    max_trial_cost=40.0,
    cpus_per_trial=1,
    seeder_timeout=3600 * 10,  # None: wait until finished, 0: don't wait
    # -----------------------------------------------
    num_random_trials=0,
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

# transfer_learning = TransferLearningConfig(
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

llms: T.List[str] = LOCAL_LLMS

embedding_models = [
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
    mode=EVAL_MODE,
    llms=[
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "Qwen/Qwen2.5",
        "google/gemma-3-27b-it",
        "nvidia/Llama-3_3-Nemotron-Super-49B",
    ],
    raise_on_exception=False,
)

datasets = [
    FinanceBenchHF(),
    # -----------------------------------------------
    # BrightHF(subset="biology"),
    # CragTask3HF(subset="music"),
    # CragTask3HF(subset="sports"),
    # DRDocsHF(),
    # HotPotQAHF(subset="train_hard"),
    # InfiniteBenchHF(),
    # MultiHopRAGHF(),
    # PhantomWikiv050(),
    # -----------------------------------------------
    # BrightHF(subset="earth_science"),
    # BrightHF(subset="economics"),
    # BrightHF(subset="psychology"),
    # BrightHF(subset="robotics"),
    # BrightHF(subset="stackoverflow"),
    # BrightHF(subset="sustainable_living"),
    # BrightHF(subset="pony"),
    # -----------------------------------------------
    # SyntheticHotPotQAHF(subset="train_hard"),
    # SyntheticFinanceBenchHF(),
    # SyntheticCragTask3HF(subset="sports"),
    # CragTask3HF(subset="movie"),
    # SyntheticCragTask3HF(subset="movie"),
    # SyntheticCragTask3HF(subset="music"),
    # -----------------------------------------------
    # CragTask3HF(subset="finance"),
    # SyntheticCragTask3HF(subset="finance"),
    # -----------------------------------------------
    # CragTask3HF(subset="open"),ms: List[str] = [
    #     "anthropic-haiku-35",
    #     "gemini-flash",
    #     # "gemini-flash2",
    #     # "gemini-pro",
    #     "gpt-4o-mini",
    #     # "llama-33-70B",   # not enough capacity
    #     # "mistral-large",  # not enough capacity
    #     # "phi-4",          # not enough capacity
    #     # "anthropic-sonnet-35",
    #     # "gpt-4o-std",
    #     "o3-mini",
    # ]
    # SyntheticCragTask3HF(subset="open"),
    # -----------------------------------------------
]
assert datasets, "No datasets found. Please check the dataset list."


def derived_representer(dumper, data):
    return dumper.represent_dict({"description": data.description})


async def iter_job_logs(job_logs: T.AsyncIterable):
    async for lines in job_logs:
        print(lines, end="")


async def iter_all_job_logs(tailers: T.List[T.AsyncIterable]):
    log_iters = [iter_job_logs(tailer) for tailer in tailers]
    await asyncio.gather(*log_iters)


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
        transfer_learning=None,
    )

    if DRY_RUN:
        print("Not submitting jobs because DRY_RUN is set to True")
        return

    delete_confirmed = user_confirm_delete(configs[0])

    # launch benchmarks
    assert delete_confirmed

    client = get_client()
    job_ids = []
    for i, (config, path) in enumerate(zip(configs, paths)):
        job_id = start_study(client, path, config, delete_confirmed=delete_confirmed)
        job_ids.append(job_id)
        logger.info("Started job %s", job_id)
        if i + 1 < len(configs):
            # I think this might help the checkpointing bug
            logger.info("Sleeping for 60 seconds before the next submission")
            time.sleep(60)

    # monitor benchmarks
    log_tailers = [client.tail_job_logs(job) for job in job_ids]

    asyncio.run(iter_all_job_logs(log_tailers))


def attach_logs(prefix: str = "<doesntmatch>", remote: bool = True):
    cfg.ray.local = False if remote else cfg.ray.local
    client = get_client()
    job_details = client.list_jobs()
    jobs_to_tail = [
        job
        for job in job_details
        if job.submission_id is not None
        and job.submission_id.startswith(prefix)
        and job.status not in {JobStatus.STOPPED, JobStatus.SUCCEEDED, JobStatus.FAILED}
    ]
    log_tailers = [client.tail_job_logs(job.job_id) for job in jobs_to_tail]
    asyncio.run(iter_all_job_logs(log_tailers))


if __name__ == "__main__":
    main()
