import itertools
import typing as T

from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from syftr.configuration import UNSUPPORTED_PARAMS
from syftr.logger import logger
from syftr.studies import SearchSpace, StudyConfig
from syftr.transfer_learning import get_examples
from syftr.validation import are_valid_parameters

BASELINE_LLM = "anthropic-haiku-35"
BASELINE_RAG_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

SIMPLE_FLOW_TEMPLATE: T.Dict[str, T.Any] = {
    "response_synthesizer_llm": BASELINE_LLM,
    "rag_mode": "no_rag",
    "template_name": "default",
}

SIMPLE_COT_TEMPLATE: T.Dict[str, T.Any] = {
    "response_synthesizer_llm": BASELINE_LLM,
    "rag_mode": "no_rag",
    "template_name": "CoT",
}

SIMPLE_CONCISE_TEMPLATE: T.Dict[str, T.Any] = {
    "response_synthesizer_llm": BASELINE_LLM,
    "rag_mode": "no_rag",
    "template_name": "concise",
}

SIMPLE_FEW_SHOT_TEMPLATE: T.Dict[str, T.Any] = {
    "response_synthesizer_llm": BASELINE_LLM,
    "rag_mode": "no_rag",
    "template_name": "default",
    "few_shot_enabled": True,
    "few_shot_top_k": 3,
    "few_shot_embedding_model": BASELINE_RAG_EMBEDDING_MODEL,
}

DENSE_RAG_TEMPLATE: T.Dict[str, T.Any] = {
    "response_synthesizer_llm": BASELINE_LLM,
    "rag_mode": "rag",
    "rag_method": "dense",
    "rag_embedding_model": BASELINE_RAG_EMBEDDING_MODEL,
    "rag_top_k": 5,
    "rag_query_decomposition_enabled": False,
    "additional_context_enabled": True,
    "additional_context_num_nodes": 2,
    "template_name": "default",
    "splitter_method": "sentence",
    "splitter_chunk_exp": 10,
    "splitter_chunk_overlap_frac": 0.25,
    "reranker_enabled": False,
    "hyde_enabled": False,
}

RERANKER_RAG_TEMPLATE: T.Dict[str, T.Any] = {
    "response_synthesizer_llm": BASELINE_LLM,
    "rag_mode": "rag",
    "rag_method": "dense",
    "template_name": "default",
    "splitter_method": "sentence",
    "rag_embedding_model": BASELINE_RAG_EMBEDDING_MODEL,
    "splitter_chunk_exp": 10,
    "splitter_chunk_overlap_frac": 0.25,
    "rag_top_k": 128,
    "rag_query_decomposition_enabled": False,
    "reranker_enabled": True,
    "reranker_llm_name": BASELINE_LLM,
    "reranker_top_k": 32,
    "hyde_enabled": False,
    "additional_context_enabled": False,
}

SPARSE_RAG_TEMPLATE: T.Dict[str, T.Any] = {
    "response_synthesizer_llm": BASELINE_LLM,
    "rag_mode": "rag",
    "rag_method": "sparse",
    "template_name": "default",
    "splitter_method": "sentence",
    "splitter_chunk_exp": 10,
    "splitter_chunk_overlap_frac": 0.25,
    "rag_top_k": 128,
    "rag_query_decomposition_enabled": False,
    "reranker_enabled": False,
    "hyde_enabled": False,
    "additional_context_enabled": False,
}

HYBRID_RAG_TEMPLATE: T.Dict[str, T.Any] = {
    "response_synthesizer_llm": BASELINE_LLM,
    "rag_mode": "rag",
    "rag_method": "hybrid",
    "rag_top_k": 32,
    "rag_embedding_model": BASELINE_RAG_EMBEDDING_MODEL,
    "template_name": "finance-expert",
    "splitter_method": "sentence",
    "splitter_chunk_exp": 10,
    "splitter_chunk_overlap_frac": 0.25,
    "rag_fusion_mode": "simple",
    "rag_query_decomposition_enabled": True,
    "rag_query_decomposition_llm_name": BASELINE_LLM,
    "rag_query_decomposition_num_queries": 2,
    "rag_hybrid_bm25_weight": 0.5,
    "reranker_enabled": False,
    "hyde_enabled": False,
    "additional_context_enabled": False,
}

RAG_W_FEW_SHOT_TEMPLATE: T.Dict[str, T.Any] = {
    "response_synthesizer_llm": BASELINE_LLM,
    "rag_mode": "rag",
    "rag_method": "dense",
    "rag_embedding_model": BASELINE_RAG_EMBEDDING_MODEL,
    "rag_top_k": 5,
    "rag_query_decomposition_enabled": False,
    "template_name": "default",
    "few_shot_enabled": True,
    "few_shot_top_k": 3,
    "few_shot_embedding_model": "sentence-transformers/all-MiniLM-L12-v2",
    "splitter_method": "sentence",
    "splitter_chunk_exp": 9,
    "splitter_chunk_overlap_frac": 0.5,
    "reranker_enabled": False,
    "hyde_enabled": False,
    "additional_context_enabled": False,
}


TOY_BASELINES: T.List[T.Dict[str, T.Any]] = [
    {
        "response_synthesizer_llm": BASELINE_LLM,
        "rag_mode": "no_rag",
        "template_name": "default",
        "enforce_full_evaluation": True,
    },
]


INDIVIDUAL_BASELINES: T.List[T.Dict[str, T.Any]] = [
    {
        "response_synthesizer_llm": BASELINE_LLM,
        "rag_mode": "no_rag",
        "template_name": "default",
        "enforce_full_evaluation": True,
    },
    {
        "response_synthesizer_llm": BASELINE_LLM,
        "rag_mode": "rag",
        "template_name": "default",
        "splitter_method": "recursive",
        "rag_embedding_model": BASELINE_RAG_EMBEDDING_MODEL,
        "splitter_chunk_exp": 10,
        "splitter_chunk_overlap_frac": 0.75,
        "rag_method": "dense",
        "rag_top_k": 5,
        "rag_query_decomposition_enabled": False,
        "reranker_enabled": False,
        "hyde_enabled": False,
        "enforce_full_evaluation": True,
        "additional_context_enabled": False,
    },
    {
        "response_synthesizer_llm": BASELINE_LLM,
        "rag_mode": "rag",
        "template_name": "default",
        "splitter_method": "token",
        "rag_embedding_model": BASELINE_RAG_EMBEDDING_MODEL,
        "splitter_chunk_exp": 10,
        "splitter_chunk_overlap_frac": 0.0,
        "rag_method": "dense",
        "rag_top_k": 5,
        "rag_query_decomposition_enabled": False,
        "reranker_enabled": False,
        "hyde_enabled": False,
        "enforce_full_evaluation": True,
        "additional_context_enabled": False,
    },
    {
        "response_synthesizer_llm": BASELINE_LLM,
        "rag_mode": "rag",
        "template_name": "default",
        "splitter_method": "sentence",
        "rag_embedding_model": BASELINE_RAG_EMBEDDING_MODEL,
        "splitter_chunk_exp": 10,
        "splitter_chunk_overlap_frac": 0.25,
        "rag_method": "dense",
        "rag_top_k": 80,
        "rag_query_decomposition_enabled": False,
        "reranker_enabled": False,
        "hyde_enabled": False,
        "enforce_full_evaluation": True,
        "additional_context_enabled": False,
    },
    {
        "response_synthesizer_llm": BASELINE_LLM,
        "rag_mode": "rag",
        "template_name": "default",
        "splitter_method": "sentence",
        "rag_embedding_model": BASELINE_RAG_EMBEDDING_MODEL,
        "splitter_chunk_exp": 10,
        "splitter_chunk_overlap_frac": 0.25,
        "rag_method": "dense",
        "rag_top_k": 80,
        "rag_query_decomposition_enabled": True,
        "rag_query_decomposition_llm_name": BASELINE_LLM,
        "rag_query_decomposition_num_queries": 4,
        "rag_fusion_mode": "dist_based_score",
        "reranker_enabled": False,
        "hyde_enabled": False,
        "enforce_full_evaluation": True,
        "additional_context_enabled": False,
    },
    {
        "response_synthesizer_llm": BASELINE_LLM,
        "rag_mode": "rag",
        "template_name": "default",
        "splitter_method": "sentence",
        "splitter_chunk_exp": 10,
        "splitter_chunk_overlap_frac": 0.25,
        "rag_method": "sparse",
        "rag_top_k": 50,
        "rag_query_decomposition_enabled": False,
        "reranker_enabled": False,
        "hyde_enabled": False,
        "enforce_full_evaluation": True,
        "additional_context_enabled": False,
    },
]


AGENT_BASELINES: T.List[T.Dict[str, T.Any]] = [
    {
        "rag_mode": "react_rag_agent",
        "template_name": "concise",
        "response_synthesizer_llm": BASELINE_LLM,
        "subquestion_engine_llm": BASELINE_LLM,
        "subquestion_response_synthesizer_llm": BASELINE_LLM,
        "rag_method": "dense",
        "rag_embedding_model": "BAAI/bge-base-en-v1.5",
        "rag_query_decomposition_enabled": False,
        "rag_top_k": 10,
        "splitter_method": "sentence",
        "splitter_chunk_exp": 11,
        "splitter_chunk_overlap_frac": 0.5,
        "reranker_enabled": False,
        "hyde_enabled": True,
        "hyde_llm_name": BASELINE_LLM,
        "additional_context_enabled": False,
        "max_iterations": 11,
    },
    {
        "rag_mode": "critique_rag_agent",
        "template_name": "concise",
        "response_synthesizer_llm": BASELINE_LLM,
        "subquestion_engine_llm": BASELINE_LLM,
        "subquestion_response_synthesizer_llm": BASELINE_LLM,
        "critique_agent_llm": BASELINE_LLM,
        "reflection_agent_llm": BASELINE_LLM,
        "rag_method": "dense",
        "rag_embedding_model": "BAAI/bge-base-en-v1.5",
        "rag_query_decomposition_enabled": False,
        "rag_top_k": 10,
        "splitter_method": "sentence",
        "splitter_chunk_exp": 11,
        "splitter_chunk_overlap_frac": 0.5,
        "reranker_enabled": False,
        "hyde_enabled": True,
        "hyde_llm_name": BASELINE_LLM,
        "additional_context_enabled": False,
        "max_iterations": 11,
    },
    {
        "rag_mode": "lats_rag_agent",
        "template_name": "concise",
        "response_synthesizer_llm": BASELINE_LLM,
        "lats_max_rollouts": 2,
        "lats_num_expansions": 2,
        "rag_method": "dense",
        "rag_embedding_model": "BAAI/bge-base-en-v1.5",
        "rag_query_decomposition_enabled": False,
        "rag_top_k": 10,
        "splitter_method": "sentence",
        "splitter_chunk_exp": 11,
        "splitter_chunk_overlap_frac": 0.5,
        "reranker_enabled": False,
        "hyde_enabled": True,
        "hyde_llm_name": BASELINE_LLM,
        "additional_context_enabled": False,
    },
]


def find_param_search_space(param: str, search_space: SearchSpace) -> T.List[T.Any]:
    """Finds the search space from param name."""
    distributions = search_space.build_distributions()
    dist = distributions[param]
    if isinstance(dist, CategoricalDistribution):
        return list(dist.choices)
    if isinstance(dist, (FloatDistribution, IntDistribution)):
        assert dist.step is not None, "step must not be None"
        return list(
            dist.low + i * dist.step
            for i in range(int((dist.high - dist.low) / dist.step) + 1)
        )
    raise ValueError(
        f"Don't know how to handle {dist=}'s type. Please extend this function."
    )


def get_clean_flows(
    flows: T.List[T.Dict[str, T.Any]],
) -> T.List[T.Dict[str, T.Any]]:
    new_flows = []
    for baseline in flows:
        new_baseline = {}
        for k, v in baseline.items():
            if k in UNSUPPORTED_PARAMS:
                logger.warning("Deleting parameter from baseline: %s", k)
                continue
            else:
                new_baseline[k] = v
        new_flows.append(new_baseline)
    return new_flows


def _add_baseline(study_config: StudyConfig, baseline: T.Dict[str, T.Any]):
    b = T.cast(T.Dict[str, T.Any], baseline)

    if b not in study_config.optimization.baselines:
        if are_valid_parameters(study_config.search_space, baseline):
            logger.info("Adding baseline: %s", b)
            study_config.optimization.baselines.append(b)
        else:
            if study_config.optimization.raise_on_invalid_baseline:
                raise ValueError(f"Invalid baseline: {b}")
            logger.warning("Skipping invalid baseline: %s", b)


def _add_variations(
    study_config: StudyConfig,
    template: dict,
    field_name: str,
    options: T.List[T.Any],
):
    for option in options:
        baseline = template.copy()
        baseline[field_name] = option
        _add_baseline(study_config, baseline)


def _add_baselines(study_config: StudyConfig, baselines: T.List[T.Dict[str, T.Any]]):
    for baseline in baselines:
        _add_baseline(study_config, baseline)


def add_toy_baselines(study_config: StudyConfig):
    if not study_config.optimization.use_toy_baselines:
        return
    _add_baselines(study_config, TOY_BASELINES)


def add_individual_baselines(study_config: StudyConfig):
    if not study_config.optimization.use_individual_baselines:
        return
    _add_baselines(study_config, INDIVIDUAL_BASELINES)


def add_agent_baselines(study_config: StudyConfig):
    if not study_config.optimization.use_agent_baselines:
        return
    _add_baselines(study_config, AGENT_BASELINES)


def add_variations_of_baselines(study_config: StudyConfig):
    if not study_config.optimization.use_variations_of_baselines:
        return
    _add_variations(
        study_config=study_config,
        template=SIMPLE_FLOW_TEMPLATE,
        field_name="response_synthesizer_llm",
        options=study_config.search_space.response_synthesizer_llms,
    )
    _add_variations(
        study_config=study_config,
        template=SIMPLE_CONCISE_TEMPLATE,
        field_name="response_synthesizer_llm",
        options=study_config.search_space.response_synthesizer_llms,
    )
    _add_variations(
        study_config=study_config,
        template=SIMPLE_COT_TEMPLATE,
        field_name="response_synthesizer_llm",
        options=study_config.search_space.response_synthesizer_llms,
    )
    _add_variations(
        study_config=study_config,
        template=SPARSE_RAG_TEMPLATE,
        field_name="response_synthesizer_llm",
        options=study_config.search_space.response_synthesizer_llms,
    )
    _add_variations(
        study_config=study_config,
        template=SIMPLE_FEW_SHOT_TEMPLATE,
        field_name="response_synthesizer_llm",
        options=study_config.search_space.response_synthesizer_llms,
    )
    _add_variations(
        study_config=study_config,
        template=RAG_W_FEW_SHOT_TEMPLATE,
        field_name="response_synthesizer_llm",
        options=study_config.search_space.response_synthesizer_llms,
    )
    _add_variations(
        study_config=study_config,
        template=RERANKER_RAG_TEMPLATE,
        field_name="reranker_llm_name",
        options=study_config.search_space.reranker.llms,
    )
    _add_variations(
        study_config=study_config,
        template=DENSE_RAG_TEMPLATE,
        field_name="rag_embedding_model",
        options=study_config.search_space.rag_retriever.embedding_models,
    )
    _add_variations(
        study_config=study_config,
        template=HYBRID_RAG_TEMPLATE,
        field_name="rag_fusion_mode",
        options=study_config.search_space.rag_retriever.fusion.fusion_modes,
    )


def add_pareto_baselines(study_config: StudyConfig):
    if not study_config.optimization.use_pareto_baselines:
        return
    examples = get_examples(study_config)
    study_config.optimization.baselines.extend(examples)


def set_baselines(study_config: StudyConfig) -> StudyConfig:
    """Add baselines to study_config object."""

    n_initial_bl = len(study_config.optimization.baselines)
    if n_initial_bl > 0:
        # use only valid baselines from the initially provided baselines
        logger.info("Using baselines provided in study config")
        baselines = study_config.optimization.baselines.copy()
        baselines = get_clean_flows(baselines)
        study_config.optimization.baselines = []
        _add_baselines(study_config=study_config, baselines=baselines)
        n_valid_bl = len(study_config.optimization.baselines)
        if n_valid_bl < n_initial_bl:
            n_invalid = n_initial_bl - n_valid_bl
            logger.warning(
                "%i of the %i provided baselines are invalid",
                n_invalid,
                n_initial_bl,
            )
        return study_config

    logger.info("Setting baselines")

    if study_config.toy_mode:
        add_toy_baselines(study_config)
    elif study_config.is_retriever_study:
        pass
    else:
        add_individual_baselines(study_config)
        add_agent_baselines(study_config)
        add_variations_of_baselines(study_config)
        add_pareto_baselines(study_config)

    def _seed_llms(study_config, baseline):
        """Tweak the baselines to use LLMs from the search space."""
        ss = study_config.search_space
        if getattr(ss, "__cyclers__", None) is None:
            ss.__cyclers__ = {}
        for param in baseline.keys():
            if "llm" in param:
                if param not in ss.__cyclers__:
                    ss.__cyclers__[param] = itertools.cycle(
                        find_param_search_space(param, ss)
                    )
                baseline[param] = next(ss.__cyclers__[param])
        return baseline

    if study_config.optimization.baselines_cycle_llms:
        study_config.optimization.baselines = [
            _seed_llms(study_config, baseline)
            for baseline in study_config.optimization.baselines
        ]

    n_valid_bl = len(study_config.optimization.baselines)
    logger.info(
        "%i baselines have been added",
        n_valid_bl,
    )
    return study_config
