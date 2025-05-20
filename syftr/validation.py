import typing as T

import optuna

from syftr.logger import logger
from syftr.studies import RetrieverSearchSpace, SearchSpace


def parameters_do_exist(trial_params: T.Dict[str, T.Any], parameters: T.List[str]):
    for param in parameters:
        if param not in trial_params:
            logger.warning("Parameter missing: %s", param)
            return False
        if trial_params[param] is None:
            logger.warning("Parameter value missing: %s", param)
            return False
    return True


def parameters_do_not_exist(
    trial_params: T.Dict[str, T.Any],
    parameters: T.List[str],
):
    for param in parameters:
        if param in trial_params:
            logger.warning("Parameter should not be provided: %s", param)
            return False
    return True


def has_valid_minimum_params(
    search_space: SearchSpace,
    trial_params: T.Dict[str, T.Any],
):
    assert search_space, "Search space is missing"
    assert trial_params, "Trial parameters are missing"

    ss = search_space
    tp = trial_params

    required = ["rag_mode"]
    if not parameters_do_exist(tp, required):
        return False

    if tp["rag_mode"] not in ss.rag_modes:
        logger.warning("RAG mode is not supported: %s", tp["rag_mode"])
        return False

    return True


def has_valid_few_shot_params(
    search_space: SearchSpace,
    trial_params: T.Dict[str, T.Any],
):
    assert search_space, "Search space is missing"
    assert trial_params, "Trial parameters are missing"

    ss = search_space
    tp = trial_params

    if not ss.is_few_shot(tp):
        not_allowed = [
            "few_shot_embedding_model",
            "few_shot_top_k",
        ]
        return parameters_do_not_exist(tp, not_allowed)

    else:
        params = ["few_shot_embedding_model", "few_shot_top_k"]
        if not parameters_do_exist(tp, params):
            return False

        if tp["few_shot_embedding_model"] not in ss.few_shot_retriever.embedding_models:
            logger.warning(
                "Few-shot embedding model is not supported: %s",
                tp["few_shot_embedding_model"],
            )
            return False

        distributions = ss.few_shot_retriever.top_k.build_distributions(
            prefix="few_shot_"
        )
        if not distributions["few_shot_top_k"]._contains(tp["few_shot_top_k"]):
            logger.warning(
                "The value %s is out-of-distribution for the parameter '%s'",
                str(tp["few_shot_top_k"]),
                "few_shot_top_k",
            )
            return False

        if not parameters_do_not_exist(tp, ["few_shot_hybrid_bm25_weight"]):
            logger.warning("Few-shot method does not support hybrid retrieval")
            return False
    return True


def has_valid_base_rag_params(
    search_space: SearchSpace,
    trial_params: T.Dict[str, T.Any],
) -> bool:
    assert trial_params, "Trial parameters are missing"

    # No rag means none of the following are set
    if trial_params["rag_mode"] == "no_rag":
        for param in sorted(
            set(
                [
                    "additional_context_enabled",
                    "additional_context_num_nodes",
                    "critique_agent_llm",
                    "hyde_enabled",
                    "hyde_llm_name",
                    "lats_max_rollouts",
                    "lats_num_expansions",
                    "rag_embedding_model",
                    "rag_fusion_mode",
                    "rag_hybrid_bm25_weight",
                    "rag_method",
                    "rag_query_decomposition_enabled",
                    "rag_query_decomposition_llm_name",
                    "rag_query_decomposition_num_queries",
                    "rag_top_k",
                    "reflection_agent_llm",
                    "reranker_enabled",
                    "reranker_llm_name",
                    "reranker_top_k",
                    "splitter_chunk_overlap_frac",
                    "splitter_chunk_exp",
                    "splitter_method",
                    "subquestion_engine_llm",
                    "subquestion_response_synthesizer_llm",
                ]
            )
        ):
            if param in trial_params:
                logger.warning(
                    "Parameter not supported for RAG mode 'no_rag': %s", param
                )
                return False
    else:
        for param in [
            "template_name",
            "response_synthesizer_llm",
            "additional_context_enabled",
            "rag_method",
            "rag_top_k",
            "rag_query_decomposition_enabled",
            "splitter_method",
            "splitter_chunk_exp",
            "splitter_chunk_overlap_frac",
            "reranker_enabled",
            "hyde_enabled",
        ]:
            if param not in trial_params:
                logger.warning("RAG parameter missing: %s", param)
                return False

    return True


def has_valid_rag_embedding_model_params(
    search_space: SearchSpace,
    trial_params: T.Dict[str, T.Any],
) -> bool:
    assert search_space, "Search space is missing"
    assert trial_params, "Trial parameters are missing"

    ss = search_space
    tp = trial_params

    if tp["rag_mode"] == "no_rag":
        logger.debug("RAG mode is 'no_rag'")
        return True

    if not parameters_do_exist(tp, ["rag_mode", "rag_method", "rag_top_k"]):
        return False

    distributions = ss.rag_retriever.top_k.build_distributions(prefix="rag_")
    if not distributions["rag_top_k"]._contains(tp["rag_top_k"]):
        logger.warning(
            "The value %s is out-of-distribution for the parameter '%s'",
            str(tp["rag_top_k"]),
            "rag_top_k",
        )
        return False

    rag_method = tp["rag_method"]
    if rag_method in ["dense", "hybrid"]:
        if "rag_embedding_model" not in tp:
            logger.warning("RAG embedding model is missing")
            return False

        rag_embedding_model = tp["rag_embedding_model"]
        if rag_embedding_model not in ss.rag_retriever.embedding_models:
            logger.warning(
                "RAG embedding model is not supported: %s",
                rag_embedding_model,
            )
            return False

        if rag_method == "hybrid":
            if "rag_hybrid_bm25_weight" not in tp:
                logger.warning("Hybrid RAG BM25 weight is missing")
                return False
    else:
        if "rag_embedding_model" in tp:
            logger.warning("RAG embedding model should not be provided")
            return False

    if rag_method not in ss.rag_retriever.methods:
        logger.warning("RAG method is not supported: %s", rag_method)
        return False

    return True


def has_valid_rag_hybrid_params(
    search_space: SearchSpace,
    trial_params: T.Dict[str, T.Any],
) -> bool:
    assert search_space, "Search space is missing"
    assert trial_params, "Trial parameters are missing"

    ss = search_space
    tp = trial_params

    if "rag_mode" not in tp:
        logger.warning("RAG mode is missing")
        return False

    if tp["rag_mode"] == "no_rag":
        logger.debug("RAG mode is 'no_rag'")
        return True

    if "rag_method" not in tp:
        logger.warning("RAG method is missing")
        return False

    hybrid_params = [
        "rag_hybrid_bm25_weight",
    ]
    if tp["rag_method"] != "hybrid":
        if not parameters_do_not_exist(tp, hybrid_params):
            return False
    else:
        if not parameters_do_exist(tp, hybrid_params):
            return False

        distributions = ss.rag_retriever.hybrid.build_distributions(prefix="rag_")
        if not distributions["rag_hybrid_bm25_weight"]._contains(
            tp["rag_hybrid_bm25_weight"]
        ):
            logger.warning(
                "The value %s is out-of-distribution for the parameter '%s'",
                str(tp["rag_hybrid_bm25_weight"]),
                "rag_hybrid_bm25_weight",
            )
            return False

        if not parameters_do_exist(tp, ["rag_fusion_mode"]):
            return False

        if tp["rag_fusion_mode"] not in ss.rag_retriever.fusion.fusion_modes:
            logger.warning(
                "RAG fusion mode is not supported: %s",
                tp["rag_fusion_mode"],
            )
            return False

    return True


def has_valid_rag_query_decomposition_params(
    search_space: SearchSpace,
    trial_params: T.Dict[str, T.Any],
) -> bool:
    assert search_space, "Search space is missing"
    assert trial_params, "Trial parameters are missing"

    ss = search_space
    tp = trial_params

    if "rag_mode" not in tp:
        logger.warning("RAG mode is missing")
        return False

    if tp["rag_mode"] == "no_rag":
        logger.debug("RAG mode is 'no_rag'")
        return True

    if "rag_method" not in tp:
        logger.warning("RAG method is missing")
        return False

    query_decomposition_params = [
        "rag_query_decomposition_llm_name",
        "rag_query_decomposition_num_queries",
    ]
    if not tp["rag_query_decomposition_enabled"]:
        if not parameters_do_not_exist(tp, query_decomposition_params):
            return False
    else:
        if not parameters_do_exist(tp, query_decomposition_params):
            return False

        if (
            tp["rag_query_decomposition_llm_name"]
            not in ss.rag_retriever.query_decomposition.llm_names
        ):
            logger.warning(
                "RAG query_decomposition LLM name '%s' is not in: %s",
                tp["rag_query_decomposition_llm_name"],
                ss.rag_retriever.query_decomposition.llm_names,
            )
            return False

        distributions = ss.rag_retriever.query_decomposition.build_distributions(
            prefix="rag_"
        )
        if not distributions["rag_query_decomposition_num_queries"]._contains(
            tp["rag_query_decomposition_num_queries"]
        ):
            logger.warning(
                "The value %s is out-of-distribution for the parameter '%s'",
                str(tp["rag_query_decomposition_num_queries"]),
                "rag_query_decomposition_num_queries",
            )
            return False

        if not parameters_do_exist(tp, ["rag_fusion_mode"]):
            return False

        if tp["rag_fusion_mode"] not in ss.rag_retriever.fusion.fusion_modes:
            logger.warning(
                "RAG fusion mode is not supported: %s",
                tp["rag_fusion_mode"],
            )
            return False

    return True


def has_valid_rag_fusion_params(
    search_space: SearchSpace,
    trial_params: T.Dict[str, T.Any],
) -> bool:
    assert search_space, "Search space is missing"
    assert trial_params, "Trial parameters are missing"

    ss = search_space
    tp = trial_params

    if "rag_mode" not in tp:
        logger.warning("RAG mode is missing")
        return False

    if tp["rag_mode"] == "no_rag":
        logger.debug("RAG mode is 'no_rag'")
        return True

    if "rag_method" not in tp:
        logger.warning("RAG method is missing")
        return False

    fusion_params = [
        "rag_fusion_mode",
    ]
    if tp["rag_method"] != "hybrid" and not tp["rag_query_decomposition_enabled"]:
        if not parameters_do_not_exist(tp, fusion_params):
            return False
    else:
        if not parameters_do_exist(tp, fusion_params):
            return False

        if tp["rag_fusion_mode"] not in ss.rag_retriever.fusion.fusion_modes:
            logger.warning(
                "RAG fusion mode is not supported: %s",
                tp["rag_fusion_mode"],
            )
            return False

    return True


def has_valid_rag_reranker_params(
    search_space: SearchSpace,
    trial_params: T.Dict[str, T.Any],
) -> bool:
    assert search_space, "Search space is missing"
    assert trial_params, "Trial parameters are missing"

    ss = search_space
    tp = trial_params

    if "rag_mode" not in tp:
        logger.warning("RAG mode is missing")
        return False

    if tp["rag_mode"] == "no_rag":
        logger.debug("RAG mode is 'no_rag'")
        return True

    reranker_params = ["reranker_llm_name", "reranker_top_k"]
    if "reranker_enabled" in tp and tp["reranker_enabled"]:
        if not parameters_do_exist(tp, reranker_params):
            return False

        if tp["reranker_llm_name"] not in ss.reranker.llms:
            logger.warning(
                "Reranker LLM name '%s' is not in: %s",
                tp["reranker_llm_name"],
                ss.reranker.llms,
            )
            return False

        distributions = ss.reranker.top_k.build_distributions(prefix="reranker_")
        if not distributions["reranker_top_k"]._contains(tp["reranker_top_k"]):
            logger.warning(
                "The value %s is out-of-distribution for the parameter '%s'",
                str(tp["reranker_top_k"]),
                "reranker_top_k",
            )
            return False

    else:
        if not parameters_do_not_exist(tp, reranker_params):
            return False

    return True


def has_valid_hyde_params(
    search_space: SearchSpace,
    trial_params: T.Dict[str, T.Any],
) -> bool:
    assert search_space, "Search space is missing"
    assert trial_params, "Trial parameters are missing"

    ss = search_space
    tp = trial_params

    hyde_params = ["hyde_llm_name"]
    if "hyde_enabled" in tp and tp["hyde_enabled"]:
        if not parameters_do_exist(tp, hyde_params):
            return False

        if tp["hyde_llm_name"] not in ss.hyde.llms:
            logger.warning(
                "Hyde LLM name is not supported: %s",
                tp["hyde_llm_name"],
            )
            return False

    else:
        if not parameters_do_not_exist(tp, hyde_params):
            return False

    return True


def has_valid_splitter_params(
    search_space: SearchSpace,
    trial_params: T.Dict[str, T.Any],
) -> bool:
    assert search_space, "Search space is missing"
    assert trial_params, "Trial parameters are missing"

    ss = search_space
    tp = trial_params

    if "rag_mode" not in tp:
        logger.warning("RAG mode is missing")
        return False

    splitter_params = [
        "splitter_chunk_overlap_frac",
        "splitter_chunk_exp",
        "splitter_method",
    ]
    if tp["rag_mode"] == "no_rag":
        return True

    if not parameters_do_exist(tp, splitter_params):
        return False

    if tp["splitter_method"] not in ss.splitter.methods:
        logger.warning(
            "Splitter method is not supported: %s",
            tp["splitter_method"],
        )
        return False

    distributions = ss.splitter.build_distributions()
    if not distributions["splitter_chunk_exp"]._contains(int(tp["splitter_chunk_exp"])):
        logger.warning(
            "The value %s is out-of-distribution for the parameter '%s'",
            str(tp["splitter_chunk_exp"]),
            "splitter_chunk_exp",
        )
        return False
    if not distributions["splitter_chunk_overlap_frac"]._contains(
        tp["splitter_chunk_overlap_frac"]
    ):
        logger.warning(
            "The value %s is out-of-distribution for the parameter '%s'",
            str(tp["splitter_chunk_overlap_frac"]),
            "splitter_chunk_overlap_frac",
        )
        return False

    return True


def has_valid_additional_context_params(
    search_space: SearchSpace,
    trial_params: T.Dict[str, T.Any],
) -> bool:
    assert search_space, "Search space is missing"
    assert trial_params, "Trial parameters are missing"

    if trial_params["rag_mode"] == "no_rag":
        return parameters_do_not_exist(
            trial_params, ["additional_context_enabled", "additional_context_num_nodes"]
        )
    else:
        if not parameters_do_exist(trial_params, ["additional_context_enabled"]):
            return False

        if trial_params["additional_context_enabled"]:
            if not parameters_do_exist(trial_params, ["additional_context_num_nodes"]):
                return False
            num_nodes = trial_params.get("additional_context_num_nodes")
            if num_nodes is None:
                logger.warning("Additional context enabled but num nodes not set")
                return False
            if num_nodes < 1:
                logger.warning(
                    "Additional context enabled but num nodes isn't a positive integer"
                )
                return False
        else:
            # Feature disabled, no param set
            return parameters_do_not_exist(
                trial_params, ["additional_context_num_nodes"]
            )

    return True


def has_valid_sub_question_params(
    search_space: SearchSpace,
    trial_params: T.Dict[str, T.Any],
) -> bool:
    if trial_params["rag_mode"] != "sub_question_rag":
        return True

    if not parameters_do_exist(
        trial_params,
        [
            "response_synthesizer_llm",
            "subquestion_engine_llm",
            "template_name",
        ],
    ):
        logger.warning("Sub Question RAG parameters are missing")
        return False

    return True


def has_valid_unique_params(
    search_space: SearchSpace,
    trial_params: T.Dict[str, T.Any],
    rag_mode: str,
    unique_params: T.List[str],
) -> bool:
    if trial_params["rag_mode"] != rag_mode:
        return parameters_do_not_exist(trial_params, unique_params)

    if not parameters_do_exist(trial_params, unique_params):
        logger.warning("Parameters are missing for RAG mode '%s'", rag_mode)
        return False

    match rag_mode:
        case "react_rag_agent":
            distributions = search_space.react_rag_agent.build_distributions()
        case "critique_rag_agent":
            distributions = search_space.critique_rag_agent.build_distributions()
        case "sub_question_rag":
            distributions = search_space.sub_question_rag.build_distributions()
        case "lats_rag_agent":
            distributions = search_space.lats_rag_agent.build_distributions()
        case _:
            pass

    for param in unique_params:
        if isinstance(trial_params[param], (int, float)):
            if not distributions[param]._contains(trial_params[param]):
                logger.warning(
                    "The value %s is out-of-distribution for the parameter '%s'",
                    str(trial_params[param]),
                    param,
                )
                return False
        elif isinstance(
            distributions[param], optuna.distributions.CategoricalDistribution
        ):
            if trial_params[param] not in distributions[param].choices:  # type: ignore
                logger.warning(
                    "The value %s is out-of-distribution for the parameter '%s'",
                    str(trial_params[param]),
                    param,
                )
                return False
        else:
            raise ValueError(
                f"Unsupported distribution type for parameter '{trial_params[param]}'"
            )

    return True


def are_valid_parameters(
    search_space: SearchSpace | RetrieverSearchSpace,
    trial_params: T.Dict[str, T.Any],
) -> bool:
    assert search_space, "Search space is missing"
    assert trial_params, "Trial parameters are missing"

    ss = search_space
    tp = trial_params

    if isinstance(ss, RetrieverSearchSpace):
        tests = [
            has_valid_minimum_params,
            has_valid_rag_embedding_model_params,
            has_valid_rag_fusion_params,
            has_valid_rag_query_decomposition_params,
            has_valid_rag_hybrid_params,
            has_valid_splitter_params,
        ]
    else:
        tests = [
            has_valid_minimum_params,
            has_valid_few_shot_params,
            has_valid_base_rag_params,
            has_valid_rag_embedding_model_params,
            has_valid_rag_fusion_params,
            has_valid_rag_query_decomposition_params,
            has_valid_rag_hybrid_params,
            has_valid_rag_reranker_params,
            has_valid_hyde_params,
            has_valid_splitter_params,
            has_valid_sub_question_params,
            has_valid_additional_context_params,
            lambda x, y: has_valid_unique_params(  # type: ignore
                x,
                y,
                "critique_rag_agent",
                [
                    "critique_agent_llm",
                    "reflection_agent_llm",
                ],
            ),
            lambda x, y: has_valid_unique_params(  # type: ignore
                x,
                y,
                "lats_rag_agent",
                [
                    "lats_max_rollouts",
                    "lats_num_expansions",
                ],
            ),
        ]

    test: T.Callable[[SearchSpace | RetrieverSearchSpace, T.Dict[str, T.Any]], bool]
    for test in tests:  # type: ignore
        if not test(ss, tp):
            logger.error(f"Parameter validation function `{test.__name__}` failed.")
            return False

    logger.debug("The parameter check did not detect any violations")

    return True
