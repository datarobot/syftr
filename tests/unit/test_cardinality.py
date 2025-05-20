import math
import typing as T

from syftr.studies import (
    FewShotRetriever,
    Hybrid,
    QueryDecomposition,
    Reranker,
    Retriever,
    SearchSpace,
    Splitter,
    TopK,
)

MINIMAL_LLMS: T.List[str] = ["llmA", "llmB"]


def _get_minimal_splitter() -> Splitter:
    return Splitter(
        chunk_min_exp=12,
        chunk_max_exp=12,
        chunk_overlap_frac_min=0.5,
        chunk_overlap_frac_max=0.75,
        chunk_overlap_frac_step=0.25,
        methods=["html", "recursive"],
    )


def test_splitter_cardinality():
    assert _get_minimal_splitter().get_cardinality() == 4


def _get_minimal_topk() -> TopK:
    return TopK(
        kmin=2,
        kmax=2,
        step=1,
    )


def test_topk_cardinality():
    assert _get_minimal_topk().get_cardinality() == 1


def _get_minimal_qd() -> QueryDecomposition:
    return QueryDecomposition(
        llm_names=MINIMAL_LLMS,
        num_queries_min=2,
        num_queries_max=4,
        num_queries_step=2,
    )


def test_query_decomposition_cardinality():
    assert _get_minimal_qd().get_cardinality() == 4


def _get_minimal_hybrid() -> Hybrid:
    return Hybrid(
        bm25_weight_min=0.1,
        bm25_weight_max=0.1,
        bm25_weight_step=0.1,
    )


def test_hybrid_cardinality():
    assert _get_minimal_hybrid().get_cardinality() == 1


def _get_minimal_retriever() -> Retriever:
    return Retriever(
        top_k=_get_minimal_topk(),
        methods=["hybrid"],
        embedding_models=["embA"],
        hybrid=_get_minimal_hybrid(),
        query_decomposition=_get_minimal_qd(),
    )


def test_retiever_cardinality():
    r = _get_minimal_retriever()
    assert (
        r.get_cardinality()
        == r.top_k.get_cardinality()
        * len(r.methods)
        * len(r.query_decomposition_enabled)
        * r.hybrid.get_cardinality()
        * r.query_decomposition.get_cardinality()
        * r.fusion.get_cardinality()
    )


def _get_minimal_few_shot_retriever() -> FewShotRetriever:
    return FewShotRetriever(
        top_k=_get_minimal_topk(),
        embedding_models=["embA"],
    )


def _get_minimal_reranker() -> Reranker:
    return Reranker(
        top_k=_get_minimal_topk(),
        llms=MINIMAL_LLMS,
    )


def test_search_space_cardinality():
    # Construct a rag and no_rag search space and check that they match the size
    # of a hybrid rag + no_rag space.
    ss = SearchSpace(
        rag_modes=["rag"],
        template_names=["templateA", "templateB"],
        response_synthesizer_llms=MINIMAL_LLMS,
        few_shot_retriever=_get_minimal_few_shot_retriever(),
        splitter=_get_minimal_splitter(),
        reranker=_get_minimal_reranker(),
        rag_retriever=_get_minimal_retriever(),
    )
    components = [
        ss.few_shot_retriever,
        ss.rag_retriever,
        ss.splitter,
        ss.reranker,
        ss.hyde,
        ss.additional_context,
    ]
    expected_rag_card = math.prod(
        [component.get_cardinality() for component in components]
    )
    expected_rag_card *= (
        len(ss.template_names)
        * len(ss.response_synthesizer_llms)
        * len(ss.few_shot_enabled)
        * len(ss.hyde_enabled)
        * len(ss.additional_context_enabled)
        * len(ss.reranker_enabled)
    )
    assert ss.get_cardinality() == expected_rag_card

    ss = SearchSpace(
        rag_modes=["no_rag"],
        template_names=["templateA", "templateB"],
        response_synthesizer_llms=MINIMAL_LLMS,
        few_shot_retriever=_get_minimal_few_shot_retriever(),
        splitter=_get_minimal_splitter(),
        reranker=_get_minimal_reranker(),
        rag_retriever=_get_minimal_retriever(),
    )
    expected_no_rag_card = (
        len(ss.template_names)
        * len(ss.response_synthesizer_llms)
        * len(ss.few_shot_enabled)
        * len(ss.hyde_enabled)
        * len(ss.additional_context_enabled)
        * len(ss.reranker_enabled)
        * ss.few_shot_retriever.get_cardinality()
    )
    assert ss.get_cardinality() == expected_no_rag_card

    # Now test with both rag and no rag
    ss = SearchSpace(
        rag_modes=["no_rag", "rag"],
        template_names=["templateA", "templateB"],
        response_synthesizer_llms=MINIMAL_LLMS,
        few_shot_retriever=_get_minimal_few_shot_retriever(),
        splitter=_get_minimal_splitter(),
        reranker=_get_minimal_reranker(),
        rag_retriever=_get_minimal_retriever(),
    )
    assert ss.get_cardinality() == (expected_rag_card + expected_no_rag_card)
