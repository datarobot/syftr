import typing as T

import pytest
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.storage.docstore.types import BaseDocumentStore

from syftr.agent_flows import LlamaIndexReactRAGAgentFlow
from syftr.flows import (
    CritiqueAgentFlow,
    Flow,
    RAGFlow,
    ReActAgentFlow,
    SubQuestionRAGFlow,
)
from syftr.llm import LLMs, get_llm, is_function_calling
from syftr.retrievers.build import _build_dense_index, _build_sparse_index
from syftr.storage import SyftrQADataset
from syftr.studies import StudyConfig
from syftr.templates import get_agent_template, get_template
from syftr.tuner.core import build_splitter
from syftr.tuner.qa_tuner import _get_example_retriever


@pytest.fixture
def tiny_sparse_flow(tiny_sparse_retriever, gpt_4o_mini, rag_template) -> RAGFlow:
    llm, _ = gpt_4o_mini
    return RAGFlow(
        retriever=tiny_sparse_retriever,
        response_synthesizer_llm=llm,
        template=rag_template,
        params={},
    )


@pytest.fixture
def real_sparse_flow(
    real_sparse_retriever, gpt_4o_mini, rag_template
) -> T.Tuple[RAGFlow, StudyConfig]:
    llm, _ = gpt_4o_mini
    retriever, docstore, study_config = real_sparse_retriever
    return RAGFlow(
        retriever=retriever,
        docstore=docstore,
        response_synthesizer_llm=llm,
        template=rag_template,
        additional_context_num_nodes=2,
        params={},
    ), study_config


@pytest.fixture
def tiny_dense_flow(tiny_dense_retriever, gpt_4o_mini, rag_template) -> RAGFlow:
    llm, _ = gpt_4o_mini
    return RAGFlow(
        retriever=tiny_dense_retriever,
        response_synthesizer_llm=llm,
        template=rag_template,
        params={},
    )


@pytest.fixture
def real_dense_flow(
    real_dense_retriever, gpt_4o_mini, rag_template
) -> T.Tuple[RAGFlow, StudyConfig]:
    llm, _ = gpt_4o_mini
    retriever, docstore, study_config = real_dense_retriever
    return RAGFlow(
        retriever=retriever,
        docstore=docstore,
        additional_context_num_nodes=2,
        response_synthesizer_llm=llm,
        template=rag_template,
        params={},
    ), study_config


@pytest.fixture
def real_sparse_flow_few_shot(
    real_sparse_retriever, gpt_4o_mini, few_shot_rag_template, bge_small_no_hf
) -> T.Tuple[RAGFlow, StudyConfig]:
    llm, _ = gpt_4o_mini
    retriever, docstore, study_config = real_sparse_retriever
    params = {"few_shot_top_k": 5}
    get_qa_examples = _get_example_retriever(params, study_config, bge_small_no_hf)
    return RAGFlow(
        retriever=retriever,
        docstore=docstore,
        additional_context_num_nodes=2,
        response_synthesizer_llm=llm,
        template=few_shot_rag_template,
        get_examples=get_qa_examples,
    ), study_config


@pytest.fixture
def real_hybrid_flow(
    real_hybrid_retriever, gpt_4o_mini, rag_template
) -> T.Tuple[RAGFlow, StudyConfig]:
    llm, _ = gpt_4o_mini
    retriever, docstore, study_config = real_hybrid_retriever
    return RAGFlow(
        retriever=retriever,
        docstore=docstore,
        additional_context_num_nodes=2,
        response_synthesizer_llm=llm,
        template=rag_template,
    ), study_config


@pytest.fixture(
    scope="session",
    params=list(
        [
            (name, llm)
            for name, llm in LLMs.items()
            if name
            not in [
                # Going the way of the dodo
                "gpt-35-turbo",
                # Don't have pricing info for these models yet
                "cerebras-llama-31-8B",
                "cerebras-llama-33-70B",
                "gemini-flash-think-exp",
                # Has to be manually turned on
                "datarobot-r1-70B-functions",
                "datarobot-r1-70B-reasoning",
                # Temporarily disabled for testing
                "o1",
                "o3-mini",
                "together-r1",
                "together-V3",
                # Flaky in agent flows
                "llama-33-70B",
                "phi-4",
            ]
        ]
    ),
    ids=list(
        [
            name
            for name in LLMs.keys()
            if name
            not in [
                # Going the way of the dodo
                "gpt-35-turbo",
                # Don't have pricing info for these models yet
                "cerebras-llama-31-8B",
                "cerebras-llama-33-70B",
                "gemini-flash-think-exp",
                # Has to be manually turned on
                "datarobot-r1-70B-functions",
                "datarobot-r1-70B-reasoning",
                # Temporarily disabled for testing
                "o1",
                "o3-mini",
                "together-r1",
                "together-V3",
                # Flaky in agent flows
                "llama-33-70B",
                "phi-4",
            ]
        ]
    ),
)
def llm(request) -> T.Tuple[str, FunctionCallingLLM]:
    name, llm = request.param
    return name, llm


@pytest.fixture(
    scope="session",
    params=list(
        [(name, llm) for name, llm in LLMs.items() if is_function_calling(llm)]
    ),
    ids=list([name for name, llm in LLMs.items() if is_function_calling(llm)]),
)
def critique_llm(request) -> T.Tuple[str, FunctionCallingLLM]:
    name, llm = request.param
    return name, llm


@pytest.fixture(scope="session")
def basic_flow_all_llms(llm):
    _, llm = llm
    return Flow(response_synthesizer_llm=llm, template=get_template("default"))


@pytest.fixture
def real_subquestion_flow(
    llm, real_dense_retriever, rag_template
) -> T.Tuple[SubQuestionRAGFlow, StudyConfig]:
    _, llm = llm
    retriever, docstore, study_config = real_dense_retriever
    return SubQuestionRAGFlow(
        subquestion_engine_llm=llm,
        subquestion_response_synthesizer_llm=llm,
        retriever=retriever,
        docstore=docstore,
        additional_context_num_nodes=2,
        response_synthesizer_llm=llm,
        template=rag_template,
        dataset_name=study_config.dataset.name,
        dataset_description=study_config.dataset.description,
    ), study_config


@pytest.fixture
def react_agent_flow(
    llm, real_dense_retriever, rag_template
) -> T.Tuple[ReActAgentFlow, StudyConfig]:
    _, llm = llm
    retriever, docstore, study_config = real_dense_retriever
    return ReActAgentFlow(
        retriever=retriever,
        docstore=docstore,
        additional_context_num_nodes=2,
        response_synthesizer_llm=llm,
        subquestion_engine_llm=llm,
        subquestion_response_synthesizer_llm=llm,
        template=rag_template,
        dataset_name=study_config.dataset.name,
        dataset_description=study_config.dataset.description,
    ), study_config


@pytest.fixture
def critique_agent_flow(
    critique_llm, real_dense_retriever, rag_template
) -> T.Tuple[CritiqueAgentFlow, StudyConfig]:
    _, llm = critique_llm
    retriever, docstore, study_config = real_dense_retriever
    return CritiqueAgentFlow(
        retriever=retriever,
        docstore=docstore,
        additional_context_num_nodes=2,
        response_synthesizer_llm=llm,
        subquestion_engine_llm=llm,
        subquestion_response_synthesizer_llm=llm,
        critique_agent_llm=llm,
        reflection_agent_llm=llm,
        template=rag_template,
        dataset_name=study_config.dataset.name,
        dataset_description=study_config.dataset.description,
    ), study_config


@pytest.fixture
def react_agent_flow_hybrid_hyde_reranker_few_shot(
    real_hybrid_retriever, gpt_4o_mini, few_shot_rag_template, bge_small_no_hf
) -> T.Tuple[ReActAgentFlow, StudyConfig]:
    llm, _ = gpt_4o_mini
    retriever, docstore, study_config = real_hybrid_retriever
    params = {"few_shot_top_k": 5}
    get_qa_examples = _get_example_retriever(params, study_config, bge_small_no_hf)
    return ReActAgentFlow(
        retriever=retriever,
        docstore=docstore,
        additional_context_num_nodes=2,
        response_synthesizer_llm=llm,
        subquestion_engine_llm=llm,
        subquestion_response_synthesizer_llm=llm,
        template=few_shot_rag_template,
        get_examples=get_qa_examples,
        dataset_name=study_config.dataset.name,
        dataset_description=study_config.dataset.description,
        hyde_llm=llm,
        reranker_llm=llm,
        reranker_top_k=5,
    ), study_config


@pytest.fixture(scope="session")
def cot_template():
    return get_template("CoT", with_context=False)


@pytest.fixture(scope="session")
def cot_template_rag():
    return get_template("CoT", with_context=True)


@pytest.fixture(scope="session")
def sentence_splitter() -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=256,
        chunk_overlap=64,
    )


@pytest.fixture(scope="session")
def recursive_splitter(bge_small_no_hf, real_dataset_study_config) -> NodeParser:
    return build_splitter(
        real_dataset_study_config,
        {
            "splitter_method": "recursive",
            "splitter_chunk_exp": 10,
            "splitter_chunk_overlap_frac": 0.5,
        },
    )


@pytest.fixture(scope="session")
def simple_dense_index(
    tiny_dataset, sentence_splitter, bge_small_no_hf
) -> T.Tuple[VectorStoreIndex, BaseDocumentStore]:
    return _build_dense_index(
        documents=list(tiny_dataset.iter_grounding_data()),
        transforms=[sentence_splitter],
        embedding_model=bge_small_no_hf,
    )


@pytest.fixture(scope="session")
def real_dense_index(
    real_dataset, sentence_splitter, bge_small_no_hf
) -> T.Tuple[VectorStoreIndex, BaseDocumentStore, SyftrQADataset]:
    index, docstore = _build_dense_index(
        documents=list(real_dataset.iter_grounding_data()),
        transforms=[sentence_splitter],
        embedding_model=bge_small_no_hf,
        max_chunks=100,
    )
    return index, docstore, real_dataset


@pytest.fixture(scope="session")
def hotpot_index(sentence_splitter, bge_small_no_hf):
    hotpot = StudyConfig.from_file(
        "tests/functional/data/studies/test-hotpot-toy.yaml"
    ).dataset
    return (
        _build_dense_index(
            documents=list(hotpot.iter_grounding_data()),
            transforms=[sentence_splitter],
            embedding_model=bge_small_no_hf,
        ),
        hotpot,
    )


@pytest.fixture(scope="session")
def financebench_index(
    sentence_splitter, bge_small_no_hf
) -> T.Tuple[VectorStoreIndex, BaseDocumentStore, SyftrQADataset]:
    financebench = StudyConfig.from_file(
        "tests/functional/data/studies/financebench-pepsi.yaml"
    ).dataset
    index, docstore = _build_dense_index(
        documents=list(financebench.iter_grounding_data()),
        transforms=[sentence_splitter],
        embedding_model=bge_small_no_hf,
        max_chunks=100,
    )
    return index, docstore, financebench


@pytest.fixture(scope="session")
def real_sparse_index(real_dataset, recursive_splitter):
    return _build_sparse_index(
        documents=real_dataset.iter_grounding_data(),
        transforms=[recursive_splitter],
        top_k=5,
    )


@pytest.fixture
def llama_index_flow(
    cot_template_rag,
    simple_dense_index,
    bge_small_no_hf,
) -> RAGFlow:
    """Simple RAG flow."""
    name = "gpt-4o-mini"
    llm = get_llm(name)
    dense_retriever = simple_dense_index.as_retriever(
        embed_model=bge_small_no_hf, similarity_top_k=5
    )
    return RAGFlow(
        retriever=dense_retriever,
        response_synthesizer_llm=llm,
        template=cot_template_rag,
    )


@pytest.fixture
def react_agent_rag_flow(
    hotpot_index, financebench_index
) -> LlamaIndexReactRAGAgentFlow:
    fb_index, _ = financebench_index
    hp_index, _ = hotpot_index
    llm = get_llm("gpt-4o-mini")
    return LlamaIndexReactRAGAgentFlow(
        indexes=[
            ("hotpot", "A collection of trivia from Wikipedia", hp_index),
            (
                "financebench",
                "Finacial reports and accounting data",
                fb_index,
            ),
        ],
        llm=llm,
        template=get_template("default", with_context=True),
        verbose=True,
    )


@pytest.fixture
def react_agent_rag_flow_system_prompt(
    hotpot_index, financebench_index
) -> LlamaIndexReactRAGAgentFlow:
    fb_index, _ = financebench_index
    hp_index, _ = hotpot_index
    llm = get_llm("gpt-4o-mini")
    prompt = get_agent_template("aggressive")
    return LlamaIndexReactRAGAgentFlow(
        indexes=[
            ("hotpot", "A collection of trivia from Wikipedia", hp_index),
            (
                "financebench",
                "Finacial reports and accounting data",
                fb_index,
            ),
        ],
        llm=llm,
        system_prompt=PromptTemplate(prompt),
        template=get_template("default", with_context=True),
        verbose=True,
    )
