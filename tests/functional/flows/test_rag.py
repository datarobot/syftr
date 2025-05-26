import typing as T

import pytest
from llama_index.core.callbacks import CBEventType

from syftr.flows import RAGFlow
from syftr.storage import SyftrQADataset
from syftr.studies import StudyConfig


@pytest.mark.xfail(reason="RAG not guaranteed to provide supporting facts")
def test_basic_rag(tiny_flow: RAGFlow, tiny_dataset: SyftrQADataset):
    for qa_pair in tiny_dataset.iter_examples():
        response, duration, call_data = tiny_flow.generate(qa_pair.question)  # type: ignore
        # Check RAG results
        source_texts = [
            node.text.lower().strip()
            for node in response.additional_kwargs["source_nodes"]
        ]  # type: ignore
        assert qa_pair.supporting_facts, (
            "Your test data should include expected supporting text from the source"
        )
        for fact in qa_pair.supporting_facts:
            print(f"Checking fact: {fact}")
            assert any(fact.lower().strip() in text for text in source_texts)


def test_real_data_sparse_flow(real_sparse_flow: T.Tuple[RAGFlow, StudyConfig]):
    flow, study_config = real_sparse_flow
    for qa_pair in list(study_config.dataset.iter_examples())[:5]:
        response, duration, call_data = flow.generate(qa_pair.question)  # type: ignore


def test_real_data_dense_flow(real_dense_flow: T.Tuple[RAGFlow, StudyConfig]):
    flow, study_config = real_dense_flow
    for qa_pair in list(study_config.dataset.iter_examples())[:5]:
        response, duration, call_data = flow.generate(qa_pair.question)  # type: ignore


def test_real_data_few_shot_flow(
    real_sparse_flow_few_shot: T.Tuple[RAGFlow, StudyConfig],
):
    flow, study_config = real_sparse_flow_few_shot
    for qa_pair in list(study_config.dataset.iter_examples())[:5]:
        response, duration, call_data = flow.generate(qa_pair.question)  # type: ignore


def test_real_data_hybrid_flow(
    real_hybrid_flow: T.Tuple[RAGFlow, StudyConfig], llama_debug
):
    flow, _ = real_hybrid_flow

    question = "Which French ace pilot and adventurer fly L'Oiseau Blanc?"

    response, duration, call_data = flow.generate(question)
    assert response is not None
    assert duration > 0
    assert llama_debug.get_event_pairs(CBEventType.QUERY)
