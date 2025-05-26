import typing as T

import pytest
from llama_index.core.callbacks import CBEventType

from syftr.flows import CritiqueAgentFlow, ReActAgentFlow, SubQuestionRAGFlow
from syftr.logger import logger
from syftr.studies import StudyConfig

QA_PAIRS = {
    "hotpotqa_hf/dev": [
        ("Were Scott Derrickson and Ed Wood both Americans?", "yes"),
    ],
    "hotpot/dev": [
        ("Were Scott Derrickson and Ed Wood both Americans?", "yes"),
    ],
    "crag/music": [
        ("what is the song that dua lipa did with elton john?", "cold heart")
    ],
    "financebench": [
        (
            "As of FY2023Q1, why did Pepsico raise full year guidance for FY2023?",
            "Pepsico experienced a strong start to FY2023.",
        ),
    ],
}


@pytest.mark.xfail(reason="Tool use not guaranteed for all models")
def test_critique_agent_flow(
    critique_agent_flow: T.Tuple[CritiqueAgentFlow, StudyConfig],
    llama_debug,
):
    flow, study_config = critique_agent_flow
    for question, _ in QA_PAIRS[study_config.dataset.name]:
        _, _, call_data = flow.generate(question)
        assert call_data
        assert llama_debug.get_event_pairs(CBEventType.LLM)
        assert llama_debug.get_event_pairs(CBEventType.AGENT_STEP)
        if not llama_debug.get_event_pairs(CBEventType.SUB_QUESTION):
            logger.warning("Critique agent did not use sub-questions.")
        assert llama_debug.get_event_pairs(CBEventType.SYNTHESIZE)


@pytest.mark.xfail(reason="Some models hit the max iterations limit")
def test_react_flow(
    react_agent_flow: T.Tuple[ReActAgentFlow, StudyConfig],
    llama_debug,
):
    flow, study_config = react_agent_flow
    for question, _ in QA_PAIRS[study_config.dataset.name]:
        _, _, call_data = flow.generate(question)  # type: ignore
        assert call_data
        assert llama_debug.get_event_pairs(CBEventType.LLM)
        assert llama_debug.get_event_pairs(CBEventType.AGENT_STEP)
        assert llama_debug.get_event_pairs(CBEventType.FUNCTION_CALL)
        assert llama_debug.get_event_pairs(CBEventType.SYNTHESIZE)


@pytest.mark.flaky(reruns=4, reruns_delay=2)
def test_subquestion_flow(
    real_subquestion_flow: T.Tuple[SubQuestionRAGFlow, StudyConfig],
    llama_debug,
):
    flow, study_config = real_subquestion_flow
    for question, _ in QA_PAIRS[study_config.dataset.name]:
        _, _, call_data = flow.generate(question)  # type: ignore
        assert len(call_data) > 1, call_data
        assert sum(call.cost for call in call_data) > 0, call_data
        assert sum(call.input_tokens for call in call_data) > 0, call_data
        assert sum(call.output_tokens for call in call_data) > 0, call_data
        assert llama_debug.get_event_pairs(CBEventType.LLM)
        assert llama_debug.get_event_pairs(CBEventType.SUB_QUESTION)


@pytest.mark.flaky(reruns=4, reruns_delay=2)
def test_react_agent_flow_hybrid_hyde_reranker_few_shot(
    react_agent_flow_hybrid_hyde_reranker_few_shot, llama_debug
):
    flow, study_config = react_agent_flow_hybrid_hyde_reranker_few_shot
    for question, _ in QA_PAIRS[study_config.dataset.name]:
        flow.generate(question)
    assert llama_debug.get_event_pairs(CBEventType.AGENT_STEP)
