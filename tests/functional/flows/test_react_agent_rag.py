import pytest
from llama_index.core.callbacks import CBEventType


@pytest.mark.xfail(reason="Will be decommissioned")
def test_react_rag_agent_simple(
    react_agent_rag_flow, correctness_evaluator, llama_debug
):
    question = "Who is Adam Collis?"
    react_agent_rag_flow.agent.chat(question)
    assert llama_debug.get_event_pairs(CBEventType.AGENT_STEP)


@pytest.mark.xfail(reason="Will be decommissioned")
def test_react_rag_agent_system_prompt(
    react_agent_rag_flow_system_prompt, correctness_evaluator, llama_debug
):
    question = "Who is Adam Collis?"
    react_agent_rag_flow_system_prompt.agent.chat(question)
    assert llama_debug.get_event_pairs(CBEventType.AGENT_STEP)
