import pytest

from syftr.tuner.core import get_flow_name


@pytest.mark.parametrize(
    "rag_mode, expected",
    [
        ("no_rag", "Flow"),
        ("rag", "RAGFlow"),
        ("react_rag_agent", "ReActAgentFlow"),
        ("critique_rag_agent", "CritiqueAgentFlow"),
        ("sub_question_rag", "SubQuestionRAGFlow"),
    ],
)
def test_flow_name(rag_mode, expected):
    flow_name = get_flow_name(rag_mode)
    assert flow_name == expected, "Bad flow name"
