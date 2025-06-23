import pytest

from syftr.evaluation.evaluator_factory import json_parser_function


@pytest.mark.parametrize(
    "data,expected_score,expected_reasoning",
    [
        (
            'Some text before {"score": 0.8, "reasoning": "Looks good."} Some text after',
            0.8,
            "Looks good.",
        ),
        (
            'First {"score": 0.5, "reasoning": "First."} Second {"score": 0.9, "reasoning": "Second."}',
            0.9,
            "Second.",
        ),
        (
            'First {"score": 0.5, "reasoning": """{"score": 0.9, "reasoning": "Second."}""""} Second ',
            None,
            None,
        ),
        ("No JSON here!", None, None),
        ('Here is an invalid JSON: {"score": 0.7, "reasoning": "Oops",}', None, None),
        ('Some text {"foo": 1, "bar": 2}', None, None),
        (
            'Some text {"score": "1.0", "reasoning": "String score"}',
            1.0,
            "String score",
        ),
    ],
)
def test_json_parser(data, expected_score, expected_reasoning):
    score, reasoning = json_parser_function(data)
    assert score == expected_score
    assert reasoning == expected_reasoning
