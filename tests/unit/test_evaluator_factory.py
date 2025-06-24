import pytest

from syftr.evaluation.evaluator_factory import json_parser_function


@pytest.mark.parametrize(
    "data,expected_score,expected_reasoning",
    [
        (
            '{"score": 1e2, "reasoning": "Looks good."}',
            1e2,
            "Looks good.",
        ),
        (
            'Reasoning missing {"score": 1e2}',
            None,
            None,
        ),
        (
            'Reasoning empty {"score": 1e2, "reasoning": ""}',
            None,
            None,
        ),
        (
            'Bad score {"score": "top", "reasoning": "best I have seen so far"}',
            None,
            None,
        ),
        (
            'Score missing {"reasoning": "best I have seen so far"}',
            None,
            None,
        ),
        (
            'With additional text {"score": 0.8, "reasoning": "Looks good."} Some text after',
            0.8,
            "Looks good.",
        ),
        (
            'Two JSONs {"score": 0.5, "reasoning": "First."} Second {"score": 0.9, "reasoning": "Second."}',
            0.9,
            "Second.",
        ),
        (
            """Proper nesting {"score": 0.5, "reasoning": {"score": 0.9, "reasoning": "Second."}} Second""",
            None,
            None,
        ),
        ("No JSON here!", None, None),
        ('Invalid JSON {"score": 0.7, "reasoning": "Oops",}', None, None),
        ('Bad keys {"foo": 1, "bar": 2}', None, None),
        (
            'With string score {"score": "1.0", "reasoning": "String score"}',
            1.0,
            "String score",
        ),
        (
            """Nested and normal {"score": 0.5, "reasoning": {"score": 0.9, "reasoning": "Second."}} Some text {"score": 0.8, "reasoning": "Looks good."} Some text after""",
            0.8,
            "Looks good.",
        ),
        (
            """Bad formatting and newlines
{

  "score": 1.0,

   "reasoning": "with bad formatting and newlines"

}""",
            1.0,
            "with bad formatting and newlines",
        ),
        (
            '''Nesting and formatting 
{
  "score": 0.5, 
  "reasoning": """{"score": 0.9, "reasoning": "Second."}""""
} 
Some text 
{
  "score": 0.8, 
  "reasoning": "Looks good."
}
Some text after''',
            0.8,
            "Looks good.",
        ),
        (
            """Last one nested
{
  "score": 0.8, 
  "reasoning": "Looks good."
}
{
  "score": 0.5, 
  "reasoning": {"score": 0.5, "reasoning": "all fine?"}
} """,
            None,
            None,
        ),
    ],
)
def test_json_parser(data, expected_score, expected_reasoning):
    score, reasoning = json_parser_function(data)
    assert score == expected_score
    assert reasoning == expected_reasoning
