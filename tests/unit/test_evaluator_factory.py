from syftr.evaluation.evaluator_factory import json_parser_function


def test_json_parser__valid_json():
    response = (
        'Some text before {"score": 0.8, "reasoning": "Looks good."} some text after'
    )
    score, reasoning = json_parser_function(response)
    assert score == 0.8
    assert reasoning == "Looks good."


def test_json_parser__incomplete_json():
    response = 'Some text before {"score": 0.8, "reasoning": "Looks good."'
    score, reasoning = json_parser_function(response)
    assert score is None
    assert reasoning is None


def test_json_parser__multiple_json_objects():
    response = 'First {"score": 0.5, "reasoning": "First."} Second {"score": 0.9, "reasoning": "Second."}'
    score, reasoning = json_parser_function(response)
    assert score == 0.9
    assert reasoning == "Second."


def test_json_parser__no_nesting():
    response = 'First {"score": 0.5, "reasoning": """{"score": 0.9, "reasoning": "Second."}""""} Second '
    score, reasoning = json_parser_function(response)
    assert score is None
    assert reasoning is None


def test_json_parser__no_json():
    response = "No JSON here!"
    score, reasoning = json_parser_function(response)
    assert score is None
    assert reasoning is None


def test_json_parser__invalid_json():
    response = 'Here is an invalid JSON: {"score": 0.7, "reasoning": "Oops",}'
    score, reasoning = json_parser_function(response)
    assert score is None
    assert reasoning is None


def test_json_parser__missing_score_and_reasoning():
    response = 'Some text {"foo": 1, "bar": 2}'
    score, reasoning = json_parser_function(response)
    assert score is None
    assert reasoning is None


def test_json_parser__score_is_string():
    response = 'Some text {"score": "1.0", "reasoning": "String score"}'
    score, reasoning = json_parser_function(response)
    assert score == 1.0
    assert reasoning == "String score"
