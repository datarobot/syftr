from llama_index.core.evaluation import EvaluationResult

from syftr.evaluation.evaluator_factory import json_parser_function


def parse_correctness_evaluation(
    query: str, response: str, threshold: float = 4.0
) -> EvaluationResult:
    score, feedback = json_parser_function(response)
    if score is None:
        raise ValueError(f"Could not parse score from response: {response}")
    passing = score >= threshold
    return EvaluationResult(
        query=query, response=response, passing=passing, score=score, feedback=feedback
    )


def parse_correctness_evaluation_ten(query: str, response: str) -> EvaluationResult:
    return parse_correctness_evaluation(query, response, threshold=8.0)


def parse_correctness_evaluation_comparison(
    query: str, response: str
) -> EvaluationResult:
    return parse_correctness_evaluation(query, response, threshold=2.0)


def parse_correctness_evaluation_simple(query: str, response: str) -> EvaluationResult:
    response_formatted = response.upper().replace(" ", "")
    passing = "*YES*" in response_formatted and "*NO*" not in response_formatted
    feedback = response
    return EvaluationResult(
        query=query, response=response, passing=passing, feedback=feedback
    )
