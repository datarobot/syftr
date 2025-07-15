from llama_index.core.evaluation import EvaluationResult

from syftr.evaluation.evaluator_factory import json_parser_function


def parse_correctness_evaluation(query: str, response: str) -> EvaluationResult:
    score, feedback = json_parser_function(response)
    if score is None:
        raise ValueError(f"Could not parse score from response: {response}")
    passing = score >= 4.0
    return EvaluationResult(
        query=query, response=response, passing=passing, score=score, feedback=feedback
    )


def parse_correctness_evaluation_ten(query: str, response: str) -> EvaluationResult:
    score, feedback = json_parser_function(response)
    if score is None:
        raise ValueError(f"Could not parse score from response: {response}")
    passing = score >= 8.0
    return EvaluationResult(
        query=query, response=response, passing=passing, score=score, feedback=feedback
    )


def parse_correctness_evaluation_simple(query: str, response: str) -> EvaluationResult:
    passing = response.split("\n")[0].strip().lower() == "yes"
    feedback = "\n".join(response.split("\n")[1:]).strip()
    return EvaluationResult(
        query=query, response=response, passing=passing, feedback=feedback
    )
