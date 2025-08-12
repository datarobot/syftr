import json
import re
import typing as T
from itertools import product

from llama_index.core.evaluation.base import BaseEvaluator
from llama_index.core.evaluation.correctness import CorrectnessEvaluator
from llama_index.core.prompts import (
    ChatMessage,
    ChatPromptTemplate,
    MessageRole,
)

from syftr.llm import get_llm
from syftr.logger import logger
from syftr.studies import AgentStudyConfig, StudyConfig


def json_parser_function(response: str) -> T.Tuple[T.Optional[float], T.Optional[str]]:
    """
    Parse the response from the evaluator to extract score and reasoning.
    In case the response is not valid, it returns None for both score and reasoning.
    The last JSON-like substring of the response is expected to be non-nested and
    should contain a "score" and "reasoning" key.
    The score can be a float or a string that can be converted to a float.
    The reasoning is expected to be a string.
    """

    # Check if the response ends with a nested JSON
    if re.search(r"\{[^{}]*\{[^{}]*\}[^{}]*\}[^{}]*$", response):
        logger.error("Nested JSON found at the end of evaluator response: %s", response)
        return None, None

    # Find all JSON-like objects in the response
    json_pattern = r"\{[^{}]*\}"
    matches = re.findall(json_pattern, response)

    if matches:
        last_json_str = matches[-1]
        try:
            response_dict = json.loads(last_json_str)
        except json.JSONDecodeError:
            logger.error("Invalid JSON response from evaluator: %s", response)
            return None, None
    else:
        logger.error("No JSON found in evaluator response: %s", response)
        return None, None

    score = response_dict.get("score")
    if score is not None:
        try:
            score = float(score)
        except ValueError:
            logger.error("Invalid score in evaluator response: %s", response)
            return None, None
    else:
        logger.error("Score missing in evaluator response: %s", response)
        return None, None
    reasoning = response_dict.get("reasoning")
    if not reasoning or not isinstance(reasoning, str):
        logger.error("Invalid reasoning in evaluator response: %s", response)
        return None, None
    return score, reasoning


class CorrectnessEvaluatorFactory:
    """Factory class to create LLM judges of type CorrectnessEvaluator based on the study configuration."""

    def __init__(
        self,
        study_config: T.Union[StudyConfig, AgentStudyConfig],
    ):
        assert isinstance(study_config, StudyConfig), (
            "AgentStudyConfig needs to provide dataset information."
        )

        self.llm_config = study_config.evaluation.llm_config
        self.eval_type = study_config.evaluation.eval_type

        system_template = study_config.evaluation.eval_system_template
        if self.llm_config.llm_use_reasoning is not None:
            system_template += (
                " /think" if self.llm_config.llm_use_reasoning else " /no_think"
            )

        self.eval_system_template = system_template
        self.eval_user_template = study_config.dataset.eval_user_template
        self.score_threshold = study_config.evaluation.score_threshold

        assert self.eval_type == "correctness", (
            f"Unsupported evaluation type: {self.eval_type}. "
            "Currently only 'correctness' is supported."
        )

    def _get_correctness_evaluators(self) -> T.List[BaseEvaluator]:
        llm_names = self.llm_config.llm_names
        llm_temperatures = [
            self.llm_config.llm_temperature_min
            + k * self.llm_config.llm_temperature_step
            for k in range(
                0,
                int(
                    (
                        self.llm_config.llm_temperature_max
                        - self.llm_config.llm_temperature_min
                    )
                    / self.llm_config.llm_temperature_step
                )
                + 1,
            )
        ]
        llm_top_ps = [
            self.llm_config.llm_top_p_min + k * self.llm_config.llm_top_p_step
            for k in range(
                0,
                int(
                    (self.llm_config.llm_top_p_max - self.llm_config.llm_top_p_min)
                    / self.llm_config.llm_top_p_step
                )
                + 1,
            )
        ]
        eval_llms = [
            get_llm(name, temperature=float(temp), top_p=float(top_p))
            for name, temp, top_p in product(llm_names, llm_temperatures, llm_top_ps)
        ]
        eval_template = ChatPromptTemplate(
            message_templates=[
                ChatMessage(role=MessageRole.SYSTEM, content=self.eval_system_template),
                ChatMessage(role=MessageRole.USER, content=self.eval_user_template),
            ]
        )
        evaluators: T.List[BaseEvaluator] = [
            CorrectnessEvaluator(
                llm=eval_llms[0],
                eval_template=eval_template,
                score_threshold=self.score_threshold,
                parser_function=json_parser_function,
            )
        ]
        return evaluators

    def get_evaluators(self) -> T.List[BaseEvaluator]:
        match self.eval_type:
            case "correctness":
                return self._get_correctness_evaluators()
            case _:
                raise ValueError(
                    f"Unsupported evaluation type: {self.eval_type}. "
                    "Currently only 'correctness' is supported."
                )
