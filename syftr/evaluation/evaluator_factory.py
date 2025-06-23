import json
import re
import typing as T

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
    if re.search(r"\{[^{}]*\{", response):
        logger.error("Nested JSON found in evaluator response: %s", response)
        return None, None
    json_pattern = r"\{[^{}]*\}"
    matches = re.findall(json_pattern, response)
    if matches:
        json_str = matches[-1]
        try:
            response_dict = json.loads(json_str)
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
            score = None
    reasoning = response_dict.get("reasoning")
    return score, reasoning


class EvaluatorFactory:
    def __init__(
        self,
        study_config: T.Union[StudyConfig, AgentStudyConfig],
    ):
        assert isinstance(study_config, StudyConfig), (
            "AgentStudyConfig needs to provide dataset information."
        )

        self.llm_names = study_config.evaluation.llms
        self.eval_type = study_config.evaluation.eval_type
        self.eval_system_template = study_config.evaluation.eval_system_template
        self.eval_user_template = study_config.dataset.eval_user_template
        self.score_threshold = study_config.evaluation.score_threshold

        assert self.eval_type == "correctness", (
            f"Unsupported evaluation type: {self.eval_type}. "
            "Currently only 'correctness' is supported."
        )

    def _get_correctness_evaluators(self) -> T.List[BaseEvaluator]:
        eval_llms = [get_llm(name) for name in self.llm_names]
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
