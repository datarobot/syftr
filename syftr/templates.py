import json

from syftr.configuration import cfg
from syftr.logger import logger


def _load_json_file(file_path):
    logger.debug(f"Loading: {file_path}")
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


_TEMPLATES_WITHOUT_CONTEXT = _load_json_file(cfg.paths.templates_without_context)
_TEMPLATES_WITH_CONTEXT = _load_json_file(cfg.paths.templates_with_context)
_AGENTIC_TEMPLATES = _load_json_file(cfg.paths.agentic_templates)

_TEMPLATE_NAMES_WITHOUT_CONTEXT = sorted(list(_TEMPLATES_WITHOUT_CONTEXT.keys()))
_TEMPLATE_NAMES_WITH_CONTEXT = sorted(list(_TEMPLATES_WITH_CONTEXT.keys()))

assert _TEMPLATE_NAMES_WITHOUT_CONTEXT == _TEMPLATE_NAMES_WITH_CONTEXT, (
    "Bad prompt data"
)

_FEW_SHOT_PROMPT_TEMPLATE = "Consider the examples below.\n\n{{few_shot_examples}}\n\n"


def get_template_names():
    return _TEMPLATE_NAMES_WITHOUT_CONTEXT


def get_template(
    template_name: str, with_context: bool = False, with_few_shot_prompt: bool = False
):
    template = (
        _TEMPLATES_WITH_CONTEXT[template_name]
        if with_context
        else _TEMPLATES_WITHOUT_CONTEXT[template_name]
    )
    if with_few_shot_prompt:
        template = f"{_FEW_SHOT_PROMPT_TEMPLATE}{template}"
    return template


def get_agent_template(prompt_name: str):
    return _AGENTIC_TEMPLATES[prompt_name]
