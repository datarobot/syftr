from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
)

from llama_index.llms.openai.utils import O1_MODELS

if TYPE_CHECKING:
    pass


def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
    out = {
        **self._model_kwargs,
        **kwargs,
    }
    if out.get("tools"):
        out["tool_choice"] = {"type": "any"}
    return out


def _get_model_kwargs_responses(self, **kwargs: Any) -> Dict[str, Any]:
    """Patch for OpenAI Responses API for tool_choice kwarg."""
    initial_tools = self.built_in_tools or []
    model_kwargs = {
        "model": self.model,
        "include": self.include,
        "instructions": self.instructions,
        "max_output_tokens": self.max_output_tokens,
        "metadata": self.call_metadata,
        "previous_response_id": self._previous_response_id,
        "store": self.store,
        "temperature": self.temperature,
        "tools": [*initial_tools, *kwargs.pop("tools", [])],
        # Hardcode auto tool choice to override other settings
        # Responses API only supports auto tool choice
        "tool_choice": "auto",
        "top_p": self.top_p,
        "truncation": self.truncation,
        "user": self.user,
    }

    if self.model in O1_MODELS and self.reasoning_options is not None:
        model_kwargs["reasoning"] = self.reasoning_options

    # priority is class args > additional_kwargs > runtime args
    model_kwargs.update(self.additional_kwargs)

    kwargs = kwargs or {}
    model_kwargs.update(kwargs)

    return model_kwargs
