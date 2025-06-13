import inspect
import json
import time
from typing import Any, Dict, Optional, Type, Union

import google
from azure.ai.inference.models._models import ChatCompletionsToolCall
from llama_index.core.base.llms.types import ChatResponse, CompletionResponse
from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
)
from openinference.instrumentation.llama_index._handler import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    INPUT_VALUE,
    JSON,
    LLM_INVOCATION_PARAMETERS,
    LLM_MODEL_NAME,
    LLM_TOKEN_COUNT_COMPLETION,
    LLM_TOKEN_COUNT_PROMPT,
    LLM_TOKEN_COUNT_TOTAL,
    OUTPUT_MIME_TYPE,
    OUTPUT_VALUE,
    EventHandler,
    _get_token_counts,
    _init_span_kind,
    _Span,
    _SpanHandler,
    context_api,
    get_attributes_from_context,
    time_ns,
)
from opentelemetry.trace import NoOpTracer
from pydantic import BaseModel, PrivateAttr
from vertexai.preview.tokenization import get_tokenizer_for_model

from syftr.configuration import LLMCostCharacters, LLMCostHourly, LLMCostTokens, cfg
from syftr.logger import logger

# Unit: $ / token
MODEL_PRICING_INFO: Dict[str, Dict[str, Dict[str, float]]] = {
    "tokens": {
        "gpt-4o-mini": {"input": 0.15 / 1e6, "output": 0.6 / 1e6},
        "gpt-4o": {"input": 2.5 / 1e6, "output": 10.0 / 1e6},
        "o1": {"input": 15.0 / 1e6, "output": 60.0 / 1e6},
        "o3-mini": {"input": 3.0 / 1e6, "output": 12 / 1e6},
        "gpt-35-turbo": {"input": 1.00 / 1e6, "output": 2.00 / 1e6},
        "mistral-large-2411": {"input": 3.00 / 1e6, "output": 9.00 / 1e6},
        "Llama-3.3-70B-Instruct": {"input": 0.71 / 1e6, "output": 0.71 / 1e6},
        "Phi-4": {"input": 0.125 / 1e6, "output": 0.50 / 1e6},
        "claude-3-5-sonnet-v2@20241022": {"input": 3.00 / 1e6, "output": 15.0 / 1e6},
        "claude-3-5-haiku@20241022": {"input": 0.80 / 1e6, "output": 4.00 / 1e6},
        # Azure costs not released yet - using Together.AI pricing for now
        "Deepseek-R1": {"input": 7.00 / 1e6, "output": 7.00 / 1e6},
        "deepseek-ai/DeepSeek-R1": {"input": 7.00 / 1e6, "output": 7.00 / 1e6},
        "deepseek-ai/DeepSeek-V3": {"input": 1.25 / 1e6, "output": 1.25 / 1e6},
        "gemini-2.0-flash-lite-preview-02-05": {
            "input": 0.075 / 1e6,
            "output": 0.3 / 1e6,
        },
        "gemini-flash-think-exp": {"input": 0.15 / 1e6, "output": 0.60 / 1e6},
        "gemini-flash-lite-preview-02-05": {"input": 0.075 / 1e6, "output": 0.3 / 1e6},
        "gemini-2.0-pro-exp-02-05": {"input": 1.50 / 1e6, "output": 1.50 / 1e6},
        "gemini-2.0-flash-001": {"input": 0.15 / 1e6, "output": 0.60 / 1e6},
        # New Gemini prices not yet available
        # "gemini-2.0-flash-thinking-exp-01-21": {
        #     "input": 0.15 / 1e6,
        #     "output": 0.60 / 1e6,
        # },
        # Pricing not yet known/available
        # "llama3.1-8b": {"input": 1.50 / 1e6, "output": 1.50 / 1e6},
        # "llama-3.3-70b": {"input": 2.68 / 1e6, "output": 3.54 / 1e6},
        "google/gemma-3-27b-it": {"input": 0.80 / 1e6, "output": 0.80 / 1e6},
        "nvidia/Llama-3_3-Nemotron-Super-49B": {
            "input": 0.90 / 1e6,
            "output": 0.90 / 1e6,
        },
        # Local models use together.ai pricing structure
        "Qwen/Qwen2.5": {"input": 0.80 / 1e6, "output": 0.80 / 1e6},
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": {
            "input": 0.88 / 1e6,
            "output": 0.88 / 1e6,
        },
        "microsoft/Phi-4-multimodal-instruct": {
            "input": 0.2 / 1e6,
            "output": 0.2 / 1e6,
        },
        "llama3.1-8b": {
            "input": 0.2 / 1e6,
            "output": 0.2 / 1e6,
        },
    },
    "characters": {
        "gemini-1.5-pro-002": {"input": 0.0003125 / 1e3, "output": 0.00125 / 1e3},
        "gemini-1.5-flash-002": {"input": 0.00001875 / 1e3, "output": 0.000075 / 1e3},
    },
    "seconds": {
        "datarobot/DeepSeek-Llama": {"cost_per_second": 12.06654 / 3600},
    },
}

if cfg.generative_models:
    logger.debug(
        f"Updating MODEL_PRICING_INFO from cfg.generative_models: {list(cfg.generative_models.keys())}"
    )
    for config_key, llm_config_item in cfg.generative_models.items():
        # model_key should match the identifier used by TokenTrackingSpan (usually llm.metadata.model_name)
        model_key = llm_config_item.metadata.model_name

        if not model_key:
            logger.warning(
                f"Model with configuration key '{config_key}' is missing 'metadata.model_name'. "
                "Skipping its cost info update for MODEL_PRICING_INFO."
            )
            continue

        cost_config = llm_config_item.cost
        cost_type = getattr(cost_config, "type", "unknown")

        if cost_type == "tokens":
            assert isinstance(cost_config, LLMCostTokens), cost_config
            # Ensure the sub-dictionary for 'tokens' exists
            MODEL_PRICING_INFO["tokens"].setdefault(model_key, {})
            # cost_config.input/output are "Cost per million tokens"
            # MODEL_PRICING_INFO stores "Cost per token"
            MODEL_PRICING_INFO["tokens"][model_key]["input"] = (
                cost_config.input / 1_000_000.0
            )
            MODEL_PRICING_INFO["tokens"][model_key]["output"] = (
                cost_config.output / 1_000_000.0
            )
            logger.debug(
                f"Updated token pricing for '{model_key}' from cfg ('{config_key}')."
            )
        elif cost_type == "characters":
            assert isinstance(cost_config, LLMCostCharacters), cost_config
            MODEL_PRICING_INFO["characters"].setdefault(model_key, {})
            # cost_config.input/output are "Cost per million characters"
            # MODEL_PRICING_INFO stores "Cost per character"
            MODEL_PRICING_INFO["characters"][model_key]["input"] = (
                cost_config.input / 1_000_000.0
            )
            MODEL_PRICING_INFO["characters"][model_key]["output"] = (
                cost_config.output / 1_000_000.0
            )
            logger.debug(
                f"Updated character pricing for '{model_key}' from cfg ('{config_key}')."
            )
        elif cost_type == "hourly":
            assert isinstance(cost_config, LLMCostHourly), cost_config
            MODEL_PRICING_INFO["seconds"].setdefault(model_key, {})
            # cost_config.rate is "Average inference cost per hour"
            # MODEL_PRICING_INFO stores "Cost per second"
            MODEL_PRICING_INFO["seconds"][model_key]["cost_per_second"] = (
                cost_config.rate / 3600.0
            )
            logger.debug(
                f"Updated hourly (per second) pricing for '{model_key}' from cfg ('{config_key}')."
            )
        else:
            logger.warning(
                f"Unknown or unsupported cost type '{cost_type}' for model '{model_key}' "
                f"(from cfg key '{config_key}'). Pricing not updated."
            )


class LLMCallData(BaseModel):
    llm_name: str
    input_tokens: int
    output_tokens: int
    input_characters: int
    output_characters: int
    total_tokens: int
    llm_call_latency: float
    invocation_parameters: str | None = None

    @property
    def cost(self) -> float:
        if price_info := MODEL_PRICING_INFO["tokens"].get(self.llm_name):
            return (
                self.input_tokens * price_info["input"]
                + self.output_tokens * price_info["output"]
            )

        if price_info := MODEL_PRICING_INFO["characters"].get(self.llm_name):
            return (
                self.input_characters * price_info["input"]
                + self.output_characters * price_info["output"]
            )

        if price_info := MODEL_PRICING_INFO["seconds"].get(self.llm_name):
            return price_info["cost_per_second"] * self.llm_call_latency

        raise ValueError(f"Don't know how to compute costs for LLM {self.llm_name=}")


class CircularReferenceEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen = set()

    def default(self, obj):
        if id(obj) in self.seen:
            return f"<Circular reference to {type(obj).__name__}>"
        self.seen.add(id(obj))
        if isinstance(obj, google.cloud.aiplatform_v1beta1.types.tool.FunctionCall):
            return obj.to_dict()
        if isinstance(obj, ChatCompletionsToolCall):
            return obj.as_dict()
        return super().default(obj)


class TokenTrackingSpan(_Span):
    _instance: Any = PrivateAttr()
    _bound_args: inspect.BoundArguments = PrivateAttr()

    def __init__(
        self,
        *args,
        instance: Any,
        bound_args: inspect.BoundArguments,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._instance = instance
        self._bound_args = bound_args

    def process_event(self, event: BaseEvent, **kwargs: Any) -> Any:
        # For LLM events, the super class will attach token counts and
        # to the span (this object)
        # Invocation params and model name should have already been attached
        # by self.process_instance on span creation
        super().process_event(event)

        if isinstance(event, (LLMChatStartEvent, LLMCompletionStartEvent)):
            self["call_start"] = time.time()

        if not isinstance(event, (LLMChatEndEvent, LLMCompletionEndEvent)):
            return

        self["call_end"] = time.time()
        latency = self._attributes["call_end"] - self._attributes["call_start"]  # type: ignore

        logger.debug("Model input: %s", self._attributes[INPUT_VALUE])
        logger.debug("Model output: %s", self._attributes[OUTPUT_VALUE])

        call_data = LLMCallData(
            llm_name=self._attributes[LLM_MODEL_NAME],  # type: ignore
            input_tokens=self._attributes[LLM_TOKEN_COUNT_PROMPT],  # type: ignore
            output_tokens=self._attributes[LLM_TOKEN_COUNT_COMPLETION],  # type: ignore
            total_tokens=self._attributes[LLM_TOKEN_COUNT_TOTAL],  # type: ignore
            input_characters=len(self._attributes[INPUT_VALUE]),  # type: ignore
            output_characters=len(self._attributes[OUTPUT_VALUE]),  # type: ignore
            invocation_parameters=self._attributes[LLM_INVOCATION_PARAMETERS],  # type: ignore
            llm_call_latency=latency,
        )
        # Recursively get the parent span until we find the Flow instance
        root_flow_span = self._find_root_flow_span()
        if (
            invocation_id := root_flow_span._bound_args.arguments.get("invocation_id")
        ) is not None:
            # Store the event data in the Flow instance
            assert hasattr(root_flow_span._instance, "_llm_call_data"), (
                root_flow_span._instance
            )
            root_flow_span._instance._llm_call_data[invocation_id].append(call_data)
        return event

    def _has_token_counts(self) -> bool:
        return all(
            key in self._attributes
            for key in [
                LLM_TOKEN_COUNT_PROMPT,
                LLM_TOKEN_COUNT_COMPLETION,
                LLM_TOKEN_COUNT_TOTAL,
            ]
        )

    def _extract_token_counts(
        self, response: Union[ChatResponse, CompletionResponse]
    ) -> None:
        """Add additional response handling logic for additional LLM response schemas.

        Called by super().process_event() for LLM Chat/Completion End events
        """
        super()._extract_token_counts(response)
        if self._has_token_counts():
            # Successfully extracted counts
            return

        # GPT models on AzureOpenAI
        if (raw := getattr(response, "raw", None)) and (
            usage := getattr(raw, "usage", None)
        ):
            for k, v in _get_token_counts(usage):
                self[k] = v
            return

        # Claude Models on GCP
        if (
            (raw := getattr(response, "raw", None))
            and hasattr(raw, "get")
            and (usage := raw.get("usage"))
            and (input_tokens := getattr(usage, "input_tokens", None))
            and (output_tokens := getattr(usage, "output_tokens", None))
            and not getattr(usage, "total_tokens", None)
        ):
            self[LLM_TOKEN_COUNT_PROMPT] = input_tokens
            self[LLM_TOKEN_COUNT_COMPLETION] = output_tokens
            self[LLM_TOKEN_COUNT_TOTAL] = input_tokens + output_tokens
            return

        # GCP Gemini responses (old)
        if (
            (raw := getattr(response, "raw", None))
            and hasattr(raw, "get")
            and (raw_resp := raw.get("_raw_response"))
            and (usage := getattr(raw_resp, "usage_metadata", None))
            and (input_tokens := getattr(usage, "prompt_token_count", None))
            and (output_tokens := getattr(usage, "candidates_token_count", None))
            and (total_tokens := getattr(usage, "total_token_count", None))
        ):
            self[LLM_TOKEN_COUNT_PROMPT] = input_tokens
            self[LLM_TOKEN_COUNT_COMPLETION] = output_tokens
            self[LLM_TOKEN_COUNT_TOTAL] = total_tokens
            return

        if self._attributes[LLM_MODEL_NAME] in MODEL_PRICING_INFO["characters"].keys():
            num_input_tokens = len(str(self._attributes[INPUT_VALUE]))
            num_output_tokens = len(str(self._attributes[OUTPUT_VALUE]))
            self[LLM_TOKEN_COUNT_PROMPT] = num_input_tokens
            self[LLM_TOKEN_COUNT_COMPLETION] = num_output_tokens
            self[LLM_TOKEN_COUNT_TOTAL] = num_input_tokens + num_output_tokens
            return

        # GCP Gemini responses (new)
        if self._attributes[LLM_MODEL_NAME] in MODEL_PRICING_INFO[
            "tokens"
        ].keys() and "gemini" in str(self._attributes[LLM_MODEL_NAME]):
            try:
                try:
                    tokenizer = get_tokenizer_for_model(
                        self._attributes[LLM_MODEL_NAME]
                    )
                except Exception as e:
                    logger.debug("Vertex tokenizer error: %s", e)
                    # temporary hack
                    tokenizer = get_tokenizer_for_model("gemini-1.5-flash-002")
                num_input_tokens = tokenizer.count_tokens(
                    self._attributes[INPUT_VALUE]
                ).total_tokens
                num_output_tokens = tokenizer.count_tokens(
                    self._attributes[OUTPUT_VALUE]
                ).total_tokens
                self[LLM_TOKEN_COUNT_PROMPT] = num_input_tokens
                self[LLM_TOKEN_COUNT_COMPLETION] = num_output_tokens
                self[LLM_TOKEN_COUNT_TOTAL] = num_input_tokens + num_output_tokens
                return
            except Exception as e:
                logger.warning("Vertex tokenizer error: %s", e)

        raise ValueError(
            f"Failed to extract token counts from response object `{response=}` when using model `{self._attributes[LLM_MODEL_NAME]}`"
        )

    def _find_root_flow_span(self) -> "TokenTrackingSpan":
        parent_span = self._parent
        if parent_span is None:
            return self
        last_parent = parent_span
        while parent_span is not None:
            assert isinstance(parent_span, TokenTrackingSpan), parent_span
            instance = getattr(parent_span, "_instance", None)
            if _is_flow(instance) or _is_evaluator(instance):
                return parent_span
            last_parent = parent_span
            parent_span = parent_span._parent
        raise ValueError(
            f"Could not find Flow class in parent spans of `{self=}`. Root span is `{last_parent}`"
        )

    def model_dump_json(self, **kwargs):
        return json.dumps(self.dict(**kwargs), cls=CircularReferenceEncoder)

    def process_output(self, instance, result):
        try:
            self[OUTPUT_VALUE] = json.dumps(
                result.dict(exclude_unset=True), cls=CircularReferenceEncoder
            )
            self[OUTPUT_MIME_TYPE] = JSON
        except ValueError as e:
            logger.error(f"Error serializing to JSON: {e}")
            super().process_output(instance, result)


def _is_flow(instance: Any) -> bool:
    return any("Flow" in cls.__name__ for cls in instance.__class__.mro())


def _is_evaluator(instance: Any) -> bool:
    return isinstance(instance, CorrectnessEvaluator)


def init_span_kind(instance: Any) -> Optional[str]:
    if _is_flow(instance):
        return "SyftrFlow"
    if _is_evaluator(instance):
        return "SyftrCorrectnessEvaluator"
    return _init_span_kind(instance)


def patch_pydantic_instance(
    instance: Any,
    un_serializable_type: Type,
    *fields: str,
    to_dict_method_name: str = "to_dict",
):
    """
    Patches a Pydantic BaseModel instance to add a custom serializer
    for fields of a specific type.

    Args:
        instance: The Pydantic BaseModel instance to patch.
        un_serializable_type: The class of the type that needs custom serialization.
        to_dict_method_name: The name of the method to call on the
            unserializable object to convert it to a dictionary (default: "to_dict")
    """

    if not isinstance(instance, BaseModel):
        return instance

    for attr, value in instance.model_dump().items():
        if isinstance(value, un_serializable_type):
            setattr(instance, attr, getattr(value, to_dict_method_name)())

    return instance


class TokenTrackingSpanHandler(_SpanHandler):
    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[_Span]:
        """Copied from the base class, with custom span type and args."""
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return None
        with self.lock:
            parent = self.open_spans.get(parent_span_id) if parent_span_id else None
        otel_span = self._otel_tracer.start_span(
            name=id_.partition("-")[0],
            start_time=time_ns(),
            attributes=dict(get_attributes_from_context()),
            context=(parent.context if parent else None),
        )
        span = TokenTrackingSpan(
            otel_span=otel_span,
            span_kind=_init_span_kind(instance),
            parent=parent,
            id_=id_,
            parent_id=parent_span_id,
            instance=instance,
            bound_args=bound_args,
        )
        span.process_instance(instance)
        span.process_input(instance, bound_args)
        return span

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        result = patch_pydantic_instance(
            result, google.cloud.aiplatform_v1beta1.types.tool.FunctionCall
        )
        return super().prepare_to_exit_span(id_, bound_args, instance, result, **kwargs)


class TokenTrackingEventHandler(EventHandler):
    """Subclass of OpenInference handler."""

    def __init__(self, **kwargs) -> None:
        super().__init__(tracer=NoOpTracer())
        # Overwrite _span_handler with modified span handler
        self._span_handler = TokenTrackingSpanHandler(tracer=NoOpTracer())  # type: ignore
