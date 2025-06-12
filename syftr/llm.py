import json
import os
import typing as T
from json import JSONDecodeError

import tiktoken
from anthropic import AnthropicVertex, AsyncAnthropicVertex
from google.cloud.aiplatform_v1beta1.types import content
from google.oauth2 import service_account
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import LLM
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.azure_inference import AzureAICompletionsModel
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.cerebras import Cerebras
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.vertex import Vertex
from mypy_extensions import DefaultNamedArg

from syftr.configuration import (
    NON_OPENAI_CONTEXT_WINDOW_FACTOR,
    AnthropicVertexLLM,
    AzureAICompletionsLLM,
    AzureOpenAILLM,
    CerebrasLLM,
    OpenAILikeLLM,
    Settings,
    VertexAILLM,
    cfg,
)
from syftr.logger import logger
from syftr.patches import _get_all_kwargs

Anthropic._get_all_kwargs = _get_all_kwargs  # type: ignore


def _scale(
    context_window_length: int, factor: float = NON_OPENAI_CONTEXT_WINDOW_FACTOR
) -> int:
    return int(context_window_length * factor)


if (hf_token := cfg.hf_embeddings.api_key.get_secret_value()) != "NOT SET":
    os.environ["HF_TOKEN"] = hf_token


GCP_SAFETY_SETTINGS = {
    content.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: content.SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    content.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: content.SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    content.HarmCategory.HARM_CATEGORY_HARASSMENT: content.SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    content.HarmCategory.HARM_CATEGORY_HATE_SPEECH: content.SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

try:
    GCP_CREDS = json.loads(cfg.gcp_vertex.credentials.get_secret_value())
except JSONDecodeError:
    GCP_CREDS = {}


def add_scoped_credentials_anthropic(anthropic_llm: Anthropic) -> Anthropic:
    """Add Google service account credentials to an Anthropic LLM"""
    credentials = (
        service_account.Credentials.from_service_account_info(GCP_CREDS).with_scopes(
            ["https://www.googleapis.com/auth/cloud-platform"]
        )
        if GCP_CREDS
        else None
    )
    sync_client = anthropic_llm._client
    assert isinstance(sync_client, AnthropicVertex)
    sync_client.credentials = credentials
    anthropic_llm._client = sync_client
    async_client = anthropic_llm._aclient
    assert isinstance(async_client, AsyncAnthropicVertex)
    async_client.credentials = credentials
    anthropic_llm._aclient = async_client
    return anthropic_llm


def _construct_azure_openai_llm(name: str, llm_config: AzureOpenAILLM) -> AzureOpenAI:
    return AzureOpenAI(
        model=llm_config.metadata.model_name,
        deployment_name=llm_config.deployment_name or llm_config.metadata.model_name,
        api_key=cfg.azure_oai.api_key.get_secret_value(),
        azure_endpoint=cfg.azure_oai.api_url.unicode_string(),
        api_version=llm_config.api_version or cfg.azure_oai.api_version,
        temperature=llm_config.temperature,
        max_tokens=llm_config.metadata.num_output,
        max_retries=llm_config.max_retries,
        additional_kwargs=llm_config.additional_kwargs or {},
    )


def _construct_vertex_ai_llm(name: str, llm_config: VertexAILLM) -> Vertex:
    credentials = (
        service_account.Credentials.from_service_account_info(GCP_CREDS)
        if GCP_CREDS
        else {}
    )
    return Vertex(
        model=llm_config.model or llm_config.metadata.model_name,
        project=cfg.gcp_vertex.project_id,
        credentials=credentials,
        temperature=llm_config.temperature,
        safety_settings=llm_config.safety_settings or GCP_SAFETY_SETTINGS,
        max_tokens=llm_config.metadata.num_output,
        context_window=_scale(llm_config.metadata.context_window),
        max_retries=llm_config.max_retries,
        additional_kwargs=llm_config.additional_kwargs or {},
        location=cfg.gcp_vertex.region,
    )


def _construct_anthropic_vertex_llm(
    name: str, llm_config: AnthropicVertexLLM
) -> Anthropic:
    anthropic_llm = Anthropic(
        model=llm_config.model,
        project_id=llm_config.project_id or cfg.gcp_vertex.project_id,
        region=llm_config.region or cfg.gcp_vertex.region,
        temperature=llm_config.temperature,
        max_tokens=llm_config.metadata.num_output,
        max_retries=llm_config.max_retries,
        additional_kwargs=llm_config.additional_kwargs or {},
    )
    return add_scoped_credentials_anthropic(anthropic_llm)


def _construct_azure_ai_completions_llm(
    name: str, llm_config: AzureAICompletionsLLM
) -> AzureAICompletionsModel:
    return AzureAICompletionsModel(
        credential=llm_config.api_key.get_secret_value(),
        endpoint=llm_config.endpoint.unicode_string(),
        model_name=llm_config.model_name,
        temperature=llm_config.temperature,
        metadata=llm_config.metadata.model_dump(),
    )


def _construct_cerebras_llm(name: str, llm_config: CerebrasLLM) -> Cerebras:
    return Cerebras(
        model=llm_config.model,
        api_key=cfg.cerebras.api_key.get_secret_value(),
        api_base=cfg.cerebras.api_url.unicode_string(),
        temperature=llm_config.temperature,
        max_tokens=llm_config.metadata.num_output,
        context_window=llm_config.metadata.context_window,  # Use raw value as per existing Cerebras configs
        is_function_calling_model=llm_config.metadata.is_function_calling_model,
        max_retries=llm_config.max_retries,
        additional_kwargs=llm_config.additional_kwargs or {},
    )


def _construct_openai_like_llm(name: str, llm_config: OpenAILikeLLM) -> OpenAILike:
    return OpenAILike(
        model=llm_config.model,
        api_base=str(llm_config.api_base),
        api_key=llm_config.api_key.get_secret_value(),
        api_version=llm_config.api_version,  # type: ignore
        max_tokens=llm_config.metadata.num_output,
        context_window=_scale(llm_config.metadata.context_window),
        is_chat_model=llm_config.metadata.is_chat_model,
        is_function_calling_model=llm_config.metadata.is_function_calling_model,
        timeout=llm_config.timeout,
        max_retries=llm_config.max_retries,
        additional_kwargs=llm_config.additional_kwargs or {},
    )


def load_configured_llms(config: Settings) -> T.Dict[str, FunctionCallingLLM]:
    _dynamically_loaded_llms: T.Dict[str, FunctionCallingLLM] = {}
    if not config.generative_models:
        return {}
    logger.debug(
        f"Loading LLMs from 'generative_models' configuration: {list(config.generative_models.keys())}"
    )
    for name, llm_config_instance in config.generative_models.items():
        llm_instance: T.Optional[FunctionCallingLLM] = None
        try:
            provider = getattr(llm_config_instance, "provider", None)

            if provider == "azure_openai" and isinstance(
                llm_config_instance, AzureOpenAILLM
            ):
                llm_instance = _construct_azure_openai_llm(name, llm_config_instance)
            elif provider == "vertex_ai" and isinstance(
                llm_config_instance, VertexAILLM
            ):
                llm_instance = _construct_vertex_ai_llm(name, llm_config_instance)
            elif provider == "anthropic_vertex" and isinstance(
                llm_config_instance, AnthropicVertexLLM
            ):
                llm_instance = _construct_anthropic_vertex_llm(
                    name, llm_config_instance
                )
            elif provider == "azure_ai" and isinstance(
                llm_config_instance, AzureAICompletionsLLM
            ):
                llm_instance = _construct_azure_ai_completions_llm(
                    name, llm_config_instance
                )
            elif provider == "cerebras" and isinstance(
                llm_config_instance, CerebrasLLM
            ):
                llm_instance = _construct_cerebras_llm(name, llm_config_instance)
            elif provider == "openai_like" and isinstance(
                llm_config_instance, OpenAILikeLLM
            ):
                llm_instance = _construct_openai_like_llm(name, llm_config_instance)
            else:
                raise ValueError(
                    f"Unsupported provider type '{provider}' or "
                    f"mismatched Pydantic config model type for model '{name}'."
                )
                continue

            if llm_instance:
                _dynamically_loaded_llms[name] = llm_instance
                logger.debug(f"Successfully loaded LLM '{name}' from configuration.")
        except Exception as e:
            # Log with traceback for easier debugging
            logger.error(
                f"Failed to load configured LLM '{name}' due to: {e}", exc_info=True
            )
            raise
    return _dynamically_loaded_llms


# When you add model, make sure all tests pass successfully
LLMs = {}

LLMs = load_configured_llms(cfg)


def get_llm(name: str | None = None):
    if not name:
        logger.warning("No LLM name specified.")
        return None
    assert name in LLMs, (
        f"Invalid LLM name specified: {name}. Valid options are: {list(LLMs.keys())}"
    )
    return LLMs[name]


def get_llm_name(llm: LLM | FunctionCallingLLM | None = None):
    for llm_name, llm_instance in LLMs.items():
        if llm == llm_instance:
            return llm_name
    raise ValueError(
        "Cannot find name for llm `{llm}`. Expected one of {set(LLMs.keys())}"
    )


def is_function_calling(llm: LLM):
    try:
        if getattr(llm.metadata, "is_function_calling_model", False):
            if "flash" in llm.metadata.model_name:
                return False
            return True
    except ValueError:
        return False


def get_tokenizer(
    name: str,
) -> T.Callable[
    [
        str,
        DefaultNamedArg(T.Literal["all"] | T.AbstractSet[str], "allowed_special"),
        DefaultNamedArg(T.Literal["all"] | T.Collection[str], "disallowed_special"),
    ],
    list[int],
]:
    if name == "gpt-35-turbo":
        return tiktoken.encoding_for_model("gpt-35-turbo").encode
    else:
        return tiktoken.encoding_for_model("gpt-4o-mini").encode
    raise ValueError("Invalid tokenizer specified")
