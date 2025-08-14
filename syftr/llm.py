import json
import os
import typing as T
from json import JSONDecodeError

import tiktoken
from anthropic import AnthropicVertex, AsyncAnthropicVertex
from google.cloud.aiplatform_v1beta1.types import content
from google.oauth2 import service_account

# from llama_index.core.base.llms.types import LLMMetadata, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import LLM

# from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
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

BASELINE_RAG_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

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
    llm_config.additional_kwargs = llm_config.additional_kwargs or {}
    llm_config.additional_kwargs
    return AzureOpenAI(
        model=llm_config.model_name,
        temperature=llm_config.temperature,
        top_p=llm_config.top_p,
        max_tokens=llm_config.max_tokens,
        max_retries=0,
        system_prompt=llm_config.system_prompt,
        engine=llm_config.deployment_name,
        api_key=llm_config.api_key.get_secret_value()
        if llm_config.api_key
        else cfg.azure_oai.api_key.get_secret_value(),
        azure_endpoint=llm_config.api_url.unicode_string()
        if llm_config.api_url
        else cfg.azure_oai.api_url.unicode_string(),
        api_version=llm_config.api_version or cfg.azure_oai.api_version,
        additional_kwargs=llm_config.additional_kwargs or {},
    )


def _construct_vertex_ai_llm(name: str, llm_config: VertexAILLM) -> Vertex:
    credentials = (
        service_account.Credentials.from_service_account_info(GCP_CREDS)
        if GCP_CREDS
        else {}
    )
    return Vertex(
        model=llm_config.model_name,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        max_retries=0,
        system_prompt=llm_config.system_prompt,
        project=llm_config.project_id or cfg.gcp_vertex.project_id,
        location=llm_config.region or cfg.gcp_vertex.region,
        safety_settings=llm_config.safety_settings or GCP_SAFETY_SETTINGS,
        credentials=credentials,
        context_window=llm_config.context_window,
        additional_kwargs=llm_config.additional_kwargs or {},
    )


def _construct_anthropic_vertex_llm(
    name: str, llm_config: AnthropicVertexLLM
) -> Anthropic:
    anthropic_llm = Anthropic(
        model=llm_config.model_name,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        max_retries=0,
        system_prompt=llm_config.system_prompt,
        project_id=llm_config.project_id or cfg.gcp_vertex.project_id,
        region=llm_config.region or cfg.gcp_vertex.region,
        thinking_dict=llm_config.thinking_dict,
        additional_kwargs=llm_config.additional_kwargs or {},
    )
    return add_scoped_credentials_anthropic(anthropic_llm)


def _construct_azure_ai_completions_llm(
    name: str, llm_config: AzureAICompletionsLLM
) -> AzureAICompletionsModel:
    return AzureAICompletionsModel(
        model_name=llm_config.model_name,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        system_prompt=llm_config.system_prompt,
        endpoint=llm_config.api_url.unicode_string(),
        credential=llm_config.api_key.get_secret_value(),
        client_kwargs=llm_config.client_kwargs,
        api_version=llm_config.api_version,
    )


def _construct_cerebras_llm(name: str, llm_config: CerebrasLLM) -> Cerebras:
    return Cerebras(
        model=llm_config.model_name,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        max_retries=0,
        system_prompt=llm_config.system_prompt,
        api_key=cfg.cerebras.api_key.get_secret_value(),
        api_base=cfg.cerebras.api_url.unicode_string(),
        context_window=llm_config.context_window,  # Use raw value as per existing Cerebras configs
        is_chat_model=llm_config.is_chat_model,
        is_function_calling_model=llm_config.is_function_calling_model,
        additional_kwargs=llm_config.additional_kwargs or {},
    )


def _construct_openai_like_llm(name: str, llm_config: OpenAILikeLLM) -> OpenAILike:
    return OpenAILike(
        model=llm_config.model_name,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        max_retries=0,
        system_prompt=llm_config.system_prompt,
        api_base=str(llm_config.api_base),
        api_key=llm_config.api_key.get_secret_value(),
        api_version=llm_config.api_version,  # type: ignore
        context_window=llm_config.context_window,
        is_chat_model=llm_config.is_chat_model,
        is_function_calling_model=llm_config.is_function_calling_model,
        timeout=llm_config.timeout,
        additional_kwargs=llm_config.additional_kwargs or {},
    )


LLM_NAMES__LOCAL_MODELS: T.List[str] = [
    model.model_name for model in cfg.local_models.generative or []
]
LLM_NAMES__GENERATIVE_MODELS: T.List[str] = [
    model for model in cfg.generative_models.keys()
]


if LLM_NAMES__GENERATIVE_MODELS and "gpt-4o-mini" not in LLM_NAMES__GENERATIVE_MODELS:
    BASELINE_LLM = LLM_NAMES__GENERATIVE_MODELS[0]

LLM_NAMES: T.List[str] = LLM_NAMES__LOCAL_MODELS + LLM_NAMES__GENERATIVE_MODELS
assert len(LLM_NAMES) == len(set(LLM_NAMES)), (
    "Duplicate LLM names found in configuration. Please ensure all LLM names are unique."
)

# at least one model is required for unit testing
BASELINE_LLM = "gpt-4o-mini"
if BASELINE_LLM not in LLM_NAMES:
    if LLM_NAMES:
        BASELINE_LLM = LLM_NAMES[0]
    else:
        LLM_NAMES = LLM_NAMES__GENERATIVE_MODELS = [BASELINE_LLM]


def get_generative_llm(
    name: str,
    temperature: float | None,
    top_p: float | None,
) -> FunctionCallingLLM:
    if name not in LLM_NAMES__GENERATIVE_MODELS:
        raise ValueError(
            f"LLM '{name}' not found in configured generative models. "
            f"Available generative models: {LLM_NAMES__GENERATIVE_MODELS}"
        )
    llm_instance: T.Optional[FunctionCallingLLM] = None
    llm_config_instance = cfg.generative_models[name]

    if temperature is not None:
        llm_config_instance.temperature = temperature
    if top_p is not None:
        llm_config_instance.top_p = top_p

    provider = getattr(llm_config_instance, "provider", None)

    if provider == "azure_openai" and isinstance(llm_config_instance, AzureOpenAILLM):
        llm_instance = _construct_azure_openai_llm(name, llm_config_instance)
    elif provider == "vertex_ai" and isinstance(llm_config_instance, VertexAILLM):
        llm_instance = _construct_vertex_ai_llm(name, llm_config_instance)
    elif provider == "anthropic_vertex" and isinstance(
        llm_config_instance, AnthropicVertexLLM
    ):
        llm_instance = _construct_anthropic_vertex_llm(name, llm_config_instance)
    elif provider == "azure_ai" and isinstance(
        llm_config_instance, AzureAICompletionsLLM
    ):
        llm_instance = _construct_azure_ai_completions_llm(name, llm_config_instance)
    elif provider == "cerebras" and isinstance(llm_config_instance, CerebrasLLM):
        llm_instance = _construct_cerebras_llm(name, llm_config_instance)
    elif provider == "openai_like" and isinstance(llm_config_instance, OpenAILikeLLM):
        llm_instance = _construct_openai_like_llm(name, llm_config_instance)
    else:
        raise ValueError(
            f"Unsupported provider type '{provider}' or "
            f"mismatched Pydantic config model type for model '{name}'."
        )
    return llm_instance


def get_local_llm(
    name: str,
    temperature: float | None = None,
    top_p: float | None = None,
) -> FunctionCallingLLM:
    assert name in LLM_NAMES__LOCAL_MODELS, (
        f"LLM '{name}' not found in configured local models. "
        f"Available local models: {LLM_NAMES__LOCAL_MODELS}"
    )
    model = next(
        model for model in cfg.local_models.generative or [] if model.model_name == name
    )
    if top_p is not None:
        model.additional_kwargs["top_p"] = top_p
    return OpenAILike(  # type: ignore
        api_base=str(model.api_base),
        api_key=model.api_key.get_secret_value()
        if model.api_key is not None
        else cfg.local_models.default_api_key.get_secret_value(),
        model=model.model_name,
        temperature=temperature if temperature is not None else model.temperature,
        max_tokens=model.max_tokens,
        context_window=_scale(model.context_window),
        is_chat_model=model.is_chat_model,
        is_function_calling_model=model.is_function_calling_model,
        timeout=model.timeout,
        max_retries=model.max_retries,
        additional_kwargs=model.additional_kwargs,
    )


def get_llm(
    name: str,
    temperature: float | None = None,
    top_p: float | None = None,
) -> FunctionCallingLLM:
    if name in LLM_NAMES__GENERATIVE_MODELS:
        return get_generative_llm(name, temperature, top_p)
    if name in LLM_NAMES__LOCAL_MODELS:
        return get_local_llm(name, temperature, top_p)
    raise ValueError(
        f"LLM '{name}' not found in configured LLMs. Available LLMs: {LLM_NAMES}"
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
            llm_instance = get_llm(
                name=name,
                temperature=llm_config_instance.temperature,
                top_p=llm_config_instance.top_p,
            )
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


def get_all_llms_with_defaults() -> T.List[FunctionCallingLLM]:
    return [get_llm(name, temperature=None, top_p=None) for name in LLM_NAMES]


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
        DefaultNamedArg(T.Literal["all"] | T.AbstractSet[str], "allowed_special"),  # type: ignore
        DefaultNamedArg(T.Literal["all"] | T.Collection[str], "disallowed_special"),  # type: ignore
    ],
    list[int],
]:
    if name in [
        "o1",
        "o3-mini",
        "gpt-4o-mini",
        "gpt-4o-std",
        "gpt-4o",
        "anthropic-sonnet-35",
        "anthropic-haiku-35",
        "llama-33-70B",
        "mistral-large",
        "gemini-pro",
        "gemini-flash",
        "gemini-flash2",
        "gemini-pro-exp",
        "gemini-flash-exp",
        "gemini-flash-think-exp",
        "cerebras-llama33-70B",
        "cerebras-qwen-3",
        "cerebras-scout",
        "cerebras-llama31-8B",
        "cerebras-deepseek",
        "phi-4",
        "azure-llama-33-70b",
        "azure-mistral-large",
        "azure-phi-4",
        "azure-r1",
        "together-r1",
        "together-v3",
        "together-V3",
        "datarobot-deployed",
    ]:
        return tiktoken.encoding_for_model("gpt-4o-mini").encode  # type: ignore
    if name == "gpt-35-turbo":
        return tiktoken.encoding_for_model("gpt-35-turbo").encode  # type: ignore
    if name in LLM_NAMES__LOCAL_MODELS:
        return tiktoken.encoding_for_model("gpt-4o-mini").encode  # type: ignore
    raise ValueError("No tokenizer for specified model: %s" % name)
