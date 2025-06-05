import json
import os
import typing as T
from json import JSONDecodeError

import tiktoken
from anthropic import AnthropicVertex, AsyncAnthropicVertex
from google.cloud.aiplatform_v1beta1.types import content
from google.oauth2 import service_account
from llama_index.core.base.llms.types import LLMMetadata, MessageRole
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import LLM
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
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


LOCAL_MODELS = (
    {
        model.model_name: OpenAILike(  # type: ignore
            api_base=str(model.api_base),
            api_key=model.api_key.get_secret_value()
            if model.api_key is not None
            else cfg.local_models.default_api_key.get_secret_value(),
            model=model.model_name,
            max_tokens=model.max_tokens,
            context_window=_scale(model.context_window),
            is_chat_model=model.is_chat_model,
            is_function_calling_model=model.is_function_calling_model,
            timeout=model.timeout,
            max_retries=model.max_retries,
            additional_kwargs=model.additional_kwargs,
        )
        for model in cfg.local_models.generative
    }
    if cfg.local_models.generative
    else {}
)

AZURE_GPT35_TURBO = AzureOpenAI(
    model="gpt-3.5-turbo",
    deployment_name="gpt-35",
    api_key=cfg.azure_oai.api_key.get_secret_value(),
    azure_endpoint=str(cfg.azure_oai.api_url),
    api_version="2024-06-01",
    temperature=0,
    max_retries=0,
    additional_kwargs={"user": "syftr"},
)

AZURE_GPT4O_MINI = AzureOpenAI(
    model="gpt-4o-mini",
    deployment_name="gpt-4o-mini",
    api_key=cfg.azure_oai.api_key.get_secret_value(),
    azure_endpoint=str(cfg.azure_oai.api_url),
    api_version="2024-06-01",
    temperature=0,
    max_retries=0,
    additional_kwargs={"user": "syftr"},
)

AZURE_GPT4O_STD = AzureOpenAI(
    model="gpt-4o",
    deployment_name="gpt-4o",
    api_key=cfg.azure_oai.api_key.get_secret_value(),
    azure_endpoint=str(cfg.azure_oai.api_url),
    api_version="2024-06-01",
    temperature=0,
    max_retries=0,
    additional_kwargs={"user": "syftr"},
)

AZURE_o1 = AzureOpenAI(
    model="o1",
    deployment_name="o1",
    api_key=cfg.azure_oai.api_key.get_secret_value(),
    azure_endpoint=str(cfg.azure_oai.api_url),
    api_version="2024-12-01-preview",
    temperature=0,
    max_retries=0,
    additional_kwargs={"user": "syftr"},
)

AZURE_o3_MINI = AzureOpenAI(
    model="o3-mini",
    deployment_name="o3-mini",
    api_key=cfg.azure_oai.api_key.get_secret_value(),
    azure_endpoint=str(cfg.azure_oai.api_url),
    api_version="2024-12-01-preview",
    temperature=0,
    max_retries=0,
    additional_kwargs={"user": "syftr"},
)

AZURE_GPT35_TURBO_1106 = AzureOpenAI(
    model="gpt-35-turbo",
    deployment_name="gpt-35-turbo",
    api_key=cfg.azure_oai.api_key.get_secret_value(),
    azure_endpoint=str(cfg.azure_oai.api_url),
    api_version="2024-06-01",
    temperature=0,
    max_retries=0,
    additional_kwargs={"user": "syftr"},
)

AZURE_TEXT_EMBEDDING_ADA_002 = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-ada-002",
    api_key=cfg.azure_oai.api_key.get_secret_value(),
    azure_endpoint=str(cfg.azure_oai.api_url),
    api_version="2023-03-15-preview",
)

AZURE_TEXT_EMBEDDING_3_LARGE = AzureOpenAIEmbedding(
    model="text-embedding-3-large",
    deployment_name="text-embedding-3-large",
    api_key=cfg.azure_oai.api_key.get_secret_value(),
    azure_endpoint=str(cfg.azure_oai.api_url),
    api_version="2024-06-01",
)

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

GCP_GEMINI_PRO = Vertex(
    model="gemini-1.5-pro-002",
    project=cfg.gcp_vertex.project_id,
    credentials=service_account.Credentials.from_service_account_info(GCP_CREDS)
    if GCP_CREDS
    else {},
    temperature=0,
    safety_settings=GCP_SAFETY_SETTINGS,
    max_tokens=8000,
    context_window=_scale(2000000),
    max_retries=0,
    additional_kwargs={},
)

GCP_GEMINI_FLASH = Vertex(
    model="gemini-1.5-flash-002",
    project=cfg.gcp_vertex.project_id,
    credentials=service_account.Credentials.from_service_account_info(GCP_CREDS)
    if GCP_CREDS
    else {},
    temperature=0,
    safety_settings=GCP_SAFETY_SETTINGS,
    context_window=_scale(1048000),
    max_tokens=8000,
    max_retries=0,
    additional_kwargs={},
)

GCP_GEMINI_FLASH_EXP = Vertex(
    model="gemini-2.0-flash-lite-preview-02-05",
    project=cfg.gcp_vertex.project_id,
    credentials=service_account.Credentials.from_service_account_info(GCP_CREDS)
    if GCP_CREDS
    else {},
    temperature=0,
    max_tokens=8000,
    context_window=_scale(1048000),
    max_retries=0,
    safety_settings=GCP_SAFETY_SETTINGS,
    additional_kwargs={},
)


class VertexFlashThink(Vertex):
    def __init__(
        self,
        model: str = "text-bison",
        project: T.Optional[str] = None,
        location: T.Optional[str] = None,
        credentials: T.Optional[T.Any] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            project=project,
            location=location,
            credentials=credentials,
            **kwargs,
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            num_output=self.max_tokens,
            context_window=self.context_window,
            is_chat_model=self._is_chat_model,
            is_function_calling_model=False,
            model_name=self.model,
            system_role=(
                MessageRole.USER if self._is_gemini else MessageRole.SYSTEM
            ),  # Gemini does not support the default: MessageRole.SYSTEM
        )


GCP_GEMINI_FLASH_THINK_EXP = VertexFlashThink(
    model="gemini-2.0-flash-thinking-exp-01-21",
    project=cfg.gcp_vertex.project_id,
    credentials=service_account.Credentials.from_service_account_info(GCP_CREDS)
    if GCP_CREDS
    else {},
    temperature=0,
    max_tokens=8000,
    context_window=_scale(32000),
    max_retries=0,
    safety_settings=GCP_SAFETY_SETTINGS,
    additional_kwargs={},
)

GCP_GEMINI_PRO_EXP = Vertex(
    model="gemini-2.0-pro-exp-02-05",
    project=cfg.gcp_vertex.project_id,
    credentials=service_account.Credentials.from_service_account_info(GCP_CREDS)
    if GCP_CREDS
    else {},
    temperature=0,
    safety_settings=GCP_SAFETY_SETTINGS,
    max_tokens=8000,
    context_window=_scale(1048000),
    max_retries=0,
    additional_kwargs={},
)


GCP_GEMINI_FLASH2 = Vertex(
    model="gemini-2.0-flash-001",
    project=cfg.gcp_vertex.project_id,
    credentials=service_account.Credentials.from_service_account_info(GCP_CREDS)
    if GCP_CREDS
    else {},
    temperature=0,
    max_tokens=8000,
    context_window=_scale(1048000),
    max_retries=0,
    safety_settings=GCP_SAFETY_SETTINGS,
    additional_kwargs={},
)

TogetherLlamaSmall = OpenAILike(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    api_base="https://api.together.xyz/v1",
    api_key=cfg.togetherai.api_key.get_secret_value(),
    api_version=None,  # type: ignore
    max_tokens=2000,
    context_window=_scale(131072),
    is_chat_model=True,
    is_function_calling_model=True,
    timeout=3600,
    max_retries=0,
)

TogetherDeepseekR1 = OpenAILike(
    model="deepseek-ai/DeepSeek-R1",
    api_base="https://api.together.xyz/v1",
    api_key=cfg.togetherai.api_key.get_secret_value(),
    api_version=None,  # type: ignore
    max_tokens=5000,
    context_window=_scale(16384),
    is_chat_model=True,
    is_function_calling_model=False,
    timeout=3600,
    max_retries=0,
)

TogetherDeepseekV3 = OpenAILike(
    model="deepseek-ai/DeepSeek-V3",
    api_base="https://api.together.xyz/v1",
    api_key=cfg.togetherai.api_key.get_secret_value(),
    api_version=None,  # type: ignore
    max_tokens=2048,
    context_window=_scale(16384),
    is_chat_model=True,
    is_function_calling_model=False,
    timeout=3600,
    max_retries=0,
)


class DROpenAILike(OpenAILike):
    pass


DataRobotDeployedLLM = DROpenAILike(
    model="datarobot/model-name",
    api_base=str(cfg.datarobot.endpoint),
    api_key=cfg.datarobot.api_key.get_secret_value(),
    api_version=None,  # type: ignore
    max_tokens=2000,
    is_chat_model=True,
    context_window=_scale(14000),
    is_function_calling_model=True,
    timeout=3600,
    max_retries=0,
)


DRDeepseekLlama70BReasoning = DROpenAILike(
    model="datarobot/DeepSeek-Llama",
    api_base=str(cfg.datarobot.endpoint),
    api_key=cfg.datarobot.api_key.get_secret_value(),
    api_version=None,  # type: ignore
    max_tokens=3000,
    is_chat_model=True,
    context_window=_scale(14000),
    is_function_calling_model=False,
    timeout=3600,
    max_retries=0,
)


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


ANTHROPIC_CLAUDE_SONNET_35 = add_scoped_credentials_anthropic(
    Anthropic(
        model="claude-3-5-sonnet-v2@20241022",
        project_id=str(cfg.gcp_vertex.project_id),
        region="us-east5",
        temperature=0,
    )
)


ANTHROPIC_CLAUDE_HAIKU_35 = add_scoped_credentials_anthropic(
    Anthropic(
        model="claude-3-5-haiku@20241022",
        project_id=str(cfg.gcp_vertex.project_id),
        region="us-east5",
        temperature=0,
    )
)


class AzureAICompletionsModelLlama(AzureAICompletionsModel):
    def __init__(self, credential, model_name, endpoint, temperature=0):
        super().__init__(
            credential=credential,
            model_name=model_name,
            endpoint=endpoint,
            temperature=temperature,
        )

    @property
    def metadata(self):
        return LLMMetadata(
            context_window=120000,
            num_output=1000,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name="Llama-3.3-70B-Instruct",
        )


class AzureAICompletionsModelPhi4(AzureAICompletionsModel):
    def __init__(self, credential, model_name, endpoint, temperature=0):
        super().__init__(
            credential=credential,
            model_name=model_name,
            endpoint=endpoint,
            temperature=temperature,
        )

    @property
    def metadata(self):
        return LLMMetadata(
            context_window=14000,
            num_output=1000,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name="Phi-4",
        )


class AzureAICompletionsModelR1(AzureAICompletionsModel):
    def __init__(self, credential, model_name, endpoint, temperature=0):
        super().__init__(
            credential=credential,
            model_name=model_name,
            endpoint=endpoint,
            temperature=temperature,
        )

    @property
    def metadata(self):
        return LLMMetadata(
            context_window=120000,
            num_output=8000,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name="Deepseek-R1",
        )


AZURE_LLAMA33_70B = AzureAICompletionsModelLlama(
    credential=cfg.azure_inference_llama33.api_key.get_secret_value(),  # type: ignore
    model_name=str(cfg.azure_inference_llama33.model_name),  # type: ignore
    endpoint=(  # type: ignore
        "https://"
        + str(cfg.azure_inference_llama33.default_deployment)
        + "."
        + str(cfg.azure_inference_llama33.region_name)
        + ".models.ai.azure.com"
    ),
    temperature=0,  # type: ignore
)

AZURE_PHI4 = AzureAICompletionsModelPhi4(
    credential=cfg.azure_inference_phi4.api_key.get_secret_value(),  # type: ignore
    model_name=str(cfg.azure_inference_phi4.model_name),  # type: ignore
    endpoint=(  # type: ignore
        "https://"
        + str(cfg.azure_inference_phi4.default_deployment)
        + "."
        + str(cfg.azure_inference_phi4.region_name)
        + ".models.ai.azure.com"
    ),
    temperature=0,  # type: ignore
)

AZURE_R1 = AzureAICompletionsModelR1(
    credential=cfg.azure_inference_r1.api_key.get_secret_value(),  # type: ignore
    model_name=str(cfg.azure_inference_r1.model_name),  # type: ignore
    endpoint=(  # type: ignore
        "https://"
        + str(cfg.azure_inference_r1.default_deployment)
        + "."
        + str(cfg.azure_inference_r1.region_name)
        + ".models.ai.azure.com"
    ),
    temperature=0,  # type: ignore
)


class AzureAICompletionsModelMistral(AzureAICompletionsModel):
    def __init__(self, credential, model_name, endpoint, temperature=0):
        super().__init__(
            credential=credential,
            model_name=model_name,
            endpoint=endpoint,
            temperature=temperature,
        )

    @property
    def metadata(self):
        return LLMMetadata(
            context_window=120000,
            num_output=2056,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name="mistral-large-2411",
        )


MISTRAL_LARGE = AzureAICompletionsModelMistral(
    credential=cfg.azure_inference_mistral.api_key.get_secret_value(),  # type: ignore
    model_name=str(cfg.azure_inference_mistral.model_name),  # type: ignore
    endpoint=(  # type: ignore
        "https://"
        + str(cfg.azure_inference_mistral.default_deployment)
        + "."
        + str(cfg.azure_inference_mistral.region_name)
        + ".models.ai.azure.com"
    ),
    temperature=0,  # type: ignore
)


CEREBRAS_LLAMA_31_8B = Cerebras(
    model="llama3.1-8b",
    api_key=cfg.cerebras.api_key.get_secret_value(),
)

CEREBRAS_LLAMA_33_70B = Cerebras(
    model="llama-3.3-70b",
    api_key=cfg.cerebras.api_key.get_secret_value(),
    # Cerebras API doesn't support 'any' so we can't guarantee tool call
    # API seems to want 'required', which may be a way to fix this
    # https://inference-docs.cerebras.ai/api-reference/chat-completions#param-tool-choice
    is_function_calling_model=False,
    context_window=8000,
)


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
LLMs = {
    # "o1": AZURE_o1,
    "o3-mini": AZURE_o3_MINI,
    "gpt-4o-mini": AZURE_GPT4O_MINI,
    "gpt-4o-std": AZURE_GPT4O_STD,
    "gpt-35-turbo": AZURE_GPT35_TURBO_1106,
    "anthropic-sonnet-35": ANTHROPIC_CLAUDE_SONNET_35,
    "anthropic-haiku-35": ANTHROPIC_CLAUDE_HAIKU_35,
    "gemini-pro": GCP_GEMINI_PRO,
    "gemini-flash": GCP_GEMINI_FLASH,
    "gemini-flash2": GCP_GEMINI_FLASH2,
    # "gemini-pro-exp": GCP_GEMINI_PRO_EXP,
    # "gemini-flash-exp": GCP_GEMINI_FLASH_EXP,
    # "gemini-flash-think-exp": GCP_GEMINI_FLASH_THINK_EXP,
    "llama-33-70B": AZURE_LLAMA33_70B,
    "mistral-large": MISTRAL_LARGE,
    "cerebras-llama-31-8B": CEREBRAS_LLAMA_31_8B,
    "cerebras-llama-33-70B": CEREBRAS_LLAMA_33_70B,
    "phi-4": AZURE_PHI4,
    # "azure-r1": AZURE_R1,
    "together-r1": TogetherDeepseekR1,
    "together-V3": TogetherDeepseekV3,
    # "datarobot-deployed": DataRobotDeployedLLM
    **LOCAL_MODELS,
}

LLMs.update(load_configured_llms(cfg))


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
    raise ValueError("Invalid LLM specified")


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
    if name in [
        "o1",
        "o3-mini",
        "gpt-4o-mini",
        "gpt-4o-std",
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
        "cerebras-llama-31-8B",
        "cerebras-llama-33-70B",
        "phi-4",
        "azure-r1",
        "together-r1",
        "together-V3",
        "datarobot-deployed",
    ]:
        return tiktoken.encoding_for_model("gpt-4o-mini").encode
    if name == "gpt-35-turbo":
        return tiktoken.encoding_for_model("gpt-35-turbo").encode
    if name in LOCAL_MODELS:
        return tiktoken.encoding_for_model("gpt-4o-mini").encode
    raise ValueError("Invalid tokenizer specified")


if __name__ == "__main__":
    print(get_llm_name(MISTRAL_LARGE))
