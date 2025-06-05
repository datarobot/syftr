"""Main settings for Syftr code.

# Usage

## Standard usage
    from syftr.configuration import cfg
    myval = cfg.optuna.study_name

## Customize how settings are loaded etc.
    from syftr.configuration import Settings
    custom_cfg = Settings(...)

# Development

- Keep the Settings class clean and put settings in namespaced models.
- You must add your model to the Settings class for it to be registered.
- Use Pydantic's built-in types where relevant for improved parsing and validation.
    https://docs.pydantic.dev/latest/api/types/
- Set sane defaults for local development wherever possible.

See https://docs.pydantic.dev/latest/concepts/pydantic_settings/ for documentation.
Also see https://github.com/makukha/pydantic-file-secrets for the secrets loader

# Configuration directory

By default, configuration files and secrets are located in the directory above this one.
To change where configuration files are loaded, set the environment variable (case-insensitive)
    export SYFTR_CONFIG_DIR=/path/to/configs
or use the command-line flag
    $ python syftr/mymodule.py --config_dir /path/to/configs

# Setting configuration values

How to override default configuration values (in resolution priority order):
    1. Pass to init:
        cfg = Settings(azure_oai={'api_version': 'foobar'})

    2. Set environment variable with prefix and __ nested separator (case-insensitive)
        export SYFTR_AZURE_OAI__API_VERSION=foobar
        export syftr_azure_oai__api_version=foobar

    3. Set environment variables in .env file using naming scheme shown above.

    4. Store secret variables in the runtime-secrets directory. Filenames are
       the variable name without the SYFTR_ prefix

        $ cat runtime-secrets/azure_oai__api_key
        asdf1324asdf1234asdf1234
        $ cat runtime-secrets/azure_oai/api_key  # Alternative file structure
        asdf1324asdf1234asdf1234

       Secrets will also be loaded from other sources if present.

    5. (Preferred method) Set the variable in the YAML file config.yaml
        azure_oai:
          api_version: foobar
        logging:
          name: mylogs
"""

import logging
import os
import socket
import typing as T
from pathlib import Path

from google.cloud.aiplatform_v1beta1.types import content
from llama_index.core.base.llms.types import LLMMetadata
from optuna.storages import RDBStorage
from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    SecretStr,
    field_serializer,
    field_validator,
)
from pydantic_file_secrets import FileSecretsSettingsSource
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SecretsSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)
from sqlalchemy import Engine, create_engine
from typing_extensions import Annotated

REPO_ROOT = Path(os.getenv("REPO_ROOT", Path(__file__).parent.parent))
logging.info(f"Repository root is: {REPO_ROOT}")
HOSTNAME = socket.gethostname()
EVAL__RAISE_ON_EXCEPTION = False
EVAL__ENABLE_RETRY = True
S3_TIMEOUT = 3600
NON_OPENAI_CONTEXT_WINDOW_FACTOR = 0.85
NDIGITS = 4
UNSUPPORTED_PARAMS = ["splitter_chunk_size"]
SYFTR_CONFIG_FILE_ENV_NAME = "SYFTR_CONFIG_FILE"


class APIKeySerializationMixin:
    @field_serializer("api_key", "credentials", "default_api_key", check_fields=False)
    def serialize_api_key(self, api_key: SecretStr | None, _info) -> str:
        if api_key is None:
            return "NOT SET"
        return api_key.get_secret_value()


"""
Namespaced configuration classes.

These must be attached to the Settings model below to be
loaded as configuration.
"""


class Paths(BaseModel):
    root_dir: Path = REPO_ROOT
    syftr_dir: Path = REPO_ROOT / "syftr"
    data_dir: Annotated[Path, Field(validate_default=True)] = syftr_dir / "data"
    templates_dir: Path = data_dir / "templates"
    results_dir: Annotated[Path, Field(validate_default=True)] = REPO_ROOT / "results"
    studies_dir: Annotated[Path, Field(validate_default=True)] = REPO_ROOT / "studies"
    test_studies_dir: Path = REPO_ROOT / "tests/studies"
    tmp_dir: Path = Path("/tmp/syftr")
    huggingface_cache: Annotated[Path, Field(validate_default=True)] = (
        tmp_dir / "huggingface"
    )
    index_cache: Annotated[Path, Field(validate_default=True)] = tmp_dir / "indexcache"
    onnx_dir: Annotated[Path, Field(validate_default=True)] = tmp_dir / "onnx"
    sota_dir: Annotated[Path, Field(validate_default=True)] = data_dir / "sota"
    lock_dir: Annotated[Path, Field(validate_default=True)] = tmp_dir / "syftr-locks"
    nltk_dir: Annotated[Path, Field(validate_default=True)] = tmp_dir / "nltk-data"
    sqlite_dir: Annotated[Path, Field(validate_default=True)] = Path.home() / ".syftr"

    @property
    def templates_without_context(self) -> Path:
        return self.templates_dir / "templates_without_context.json"

    @property
    def templates_with_context(self) -> Path:
        return self.templates_dir / "templates_with_context.json"

    @property
    def agentic_templates(self) -> Path:
        return self.templates_dir / "agentic_templates.json"

    @field_validator(
        "data_dir",
        "results_dir",
        "studies_dir",
        "huggingface_cache",
        "onnx_dir",
        "index_cache",
        "lock_dir",
        "nltk_dir",
        "sqlite_dir",
        mode="after",
    )
    @classmethod
    def path_exists(cls, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        try:
            path.chmod(0o775)
        except PermissionError:
            logging.debug(f"PermissionError: Unable to change permissions for {path}.")
        return path


class Logging(BaseModel):
    name: str = "syftr"
    filename: str = "syftr.log"
    level: int = logging.INFO
    use_colors: bool = False
    color_format: str = (
        "%(log_color)s[%(levelname)1.1s %(asctime)s]%(reset)s %(message)s"
    )
    normal_format: str = "[%(levelname)1.1s %(asctime)s] %(message)s"


class Plotting(BaseModel):
    target_accuracy_name: str = "Accuracy"
    target_accuracy_unit: str = "Accuracy"
    target_latency_name: str = "Latency"
    target_latency_unit: str = "Seconds per Generation"
    target_cost_name: str = "Cost"
    target_cost_unit: str = "¢ per 100 calls"
    do_plot: bool = True
    datarobot_green: T.Tuple[float, float, float, float] = (
        0.60784316,
        0.96862745,
        0.6666667,
        1.0,
    )
    datarobot_purple: T.Tuple[float, float, float, float] = (
        0.654902,
        0.6901961,
        0.9843137,
        1.0,
    )
    datarobot_yellow: T.Tuple[float, float, float, float] = (
        1.0,
        1.0,
        0.3294,
        1.0,
    )


class Storage(BaseModel):
    remote_protocol: str = "s3"
    data_bucket: str = "NOT SET"
    s3_cache_enabled: bool = False
    cache_bucket: str = "NOT SET"
    local_cache_dir: Path = Path("benchmarking/data/cache")
    local_cache_max_size_gb: int | float = 10


class HFEmbedding(BaseModel):
    embedding_model_name: str
    max_length: int = 512
    hf_embedding_batch_size: int = 32
    query_prefix: str = ""
    text_prefix: str = ""
    api_url: HttpUrl


class HFEmbeddings(BaseModel, APIKeySerializationMixin):
    api_key: SecretStr = SecretStr("NOT SET")
    models_config_map: T.Dict[str, HFEmbedding] = {
        "BAAI/bge-small-en-v1.5": HFEmbedding(
            embedding_model_name="BAAI/bge-small-en-v1.5",
            query_prefix="Represent this sentence for searching relevant passages: ",
            api_url=HttpUrl(
                "https://arvpb242d0oeut3e.us-east-1.aws.endpoints.huggingface.cloud"
            ),
        ),
        "BAAI/bge-large-en-v1.5": HFEmbedding(
            embedding_model_name="BAAI/bge-large-en-v1.5",
            query_prefix="Represent this sentence for searching relevant passages: ",
            api_url=HttpUrl(
                "https://fmsmp8yiggtswqzx.us-east-1.aws.endpoints.huggingface.cloud"
            ),
        ),
        "thenlper/gte-large": HFEmbedding(
            embedding_model_name="thenlper/gte-large",
            api_url=HttpUrl(
                "https://aml91apln84nfd2a.us-east-1.aws.endpoints.huggingface.cloud"
            ),
        ),
        "mixedbread-ai/mxbai-embed-large-v1": HFEmbedding(
            embedding_model_name="mixedbread-ai/mxbai-embed-large-v1",
            api_url=HttpUrl(
                "https://op6gmi8fb0qvie9q.us-east-1.aws.endpoints.huggingface.cloud"
            ),
            query_prefix="Represent this sentence for searching relevant passages: ",
        ),
        "WhereIsAI/UAE-Large-V1": HFEmbedding(
            embedding_model_name="WhereIsAI/UAE-Large-V1",
            api_url=HttpUrl(
                "https://ddrmi1khg5f96d27.us-east-1.aws.endpoints.huggingface.cloud"
            ),
            query_prefix="Represent this sentence for searching relevant passages: ",
        ),
        "avsolatorio/GIST-large-Embedding-v0": HFEmbedding(
            embedding_model_name="avsolatorio/GIST-large-Embedding-v0",
            api_url=HttpUrl(
                "https://gaaymsna8pbdbgyd.us-east-1.aws.endpoints.huggingface.cloud"
            ),
        ),
        "w601sxs/b1ade-embed": HFEmbedding(
            embedding_model_name="w601sxs/b1ade-embed",
            api_url=HttpUrl(
                "https://qluvw1j4bm9cpjdg.us-east-1.aws.endpoints.huggingface.cloud"
            ),
        ),
        "Labib11/MUG-B-1.6": HFEmbedding(
            embedding_model_name="Labib11/MUG-B-1.6",
            api_url=HttpUrl(
                "https://f4x3liz6kzef0xu8.us-east-1.aws.endpoints.huggingface.cloud"
            ),
        ),
        "sentence-transformers/all-MiniLM-L12-v2": HFEmbedding(
            embedding_model_name="sentence-transformers/all-MiniLM-L12-v2",
            api_url=HttpUrl(
                "https://z770ejw5u6shi15r.us-east-1.aws.endpoints.huggingface.cloud"
            ),
            max_length=128,
        ),
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": HFEmbedding(
            embedding_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            api_url=HttpUrl(
                "https://e76i9w21sfwymoga.us-east-1.aws.endpoints.huggingface.cloud"
            ),
            max_length=128,
        ),
        "BAAI/bge-base-en-v1.5": HFEmbedding(
            embedding_model_name="BAAI/bge-base-en-v1.5",
            query_prefix="Represent this sentence for searching relevant passages: ",
            api_url=HttpUrl(
                "https://o6fu3avitwffxgvw.us-east-1.aws.endpoints.huggingface.cloud"
            ),
        ),
        "thomaskim1130/stella_en_400M_v5-FinanceRAG-v2": HFEmbedding(
            embedding_model_name="thomaskim1130/stella_en_400M_v5-FinanceRAG-v2",
            query_prefix="Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery:",
            api_url=HttpUrl(
                "https://sb421185vljmt9p5.us-east-1.aws.endpoints.huggingface.cloud"
            ),
            max_length=512,
        ),
        "baconnier/Finance2_embedding_small_en-V1.5": HFEmbedding(
            embedding_model_name="baconnier/Finance2_embedding_small_en-V1.5",
            query_prefix="Represent this sentence for searching relevant passages: ",
            api_url=HttpUrl(
                "https://lnz1dqk5e2qn0sp3.us-east-1.aws.endpoints.huggingface.cloud"
            ),
            max_length=512,
        ),
        "FinLang/finance-embeddings-investopedia": HFEmbedding(
            embedding_model_name="FinLang/finance-embeddings-investopedia",
            query_prefix="Represent this sentence for searching relevant passages: ",
            api_url=HttpUrl(
                "https://yr4h7d6dyejrhtwg.us-east-1.aws.endpoints.huggingface.cloud"
            ),
        ),
    }


class AzureOAI(BaseModel, APIKeySerializationMixin):
    # Use cfg.azure_oai.api_key.get_secret_value() to get value
    api_key: SecretStr = SecretStr("NOT SET")
    default_deployment: str = "gpt-4o-mini"
    api_url: HttpUrl = HttpUrl("http://NOT.SET")

    api_version: str = "2024-07-18"
    api_type: str = "azure"


class GCPVertex(BaseModel, APIKeySerializationMixin):
    # Use cfg.gcp_vertex.credentials.get_secret_value() to get value
    # Note credentials are a string, typically will need to do a json.loads
    project_id: str = "NOT SET"
    region: str = "NOT SET"
    credentials: SecretStr = SecretStr("NOT SET")


class LLMCostTokens(BaseModel):
    type: T.Literal["tokens"] = "tokens"
    input: float = Field(description="Cost per million input tokens")
    output: float = Field(description="Cost per million output tokens")


class LLMCostCharacters(BaseModel):
    type: T.Literal["characters"] = "characters"
    input: float = Field(description="Cost per million input characters")
    output: float = Field(description="Cost per million output characters")


class LLMCostHourly(BaseModel):
    type: T.Literal["hourly"] = "hourly"
    rate: float = Field(
        description="Average inference cost per hour "
        "(eg. machine hourly rate divided by average number of concurrent requests)"
    )


class LLMConfig(BaseModel):
    cost: Annotated[
        T.Union[LLMCostTokens, LLMCostCharacters, LLMCostHourly],
        Field(discriminator="type"),
    ] = Field(description="LLM inference costs by token, character, or hourly rate.")

    metadata: LLMMetadata = Field(
        description="LLM Metadata such as context window and capabilities"
    )

    temperature: float = 0.0

    model_config = ConfigDict(extra="forbid")


# Specific LLM Instance Configurations
class AzureOpenAILLM(LLMConfig):
    provider: T.Literal["azure_openai"] = "azure_openai"
    deployment_name: T.Optional[str] = None
    api_version: T.Optional[str] = None
    max_retries: int = 0
    additional_kwargs: T.Optional[T.Dict[str, T.Any]] = None


GCP_SAFETY_SETTINGS = {
    content.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: content.SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    content.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: content.SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    content.HarmCategory.HARM_CATEGORY_HARASSMENT: content.SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    content.HarmCategory.HARM_CATEGORY_HATE_SPEECH: content.SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}


class VertexAILLM(LLMConfig):
    provider: T.Literal["vertex_ai"] = "vertex_ai"
    model: T.Optional[str] = None
    safety_settings: dict = GCP_SAFETY_SETTINGS
    max_retries: int = 0
    additional_kwargs: T.Optional[T.Dict[str, T.Any]] = None


class AnthropicVertexLLM(LLMConfig):
    provider: T.Literal["anthropic_vertex"] = Field(
        "anthropic_vertex", description="Provider identifier."
    )
    model: str = Field(
        description="Name of the Anthropic model on Vertex AI (e.g., claude-3-5-sonnet-v2@20241022)."
    )
    project_id: T.Optional[str] = Field(
        default=None,
        description="GCP Project ID. If None, uses global cfg.gcp_vertex.project_id.",
    )
    region: T.Optional[str] = Field(
        default=None,
        description="GCP Region. If None, uses global cfg.gcp_vertex.region.",
    )
    max_retries: int = Field(
        default=0, description="Maximum number of retries for API calls."
    )
    additional_kwargs: T.Dict[str, T.Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments for the Anthropic model.",
    )


class AzureAICompletionsLLM(LLMConfig):
    provider: T.Literal["azure_ai"] = Field(
        "azure_ai", description="Provider identifier."
    )
    model_name: str = Field(
        description="Model name as recognized by Azure AI Completions (e.g., Llama-3.3-70B-Instruct)."
    )
    endpoint: HttpUrl = Field(description="API URL for this model")
    api_key: SecretStr = Field(description="API key for this model")
    max_retries: int = Field(
        default=0, description="Maximum number of retries for API calls."
    )
    additional_kwargs: T.Dict[str, T.Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments for the Azure AI Completions model.",
    )


class CerebrasLLM(LLMConfig):
    provider: T.Literal["cerebras"] = Field(
        "cerebras", description="Provider identifier."
    )
    model: str = Field(description="Name of the Cerebras model (e.g., llama3.1-8b).")
    # API key and URL are typically derived from cfg.cerebras.
    max_retries: int = Field(
        default=0, description="Maximum number of retries for API calls."
    )
    additional_kwargs: T.Dict[str, T.Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments for the Cerebras model.",
    )


class OpenAILikeLLM(LLMConfig):
    provider: T.Literal["openai_like"] = Field(
        "openai_like", description="Provider identifier for OpenAI-compatible APIs."
    )
    model: str = Field(
        description="Name of the OpenAI-like model (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo)."
    )
    api_base: HttpUrl = Field(description="API base URL for the OpenAI-like model.")
    api_key: SecretStr = Field(description="API key for this endpoint")
    api_version: T.Optional[str] = Field(
        default=None, description="API version to use for this endpoint"
    )
    timeout: int = Field(
        default=120, description="Timeout in seconds for API requests."
    )
    max_retries: int = Field(
        default=0, description="Maximum number of retries for API calls."
    )
    additional_kwargs: T.Dict[str, T.Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments for the OpenAI-like model.",
    )


# Update LLMConfigUnion by adding the new classes
LLMConfigUnion = Annotated[
    T.Union[
        AzureOpenAILLM,
        VertexAILLM,
        AnthropicVertexLLM,
        AzureAICompletionsLLM,
        CerebrasLLM,
        OpenAILikeLLM,
    ],
    Field(discriminator="provider"),
]


class AzureInferenceLlama33(BaseModel, APIKeySerializationMixin):
    # Use cfg.azure_inference_llama.api_key.get_secret_value() to get value
    api_key: SecretStr = SecretStr("NOT SET")
    region_name: str = "NOT SET"
    default_deployment: str = "NOT SET"
    model_name: str = "NOT SET"


class AzureInferenceMistral(BaseModel, APIKeySerializationMixin):
    # Use cfg.azure_inference_mistral.api_key.get_secret_value() to get value
    api_key: SecretStr = SecretStr("NOT SET")
    region_name: str = "NOT SET"
    default_deployment: str = "NOT SET"
    model_name: str = "NOT SET"


class AzureInferencePhi4(BaseModel, APIKeySerializationMixin):
    # Use cfg.azure_inference_phi4.api_key.get_secret_value() to get value
    api_key: SecretStr = SecretStr("NOT SET")
    region_name: str = "NOT SET"
    default_deployment: str = "NOT SET"
    model_name: str = "NOT SET"


class AzureInferenceR1(BaseModel, APIKeySerializationMixin):
    # Use cfg.azure_inference_phi4.api_key.get_secret_value() to get value
    api_key: SecretStr = SecretStr("NOT SET")
    region_name: str = "NOT SET"
    default_deployment: str = "NOT SET"
    model_name: str = "NOT SET"


class Cerebras(BaseModel, APIKeySerializationMixin):
    # Use cfg.azure_oai.api_key.get_secret_value() to get value
    api_key: SecretStr = SecretStr("NOT SET")
    api_url: HttpUrl = HttpUrl("https://api.cerebras.ai/v1")


class TogetherAI(BaseModel, APIKeySerializationMixin):
    # Use cfg.azure_oai.api_key.get_secret_value() to get value
    api_key: SecretStr = SecretStr("NOT SET")


class DataRobot(BaseModel, APIKeySerializationMixin):
    api_key: SecretStr = SecretStr("NOT SET")
    endpoint: HttpUrl = HttpUrl("http://NOT.SET")


class LocalOpenAILikeModel(BaseModel, APIKeySerializationMixin):
    model_name: str
    api_base: str
    api_key: SecretStr | None = None
    max_tokens: int
    context_window: int
    is_chat_model: bool = True
    is_function_calling_model: bool = False
    timeout: int = 300
    max_retries: int = 0
    additional_kwargs: dict[str, T.Any] = Field(default_factory=dict)


class LocalOpenAILikeEmbeddingModel(BaseModel, APIKeySerializationMixin):
    model_name: str
    api_base: str
    api_key: SecretStr | None = None
    dimensions: int | None = None
    timeout: int = 300
    additional_kwargs: dict[str, T.Any] | None = None


class LocalOpenAILikeModels(BaseModel, APIKeySerializationMixin):
    generative: T.Optional[T.List[LocalOpenAILikeModel]] = Field(default=None)
    embedding: T.Optional[T.List[LocalOpenAILikeEmbeddingModel]] = Field(default=None)
    default_api_key: SecretStr = SecretStr("NOT SET")


class Optuna(BaseModel):
    study_name: str = "syftr"
    # Don't confirm on delete
    noconfirm: bool = False
    show_progress: bool = True


class Database(BaseModel):
    dsn: AnyUrl = AnyUrl(
        "sqlite:////{}/syftr.db".format(Paths().sqlite_dir)
    )  # Provide default SQLite path when not specified.
    postgres_engine_kwargs: T.Dict[str, T.Any] = {
        # https://docs.sqlalchemy.org/en/20/core/pooling.html#setting-pool-recycle
        "pool_recycle": 300,
        "pool_pre_ping": True,  # Important for PostgreSQL
        "pool_size": 10,
        "max_overflow": 50,
        # https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-PARAMKEYWORDS
        "connect_args": {
            "application_name": "syftr",
            "connect_timeout": 60,
            "keepalives": 1,  # Enable TCP keepalives
            "keepalives_idle": 30,  # Send keepalives after 30s of idle time
            "keepalives_interval": 10,  # Resend after 10s if no response
            "keepalives_count": 5,  # Give up after 5 failed keepalives
        },
    }

    def get_engine(self) -> Engine:
        kwargs = (
            {} if "sqlite" in self.dsn.unicode_string() else self.postgres_engine_kwargs
        )
        return create_engine(self.dsn.unicode_string(), **kwargs)

    def get_optuna_storage(self) -> RDBStorage:
        kwargs = (
            {} if "sqlite" in self.dsn.unicode_string() else self.postgres_engine_kwargs
        )
        return RDBStorage(self.dsn.unicode_string(), engine_kwargs=kwargs)

    @field_serializer("dsn")
    def serialize_dsn(self, dsn: AnyUrl):
        return dsn.unicode_string()


class Ray(BaseModel):
    local_endpoint: T.Optional[str] = "http://127.0.0.1:8265"
    remote_endpoint: str = "ray://localhost:10001"
    local: bool = True  # Set to False to use the remote cluster
    remote_root: str = "NOT SET"
    fail_fast: bool = False
    num_gpus: int = 0


class AWS(BaseModel):
    access_key_ssm_path: str = "NOT SET"
    secret_key_ssm_path: str = "NOT SET"
    assume_role_arn: str = "NOT SET"
    region: str = "NOT SET"


class Instrumentation(BaseModel):
    tracing_enabled: bool = Field(
        default=False,
        description=(
            "Enable OpenTelementry tracing for debug. "
            "Requires running otel traces endpoint. "
            "Start up a local instance with `phoenix serve`, "
            "or install and configure otel-collector."
        ),
    )
    otel_endpoint: str = "http://127.0.0.1:6006/v1/traces"


class LlamaIndexGeneral(BaseModel):
    default_tool_choice: str = "auto"


"""
Build the main Settings class
"""


class Settings(BaseSettings):
    """Syftr Settings class. See module docstring for usage details."""

    aws: AWS = AWS()
    hf_embeddings: HFEmbeddings = HFEmbeddings()
    azure_inference_llama33: AzureInferenceLlama33 = AzureInferenceLlama33()
    azure_inference_mistral: AzureInferenceMistral = AzureInferenceMistral()
    azure_inference_phi4: AzureInferencePhi4 = AzureInferencePhi4()
    azure_inference_r1: AzureInferenceR1 = AzureInferenceR1()
    azure_oai: AzureOAI = AzureOAI()
    datarobot: DataRobot = DataRobot()
    gcp_vertex: GCPVertex = GCPVertex()
    cerebras: Cerebras = Cerebras()
    togetherai: TogetherAI = TogetherAI()
    local_models: LocalOpenAILikeModels = LocalOpenAILikeModels()
    generative_models: T.Dict[str, LLMConfigUnion] = Field(
        default_factory=dict, description="User-provided LLM definitions"
    )
    logging: Logging = Logging()
    optuna: Optuna = Optuna()
    paths: Paths = Paths()
    plotting: Plotting = Plotting()
    instrumentation: Instrumentation = Instrumentation()
    database: Database = Database()
    storage: Storage = Storage()
    ray: Ray = Ray()
    study_config_file: T.Optional[Path] = None
    llama_index: LlamaIndexGeneral = LlamaIndexGeneral()

    # Meta-configuration (where/how to load settings values)
    model_config = SettingsConfigDict(
        yaml_file=[
            # Prioritize config in cwd, then repo root, then default locations
            "/etc/syftr/config.yaml",
            Path.home() / ".syftr/config.yaml",
            REPO_ROOT / "config.yaml",
            "config.yaml",
            Path(os.environ.get(SYFTR_CONFIG_FILE_ENV_NAME, "")),
        ],
        secrets_dir="runtime-secrets",
        env_file=".env",
        env_prefix="SYFTR_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: T.Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> T.Tuple[PydanticBaseSettingsSource, ...]:
        """Return order dictates setting resolution priority order (first is highest).

        Add new config sources by inserting them in the returned tuple.

        Rationale for ordering:
        - init variables are most likely to be used for unit tests.
        - env vars are most likely to be used to override settings at runtime
        - dotenv settings should be roughly equivalent to actual env vars
        - secrets files should override any defaults, but be secondary to env vars
        - YAML config is the default mechanism for customization
        """
        # Make mypy happy and avoid weird mistakes
        assert isinstance(file_secret_settings, SecretsSettingsSource)
        # Main sources list
        sources: T.Tuple[PydanticBaseSettingsSource, ...] = (
            init_settings,
            env_settings,
            dotenv_settings,
            FileSecretsSettingsSource(  # Third-party extension to improve secrets loading
                file_secret_settings, secrets_dir_missing="ok", secrets_prefix=""
            ),
            YamlConfigSettingsSource(settings_cls),
        )
        return sources


cfg = Settings()
