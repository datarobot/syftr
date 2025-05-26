import os
import typing as T

from filelock import FileLock, Timeout
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from optimum.onnxruntime import ORTModelForFeatureExtraction
from slugify import slugify
from transformers import AutoTokenizer

from syftr.configuration import cfg
from syftr.embeddings.timeouts import EmbeddingTimeoutMixin
from syftr.hf_endpoint_models import get_hf_endpoint_embed_model
from syftr.logger import logger
from syftr.studies import EmbeddingDeviceType, TimeoutConfig


def get_hf_token():
    hf_token = str(cfg.hf_embeddings.api_key.get_secret_value())
    if not hf_token or hf_token == "NOT SET":
        return {}
    return {"HF_TOKEN": hf_token}


def load_hf_token_into_env():
    hf_token = get_hf_token()
    # only update the environment if set
    if hf_token:
        os.environ.update(hf_token)


class HuggingFaceEmbeddingWithTimeout(EmbeddingTimeoutMixin, HuggingFaceEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class OpenAILikeEmbeddingWithTimeout(EmbeddingTimeoutMixin, OpenAILikeEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


LOCAL_EMBEDDING_MODELS = (
    {
        model.model_name: OpenAILikeEmbeddingWithTimeout(
            model_name=model.model_name,
            api_base=str(model.api_base),
            api_key=model.api_key.get_secret_value()
            if model.api_key is not None
            else cfg.local_models.default_api_key.get_secret_value(),
            timeout=model.timeout,
            dimensions=model.dimensions,
            additional_kwargs=model.additional_kwargs,
        )
        for model in cfg.local_models.embedding
    }
    if cfg.local_models.embedding
    else {}
)


class OptimumEmbeddingWithTimeout(EmbeddingTimeoutMixin, OptimumEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def get_hf_embedding_model(
    name,
    timeout_config: TimeoutConfig = TimeoutConfig(),
    total_chunks: int = 0,
    device: EmbeddingDeviceType = None,
) -> BaseEmbedding | None:
    """Use generic LLamaIndex HuggingFaceEmbedding model.

    name: name of the embedding model
    device: Device type to pass in. LlamaIndex uses Torch to autodiscover if None
    """
    if not name:
        return None
    if device == "onnx-cpu":
        device = None

    logger.debug(f"Getting HuggingFace model '{name}'")
    return HuggingFaceEmbeddingWithTimeout(
        model_name=name,
        device=device,
        trust_remote_code=True,
        cache_folder=cfg.paths.huggingface_cache.as_posix(),
        timeout_config=timeout_config,
        total_chunks=total_chunks,
        use_auth_token=get_hf_token(),
    )


def get_onnx_embedding_model(
    name: str, timeout_config: TimeoutConfig = TimeoutConfig(), total_chunks: int = 0
) -> BaseEmbedding | None:
    if not name:
        return None
    logger.debug("Getting ONNX version of '%s'", name)
    model_folder = cfg.paths.onnx_dir / slugify(name)
    model_folder.mkdir(parents=True, exist_ok=True)
    try:
        if (model_folder / "model.onnx").exists():
            logger.debug("ONNX model for %s already exists. Skipping creation.", name)
        else:
            logger.info("Creating ONNX version of '%s'", name)
            write_lock = model_folder / "write.lock"
            with FileLock(write_lock, timeout=timeout_config.onnx_timeout):
                model = ORTModelForFeatureExtraction.from_pretrained(
                    name,
                    export=True,
                    trust_remote_code=True,
                    use_auth_token=get_hf_token(),
                )
                tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
                model.save_pretrained(model_folder)
                tokenizer.save_pretrained(model_folder)
            logger.info("Done creating ONNX version of '%s'", name)

        return OptimumEmbeddingWithTimeout(
            folder_name=model_folder.as_posix(),
            timeout_config=timeout_config,
            total_chunks=total_chunks,
        )
    except (ValueError, Timeout):
        logger.warning("Cannot get ONNX version of '%s'", name)
        write_lock.unlink(missing_ok=True)
        raise


def get_embedding_model(
    name: str,
    timeout_config: TimeoutConfig = TimeoutConfig(),
    total_chunks: int = 0,
    device: EmbeddingDeviceType = "cpu",
    use_hf_endpoint_models: bool = False,
) -> T.Tuple[BaseEmbedding | None, bool | None]:
    """
    Returns an embedding model based on the name and device type.
    4-stage fallback:
    1. check if the model can be served by a local vLLM Endpoint
    2. check if the model can be served by a dedicated HF endpoint
    3. if not try to get an onnx model with cpu backend
    4. if that fails, get a torch model
    """
    if not name:
        logger.warning("No embedding model name provided.")
        return None, None

    if name in LOCAL_EMBEDDING_MODELS:
        return LOCAL_EMBEDDING_MODELS[name], False

    if use_hf_endpoint_models and cfg.hf_embeddings.models_config_map.get(name, False):
        logger.info("Getting HF endpoint model: %s", name)
        return get_hf_endpoint_embed_model(name), False
    elif device == "onnx-cpu":
        try:
            logger.info("Getting ONNX model: %s", name)
            return (
                get_onnx_embedding_model(
                    name, timeout_config=timeout_config, total_chunks=total_chunks
                ),
                True,
            )
        except ValueError:
            logger.info("Getting local HF model for CPU: %s", name)
            return (
                get_hf_embedding_model(
                    name,
                    timeout_config=timeout_config,
                    total_chunks=total_chunks,
                    device="cpu",
                ),
                False,
            )
    else:
        logger.info("Getting local HF model for device '%s': %s", device, name)
        return (
            get_hf_embedding_model(
                name,
                timeout_config=timeout_config,
                total_chunks=total_chunks,
                device=device,
            ),
            False,
        )
