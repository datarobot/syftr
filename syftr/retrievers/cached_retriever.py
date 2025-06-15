import hashlib
import io
import json
from contextlib import contextmanager
from typing import Any, Dict, Optional

import cloudpickle
import diskcache
from lz4.frame import compress, decompress

from syftr.amazon import get_file_from_s3
from syftr.configuration import cfg
from syftr.logger import logger
from syftr.ray.utils import ray_cache_get, ray_cache_put
from syftr.studies import ParamDict, StudyConfig
from syftr.utils.locks import distributed_lock

# Retrieval cache constants and key builder
RETRIEVAL_CACHE_PREFIX = "retrieval_cache"
RETRIEVER_CACHE_VERSION = 1


@contextmanager
def get_retrieval_cache_key(question: str, retriever_params_dict: Dict[str, Any]):
    """
    Build a cache key from question text and retriever params.
    """
    raw_dict = {**retriever_params_dict, "question": question}
    raw = json.dumps(raw_dict, sort_keys=True).encode("utf-8")
    cache_key = hashlib.sha1(raw).hexdigest()
    host_only = not cfg.storage.s3_cache_enabled
    with distributed_lock(cache_key, host_only=host_only):
        yield cache_key


def get_retriever_fingerprint(
    study_config: StudyConfig, params: ParamDict
) -> Dict[str, Any]:
    param_names = [
        "hyde_enabled",
        "additional_context_enabled",
        "rag_method",
        "rag_query_decomposition_enabled",
        "rag_top_k",
        "rag_embedding_model",
        "rag_query_decomposition_llm_name",
        "rag_query_decomposition_num_queries",
        "rag_fusion_mode",
        "splitter_method",
        "splitter_chunk_exp",
        "splitter_chunk_overlap_frac",
        "hyde_llm_name",
        "additional_context_num_nodes",
    ]
    retriever_params = {
        key: value for key, value in params.items() if key in param_names
    }
    dataset_param_names = ["xname", "partition_map", "subset", "grounding_data_path"]
    dataset_params = {
        key: value
        for key, value in study_config.dataset.model_dump().items()
        if key in dataset_param_names
    }
    key_params = {
        "retriever": retriever_params,
        "dataset": dataset_params,
        "cache_version": RETRIEVER_CACHE_VERSION,
    }
    return key_params


@contextmanager
def local_retrieval_cache():
    with diskcache.Cache(
        cfg.paths.retrieval_cache,
        size_limit=cfg.storage.local_cache_max_size_gb * 1024**3,
    ) as cache:
        yield cache


def put_retrieval_cache(cache_key: str, obj: Any, local_only: bool = False):
    """
    Mirror to both diskcache & S3 under “retrieval_cache/{cache_key}.pkl”.
    """
    serialized = compress(cloudpickle.dumps(obj))
    # Local diskcache
    with local_retrieval_cache() as cache:
        logger.info(f"Storing object to {cache.directory} under key {cache_key}")
        cache.set(cache_key, serialized)

    # Try ray cache
    try:
        logger.info(f"Storing {cache_key} to Ray cache")
        ray_cache_put(cache_key, serialized)
    except Exception as e:
        logger.warning(f"Skipping Ray cache put due to error: {e}")

    # S3 mirror
    if not local_only and cfg.storage.s3_cache_enabled:
        s3_key = f"{RETRIEVAL_CACHE_PREFIX}/{cache_key}.pkl"
        try:
            import boto3
            from boto3.s3.transfer import TransferConfig
        except ImportError:
            logger.info("Skipping S3 cache - install boto3 to cache objects in S3")
            return
        s3 = boto3.client("s3")
        config = TransferConfig(multipart_threshold=5 * 1024**3)
        fileobj = io.BytesIO(serialized)
        logger.info(f"Storing object to S3: {s3_key}")
        s3.upload_fileobj(fileobj, cfg.storage.cache_bucket, s3_key, Config=config)
        logger.info("Done storing object to S3")


def get_retrieval_cache(cache_key: str) -> Optional[Any]:
    """
    First check diskcache, then fall back to S3. Returns the retrieved list
    (NodeWithScore) or None if missing.
    """
    s3_key = f"{RETRIEVAL_CACHE_PREFIX}/{cache_key}.pkl"
    # Try local
    with local_retrieval_cache() as cache:
        if (data := cache.get(cache_key)) is not None:
            logger.info(f"Loading cached object from {cache.directory}")
            return cloudpickle.loads(decompress(data))

    # Try Ray cache
    try:
        data = ray_cache_get(cache_key)
        if data is not None:
            logger.info(f"Loading {cache_key} from Ray cache")
            return cloudpickle.loads(decompress(data))
    except Exception as e:
        logger.warning(f"Skipping Ray cache get due to error: {e}")

    # Try S3
    if cfg.storage.s3_cache_enabled:
        if (data := get_file_from_s3(s3_key)) is not None:
            logger.info(f"Loading cached object from S3: {s3_key}")
            obj = cloudpickle.loads(decompress(data))
            # populate local cache
            put_retrieval_cache(cache_key, obj, local_only=True)
            return obj
    return None
