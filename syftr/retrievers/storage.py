import hashlib
import io
import json
from contextlib import contextmanager
from pickle import UnpicklingError
from typing import Any, Optional

import cloudpickle
import diskcache
import torch
from llama_index.core.indices.base import BaseIndex
from lz4.frame import compress, decompress

from syftr.amazon import delete_file_from_s3, get_file_from_s3
from syftr.configuration import cfg
from syftr.logger import logger
from syftr.studies import StudyConfig
from syftr.utils.locks import distributed_lock

original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    kwargs["map_location"] = torch.device("cpu")
    return original_torch_load(*args, **kwargs)


torch.load = patched_torch_load


@contextmanager
def local_cache():
    with diskcache.Cache(
        cfg.paths.index_cache, size_limit=cfg.storage.local_cache_max_size_gb * 1024**3
    ) as cache:
        yield cache


def _get_dense_index_params_hash(study_config, params, cache_version: int = 3) -> str:
    param_names = [
        "splitter_chunk_overlap_frac",
        "splitter_chunk_exp",
        "splitter_method",
        "rag_embedding_model",
    ]
    index_params = {key: value for key, value in params.items() if key in param_names}
    dataset_param_names = ["xname", "partition_map", "subset", "grounding_data_path"]
    dataset_params = {
        key: value
        for key, value in study_config.dataset.model_dump().items()
        if key in dataset_param_names
    }
    key_params = {
        "index": index_params,
        "dataset": dataset_params,
        "cache_version": cache_version,
    }
    params_hash = hashlib.sha1(
        json.dumps(key_params, sort_keys=True).encode("utf-8")
    ).hexdigest()
    logger.info(
        f"Built cache key {params_hash} for params {json.dumps(key_params, sort_keys=True)}"
    )
    return params_hash


@contextmanager
def index_cache_lock(study_config: StudyConfig, params):
    cache_key = _get_dense_index_params_hash(study_config, params)
    host_only = not cfg.storage.s3_cache_enabled
    with distributed_lock(cache_key, host_only=host_only):
        yield cache_key


def put_cache(cache_key, index, local_only: bool = False) -> None:
    serialized_obj = compress(cloudpickle.dumps(index))
    try:
        index = cloudpickle.loads(decompress(serialized_obj))
        assert isinstance(index, BaseIndex), index
    except (UnpicklingError, AssertionError) as exc:
        logger.warning(
            "Pickling index failed with exception %s. Not storing object.", exc
        )
        return

    with local_cache() as cache:
        logger.info(f"Storing index to {cache.directory}")
        cache.add(cache_key, serialized_obj)
        logger.info(f"Done storing index to {cache.directory}")

    if not local_only and cfg.storage.s3_cache_enabled:
        try:
            import boto3
            from boto3.s3.transfer import TransferConfig
        except ImportError:
            logger.info("Skipping S3 cache - install boto3 to cache objects in S3")
            return
        s3 = boto3.client("s3")
        config = TransferConfig(multipart_threshold=5 * 1024**3)  # 5GB default limit
        fileobj = io.BytesIO(initial_bytes=serialized_obj)
        s3_cache_key = f"index_cache/{cache_key}.pkl"
        logger.info(f"Storing index to S3: {s3_cache_key}")
        s3.upload_fileobj(
            fileobj, cfg.storage.cache_bucket, s3_cache_key, Config=config
        )
        logger.info(f"Done storing index to S3: {s3_cache_key}")


def get_cached(cache_key: str) -> Optional[Any]:
    s3_cache_key = f"index_cache/{cache_key}.pkl"

    try:
        with local_cache() as cache:
            if (data := cache.get(cache_key)) is not None:
                logger.info(f"Loading pre-built index from {cache.directory}")
                index = cloudpickle.loads(decompress(data))
                logger.info(f"Loaded pre-built index from {cache.directory}")
                return index

        if cfg.storage.s3_cache_enabled:
            if (data := get_file_from_s3(s3_cache_key)) is not None:
                logger.info(f"Loading pre-built index from S3: {s3_cache_key}")
                index = cloudpickle.loads(decompress(data))
                assert isinstance(index, BaseIndex), index
                logger.info(f"Loaded pre-built index from S3: {s3_cache_key}")
                put_cache(cache_key, index, local_only=True)
                return index
    except UnpicklingError as exc:
        logger.warning(
            "Unpickling index failed with exception %s. Removing object from caches.",
            exc,
        )
        try:
            del cache[cache_key]
        except Exception:
            pass
        try:
            delete_file_from_s3(s3_cache_key)
        except Exception:
            pass
        return None
    return None
