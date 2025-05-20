import logging
import typing as T

import pytest

from syftr.embeddings.timeouts import EmbeddingPreemptiveTimeoutError
from syftr.huggingface_helper import get_embedding_model
from syftr.logger import logger
from syftr.storage import FinanceBenchHF, PartitionMap
from syftr.studies import StudyConfig


def test_embedding_models():
    study_config = StudyConfig(
        name="test-embedding-models",
        dataset=FinanceBenchHF(
            partition_map=PartitionMap(test="pepsi"),
        ),
    )
    search_space = study_config.search_space

    for name in search_space.rag_retriever.embedding_models:
        logger.info(f"Testing embedding model: {name}")
        model, is_onnx = get_embedding_model(
            name,
            study_config.timeouts,
            total_chunks=0,
            use_hf_endpoint_models=False,
        )
        assert is_onnx in [True, False], "Bad is_onnx response"
        out: T.List[float] = model.get_query_embedding("Do you function?")
        assert out, f"Cannot generate embeddings with '{name}'"


def test_embedding_model_timeout(build_index_that_will_time_out):
    with pytest.raises(EmbeddingPreemptiveTimeoutError):
        build_index_that_will_time_out()


def test_embedding_model_no_timeout(build_index_that_will_not_time_out):
    build_index_that_will_not_time_out()


def test_embedding_model_skip_timeout(caplog, build_index_that_will_skip_time_out):
    with caplog.at_level(logging.WARNING):
        index, _ = build_index_that_will_skip_time_out()
        assert "Not raising embedding timeout because it is deactivated" in caplog.text
        assert index._embed_model._warning_logged
