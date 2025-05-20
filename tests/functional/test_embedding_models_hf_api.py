import os
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import (
    SentenceSplitter,
)

from syftr.configuration import cfg
from syftr.hf_endpoint_models import get_hf_endpoint_embed_model


def test_hf_endpoint_vdb_creation(tiny_dataset):
    # Dummy line to load credentials
    _ = Path(cfg.paths.test_studies_dir / "test-no-rag-seeding.yaml")

    chunk_size = 1024
    chunk_overlap = 128
    model_name = "thomaskim1130/stella_en_400M_v5-FinanceRAG-v2"

    hf_embedding_model = get_hf_endpoint_embed_model(model_name)

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    transforms = [splitter]

    pipeline = IngestionPipeline(transformations=transforms)
    docs = list(tiny_dataset.iter_grounding_data())

    # Check if docs is not empty
    assert docs, "No documents loaded from dataset"

    nodes = pipeline.run(
        documents=docs,
        show_progress=True,
        max_workers=os.cpu_count(),
    )

    # Take only first 128 nodes
    nodes = nodes[:128]

    # Check if nodes is not empty
    assert nodes, "No nodes created from documents"

    # remove nodes with empty text
    nodes = [node for node in nodes if node.text]

    # Check if nodes have text
    assert all(node.text for node in nodes), "Some nodes have empty text"

    # create VectorStoreIndex using this embedding model
    vdb = VectorStoreIndex(
        nodes=nodes,
        embed_model=hf_embedding_model,
        insert_batch_size=2048,
        show_progress=True,
    )

    # Check if vdb is not None
    assert vdb is not None, "VectorStoreIndex creation failed"

    # Check the type of vdb
    assert isinstance(vdb, VectorStoreIndex), (
        "vdb is not an instance of VectorStoreIndex"
    )
