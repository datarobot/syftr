import socket
from multiprocessing import Pool

from syftr.huggingface_helper import get_embedding_model
from syftr.storage import HotPotQAHF, PartitionMap
from syftr.studies import StudyConfig

hostname = socket.gethostname()
study_config = StudyConfig(
    name="create-onnx",
    dataset=HotPotQAHF(
        partition_map=PartitionMap(test="sample"),
    ),
)
search_space = study_config.search_space


def process_model(name):
    model, is_onnx = get_embedding_model(name, study_config.timeouts, total_chunks=0)
    assert model, f"Cannot get a model for '{name}' on {hostname}"
    assert is_onnx in [True, False], "Bad is_onnx response"
    model.get_query_embedding("Welcome!")
    if is_onnx:
        print(f"Create ONNX version of '{name}' on {hostname}")
    else:
        print(f"ONNX version of '{name}' not available on {hostname}")


if __name__ == "__main__":
    with Pool() as pool:
        pool.map(process_model, search_space.rag_retriever.embedding_models)
