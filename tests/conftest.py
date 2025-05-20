from syftr.huggingface_helper import load_hf_token_into_env
from syftr.startup import prepare_worker

prepare_worker()
load_hf_token_into_env()


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", help="Use GPU for embeddings")
