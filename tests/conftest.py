import getpass
import os

import pytest

from syftr.huggingface_helper import load_hf_token_into_env
from syftr.startup import download_nltk_data

download_nltk_data()
load_hf_token_into_env()


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", help="Use GPU for embeddings")


@pytest.fixture(scope="session", autouse=True)
def set_ray_tmpdir():
    username = getpass.getuser()
    os.environ["RAY_TMPDIR"] = f"/tmp/ray_test_{username}"
    yield
