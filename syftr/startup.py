import nltk

from syftr.configuration import cfg
from syftr.utils.locks import distributed_lock


def _download_nltk_data():
    with distributed_lock("nltk_download"):
        nltk.download("punkt", cfg.paths.nltk_dir, quiet=True)
        nltk.download("stopwords", cfg.paths.nltk_dir, quiet=True)


def prepare_worker():
    _download_nltk_data()


if __name__ == "__main__":
    prepare_worker()
