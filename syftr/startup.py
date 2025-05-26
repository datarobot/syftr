import nltk

from syftr.configuration import cfg
from syftr.utils.locks import distributed_lock


def download_nltk_data():
    with distributed_lock("nltk_download", host_only=True):
        nltk.download("punkt", cfg.paths.nltk_dir, quiet=True)
        nltk.download("stopwords", cfg.paths.nltk_dir, quiet=True)


def prepare_worker():
    download_nltk_data()


if __name__ == "__main__":
    prepare_worker()
