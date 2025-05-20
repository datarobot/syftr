import nltk

from syftr.configuration import cfg


def _download_nltk_data():
    nltk.download("punkt", cfg.paths.nltk_dir, quiet=True)
    nltk.download("stopwords", cfg.paths.nltk_dir, quiet=True)


def prepare_worker():
    _download_nltk_data()


if __name__ == "__main__":
    prepare_worker()
