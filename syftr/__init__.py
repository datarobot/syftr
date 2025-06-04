try:
    from importlib.metadata import version

    __version__ = version("syftr")
except Exception:
    import os

    __version__ = os.getenv("SYFTR_VERSION", "0.0.0")
