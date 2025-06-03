try:
    from importlib.metadata import version

    __version__ = version("syftr")
except Exception:
    __version__ = "0.0.0"
