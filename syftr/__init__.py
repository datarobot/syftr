try:
    """
    Try to load the syftr version from the package metadata to avoid using static version strings.
    This allows for dynamic versioning based on the installed package version.
    """
    from importlib.metadata import version

    __version__ = version("syftr")
except Exception:
    """
    Package metadata is not avaiable when only the syftr module is used on a ray worker.
    When a ray job is submitted, the dynamically loaded version above is stored in the environment variable SYFTR_VERSION.
    This fallback ensures that the version is still accessible in such cases.
    """
    import os

    __version__ = os.getenv("SYFTR_VERSION", "0.0.0")
