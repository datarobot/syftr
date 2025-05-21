"""Script to construct a runtime_env yaml for remote Ray Job submission.

See
https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#api-reference
for detailed documentation of this file's parameters.
"""

import shutil
import tomllib
from typing import Any, Dict, List

from syftr.configuration import cfg
from syftr.huggingface_helper import get_hf_token


def _build_pip() -> List[str]:
    with open(cfg.paths.root_dir / "pyproject.toml", "rb") as pyproject:
        pyproject_data = tomllib.load(pyproject)
    return pyproject_data["project"]["dependencies"]


def _build_env(delete_confirmed: bool) -> Dict[str, str]:
    env = {"TOKENIZERS_PARALLELISM": "true", "NLTK_DATA": cfg.paths.nltk_dir.as_posix()}
    if delete_confirmed:
        env["SYFTR_OPTUNA__NOCONFIRM"] = "true"
    env.update(get_hf_token())
    return env


def _build_excludes() -> List[str]:
    """Add excluded files from py_modules.

    Paths are relative to any py_modules path"""
    excludes = {
        "data/cache/**",
        "data/crag/**",
        "data/drdocs/**",
        "data/financebench/**",
        "data/hotpot/**",
        "data/synth/**",
    }
    return sorted(list(excludes))


def _prepare_working_dir() -> str:
    root = cfg.paths.root_dir
    dest = root / "ray_working_dir"
    secrets_dir = cfg.model_config["secrets_dir"]

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=False)

    shutil.copytree(root / secrets_dir, dest / secrets_dir)  # type: ignore
    shutil.copytree(root / "studies", dest / "studies")
    try:
        shutil.copyfile(root / ".env", dest / ".env")
    except FileNotFoundError:
        pass
    try:
        shutil.copyfile(root / "config.yaml", dest / "config.yaml")
    except FileNotFoundError:
        pass

    return dest.as_posix()


def _prepare_modules():
    import syftr

    return [syftr]


def get_runtime_env(delete_confirmed: bool = False) -> Dict[str, Any]:
    return {
        "env_vars": _build_env(delete_confirmed),
        "pip": _build_pip(),
        "py_modules": _prepare_modules(),
        "working_dir": _prepare_working_dir(),
        "excludes": _build_excludes(),
    }
