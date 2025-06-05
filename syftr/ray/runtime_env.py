"""Script to construct a runtime_env yaml for remote Ray Job submission.

See
https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#api-reference
for detailed documentation of this file's parameters.
"""

import json
import shutil
import tomllib
from pathlib import Path
from typing import Any, Dict, List

import yaml

from syftr import __version__
from syftr.configuration import cfg
from syftr.huggingface_helper import get_hf_token


def _build_pip() -> List[str]:
    try:
        with open(cfg.paths.root_dir / "pyproject.toml", "rb") as pyproject:
            pyproject_data = tomllib.load(pyproject)
        return pyproject_data["project"]["dependencies"]
    except FileNotFoundError:
        # We are not in a git repo, syftr is used as a library.
        from importlib.metadata import version

        curr_syftr_ver = version("syftr")
        return [f"syftr=={curr_syftr_ver}"]


def _build_env(delete_confirmed: bool) -> Dict[str, str]:
    env = {
        "TOKENIZERS_PARALLELISM": "true",
        "NLTK_DATA": cfg.paths.nltk_dir.as_posix(),
        "SYFTR_VERSION": __version__,
    }
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
        "results/**",
        "studies/**",
    }
    return sorted(list(excludes))


def _prepare_working_dir(study_config_path: Path) -> str:
    root = cfg.paths.root_dir
    dest = root / "ray_working_dir"

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=False)

    cfg_data = json.loads(
        # Don't send un-set configurations like default paths, as
        # these will be auto-discovered on the cluster.
        cfg.model_dump_json(exclude_unset=True, exclude_defaults=True)
    )
    with open(dest / "config.yaml", "w") as cfg_file:
        yaml.safe_dump(cfg_data, cfg_file)

    shutil.copy(study_config_path, dest)

    return dest.as_posix()


def _prepare_modules():
    import syftr

    return [syftr]


def get_runtime_env(
    study_config_path: Path, delete_confirmed: bool = False
) -> Dict[str, Any]:
    return {
        "env_vars": _build_env(delete_confirmed),
        "pip": _build_pip(),
        "py_modules": _prepare_modules(),
        "working_dir": _prepare_working_dir(study_config_path),
        "excludes": _build_excludes(),
    }
