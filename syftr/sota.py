import json
import typing as T
from dataclasses import dataclass
from pathlib import Path

from syftr.configuration import cfg


@dataclass
class SOTA:
    name: str
    accuracy: float
    link: str
    dataset: str
    subset: str | None = None
    test_setting: str | None = None
    eval_method: str | None = None
    method_name: str | None = None


def load_sota(file_path: Path) -> list[SOTA]:
    with open(file_path) as f:
        return [SOTA(**sota) for sota in json.load(f)]


_SOTAS: T.List[SOTA] = load_sota(cfg.paths.sota_dir / "sota_data.json")


def get_sota(study_name) -> SOTA | None:
    match study_name.lower():
        case name if "hotpot" in name:
            return next((s for s in _SOTAS if s.dataset == "HotpotQA"), None)
        case name if "financebench" in name:
            return next((s for s in _SOTAS if s.dataset == "FinanceBench"), None)
        case name if "crag" in name and "finance" in name:
            return next(
                (
                    s
                    for s in _SOTAS
                    if s.dataset == "CRAG Task3" and s.subset == "Finance"
                ),
                None,
            )
        case name if "crag" in name and "sports" in name:
            return next(
                (
                    s
                    for s in _SOTAS
                    if s.dataset == "CRAG Task3" and s.subset == "Sports"
                ),
                None,
            )
        case name if "crag" in name and "music" in name:
            return next(
                (
                    s
                    for s in _SOTAS
                    if s.dataset == "CRAG Task3" and s.subset == "Music"
                ),
                None,
            )
        case name if "crag" in name and "movie" in name:
            return next(
                (
                    s
                    for s in _SOTAS
                    if s.dataset == "CRAG Task3" and s.subset == "Movie"
                ),
                None,
            )
        case name if "crag" in name and "open" in name:
            return next(
                (s for s in _SOTAS if s.dataset == "CRAG Task3" and s.subset == "Open"),
                None,
            )
        case _:
            return None
