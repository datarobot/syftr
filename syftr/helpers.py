import csv
import json
import logging
import numbers
import textwrap
import traceback
import typing as T
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock
from tabulate import tabulate

from syftr.configuration import cfg
from syftr.logger import logger


def text_to_bool(text):
    true_values = {"true", "1", "yes", "on", "t"}
    false_values = {"false", "0", "no", "off", "f"}

    normalized_text = text.strip().lower()

    if normalized_text in true_values:
        return True
    elif normalized_text in false_values:
        return False
    raise ValueError(f"Invalid text input for boolean conversion: '{text}'")


def format_value(value, max_width=60):
    if isinstance(value, str):
        return "\n".join(textwrap.wrap(value, max_width))
    return value


def append_dict_to_csv(file_path: Path, data: dict):
    file_path = Path(file_path).resolve()
    lock = FileLock(file_path.with_suffix(".lock"))
    with lock:
        logger.info("Appending data to file: %s", file_path)
        file_exists = file_path.exists()
        with file_path.open(mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)


def debug(
    key_value_pairs: dict, max_width=60, headers=None, tablefmt="grid", file_path=None
):
    headers = headers or ["Parameter", "Value"]
    if cfg.logging.level > logging.DEBUG:
        return
    table = [
        [str(k), format_value(str(v), max_width=max_width)]
        for k, v in sorted(key_value_pairs.items())
    ]
    print(tabulate(table, headers=headers, tablefmt=tablefmt))
    if file_path:
        append_dict_to_csv(file_path, key_value_pairs)


def get_baselines_from_trials(df_trials: pd.DataFrame) -> T.List[T.Dict[str, T.Any]]:
    return list(df_trials["user_attrs_flow"].apply(json.loads).values)


def get_flows_from_trials(df_trials: pd.DataFrame) -> T.List[T.Dict[str, T.Any]]:
    return list(df_trials["user_attrs_flow"].apply(json.loads).values)


def is_within_range(value, min_val, max_val, step, tolerance=1e-9):
    if not (min_val <= value <= max_val):
        return False
    diff = value - min_val
    remainder = diff % step
    return remainder < tolerance or abs(remainder - step) < tolerance


def get_exception_report(
    exception: T.Union[
        BaseException,
        BaseExceptionGroup,
    ],
) -> str:
    e = exception
    ex_type = type(e).__name__
    ex_message = str(e)
    ex_traceback = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    return f"Type: {ex_type}, Message: {ex_message}, Traceback: {ex_traceback}"


def is_numeric(x):
    return isinstance(x, numbers.Number) or (
        isinstance(x, np.generic) and np.issubdtype(x, np.number)
    )


def get_unique_bools(df: pd.DataFrame, col: str) -> T.List[bool]:
    param = "params_" + col
    if param not in df.columns:
        return []
    return [value for value in df[param].unique() if isinstance(value, bool)]


def get_unique_strings(df: pd.DataFrame, col: str) -> T.List[str]:
    param = "params_" + col
    if param not in df.columns:
        return []
    return [value for value in df[param].unique() if isinstance(value, str)]


def get_unique_ints(df: pd.DataFrame, col: str) -> T.List[int]:
    param = "params_" + col
    if param not in df.columns:
        return []
    values = [
        int(round(value, 0))
        for value in df[param].unique()
        if is_numeric(value) and not pd.isna(value)
    ]
    return list(set(values))


def get_unique_floats(df: pd.DataFrame, col: str, ndigits: int) -> T.List[float]:
    param = "params_" + col
    if param not in df.columns:
        return []
    values = [
        round(float(value), ndigits)
        for value in df[param].unique()
        if is_numeric(value) and value != np.nan
    ]
    return list(set(values))


def get_min_int(df: pd.DataFrame, col: str, default: int) -> int:
    ints = get_unique_ints(df, col)
    if not ints:
        return default
    return min(ints)


def get_max_int(df: pd.DataFrame, col: str, default: int) -> int:
    ints = get_unique_ints(df, col)
    if not ints:
        return default
    return max(ints)


def get_min_float(df: pd.DataFrame, col: str, default: float, ndigits: int) -> float:
    floats = get_unique_floats(df, col, ndigits)
    if not floats:
        return default
    return min(floats)


def get_max_float(df: pd.DataFrame, col: str, default: float, ndigits: int) -> float:
    floats = get_unique_floats(df, col, ndigits)
    if not floats:
        return default
    return max(floats)
