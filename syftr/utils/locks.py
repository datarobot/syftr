import hashlib
import json
from contextlib import contextmanager
from typing import Dict, Union

import pals
from flock import LOCK_EX, Flock

from syftr.configuration import cfg

LOCKER = pals.Locker("syftr-locks", create_engine_callable=cfg.database.get_engine)


@contextmanager
def distributed_lock(
    key_data: Union[Dict, str], timeout_s: int = 60 * 60 * 12, host_only: bool = False
):
    assert key_data, f"key_data must not be empty! Got {key_data=}"
    if isinstance(key_data, Dict):
        key_data = hashlib.sha1(
            json.dumps(key_data, sort_keys=True).encode("utf-8")
        ).hexdigest()

    if host_only:
        key_path = key_data.replace("/", "-")
        with open(cfg.paths.lock_dir / key_path, "w") as lockfile:
            with Flock(lockfile, LOCK_EX):
                yield key_data
    else:
        lock = LOCKER.lock(key_data)
        lock.acquire_timeout = timeout_s * 1000
        with lock:
            yield key_data
