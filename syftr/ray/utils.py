import getpass
import os
import functools
from typing import Any, Final
import ray

from syftr.configuration import cfg
from syftr.logger import logger

NAMESPACE: Final = "syftr"
RAY_CACHE_ACTOR_NAME: Final = "ray_cache"


def ray_init(force_remote: bool = False):
    if ray.is_initialized():
        logger.warning(
            "Using existing ray client with address '%s'", ray.client().address
        )
    else:
        address = cfg.ray.remote_endpoint if force_remote else None

        if address is None:
            username = getpass.getuser()
            ray_tmpdir = f"/tmp/ray_{username}"
            logger.info(
                "Using local ray client with temporary directory '%s'", ray_tmpdir
            )
            os.environ["RAY_TMPDIR"] = ray_tmpdir

        ray.init(
            address=address,
            logging_level=cfg.logging.level,
            namespace=NAMESPACE,
        )


@ray.remote
class RayCacheActor:
    def __init__(self):
        self.cache = {}

    def get(self, key):
        return self.cache.get(key, None)

    def set(self, key, value):
        self.cache[key] = value

    def contains(self, key):
        return key in self.cache

    def clear(self):
        self.cache.clear()


@functools.lru_cache(maxsize=1)
def get_ray_cache():
    """Returns the ray cache actor."""
    if not ray.is_initialized():
        raise RuntimeError("Ray is not initialized. Cannot use Ray cache.")
    try:
        return ray.get_actor(RAY_CACHE_ACTOR_NAME)
    except ValueError:
        return RayCacheActor.options(
            name=RAY_CACHE_ACTOR_NAME, lifetime="detached", namespace=NAMESPACE
        ).remote()


def ray_cache_get(key: str):
    """Get a value from the Ray cache. Raises a RuntimeError if Ray not initialized."""
    ray_cache = get_ray_cache()
    return ray.get(ray_cache.get.remote(key))


def ray_cache_put(key: str, obj: Any) -> None:
    """Put a value in the ray cache. Raises a RuntimeError if Ray not initialized."""
    ray_cache = get_ray_cache()
    ray_cache.set.remote(key, obj)


def ray_cache_restart():
    """Kills and restarts the otherwise persistent Ray cache actor."""
    try:
        old = ray.get_actor(RAY_CACHE_ACTOR_NAME, namespace=NAMESPACE)
        ray.kill(old)
    except ValueError:
        pass
    get_ray_cache.cache_clear()
    return get_ray_cache()
