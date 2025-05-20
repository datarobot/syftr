import ray

from syftr.configuration import cfg
from syftr.logger import logger


def ray_init(force_remote: bool = False):
    if ray.is_initialized():
        logger.warning(
            "Using existing ray client with address '%s'", ray.client().address
        )
    else:
        address = cfg.ray.remote_endpoint if force_remote else None
        ray.init(
            address=address,
            logging_level=cfg.logging.level,
        )
