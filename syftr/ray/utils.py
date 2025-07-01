import getpass
import os

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
        )
