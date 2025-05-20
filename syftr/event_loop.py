import asyncio

import nest_asyncio

from syftr.logger import logger

USE_UVLOOP = False


def fix_asyncio():
    if not USE_UVLOOP:
        logger.debug("Fixing asyncio and setting a new default loop.")
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        asyncio.set_event_loop(asyncio.new_event_loop())
        nest_asyncio.apply()
