import logging
from logging.handlers import RotatingFileHandler

import colorlog

from syftr.configuration import cfg

if cfg.logging.use_colors:
    logger = colorlog.getLogger(cfg.logging.name)
else:
    logger = logging.getLogger(cfg.logging.name)


def _create_default_handler(handler_name: str) -> logging.Handler:
    """Gives the project default logging handler"""
    if cfg.logging.use_colors:
        new_handler = colorlog.StreamHandler()
        new_handler.setFormatter(colorlog.ColoredFormatter(cfg.logging.color_format))
    else:
        new_handler = logging.StreamHandler()
        new_handler.setFormatter(logging.Formatter(cfg.logging.normal_format))
    new_handler.set_name(handler_name)
    return new_handler


def has_handler(logger_instance: logging.Logger, handler_name: str) -> bool:
    if not logger.handlers:
        return False
    for han in logger_instance.handlers:
        if han.name == handler_name:
            return True
    return False


def _create_file_handler(handler_name: str, file_name: str) -> logging.Handler:
    new_handler = RotatingFileHandler(
        file_name, maxBytes=2 * 1024 * 1024, backupCount=5
    )
    normal_formatter = logging.Formatter(cfg.logging.normal_format)
    new_handler.setFormatter(normal_formatter)
    new_handler.set_name(handler_name)
    return new_handler


def add_file_handler(
    logger_instance: logging.Logger,
    file_name: str = cfg.logging.filename,
    handler_name: str = "file_handler",
):
    file_handler = None
    if logger.handlers:
        for han in logger_instance.handlers:
            if han.name == handler_name:
                file_handler = han
    if file_handler is None:
        file_handler = _create_file_handler(handler_name, file_name)
    logger.addHandler(file_handler)


def add_default_handler(logger_instance: logging.Logger):
    if not has_handler(logger_instance, "default_handler"):
        default_handler = _create_default_handler("default_handler")
        logger.addHandler(default_handler)


add_default_handler(logger)
add_file_handler(logger)
logger.setLevel(cfg.logging.level)


def io_logger(func):
    """Decorator that returns a wrapped function for logging"""

    def wrapper(*args, **kwargs):
        """Wrapped function for logging"""
        logger.debug("%s input:", func.__name__)
        for value in args:
            logger.debug("\t%s", value)
        for parameter, value in kwargs.items():
            logger.debug("\t%s=%s", parameter, value)
        result = func(*args, **kwargs)
        if result:
            logger.debug("%s output:", func.__name__)
            logger.debug("\t%s", result)
        return result

    return wrapper


if __name__ == "__main__":
    logger.info("Logging module loaded")
