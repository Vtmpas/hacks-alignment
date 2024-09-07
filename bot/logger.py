import logging
import sys
from types import MappingProxyType

from constants import LOG_DIR
from loguru import logger

LEVELS_MAP: MappingProxyType[int, str] = MappingProxyType(
    {
        logging.CRITICAL: "CRITICAL",
        logging.ERROR: "ERROR",
        logging.WARNING: "WARNING",
        logging.INFO: "INFO",
        logging.DEBUG: "DEBUG",
    }
)


class InterceptHandler(logging.Handler):
    def emit(self, record):
        logger_opt = logger.opt(depth=6, exception=record.exc_info)
        logger_opt.log(self._get_level(record), record.getMessage())

    @staticmethod
    def _get_level(record):
        return LEVELS_MAP.get(record.levelno, record.levelno)


def setup():
    logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
    logger.add(LOG_DIR / "file_{time}.log")
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)
