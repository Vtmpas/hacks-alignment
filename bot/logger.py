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
    """
    A logging handler that intercepts standard Python logging records and redirects them to Loguru.

    Methods:
        emit(record):
            Logs the provided `record` using Loguru's logger.
        _get_level(record):
            Retrieves the logging level for the record based on the `LEVELS_MAP` or returns the
            record's level number if it's not found in the map.
    """

    def emit(self, record):
        """
        Emit a log record, redirecting it to Loguru's logger with the appropriate logging level.

        Args:
            record (logging.LogRecord): The log record to be emitted.
        """
        logger_opt = logger.opt(depth=6, exception=record.exc_info)
        logger_opt.log(self._get_level(record), record.getMessage())

    @staticmethod
    def _get_level(record):
        """
        Retrieve the logging level for the given record.

        Args:
            record (logging.LogRecord): The log record whose level needs to be mapped.

        Returns:
            str or int: The mapped log level string from `LEVELS_MAP` or the log level number if
            it's not found in the map.
        """
        return LEVELS_MAP.get(record.levelno, record.levelno)


def setup():
    """
    Configures logging for the application. It sets up the Loguru logger to log to both
    standard error (`sys.stderr`) and a rotating log file in the specified `LOG_DIR`.

    It also integrates the standard Python `logging` module with Loguru by using
    `InterceptHandler` to redirect log messages from `logging` to Loguru.

    Loggers:
        - Logs to `sys.stderr` with a specified format and logs only messages with the
          "INFO" level or higher.
        - Logs to a rotating log file in the `LOG_DIR` directory.

    The function also initializes basic configuration for Python's `logging` module to use
    the `InterceptHandler` and set the default level to `INFO`.
    """
    logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
    logger.add(LOG_DIR / "file_{time}.log")
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)
