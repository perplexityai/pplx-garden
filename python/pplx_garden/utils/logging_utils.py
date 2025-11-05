"""Logging utilities."""

import logging
import os
import warnings


class LoggerFormatter(logging.Formatter):
    LEVEL_COLOURS = [
        (logging.DEBUG, "\x1b[40;1m"),
        (logging.INFO, "\x1b[34;1m"),
        (logging.WARNING, "\x1b[33;1m"),
        (logging.ERROR, "\x1b[31m"),
        (logging.CRITICAL, "\x1b[41m"),
    ]

    FORMATS = {
        level: logging.Formatter(
            f"\x1b[36;1m[%(asctime)s.%(msecs)03d]\x1b[0m %(process)d {colour}%(levelname)-8s"
            f"\x1b[0m \x1b[32m%(name)s\x1b[0m %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        for level, colour in LEVEL_COLOURS
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record."""
        formatter = self.FORMATS.get(record.levelno)
        if formatter is None:
            formatter = self.FORMATS[logging.DEBUG]

        # Override the traceback to always print in red
        if record.exc_info:
            text = formatter.formatException(record.exc_info)
            record.exc_text = f"\x1b[31m{text}\x1b[0m"

        output = formatter.format(record)

        # Remove the cache layer
        record.exc_text = None
        return output


_IS_SETUP = False


def setup(
    *,
    handler: logging.Handler | None = None,
    level: str | int | None = None,
) -> None:
    """Setup the logging."""
    global _IS_SETUP  # pylint: disable=global-statement
    if _IS_SETUP:
        return
    _IS_SETUP = True

    level = level or logging.INFO
    handler = handler or logging.StreamHandler()

    # Check if DD_ENV is set, if so use plain formatter without colors
    if os.environ.get("DD_ENV"):
        formatter = logging.Formatter(
            "[%(asctime)s.%(msecs)03d] %(process)d %(levelname)-8s %(name)s %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    else:
        formatter = LoggerFormatter()

    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)

    # Filter out UserWarning and FutureWarning
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


def get_logger(path: str) -> logging.Logger:
    """Return logger for the given path."""
    return logging.getLogger(path)
