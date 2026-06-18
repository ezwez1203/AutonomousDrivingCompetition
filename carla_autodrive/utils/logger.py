"""Small colored console logger."""
import logging
import sys

_COLORS = {
    "DEBUG": "\033[36m",     # cyan
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[41m",  # red bg
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    def format(self, record):
        color = _COLORS.get(record.levelname, "")
        record.levelname_c = f"{color}{record.levelname:<7}{_RESET}"
        return super().format(record)


def get_logger(name: str = "carla_autodrive", level: int = logging.INFO) -> logging.Logger:
    """Return a singleton logger for the given name."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ColorFormatter(
        "[%(asctime)s] %(levelname_c)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
