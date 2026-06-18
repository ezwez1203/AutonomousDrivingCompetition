"""Utility module for logging, configuration, and CARLA sessions."""
from .config import load_config
from .logger import get_logger

__all__ = ["load_config", "get_logger", "CarlaSession"]


def __getattr__(name: str):
    if name == "CarlaSession":
        from .carla_session import CarlaSession
        return CarlaSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
