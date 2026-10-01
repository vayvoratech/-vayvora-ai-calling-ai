"""Standardized logging configuration.

Provides a clean, centralized logging setup without sensitive data leaks
or over-engineering.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO", logger_name: str = "voice_agent"
) -> logging.Logger:
    """Configure and return the root logger for the application."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(numeric_level)

    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers during re-initialization
    if not logger.handlers:
        logger.addHandler(handler)
    else:
        logger.handlers.clear()
        logger.addHandler(handler)

    logger.propagate = False
    return logger


def get_logger(child_name: Optional[str] = None) -> logging.Logger:
    """Obtain a logger within the application namespace."""
    base_name = "voice_agent"
    if child_name:
        return logging.getLogger(f"{base_name}.{child_name}")
    return logging.getLogger(base_name)
