"""Unit tests for centralized logging setup."""

import logging
from src.logging import get_logger, setup_logging


class TestLogging:
    """Test logger initialization and formatting."""

    def test_setup_logging_level(self):
        logger = setup_logging(level="DEBUG", logger_name="test_voice_agent")
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_setup_logging_reentrant(self):
        # Ensure repeated calls do not duplicate handlers
        setup_logging(level="INFO", logger_name="test_voice_agent")
        logger = setup_logging(level="WARNING", logger_name="test_voice_agent")
        assert logger.level == logging.WARNING
        assert len(logger.handlers) == 1

    def test_get_logger_namespacing(self):
        base = get_logger()
        assert base.name == "voice_agent"

        child = get_logger("core.router")
        assert child.name == "voice_agent.core.router"
