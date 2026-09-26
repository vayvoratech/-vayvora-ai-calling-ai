"""Unit tests for configuration loading and validation."""

import pytest
from pydantic import ValidationError
from src.config import Settings, get_settings


class TestSettings:
    """Test Settings loading and validation."""

    def test_default_settings(self, monkeypatch):
        # Clear any environment variables that might interfere
        for key in [
            "APP_ENV",
            "LOG_LEVEL",
            "GEMINI_API_KEY",
            "GEMINI_MODEL",
            "REDIS_HOST",
            "REDIS_PORT",
            "REDIS_DB",
            "REDIS_PASSWORD",
            "MCP_SERVER_URL",
            "MCP_ENABLED",
        ]:
            monkeypatch.delenv(key, raising=False)

        settings = Settings(_env_file=None)
        assert settings.app_env == "development"
        assert settings.log_level == "INFO"
        assert settings.gemini_model == "gemini-3.5-flash"
        assert settings.gemini_api_key is None
        assert settings.redis_host == "localhost"
        assert settings.redis_port == 6379
        assert settings.redis_db == 0
        assert settings.redis_password is None
        assert settings.redis_url == "redis://localhost:6379/0"
        assert settings.mcp_enabled is True

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        monkeypatch.setenv("GEMINI_API_KEY", "secret-test-key-12345")
        monkeypatch.setenv("REDIS_HOST", "redis.internal.net")
        monkeypatch.setenv("REDIS_PORT", "6380")
        monkeypatch.setenv("REDIS_PASSWORD", "redispass123")
        monkeypatch.setenv("MCP_ENABLED", "false")

        settings = Settings(_env_file=None)
        assert settings.app_env == "production"
        assert settings.log_level == "WARNING"
        assert settings.gemini_api_key is not None
        assert settings.gemini_api_key.get_secret_value() == "secret-test-key-12345"
        assert "secret-test-key-12345" not in repr(settings.gemini_api_key)  # SecretStr masking
        assert settings.redis_host == "redis.internal.net"
        assert settings.redis_port == 6380
        assert settings.redis_url == "redis://:redispass123@redis.internal.net:6380/0"
        assert settings.mcp_enabled is False

    def test_invalid_redis_port(self, monkeypatch):
        monkeypatch.setenv("REDIS_PORT", "999999")  # Port out of bounds
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_get_settings_cached(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
