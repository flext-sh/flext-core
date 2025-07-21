"""Simple tests for Enhanced Base Configuration.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from flext_core.configuration import EnhancedBaseConfig, Environment, LogLevel


class TestEnhancedBaseConfigSimple:
    """Test EnhancedBaseConfig with isolated environment."""

    def test_basic_configuration(self) -> None:
        """Test basic configuration creation."""
        # Clear environment variables and set explicit defaults
        with patch.dict(os.environ, {}, clear=True):
            config = EnhancedBaseConfig(
                debug=False,  # Explicitly set to ensure clean test
                log_level=LogLevel.INFO,
                environment=Environment.DEVELOPMENT,
            )

            assert config.debug is False
            assert config.log_level == LogLevel.INFO
            assert config.environment == Environment.DEVELOPMENT
            assert config.app_name == "flext-application"

    def test_log_level_validation(self) -> None:
        """Test log level validation."""
        # Clear environment variables
        with patch.dict(os.environ, {}, clear=True):
            # Valid string conversion
            config = EnhancedBaseConfig(log_level=LogLevel.DEBUG)
            assert config.log_level == LogLevel.DEBUG

            # LogLevel enum directly
            config = EnhancedBaseConfig(log_level=LogLevel.ERROR)
            assert config.log_level == LogLevel.ERROR

    def test_environment_validation(self) -> None:
        """Test environment validation."""
        # Clear environment variables
        with patch.dict(os.environ, {}, clear=True):
            # Valid string conversion
            config = EnhancedBaseConfig(environment=Environment.PRODUCTION)
            assert config.environment == Environment.PRODUCTION

            # Environment enum directly
            config = EnhancedBaseConfig(environment=Environment.STAGING)
            assert config.environment == Environment.STAGING

    def test_secret_key_validation(self) -> None:
        """Test secret key validation."""
        # Clear environment variables
        with patch.dict(os.environ, {}, clear=True):
            # Valid secret key (32+ characters)
            long_key = "a" * 32
            config = EnhancedBaseConfig(secret_key=long_key)
            assert config.secret_key == long_key

            # Invalid secret key (too short)
            with pytest.raises(
                ValueError, match="Secret key must be at least 32 characters"
            ):
                EnhancedBaseConfig(secret_key="short")

    def test_environment_properties(self) -> None:
        """Test environment check properties."""
        # Clear environment variables
        with patch.dict(os.environ, {}, clear=True):
            # Production environment
            prod_config = EnhancedBaseConfig(environment=Environment.PRODUCTION)
            assert prod_config.is_production is True
            assert prod_config.is_development is False

            # Development environment
            dev_config = EnhancedBaseConfig(environment=Environment.DEVELOPMENT)
            assert dev_config.is_development is True
            assert dev_config.is_production is False

    def test_configuration_summary(self) -> None:
        """Test configuration summary with hidden sensitive data."""
        # Clear environment variables
        with patch.dict(os.environ, {}, clear=True):
            config = EnhancedBaseConfig(
                debug=False,  # Explicitly set debug to False
                secret_key="super-secret-key-that-is-long-enough",
                database_url="postgresql://user:pass@host:5432/db",
            )

            summary = config.get_config_summary()

            assert summary["secret_key"] == "[HIDDEN]"
            assert summary["database_url"] == "[HIDDEN]"
            assert summary["debug"] is False
