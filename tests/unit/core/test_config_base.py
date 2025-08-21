"""Tests for configuration base system.

Tests configuration operations, validation, defaults,
and utilities to achieve near 100% coverage.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

# Use actual available config APIs
from flext_core import (
    FlextConfig as _BaseConfig,
    FlextConstants,
    FlextSystemDefaults,
    safe_get_env_var,
    safe_load_json_file,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


@pytest.fixture
def temp_dir() -> Generator[Path]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_path:
        yield Path(temp_path)


# Note: Most configuration operation classes (_BaseConfigOps, FlextConfigValidation, etc.)
# do not exist in the current codebase. Only FlextConfig, FlextSettings, and utility functions
# like safe_get_env_var and safe_load_json_file are available.
# This file has been cleaned to only test actual existing functionality.


@pytest.mark.unit
class TestActualConfigFunctionality:
    """Test actual configuration functionality that exists."""

    def test_safe_get_env_var_exists(self) -> None:
        """Test safe_get_env_var with existing variable - REAL execution."""
        # Set real environment variable
        original_value = os.environ.get("FLEXT_TEST_VAR")
        os.environ["FLEXT_TEST_VAR"] = "test_value"

        try:
            result = safe_get_env_var("FLEXT_TEST_VAR")

            assert result.success
            if result.value != "test_value":
                raise AssertionError(f"Expected 'test_value', got {result.value}")
        finally:
            # Clean up real environment
            if original_value is not None:
                os.environ["FLEXT_TEST_VAR"] = original_value
            else:
                os.environ.pop("FLEXT_TEST_VAR", None)

    def test_safe_get_env_var_not_exists_with_default(self) -> None:
        """Test safe_get_env_var with non-existing variable and default."""
        result = safe_get_env_var(
            "NONEXISTENT_VAR",
            default="default_value",
        )

        assert result.success
        if result.value != "default_value":
            raise AssertionError(f"Expected 'default_value', got {result.value}")

    def test_safe_get_env_var_not_exists_no_default(self) -> None:
        """Test safe_get_env_var with non-existing variable and no default."""
        result = safe_get_env_var("NONEXISTENT_VAR")

        assert result.is_failure
        error_msg = result.error or ""
        if "not set" not in error_msg.lower():
            raise AssertionError(
                f"Expected 'not set' in {result.error}",
            )

    def test_safe_load_json_file_valid(self, temp_json_file: str) -> None:
        """Test safe_load_json_file with valid JSON file."""
        result = safe_load_json_file(temp_json_file)

        assert result.success
        data = result.value or {}
        if "key1" not in data:
            raise AssertionError(f"Expected 'key1' in {result.value}")
        if data.get("key1") != "value1":
            raise AssertionError(f"Expected 'value1', got {data.get('key1')}")
        assert data.get("key2") == 42

    def test_safe_load_json_file_not_exists(self) -> None:
        """Test safe_load_json_file with non-existent file."""
        result = safe_load_json_file("/nonexistent/file.json")

        assert result.is_failure
        if "File not found:" not in (result.error or ""):
            raise AssertionError(
                f"Expected 'File not found:' in {result.error}",
            )

    def test_safe_load_json_file_invalid_json(self, temp_dir: Path) -> None:
        """Test safe_load_json_file with invalid JSON."""
        invalid_json_file = temp_dir / "invalid.json"
        invalid_json_file.write_text("{ invalid json", encoding="utf-8")

        result = safe_load_json_file(invalid_json_file)

        assert result.is_failure
        if "Invalid JSON:" not in (result.error or ""):
            raise AssertionError(
                f"Expected 'Invalid JSON:' in {result.error}",
            )


@pytest.mark.unit
class TestFlextSystemDefaults:
    """Test FlextSystemDefaults constants."""

    def test_system_defaults_available(self) -> None:
        """Test that system defaults are available."""
        # Test that constants exist
        assert hasattr(FlextSystemDefaults, "Environment")
        assert hasattr(FlextSystemDefaults, "Logging")
        assert hasattr(FlextSystemDefaults, "Network")
        assert hasattr(FlextSystemDefaults, "Pagination")
        assert hasattr(FlextSystemDefaults, "Security")


@pytest.mark.unit
class TestFlextConstants:
    """Test FlextConstants constants."""

    def test_constants_available(self) -> None:
        """Test that FlextConstants are available."""
        # Test that key constants exist
        assert hasattr(FlextConstants, "Performance")
        assert hasattr(FlextConstants, "Platform")
        assert hasattr(FlextConstants, "DEFAULT_TIMEOUT")
        assert hasattr(FlextConstants, "VERSION")

    def test_performance_constants(self) -> None:
        """Test that performance constants are defined."""
        assert hasattr(FlextConstants.Performance, "TIMEOUT")
        assert hasattr(FlextConstants.Performance, "BATCH_SIZE")
        assert hasattr(FlextConstants.Performance, "MAX_CONNECTIONS")

        # Test that values are reasonable
        assert FlextConstants.Performance.TIMEOUT > 0
        assert FlextConstants.Performance.BATCH_SIZE > 0
        assert FlextConstants.Performance.MAX_CONNECTIONS > 0


@pytest.mark.unit
class TestBaseConfig:
    """Test _BaseConfig utility functions."""

    def test_config_creation(self) -> None:
        """Test FlextConfig creation."""
        config = _BaseConfig()
        assert config.name == "flext"
        assert config.environment == "development"
        assert config.debug is False

    def test_config_custom_values(self) -> None:
        """Test FlextConfig with custom values."""
        config = _BaseConfig(
            name="custom-app",
            environment="production",
            debug=True,
        )
        assert config.name == "custom-app"
        assert config.environment == "production"
        assert config.debug is True

    def test_config_validation(self) -> None:
        """Test FlextConfig validation."""
        config = _BaseConfig()
        result = config.validate_business_rules()
        assert result.is_success


@pytest.mark.integration
class TestConfigBaseIntegration:
    """Integration tests for configuration base system."""

    def test_environment_variable_integration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test environment variable integration - functional test."""
        # Set test environment variables functionally
        monkeypatch.setenv("APP_DEBUG", "true")
        monkeypatch.setenv("APP_PORT", str(FlextConstants.Platform.FLEXCORE_PORT))
        monkeypatch.setenv("APP_SECRET", "secret123")

        # Get configuration from environment
        debug_result = safe_get_env_var("APP_DEBUG", default="false")
        port_result = safe_get_env_var("APP_PORT")  # No 'required' parameter
        secret_result = safe_get_env_var("APP_SECRET")  # No 'required' parameter
        missing_result = safe_get_env_var(
            "APP_MISSING",
            default="default",
        )

        assert debug_result.success
        if debug_result.value != "true":
            raise AssertionError("Expected 'true'")
        assert port_result.success
        if port_result.value != str(FlextConstants.Platform.FLEXCORE_PORT):
            raise AssertionError("Expected matching port")
        assert secret_result.success
        if secret_result.value != "secret123":
            raise AssertionError("Expected 'secret123'")
        assert missing_result.success
        if missing_result.value != "default":
            raise AssertionError("Expected 'default'")

    def test_error_recovery_workflow(self) -> None:
        """Test error recovery in configuration workflow."""
        # Simulate config loading with fallbacks
        primary_file = "/nonexistent/primary.json"

        # Try primary config (will fail)
        primary_result = safe_load_json_file(primary_file)
        assert primary_result.is_failure

        # Fall back to creating basic config
        fallback_config = _BaseConfig()
        assert fallback_config.name == "flext"
        assert fallback_config.environment == "development"
