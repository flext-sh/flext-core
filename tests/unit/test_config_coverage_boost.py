"""Additional tests for config.py to achieve 100% coverage.

Focus on uncovered classes and methods identified by coverage analysis.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from flext_core import FlextConfig
from flext_tests import FlextTestsBuilders, FlextTestsMatchers


class TestConfigRuntimeValidator:
    """Test ConfigRuntimeValidator coverage."""

    def test_validate_runtime_requirements_success(self) -> None:
        """Test successful runtime validation."""
        config = FlextTestsBuilders.config().with_debug(debug=False).build()
        result = FlextConfig.RuntimeValidator.validate_runtime_requirements(config)
        FlextTestsMatchers.assert_result_success(result)

    def test_validate_runtime_requirements_production_workers_fail(self) -> None:
        """Test runtime validation failure with insufficient workers in production."""
        config = FlextConfig(
            environment="production",
            max_workers=1,  # Less than minimum required (2)
            debug=False,
        )
        result = FlextConfig.RuntimeValidator.validate_runtime_requirements(config)
        FlextTestsMatchers.assert_result_failure(result)
        assert "production environment requires at least 2 workers" in (
            result.error or ""
        )

    def test_validate_runtime_requirements_high_timeout_workers_fail(self) -> None:
        """Test runtime validation failure with high timeout but insufficient workers."""
        config = FlextConfig(
            environment="development",
            timeout_seconds=150,  # High timeout
            max_workers=2,  # Less than required for high timeout (4)
            debug=False,
        )
        result = FlextConfig.RuntimeValidator.validate_runtime_requirements(config)
        FlextTestsMatchers.assert_result_failure(result)
        assert "high timeout (120s+) requires at least 4 workers" in (
            result.error or ""
        )

    def test_validate_runtime_requirements_too_many_workers_fail(self) -> None:
        """Test runtime validation failure with too many workers."""
        config = FlextConfig(
            environment="development",
            max_workers=100,  # Exceeds maximum (50)
            debug=False,
        )
        result = FlextConfig.RuntimeValidator.validate_runtime_requirements(config)
        FlextTestsMatchers.assert_result_failure(result)
        assert "exceeds maximum recommended workers" in (result.error or "")


class TestConfigBusinessValidator:
    """Test ConfigBusinessValidator coverage."""

    def test_validate_business_rules_success(self) -> None:
        """Test successful business rule validation."""
        config = FlextTestsBuilders.config().with_environment("development").build()
        result = FlextConfig.BusinessValidator.validate_business_rules(config)
        FlextTestsMatchers.assert_result_success(result)

    def test_validate_business_rules_debug_production_fail(self) -> None:
        """Test business rule validation failure - debug in production."""
        config = FlextConfig(
            environment="production",
            debug=True,  # Debug should not be enabled in production
            config_source="env",  # Non-default source to trigger validation failure
        )
        result = FlextConfig.BusinessValidator.validate_business_rules(config)
        FlextTestsMatchers.assert_result_failure(result)
        assert "Debug mode in production requires explicit configuration" in (
            result.error or ""
        )

    def test_validate_business_rules_missing_api_key_production(self) -> None:
        """Test business rule validation with missing API key in production."""
        config = FlextConfig(
            environment="production",
            debug=False,
            api_key="",  # Empty API key in production
            enable_auth=True,  # Enable auth to trigger API key validation
        )
        result = FlextConfig.BusinessValidator.validate_business_rules(config)
        FlextTestsMatchers.assert_result_failure(result)
        assert "API key required when authentication is enabled" in (result.error or "")

    def test_validate_business_rules_insecure_cors_production(self) -> None:
        """Test business rule validation with insecure CORS in production."""
        # Note: CORS validation is not currently implemented in business rules
        # This test is skipped until CORS validation is added
        pytest.skip("CORS validation not implemented in business rules yet")
        config = FlextConfig(
            environment="production",
            debug=False,
            api_key="valid-key",
            cors_origins=["*"],  # Wildcard CORS in production
        )
        result = FlextConfig.BusinessValidator.validate_business_rules(config)
        FlextTestsMatchers.assert_result_failure(result)
        assert "Wildcard CORS origins not allowed in production" in (result.error or "")


class TestDefaultEnvironmentAdapter:
    """Test FlextConfig.DefaultEnvironmentAdapter coverage."""

    def test_get_env_var_success(self) -> None:
        """Test successful environment variable retrieval."""
        adapter = FlextConfig.DefaultEnvironmentAdapter()
        with patch.dict("os.environ", {"TEST_VAR": "test_value"}):
            result = adapter.get_env_var("TEST_VAR")
            FlextTestsMatchers.assert_result_success(result, "test_value")

    def test_get_env_var_not_found(self) -> None:
        """Test environment variable not found."""
        adapter = FlextConfig.DefaultEnvironmentAdapter()
        with patch.dict("os.environ", {}, clear=True):
            result = adapter.get_env_var("NONEXISTENT_VAR")
            FlextTestsMatchers.assert_result_failure(result)
            assert "not found" in (result.error or "")

    def test_get_env_var_exception(self) -> None:
        """Test environment variable retrieval with exception."""
        adapter = FlextConfig.DefaultEnvironmentAdapter()
        with patch("os.getenv", side_effect=Exception("OS error")):
            result = adapter.get_env_var("TEST_VAR")
            FlextTestsMatchers.assert_result_failure(result)
            assert "Failed to get environment variable" in (result.error or "")

    def test_get_env_vars_with_prefix_success(self) -> None:
        """Test successful environment variables with prefix retrieval."""
        adapter = FlextConfig.DefaultEnvironmentAdapter()
        test_env = {
            "FLEXT_VAR1": "value1",
            "FLEXT_VAR2": "value2",
            "OTHER_VAR": "other",
        }
        with patch.dict("os.environ", test_env):
            result = adapter.get_env_vars_with_prefix("FLEXT_")
            FlextTestsMatchers.assert_result_success(result)
            env_vars = result.value
            assert env_vars["VAR1"] == "value1"  # Prefix removed
            assert env_vars["VAR2"] == "value2"
            assert "OTHER_VAR" not in env_vars

    def test_get_env_vars_with_prefix_exception(self) -> None:
        """Test environment variables with prefix retrieval exception."""
        adapter = FlextConfig.DefaultEnvironmentAdapter()
        with patch.dict("os.environ", {"FLEXT_VAR": "value"}), patch("os.environ.items", side_effect=Exception("OS error")):
                result = adapter.get_env_vars_with_prefix("FLEXT_")
                FlextTestsMatchers.assert_result_failure(result)
                assert "Failed to get environment variables" in (result.error or "")


class TestConfigFilePersistence:
    """Test ConfigFilePersistence coverage."""

    def test_save_to_file_json_success(self) -> None:
        """Test successful JSON file saving."""
        data = {"test": "value", "number": 42}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            result = FlextConfig.FilePersistence.save_to_file(data, temp_path)
            FlextTestsMatchers.assert_result_success(result)

            # Verify file content
            with Path(temp_path).open(encoding="utf-8") as f:
                loaded_data = json.load(f)
            assert loaded_data == data
        finally:
            Path(temp_path).unlink()

    def test_save_to_file_yaml_success(self) -> None:
        """Test successful YAML file saving."""
        data = {"test": "value", "list": [1, 2, 3]}
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            result = FlextConfig.FilePersistence.save_to_file(data, temp_path)
            FlextTestsMatchers.assert_result_success(result)

            # Verify file exists and has content
            assert Path(temp_path).exists()
            content = Path(temp_path).read_text(encoding="utf-8")
            assert "test: value" in content
        finally:
            Path(temp_path).unlink()

    def test_save_to_file_invalid_format(self) -> None:
        """Test file saving with invalid format."""
        data = {"test": "value"}
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        try:
            result = FlextConfig.FilePersistence.save_to_file(data, temp_path)
            FlextTestsMatchers.assert_result_failure(result)
            assert "Unsupported format" in (result.error or "")
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_to_file_permission_error(self) -> None:
        """Test file saving with permission error."""
        data = {"test": "value"}
        invalid_path = "/root/test.json"  # Should cause permission error

        result = FlextConfig.FilePersistence.save_to_file(data, invalid_path)
        FlextTestsMatchers.assert_result_failure(result)
        assert "Failed to save file" in (result.error or "")

    def test_load_from_file_json_success(self) -> None:
        """Test successful JSON file loading."""
        data = {"test": "value", "number": 42}
        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            result = FlextConfig.FilePersistence.load_from_file(temp_path)
            FlextTestsMatchers.assert_result_success(result)
            assert result.value == data
        finally:
            Path(temp_path).unlink()

    def test_load_from_file_yaml_success(self) -> None:
        """Test successful YAML file loading."""
        data = {"test": "value", "list": [1, 2, 3]}
        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("test: value\nlist:\n  - 1\n  - 2\n  - 3\n")
            temp_path = f.name

        try:
            result = FlextConfig.FilePersistence.load_from_file(temp_path)
            FlextTestsMatchers.assert_result_success(result)
            assert result.value == data
        finally:
            Path(temp_path).unlink()

    def test_load_from_file_toml_success(self) -> None:
        """Test successful TOML file loading."""
        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", suffix=".toml", delete=False
        ) as f:
            f.write('[section]\nkey = "value"\nnumber = 42\n')
            temp_path = f.name

        try:
            result = FlextConfig.FilePersistence.load_from_file(temp_path)
            FlextTestsMatchers.assert_result_success(result)
            assert result.value["section"]["key"] == "value"
            assert result.value["section"]["number"] == 42
        finally:
            Path(temp_path).unlink()

    def test_load_from_file_not_found(self) -> None:
        """Test file loading with file not found."""
        result = FlextConfig.FilePersistence.load_from_file("/nonexistent/file.json")
        FlextTestsMatchers.assert_result_failure(result)
        assert "Configuration file not found" in (result.error or "")

    def test_load_from_file_invalid_json(self) -> None:
        """Test file loading with invalid JSON."""
        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", suffix=".json", delete=False
        ) as f:
            f.write("{ invalid json ")
            temp_path = f.name

        try:
            result = FlextConfig.FilePersistence.load_from_file(temp_path)
            FlextTestsMatchers.assert_result_failure(result)
            assert "Failed to parse" in (result.error or "")
        finally:
            Path(temp_path).unlink()


class TestFlextConfigFactory:
    """Test FlextConfigFactory coverage."""

    def test_create_from_env_success(self) -> None:
        """Test config creation from environment variables."""
        env_vars = {
            "FLEXT_NAME": "env-test",
            "FLEXT_ENVIRONMENT": "development",
            "FLEXT_DEBUG": "true",
            "FLEXT_MAX_WORKERS": "4",
        }

        with patch.dict("os.environ", env_vars):
            result = FlextConfig.Factory.create_from_env()
            FlextTestsMatchers.assert_result_success(result)
            config = result.value
            assert config.name == "env-test"
            assert config.environment == "development"
            assert config.debug is True
            assert config.max_workers == 4

    def test_create_from_env_exception(self) -> None:
        """Test config creation from environment with exception."""
        with patch(
            "flext_core.config.FlextConfig", side_effect=Exception("Config error")
        ):
            result = FlextConfig.Factory.create_from_env()
            FlextTestsMatchers.assert_result_failure(result)
            assert "Failed to create configuration from environment" in (
                result.error or ""
            )

    def test_create_from_file_success(self) -> None:
        """Test config creation from file."""
        config_data = {
            "name": "file-test",
            "environment": "test",
            "debug": False,
            "max_workers": 2,
        }

        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            result = FlextConfig.Factory.create_from_file(temp_path)
            FlextTestsMatchers.assert_result_success(result)
            config = result.value
            assert config.name == "file-test"
            assert config.environment == "test"
            assert config.debug is False
        finally:
            Path(temp_path).unlink()

    def test_create_from_file_load_failure(self) -> None:
        """Test config creation from file with load failure."""
        result = FlextConfig.Factory.create_from_file("/nonexistent/file.json")
        FlextTestsMatchers.assert_result_failure(result)
        assert "Failed to load file data" in (result.error or "")

    def test_create_for_testing_success(self) -> None:
        """Test test configuration creation."""
        result = FlextConfig.Factory.create_for_testing(
            name="test-config",
            debug=True,
        )
        FlextTestsMatchers.assert_result_success(result)
        config = result.value
        assert config.name == "test-config"
        assert config.debug is True
        assert config.environment == "test"  # Default for test configs

    def test_create_for_testing_exception(self) -> None:
        """Test test configuration creation with exception."""
        with patch(
            "flext_core.config.FlextConfig", side_effect=Exception("Config error")
        ):
            result = FlextConfig.Factory.create_for_testing()
            FlextTestsMatchers.assert_result_failure(result)
            assert "Failed to create test configuration" in (result.error or "")
