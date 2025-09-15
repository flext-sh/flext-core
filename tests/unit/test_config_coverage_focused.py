"""Focused FlextConfig coverage tests targeting specific uncovered lines.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from flext_core import FlextConfig
from flext_core.result import FlextResult


class TestFlextConfigCoverageFocused:
    """Focused tests for FlextConfig coverage improvement."""

    def test_config_validation_methods(self) -> None:
        """Test basic validation methods."""
        config = FlextConfig()

        # Test validate_all method
        result = config.validate_all()
        assert result.is_success

        # Test individual validation methods
        result = config.validate_runtime_requirements()
        assert result.is_success

        result = config.validate_business_rules()
        assert result.is_success

    def test_config_sealing(self) -> None:
        """Test configuration sealing functionality."""
        config = FlextConfig()

        # Initially not sealed
        assert not config.is_sealed()

        # Seal the configuration
        seal_result = config.seal()
        assert seal_result.is_success
        assert config.is_sealed()

    def test_default_environment_adapter(self) -> None:
        """Test DefaultEnvironmentAdapter functionality."""
        adapter = FlextConfig.DefaultEnvironmentAdapter()

        # Test with existing environment variable
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = adapter.get_env_var("TEST_VAR")
            assert result.is_success
            assert result.value == "test_value"

        # Test with missing variable
        result = adapter.get_env_var("MISSING_VAR")
        assert result.is_failure

        # Test get_env_vars_with_prefix
        with patch.dict(
            os.environ, {"PREFIX_VAR1": "val1", "PREFIX_VAR2": "val2", "OTHER": "other"}
        ):
            prefix_result: FlextResult[dict[str, object]] = (
                adapter.get_env_vars_with_prefix("PREFIX_")
            )
            assert prefix_result.is_success
            env_vars = prefix_result.value
            assert "VAR1" in env_vars
            assert "VAR2" in env_vars
            assert "OTHER" not in env_vars

    def test_config_factory_methods(self) -> None:
        """Test Factory class methods."""
        factory = FlextConfig.Factory()

        # Test create_from_env
        result = factory.create_from_env()
        assert result.is_success

        # Test create_for_testing
        result = factory.create_for_testing()
        assert result.is_success
        config = result.value
        assert config.environment == "test"

    def test_config_file_operations(self) -> None:
        """Test file save/load operations."""
        config = FlextConfig(app_name="file_test", environment="test")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            # Test save_to_file
            save_result = config.save_to_file(temp_path)
            assert save_result.is_success

            # Test load_from_file
            load_result = FlextConfig.load_from_file(temp_path)
            assert load_result.is_success
            loaded_config = load_result.value
            assert loaded_config.app_name == "file_test"

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_config_global_instance(self) -> None:
        """Test global instance management."""
        # Clear any existing global instance
        FlextConfig.clear_global_instance()

        # Get global instance (should create new one)
        global_config = FlextConfig.get_global_instance()
        assert isinstance(global_config, FlextConfig)

        # Get again (should return same instance)
        same_config = FlextConfig.get_global_instance()
        assert global_config is same_config

    def test_config_serialization(self) -> None:
        """Test serialization methods."""
        config = FlextConfig(app_name="serialize_test", debug=True)

        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict["app_name"] == "serialize_test"
        assert config_dict["debug"] is True

        # Test to_json
        json_str = config.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["app_name"] == "serialize_test"

    def test_config_metadata(self) -> None:
        """Test metadata functionality."""
        config = FlextConfig()
        metadata = config.get_metadata()
        assert isinstance(metadata, dict)

    def test_config_create_from_environment(self) -> None:
        """Test create_from_environment class method."""
        with patch.dict(os.environ, {"FLEXT_APP_NAME": "env_test"}):
            result = FlextConfig.create_from_environment()
            assert result.is_success

    def test_config_validation_edge_cases(self) -> None:
        """Test validation methods with edge cases."""
        config = FlextConfig()

        # Test validate_environment with valid values
        assert config.validate_environment("production") == "production"
        assert config.validate_environment("development") == "development"
        assert config.validate_environment("test") == "test"

        # Test validate_debug
        assert config.validate_debug(True) is True
        assert config.validate_debug(False) is False

        # Test validate_log_level
        assert config.validate_log_level("INFO") == "INFO"
        assert config.validate_log_level("DEBUG") == "DEBUG"

    def test_config_nested_validator_classes(self) -> None:
        """Test nested validator classes."""
        config = FlextConfig()

        # Test RuntimeValidator
        runtime_validator = FlextConfig.RuntimeValidator()
        result = runtime_validator.validate_runtime_requirements(config)
        assert result.is_success

        # Test BusinessValidator
        business_validator = FlextConfig.BusinessValidator()
        result = business_validator.validate_business_rules(config)
        assert result.is_success

    def test_config_file_persistence_class(self) -> None:
        """Test FilePersistence nested class."""
        config = FlextConfig(app_name="persistence_test")
        persistence = FlextConfig.FilePersistence()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            # Test FilePersistence save_to_file
            save_result = persistence.save_to_file(config, temp_path)
            assert save_result.is_success

            # Test FilePersistence load_from_file
            load_result = persistence.load_from_file(temp_path)
            assert load_result.is_success

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_config_env_var_methods(self) -> None:
        """Test environment variable methods."""
        config = FlextConfig()

        with patch.dict(os.environ, {"TEST_CONFIG_VAR": "config_value"}):
            result = config.get_env_var("TEST_CONFIG_VAR")
            assert result.is_success
            assert result.value == "config_value"

    def test_config_validation_consistency(self) -> None:
        """Test validation consistency method."""
        config = FlextConfig()
        # Should not raise exception for valid config
        # validate_configuration_consistency is a Pydantic model validator that runs automatically

    def test_config_error_handling(self) -> None:
        """Test error handling paths."""
        # Test loading from non-existent file
        result = FlextConfig.load_from_file("/non/existent/file.json")
        assert result.is_failure

        # Test factory create_from_file with non-existent file
        factory = FlextConfig.Factory()
        result = factory.create_from_file("/non/existent/file.json")
        assert result.is_failure

    def test_config_positive_validation_methods(self) -> None:
        """Test numeric validation methods."""
        config = FlextConfig()

        # Test validate_positive_integers
        assert config.validate_positive_integers(5) == 5
        assert config.validate_positive_integers(1) == 1

        # Test validate_non_negative_integers
        assert config.validate_non_negative_integers(0) == 0
        assert config.validate_non_negative_integers(10) == 10

    def test_config_host_and_url_validation(self) -> None:
        """Test host and URL validation methods."""
        config = FlextConfig()

        # Test validate_host
        assert config.validate_host("localhost") == "localhost"
        assert config.validate_host("127.0.0.1") == "127.0.0.1"

        # Test validate_base_url
        assert config.validate_base_url("http://localhost") == "http://localhost"
        assert config.validate_base_url("https://example.com") == "https://example.com"

    def test_config_source_validation(self) -> None:
        """Test config source validation."""
        config = FlextConfig()

        # Test valid config sources
        assert config.validate_config_source("env") == "env"
        assert config.validate_config_source("file") == "file"
        assert config.validate_config_source("cli") == "cli"
