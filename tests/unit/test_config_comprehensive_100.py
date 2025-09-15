"""Comprehensive unit tests for FlextConfig targeting 100% coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

from flext_core import FlextConfig, FlextResult
from flext_tests import FlextTestsMatchers


class TestFlextConfigComprehensive100:
    """Comprehensive FlextConfig tests targeting 100% coverage."""

    def test_config_validator_validate_runtime_requirements(self) -> None:
        """Test FlextConfig.ConfigValidator.validate_runtime_requirements method."""
        config = FlextConfig()
        # Test successful runtime validation
        runtime_result = FlextConfig.RuntimeValidator.validate_runtime_requirements(
            config
        )
        FlextTestsMatchers.assert_result_success(runtime_result)

    def test_config_validator_validate_business_rules(self) -> None:
        """Test FlextConfig.ConfigValidator.validate_business_rules method."""
        config = FlextConfig()
        # Test successful business validation
        business_result = FlextConfig.BusinessValidator.validate_business_rules(config)
        FlextTestsMatchers.assert_result_success(business_result)

    def test_config_persistence_save_to_file(self) -> None:
        """Test FlextConfig.ConfigPersistence.save_to_file method."""
        config = FlextConfig(app_name="test_app", environment="test")
        persistence = FlextConfig.FilePersistence()

        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", delete=False, suffix=".json"
        ) as f:
            temp_path = f.name

        try:
            # Test successful save
            save_result = persistence.save_to_file(config, temp_path)
            FlextTestsMatchers.assert_result_success(save_result)

            # Verify file was created and has content
            assert Path(temp_path).exists()
            with Path(temp_path).open(encoding="utf-8") as f:
                saved_data = json.load(f)
                assert saved_data["app_name"] == "test_app"
                assert saved_data["environment"] == "test"

        finally:
            if Path(temp_path).exists():
                Path(temp_path).unlink()

    def test_config_persistence_load_from_file(self) -> None:
        """Test FlextConfig.FilePersistence.load_from_file method."""
        persistence = FlextConfig.FilePersistence()

        # Create test config data
        config_data = {
            "app_name": "loaded_app",
            "environment": "production",
            "debug": False,
            "log_level": "INFO",
        }

        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", delete=False, suffix=".json"
        ) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            # Test FilePersistence.load_from_file (returns dict data)
            load_result = persistence.load_from_file(temp_path)
            FlextTestsMatchers.assert_result_success(load_result)

            loaded_data = cast("dict", load_result.value)
            assert loaded_data["app_name"] == "loaded_app"
            assert loaded_data["environment"] == "production"
            assert loaded_data["debug"] is False
            assert loaded_data["log_level"] == "INFO"

            # Test FlextConfig.load_from_file (returns FlextConfig object)
            config_result = FlextConfig.load_from_file(temp_path)
            FlextTestsMatchers.assert_result_success(config_result)

            loaded_config = cast("FlextConfig", config_result.value)
            assert loaded_config.app_name == "loaded_app"
            assert loaded_config.environment == "production"
            assert loaded_config.debug is False
            assert loaded_config.log_level == "INFO"

        finally:
            if Path(temp_path).exists():
                Path(temp_path).unlink()

    def test_config_factory_create_from_file(self) -> None:
        """Test FlextConfig.Factory.create_from_file method."""
        factory = FlextConfig.Factory()

        # Create a temporary config file
        config_data = {
            "app_name": "file_test_app",
            "environment": "test",
            "debug": True,
            "port": 9000,
        }

        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", delete=False, suffix=".json"
        ) as f:
            json.dump(config_data, f)
            temp_file = f.name

        try:
            # Test config creation from file
            file_result = factory.create_from_file(temp_file)
            FlextTestsMatchers.assert_result_success(file_result)

            file_config = cast("FlextConfig", file_result.value)
            assert file_config.app_name == "file_test_app"
            assert file_config.environment == "test"
            assert file_config.debug is True
            assert file_config.port == 9000

        finally:
            if Path(temp_file).exists():
                Path(temp_file).unlink()

    def test_config_factory_create_for_testing(self) -> None:
        """Test FlextConfig.Factory.create_for_testing method."""
        factory = FlextConfig.Factory()

        # Test config creation for testing with overrides
        test_config_result = factory.create_for_testing(
            app_name="test_app", environment="test", debug=True
        )
        FlextTestsMatchers.assert_result_success(test_config_result)

        test_config = cast("FlextConfig", test_config_result.value)
        assert test_config.app_name == "test_app"
        assert test_config.environment == "test"
        assert test_config.debug is True

    def test_environment_config_adapter_get_env_var(self) -> None:
        """Test FlextConfig.EnvironmentConfigAdapter.get_env_var method."""
        adapter = FlextConfig.DefaultEnvironmentAdapter()

        # Test with existing environment variable
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = adapter.get_env_var("TEST_VAR")
            FlextTestsMatchers.assert_result_success(result, "test_value")

        # Test with non-existent variable (should fail)
        result = adapter.get_env_var("NON_EXISTENT_VAR")
        FlextTestsMatchers.assert_result_failure(result)

    def test_environment_config_adapter_get_env_vars_with_prefix(self) -> None:
        """Test FlextConfig.EnvironmentConfigAdapter.get_env_vars_with_prefix method."""
        adapter = FlextConfig.DefaultEnvironmentAdapter()

        # Test with environment variables with prefix
        with patch.dict(
            os.environ,
            {
                "FLEXT_APP_NAME": "test_app",
                "FLEXT_DEBUG": "true",
                "FLEXT_LOG_LEVEL": "DEBUG",
                "OTHER_VAR": "should_not_match",
            },
        ):
            env_vars_result = adapter.get_env_vars_with_prefix("FLEXT_")
            FlextTestsMatchers.assert_result_success(env_vars_result)
            env_vars = cast("dict[str, str]", env_vars_result.value)
            # Note: prefix is stripped, so keys don't include FLEXT_ prefix
            assert env_vars["APP_NAME"] == "test_app"
            assert env_vars["DEBUG"] == "true"
            assert env_vars["LOG_LEVEL"] == "DEBUG"
            assert "OTHER_VAR" not in env_vars

    def test_default_environment_adapter_get_env_var_with_validation(self) -> None:
        """Test FlextConfig.DefaultEnvironmentAdapter.get_env_var method."""
        adapter = FlextConfig.DefaultEnvironmentAdapter()

        # Test successful environment variable retrieval
        with patch.dict(os.environ, {"TEST_VALIDATED_VAR": "test_value"}):
            result = adapter.get_env_var("TEST_VALIDATED_VAR")
            FlextTestsMatchers.assert_result_success(result)
            assert cast("str", result.value) == "test_value"

        # Test with missing environment variable (should fail)
        missing_result = adapter.get_env_var("MISSING_VAR")
        FlextTestsMatchers.assert_result_failure(missing_result)

    def test_default_environment_adapter_get_env_vars_with_prefix(self) -> None:
        """Test FlextConfig.DefaultEnvironmentAdapter.get_env_vars_with_prefix method."""
        adapter = FlextConfig.DefaultEnvironmentAdapter()

        # Test environment variables with prefix
        with patch.dict(
            os.environ,
            {
                "APP_NAME": "test_app",
                "APP_VERSION": "1.0.0",
                "APP_DEBUG": "false",
                "OTHER_CONFIG": "ignored",
            },
        ):
            result = adapter.get_env_vars_with_prefix("APP_")
            FlextTestsMatchers.assert_result_success(result)

            env_vars = cast("dict[str, str]", result.value)
            assert env_vars["NAME"] == "test_app"  # Prefix stripped
            assert env_vars["VERSION"] == "1.0.0"  # Prefix stripped
            assert env_vars["DEBUG"] == "false"  # Prefix stripped
            assert "OTHER_CONFIG" not in env_vars

    def test_runtime_validator_comprehensive(self) -> None:
        """Test FlextConfig.RuntimeValidator.validate_runtime_requirements method."""
        validator = FlextConfig.RuntimeValidator()

        # Test with valid production config
        prod_config = FlextConfig(
            environment="production",
            debug=False,
            database_url="postgresql://user:pass@localhost:5432/prod",
        )

        runtime_result = validator.validate_runtime_requirements(prod_config)
        FlextTestsMatchers.assert_result_success(runtime_result)

        # Test with invalid config (production with insufficient workers)
        invalid_config = FlextConfig(
            environment="production",
            debug=False,
            max_workers=1,  # Below minimum production workers (2)
        )

        invalid_result = validator.validate_runtime_requirements(invalid_config)
        FlextTestsMatchers.assert_result_failure(
            cast("FlextResult[Any]", invalid_result)
        )

    def test_business_validator_comprehensive(self) -> None:
        """Test FlextConfig.BusinessValidator.validate_business_rules method."""
        validator = FlextConfig.BusinessValidator()

        # Test with valid business configuration
        valid_config = FlextConfig(
            max_workers=10, timeout_seconds=30, port=8080, database_pool_size=20
        )

        business_result = validator.validate_business_rules(valid_config)
        FlextTestsMatchers.assert_result_success(business_result)

        # Test with invalid business rules (port conflicts)
        # Test with invalid business rules (production with debug=True and non-default source)
        invalid_config = FlextConfig(
            environment="production",
            debug=True,  # Invalid: debug in production
            config_source="cli",  # Non-default source triggers the check
        )

        invalid_result = validator.validate_business_rules(invalid_config)
        FlextTestsMatchers.assert_result_failure(
            cast("FlextResult[Any]", invalid_result)
        )

    def test_file_persistence_save_to_file_error_handling(self) -> None:
        """Test FlextConfig.FilePersistence.save_to_file error handling."""
        persistence = FlextConfig.FilePersistence()
        config = FlextConfig(app_name="test")

        # Test save to invalid path
        invalid_path = "/invalid/nonexistent/path/config.json"
        save_result = persistence.save_to_file(config, invalid_path)
        FlextTestsMatchers.assert_result_failure(cast("FlextResult[Any]", save_result))

    def test_file_persistence_load_from_file_error_handling(self) -> None:
        """Test FlextConfig.FilePersistence.load_from_file error handling."""
        persistence = FlextConfig.FilePersistence()

        # Test load from non-existent file
        missing_file_result = persistence.load_from_file("nonexistent.json")
        FlextTestsMatchers.assert_result_failure(missing_file_result)

        # Test load from invalid JSON file
        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", delete=False, suffix=".json"
        ) as f:
            f.write("invalid json content {")
            invalid_json_path = f.name

        try:
            invalid_result = persistence.load_from_file(invalid_json_path)
            FlextTestsMatchers.assert_result_failure(invalid_result)
        finally:
            if Path(invalid_json_path).exists():
                Path(invalid_json_path).unlink()

    def test_factory_create_from_env_comprehensive(self) -> None:
        """Test FlextConfig.Factory.create_from_env method."""
        factory = FlextConfig.Factory()

        # Test creation from environment variables
        with patch.dict(
            os.environ,
            {
                "FLEXT_APP_NAME": "env_app",
                "FLEXT_ENVIRONMENT": "test",
                "FLEXT_DEBUG": "true",
                "FLEXT_LOG_LEVEL": "DEBUG",
                "FLEXT_PORT": "8080",
                "FLEXT_MAX_WORKERS": "5",
            },
        ):
            env_result = factory.create_from_env()
            FlextTestsMatchers.assert_result_success(env_result)

            env_config = cast("FlextConfig", env_result.value)
            assert env_config.app_name == "env_app"
            assert env_config.environment == "test"
            assert env_config.debug is True
            assert env_config.log_level == "DEBUG"
            assert env_config.port == 8080
            assert env_config.max_workers == 5

    def test_factory_create_from_file_comprehensive(self) -> None:
        """Test FlextConfig.Factory.create_from_file method."""
        factory = FlextConfig.Factory()

        # Test creation from JSON file
        config_data = {
            "app_name": "file_app",
            "environment": "staging",
            "debug": False,
            "host": "staging.example.com",
            "port": 9000,
            "database_url": "postgresql://user:pass@db:5432/staging",
        }

        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", delete=False, suffix=".json"
        ) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            file_result = factory.create_from_file(config_file)
            FlextTestsMatchers.assert_result_success(file_result)

            file_config = cast("FlextConfig", file_result.value)
            assert file_config.app_name == "file_app"
            assert file_config.environment == "staging"
            assert file_config.debug is False
            assert file_config.host == "staging.example.com"
            assert file_config.port == 9000
            assert file_config.database_url == "postgresql://user:pass@db:5432/staging"

        finally:
            if Path(config_file).exists():
                Path(config_file).unlink()

    def test_factory_create_for_testing_comprehensive(self) -> None:
        """Test FlextConfig.Factory.create_for_testing method."""
        factory = FlextConfig.Factory()

        # Test testing configuration creation
        test_overrides = {
            "database_url": "sqlite:///test.db",
            "debug": True,
            "log_level": "DEBUG",
            "cache_enabled": False,
        }

        test_result = factory.create_for_testing(**test_overrides)
        FlextTestsMatchers.assert_result_success(test_result)

        test_config = cast("FlextConfig", test_result.value)
        assert test_config.environment == "test"
        assert test_config.database_url == "sqlite:///test.db"
        assert test_config.debug is True
        assert test_config.log_level == "DEBUG"
        assert test_config.cache_enabled is False

    def test_config_validation_methods_comprehensive(self) -> None:
        """Test FlextConfig validation methods comprehensively."""
        # Test validate_environment
        valid_env_result = FlextConfig.validate_environment("production")
        assert valid_env_result == "production"

        with pytest.raises(ValueError, match="Invalid environment"):
            FlextConfig.validate_environment("invalid_env")

        # Test validate_debug
        debug_result = FlextConfig.validate_debug(True)
        assert debug_result is True

        # Test validate_log_level
        log_level_result = FlextConfig.validate_log_level("ERROR")
        assert log_level_result == "ERROR"

        with pytest.raises(ValueError, match="Invalid log_level"):
            FlextConfig.validate_log_level("INVALID")

        # Test validate_config_source
        config_source_result = FlextConfig.validate_config_source("env")
        assert config_source_result == "env"

        with pytest.raises(ValueError, match="Config source must be one of"):
            FlextConfig.validate_config_source("invalid_source")

        # Test validate_positive_integers
        positive_result = FlextConfig.validate_positive_integers(10)
        assert positive_result == 10

        with pytest.raises(ValueError, match="must be positive"):
            FlextConfig.validate_positive_integers(-1)

        # Test validate_non_negative_integers
        non_negative_result = FlextConfig.validate_non_negative_integers(0)
        assert non_negative_result == 0

        # Test validate_host
        host_result = FlextConfig.validate_host("localhost")
        assert host_result == "localhost"

        with pytest.raises(ValueError, match="Host cannot be empty"):
            FlextConfig.validate_host("")

        # Test validate_base_url
        base_url_result = FlextConfig.validate_base_url("https://api.example.com")
        assert base_url_result == "https://api.example.com"

        with pytest.raises(ValueError, match="Base URL must start with"):
            FlextConfig.validate_base_url("invalid-url")

    def test_config_validation_consistency(self) -> None:
        """Test FlextConfig.validate_configuration_consistency method."""
        # Test valid configuration
        valid_config = FlextConfig(
            environment="production",
            debug=False,
            enable_metrics=True,
            port=8080,
            metrics_port=9090,
        )

        # Test successful validation (should not raise)
        try:
            consistency_result = valid_config.validate_configuration_consistency()
            assert consistency_result == valid_config  # Should return self
        except Exception as e:
            pytest.fail(f"Validation should not fail for valid config: {e}")

        # Test inconsistent configuration (should raise during construction)
        with pytest.raises(ValueError, match=r"Log level.*too restrictive"):
            FlextConfig(environment="development", log_level="CRITICAL")

    def test_config_global_instance_management(self) -> None:
        """Test global instance management methods."""
        # Clear any existing global instance
        FlextConfig.clear_global_instance()

        # Test get_global_instance when none exists
        initial_global = FlextConfig.get_global_instance()
        assert initial_global is not None
        assert isinstance(initial_global, FlextConfig)

        # Test that subsequent calls return the same instance
        second_global = FlextConfig.get_global_instance()
        assert second_global is initial_global

        # Test set_global_instance
        custom_config = FlextConfig(app_name="custom_global")
        FlextConfig.set_global_instance(custom_config)

        retrieved_custom = FlextConfig.get_global_instance()
        assert retrieved_custom is custom_config
        assert retrieved_custom.app_name == "custom_global"

        # Test clear_global_instance
        FlextConfig.clear_global_instance()
        new_global = FlextConfig.get_global_instance()
        assert new_global is not custom_config

    def test_config_create_class_method_comprehensive(self) -> None:
        """Test FlextConfig.create class method comprehensively."""
        # Test basic creation
        basic_constants = {
            "app_name": "create_test",
            "environment": "test",
            "debug": True,
        }

        basic_result = FlextConfig.create(constants=basic_constants)
        FlextTestsMatchers.assert_result_success(basic_result)

        basic_config = cast("FlextConfig", basic_result.value)
        assert basic_config.app_name == "create_test"
        assert basic_config.environment == "test"
        assert basic_config.debug is True

        # Test creation with CLI overrides
        cli_overrides = {"port": 9090, "log_level": "ERROR"}

        cli_result = FlextConfig.create(
            constants=basic_constants, cli_overrides=cli_overrides
        )
        FlextTestsMatchers.assert_result_success(cli_result)

        cli_config = cast("FlextConfig", cli_result.value)
        assert cli_config.port == 9090
        assert cli_config.log_level == "ERROR"

        # Test creation with invalid environment
        invalid_constants: dict[str, object] = {"environment": "invalid_environment"}

        invalid_result = FlextConfig.create(constants=invalid_constants)
        FlextTestsMatchers.assert_result_failure(
            cast("FlextResult[Any]", invalid_result)
        )

    def test_config_create_from_environment_class_method(self) -> None:
        """Test FlextConfig.create_from_environment class method."""
        # Test creation from environment with extra settings
        extra_settings = {"app_name": "env_test_app", "debug": False, "max_workers": 8}

        with patch.dict(
            os.environ,
            {
                "FLEXT_ENVIRONMENT": "staging",
                "FLEXT_LOG_LEVEL": "INFO",
                "FLEXT_PORT": "8080",
            },
        ):
            env_result = FlextConfig.create_from_environment(
                extra_settings=extra_settings
            )
            FlextTestsMatchers.assert_result_success(env_result)

            env_config = cast("FlextConfig", env_result.value)
            assert env_config.app_name == "env_test_app"
            assert env_config.environment == "staging"
            assert env_config.log_level == "INFO"
            assert env_config.port == 8080
            assert env_config.debug is False
            assert env_config.max_workers == 8

    def test_config_sealing_functionality(self) -> None:
        """Test configuration sealing functionality."""
        config = FlextConfig(app_name="seal_test")

        # Test initial state
        assert not config.is_sealed()

        # Test sealing
        seal_result = config.seal()
        FlextTestsMatchers.assert_result_success(seal_result)
        assert config.is_sealed()

        # Test that sealed config cannot be modified
        with pytest.raises(
            AttributeError,
            match="Cannot modify field 'app_name' - configuration is sealed",
        ):
            config.app_name = "modified_name"

    def test_config_api_payload_methods(self) -> None:
        """Test configuration API payload methods."""
        config = FlextConfig(
            app_name="api_test", environment="production", debug=False, port=8080
        )

        # Test to_api_payload
        payload_result = config.to_api_payload()
        FlextTestsMatchers.assert_result_success(payload_result)

        payload = cast("dict[str, Any]", payload_result.value)
        assert payload["app_name"] == "api_test"
        assert payload["environment"] == "production"
        assert payload["debug"] is False
        assert payload["port"] == 8080

        # Test as_api_payload (alias method)
        direct_payload_result = config.as_api_payload()
        FlextTestsMatchers.assert_result_success(direct_payload_result)

        direct_payload = cast("dict[str, Any]", direct_payload_result.value)
        assert direct_payload["app_name"] == "api_test"

    def test_config_serialization_methods(self) -> None:
        """Test configuration serialization methods."""
        config = FlextConfig(
            app_name="serialize_test", environment="test", debug=True, max_workers=4
        )

        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["app_name"] == "serialize_test"
        assert config_dict["environment"] == "test"
        assert config_dict["debug"] is True
        assert config_dict["max_workers"] == 4

        # Test to_json
        config_json = config.to_json()
        assert isinstance(config_json, str)

        # Parse and verify JSON
        parsed = json.loads(config_json)
        assert parsed["app_name"] == "serialize_test"
        assert parsed["environment"] == "test"
        assert parsed["debug"] is True
        assert parsed["max_workers"] == 4

        # Test to_json with custom parameters
        pretty_json = config.to_json(indent=2, by_alias=True)
        assert isinstance(pretty_json, str)
        assert "  " in pretty_json  # Check indentation

    def test_config_compatibility_methods(self) -> None:
        """Test configuration compatibility methods (safe_load and merge)."""
        config1 = FlextConfig(app_name="base_config", debug=False, port=8080)
        config2_dict = {"app_name": "override_config", "debug": True, "max_workers": 10}

        # Test safe_load (compatibility method - returns global instance)
        safe_load_data: dict[str, object] = {"env": "test", "log_level": "DEBUG"}
        load_result = config1.safe_load(safe_load_data)
        FlextTestsMatchers.assert_result_success(load_result)

        # safe_load is a compatibility method that returns global instance
        loaded_config = cast("FlextConfig", load_result.value)
        assert loaded_config is not None  # Should return a valid config

        # Test merge (compatibility method - returns global instance)
        merge_result = FlextConfig.merge(config1, config2_dict)
        FlextTestsMatchers.assert_result_success(merge_result)

        # merge is a compatibility method that returns global instance, not actual merge
        merged_config = cast("FlextConfig", merge_result.value)
        assert merged_config is not None  # Should return a valid config

        # Test merge_configs (actual working method)
        config1_dict = {"app_name": "app1", "debug": False, "port": 8080}
        config2_dict = {"app_name": "app2", "debug": True, "max_workers": 10}

        actual_merge_result = FlextConfig.merge_configs(config1_dict, config2_dict)
        FlextTestsMatchers.assert_result_success(actual_merge_result)

        merged_dict = cast("dict", actual_merge_result.value)
        assert merged_dict["app_name"] == "app2"  # config2 takes precedence
        assert merged_dict["debug"] is True
        assert merged_dict["port"] == 8080  # from config1
        assert merged_dict["max_workers"] == 10  # from config2

    def test_config_load_from_file_comprehensive(self) -> None:
        """Test FlextConfig.load_from_file class method comprehensively."""
        # Test JSON file loading
        json_config_data = {
            "app_name": "json_loaded_app",
            "environment": "production",
            "debug": False,
            "port": 8080,
            "database_url": "postgresql://user:pass@localhost:5432/prod",
        }

        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", delete=False, suffix=".json"
        ) as f:
            json.dump(json_config_data, f)
            json_file = f.name

        try:
            json_result = FlextConfig.load_from_file(json_file)
            FlextTestsMatchers.assert_result_success(json_result)

            json_config = cast("FlextConfig", json_result.value)
            assert json_config.app_name == "json_loaded_app"
            assert json_config.environment == "production"
            assert json_config.debug is False
            assert json_config.port == 8080

        finally:
            if Path(json_file).exists():
                Path(json_file).unlink()

        # Test file not found error
        missing_file_result = FlextConfig.load_from_file("nonexistent.json")
        FlextTestsMatchers.assert_result_failure(missing_file_result)

    def test_config_validation_all_method(self) -> None:
        """Test FlextConfig.validate_all method."""
        # Test valid configuration
        valid_config = FlextConfig(
            environment="production",
            debug=False,
            database_url="postgresql://user:pass@localhost:5432/prod",
            port=8080,
            max_workers=10,
        )

        validate_all_result = valid_config.validate_all()
        FlextTestsMatchers.assert_result_success(validate_all_result)

        # Test invalid configuration (production with debug=True and non-default config_source)
        invalid_config = FlextConfig(
            environment="production",
            debug=True,  # Debug should be False in production
            config_source="cli",  # Non-default source triggers business rule check
        )

        invalid_validate_result = invalid_config.validate_all()
        FlextTestsMatchers.assert_result_failure(
            cast("FlextResult[Any]", invalid_validate_result)
        )

    def test_config_environment_adapter_integration(self) -> None:
        """Test configuration with environment adapter integration."""
        config = FlextConfig()

        # Test get_env_var method
        with patch.dict(os.environ, {"FLEXT_TEST_VAR": "test_value"}):
            env_result = config.get_env_var("FLEXT_TEST_VAR")
            FlextTestsMatchers.assert_result_success(env_result, "test_value")

        # Test with missing environment variable
        missing_result = config.get_env_var("NONEXISTENT_VAR")
        FlextTestsMatchers.assert_result_failure(
            cast("FlextResult[Any]", missing_result)
        )

    def test_config_value_validation_comprehensive(self) -> None:
        """Test config value validation method."""
        config = FlextConfig()

        # Test valid string validation
        string_result = config.validate_config_value("test_string", str)
        FlextTestsMatchers.assert_result_success(string_result)

        # Test valid integer validation
        int_result = config.validate_config_value(42, int)
        FlextTestsMatchers.assert_result_success(int_result)

        # Test invalid type validation
        invalid_result = config.validate_config_value("string", int)
        FlextTestsMatchers.assert_result_success(invalid_result)
        assert (
            cast("bool", invalid_result.value) is False
        )  # Invalid type should return False

    def test_config_merge_configs_static_method(self) -> None:
        """Test FlextConfig.merge_configs static method."""
        config1_dict = {
            "app_name": "app1",
            "environment": "test",
            "debug": False,
            "port": 8080,
        }

        config2_dict = {
            "app_name": "app2",  # Should override
            "debug": True,  # Should override
            "max_workers": 5,  # Should be added
        }

        merge_result = FlextConfig.merge_configs(config1_dict, config2_dict)
        FlextTestsMatchers.assert_result_success(merge_result)

        merged_dict = cast("dict[str, Any]", merge_result.value)
        assert merged_dict["app_name"] == "app2"  # Overridden
        assert merged_dict["environment"] == "test"  # From config1
        assert merged_dict["debug"] is True  # Overridden
        assert merged_dict["port"] == 8080  # From config1
        assert merged_dict["max_workers"] == 5  # Added from config2
