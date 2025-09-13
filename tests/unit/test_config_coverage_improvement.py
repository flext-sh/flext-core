"""Test module for FlextConfig coverage improvement.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from flext_core import FlextConfig
from flext_tests import FlextTestsMatchers


class TestFlextConfigCoverageImprovement:
    """Tests to improve FlextConfig coverage systematically."""

    def test_basic_config_singleton_pattern(self) -> None:
        """Test basic config singleton pattern."""
        # Test singleton pattern
        config1 = FlextConfig.get_global_instance()
        config2 = FlextConfig.get_global_instance()
        assert config1 is config2
        assert isinstance(config1, FlextConfig)

    def test_config_field_access(self) -> None:
        """Test accessing configuration fields."""
        config = FlextConfig.get_global_instance()

        # Test basic field access
        assert hasattr(config, "app_name")
        assert hasattr(config, "version")
        assert hasattr(config, "debug")
        assert hasattr(config, "environment")

        # Test actual values
        app_name = config.app_name
        version = config.version
        debug = config.debug
        environment = config.environment

        assert isinstance(app_name, str)
        assert isinstance(version, str)
        assert isinstance(debug, bool)
        assert isinstance(environment, str)

    def test_config_validation_methods(self) -> None:
        """Test configuration validation methods."""
        config = FlextConfig.get_global_instance()

        # Test environment validation (returns validated string value)
        result = FlextConfig.validate_environment(config.environment)
        assert isinstance(result, str)
        assert result == config.environment

        # Test debug validation (returns validated bool value)
        result = FlextConfig.validate_debug(config.debug)
        assert isinstance(result, bool)
        assert result == config.debug

        # Test log level validation (returns validated string value)
        result = FlextConfig.validate_log_level(config.log_level)
        assert isinstance(result, str)
        assert result == config.log_level

    def test_config_create_class_method(self) -> None:
        """Test config creation class method."""
        # Test create method
        result = FlextConfig.create()
        FlextTestsMatchers.assert_result_success(result)

        new_config = result.unwrap()
        assert isinstance(new_config, FlextConfig)

    def test_config_create_from_environment(self) -> None:
        """Test creating config from environment."""
        # Test create from environment
        result = FlextConfig.create_from_environment()
        FlextTestsMatchers.assert_result_success(result)

        config = result.unwrap()
        assert isinstance(config, FlextConfig)

    def test_config_sealing_functionality(self) -> None:
        """Test configuration sealing."""
        config = FlextConfig.get_global_instance()

        # Test sealing (this might modify global state, so be careful)
        initially_sealed = config.is_sealed()
        assert isinstance(initially_sealed, bool)

        # Test seal operation
        seal_result = config.seal()
        FlextTestsMatchers.assert_result_success(seal_result)

        # Test that it's now sealed
        now_sealed = config.is_sealed()
        assert now_sealed is True

    def test_config_persistence_operations(self) -> None:
        """Test configuration save/load operations."""
        config = FlextConfig.get_global_instance()

        # Test save to file
        with tempfile.NamedTemporaryFile(
            encoding="utf-8", mode="w", delete=False, suffix=".json"
        ) as tmp:
            tmp_path = tmp.name

        try:
            save_result = config.save_to_file(tmp_path)
            FlextTestsMatchers.assert_result_success(save_result)

            # Test load from file
            load_result = FlextConfig.load_from_file(tmp_path)
            FlextTestsMatchers.assert_result_success(load_result)

            loaded_config = load_result.unwrap()
            assert isinstance(loaded_config, FlextConfig)

        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()

    def test_config_business_validation(self) -> None:
        """Test business validation methods."""
        config = FlextConfig.get_global_instance()

        # Test business rules validation
        result = config.validate_business_rules()
        FlextTestsMatchers.assert_result_success(result)

        # Test runtime requirements validation
        result = config.validate_runtime_requirements()
        FlextTestsMatchers.assert_result_success(result)

        # Test validate all
        result = config.validate_all()
        FlextTestsMatchers.assert_result_success(result)

    def test_config_merge_operations(self) -> None:
        """Test configuration merge operations."""
        config = FlextConfig.get_global_instance()

        # Test merge configs class method
        override_data = {"app_name": "merged_app", "debug": True}
        base_data = config.to_dict()
        merge_result = FlextConfig.merge_configs(base_data, override_data)
        FlextTestsMatchers.assert_result_success(merge_result)

        # Test class method merge
        class_merge_result = FlextConfig.merge(config, override_data)
        FlextTestsMatchers.assert_result_success(class_merge_result)

    def test_config_api_payload_methods(self) -> None:
        """Test API payload generation."""
        config = FlextConfig.get_global_instance()

        # Test to_api_payload
        api_payload_result = config.to_api_payload()
        FlextTestsMatchers.assert_result_success(api_payload_result)
        payload = api_payload_result.unwrap()
        assert isinstance(payload, dict)

        # Test as_api_payload (returns FlextResult)
        as_api_result = config.as_api_payload()
        assert as_api_result.is_success
        api_payload = as_api_result.unwrap()
        assert isinstance(api_payload, dict)

        # Test to_dict
        dict_result = config.to_dict()
        assert isinstance(dict_result, dict)

        # Test to_json (returns JSON string)
        json_result = config.to_json()
        assert isinstance(json_result, str)
        assert len(json_result) > 0
        # json_result is already a string, no need to unwrap
        assert '"app_name"' in json_result  # Verify it contains expected JSON content

    def test_config_environment_adapter(self) -> None:
        """Test environment adapter functionality."""
        config = FlextConfig.get_global_instance()

        # Test get_env_var returns FlextResult
        result = config.get_env_var("FLEXT_DEBUG")
        assert hasattr(result, "is_success")  # Check it's a FlextResult
        # The result will likely be a failure since FLEXT_DEBUG isn't set
        if result.is_failure:
            assert "not found" in result.error

    def test_config_validation_edge_cases(self) -> None:
        """Test validation edge cases."""
        # Test validate_config_value
        result = FlextConfig.validate_config_value("test_app", str)
        FlextTestsMatchers.assert_result_success(result)

        # Test consistency validation

    def test_config_host_and_url_validation(self) -> None:
        """Test host and URL validation methods."""
        config = FlextConfig.get_global_instance()

        # Test host validation
        # Test host validation - field validators return validated strings
        host_result = FlextConfig.validate_host(config.host)
        assert isinstance(host_result, str)

        # Test base URL validation
        url_result = FlextConfig.validate_base_url(config.base_url)
        assert isinstance(url_result, str)

    def test_config_integer_validations(self) -> None:
        """Test integer validation methods."""
        config = FlextConfig.get_global_instance()

        # Test that positive integer fields are correctly set
        assert config.port > 0
        assert config.max_workers > 0

        # Test that timeout field is non-negative
        assert config.timeout_seconds >= 0

    def test_config_safe_operations(self) -> None:
        """Test safe configuration operations."""
        config = FlextConfig.get_global_instance()

        # Test safe_load
        safe_result = config.safe_load({})
        FlextTestsMatchers.assert_result_success(safe_result)

    def test_config_metadata_operations(self) -> None:
        """Test metadata operations."""
        config = FlextConfig.get_global_instance()

        # Test get_metadata
        metadata = config.get_metadata()
        assert isinstance(metadata, dict)

    def test_config_source_validation(self) -> None:
        """Test config source validation."""
        config = FlextConfig.get_global_instance()

        # Test config source validation (returns validated string value)
        result = FlextConfig.validate_config_source(config.config_source)
        assert isinstance(result, str)
        assert result == config.config_source
