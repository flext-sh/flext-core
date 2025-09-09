"""Strategic tests to achieve 100% coverage.

Focus on uncovered code paths and edge cases in core modules.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from flext_core import (
    FlextConfig,
    FlextTypeAdapters,
    FlextUtilities,
    FlextValidations,
)

# FlextConfigFactory is now a nested class within FlextConfig
# from flext_core.config import FlextConfigFactory  # Removed loose class


class TestConfigCoverage:
    """Strategic config tests for 100% coverage."""

    def test_config_create_basic(self) -> None:
        """Test basic config creation - covers core paths."""
        config = FlextConfig()
        assert config.name == "flext"
        assert config.environment in {
            "development",
            "production",
            "staging",
            "test",
            "local",
        }

    def test_config_with_environment_validation(self) -> None:
        """Test environment validation paths."""
        # Test valid environments
        for env in ["development", "production", "staging", "test", "local"]:
            config = FlextConfig(environment=env)
            assert config.environment == env

    def test_config_factory_create_for_testing(self) -> None:
        """Test factory method for testing configs."""
        result = FlextConfig.Factory.create_for_testing(
            name="test-config", debug=True, environment="test"
        )
        assert result.is_success
        config = result.value
        assert config.name == "test-config"
        assert config.debug is True
        assert config.environment == "test"


class TestValidationsCoverage:
    """Strategic validation tests for 100% coverage."""

    def test_basic_validation_functions(self) -> None:
        """Test basic validation utilities."""
        # Test string validation
        result = FlextValidations.validate_string("test", min_length=1, max_length=10)
        assert result.is_success

        # Test empty string
        result = FlextValidations.validate_string("", required=False)
        assert result.is_success

        # Test validation failure
        result = FlextValidations.validate_string("", required=True)
        assert result.is_failure

    def test_numeric_validation(self) -> None:
        """Test numeric validation paths."""
        # Test valid number
        result = FlextValidations.validate_number(5, min_value=1, max_value=10)
        assert result.is_success

        # Test out of range
        result = FlextValidations.validate_number(15, max_value=10)
        assert result.is_failure

    def test_email_validation(self) -> None:
        """Test email validation functionality."""
        # Valid email
        result = FlextValidations.validate_email("test@example.com")
        assert result.is_success

        # Invalid email
        result = FlextValidations.validate_email("invalid-email")
        assert result.is_failure


class TestAdaptersCoverage:
    """Strategic adapter tests for 100% coverage."""

    def test_basic_adapter_functionality(self) -> None:
        """Test basic adapter creation and usage."""
        # Test type adapter creation
        adapter = FlextTypeAdapters.Foundation.create_basic_adapter(str)
        assert adapter is not None

        # Test validation
        result = FlextTypeAdapters.Foundation.validate_with_adapter("test", str)
        assert result.is_success

    def test_type_adapter_paths(self) -> None:
        """Test type adapter paths."""
        # Test type adapter validation
        adapter = FlextTypeAdapters()
        result = adapter.adapt_type("123", int)
        assert result.is_success
        assert result.value == 123

        # Test conversion failure
        result = adapter.adapt_type("invalid", int)
        assert result.is_failure

    def test_validation_adapter(self) -> None:
        """Test validation adapter functionality."""
        # Test domain validation
        result = FlextTypeAdapters.Domain.validate_entity_id("test-id")
        assert result.is_success

        # Test validation failure
        result = FlextTypeAdapters.Domain.validate_entity_id("")
        assert result.is_failure


class TestUtilitiesCoverage:
    """Strategic utilities tests for 100% coverage."""

    def test_conversion_utilities(self) -> None:
        """Test conversion utility functions."""
        # Test safe bool conversion
        result = FlextUtilities.Conversions.safe_bool("true")
        assert result

        # Test safe bool conversion with default
        result = FlextUtilities.Conversions.safe_bool("invalid", default=False)
        assert not result

    def test_string_utilities(self) -> None:
        """Test string utility functions."""
        # Test string validation
        result = FlextUtilities.Strings.is_valid_string("test")
        assert result is True

        # Test empty string
        result = FlextUtilities.Strings.is_valid_string("")
        assert result is False

    def test_collection_utilities(self) -> None:
        """Test collection utility functions."""
        # Test safe dictionary access
        data = {"key": "value", "nested": {"inner": "data"}}
        result = FlextUtilities.Collections.safe_dict_get(data, "key", "default")
        assert result == "value"

        # Test missing key with default
        result = FlextUtilities.Collections.safe_dict_get(data, "missing", "default")
        assert result == "default"

    def test_file_utilities(self) -> None:
        """Test file utility functions."""
        # Test safe path validation
        result = FlextUtilities.Files.is_valid_path("/valid/path")
        assert result is True

        # Test invalid path
        result = FlextUtilities.Files.is_valid_path("")
        assert result is False


class TestEdgeCasesCoverage:
    """Test edge cases and error paths for 100% coverage."""

    def test_exception_handling_paths(self) -> None:
        """Test exception handling in various modules."""
        # Test config with invalid data
        with pytest.raises(Exception):
            FlextConfig(max_workers=-1)  # Should raise validation error

    def test_boundary_conditions(self) -> None:
        """Test boundary conditions."""
        # Test empty collections
        result = FlextValidations.Collections.validate_list([], min_items=0)
        assert result.is_success

        result = FlextValidations.Collections.validate_list([], min_items=1)
        assert result.is_failure

    def test_none_and_empty_handling(self) -> None:
        """Test None and empty value handling."""
        # Test None handling in utilities
        result = FlextUtilities.Conversions.safe_bool(None, default=False)
        assert result is False

        # Test empty string handling
        result = FlextValidations.Fields.validate_string("", required=False)
        assert result.is_success


class TestIntegrationCoverage:
    """Integration tests to cover cross-module paths."""

    def test_config_with_validation(self) -> None:
        """Test config creation with validation."""
        # Test valid config
        config_data = {
            "name": "integration-test",
            "environment": "test",
            "debug": True,
            "max_workers": 4,
        }

        # Validate config data first
        for value in config_data.values():
            if isinstance(value, str):
                result = FlextValidations.Fields.validate_string(value, min_length=1)
                assert result.is_success
            elif isinstance(value, int):
                result = FlextValidations.Numbers.validate_integer(value, min_value=1)
                assert result.is_success

        # Create config
        config = FlextConfig(**config_data)
        assert config.name == "integration-test"

    def test_adapter_with_validation(self) -> None:
        """Test adapter usage with validation."""
        # Convert and validate data
        source_data = {"number": "42", "text": "hello"}

        # Convert number field
        adapter = FlextTypeAdapters()
        convert_result = adapter.adapt_type(source_data["number"], int)
        assert convert_result.is_success

        # Validate converted result
        validate_result = FlextValidations.Numbers.validate_integer(
            convert_result.value, min_value=1
        )
        assert validate_result.is_success

    def test_utilities_integration(self) -> None:
        """Test utilities in integration scenarios."""
        # Create test data
        test_data = {
            "config": {
                "name": "test-service",
                "settings": {"timeout": "30", "retries": "3"},
            }
        }

        # Use utilities to extract and convert
        name_result = FlextUtilities.Collections.safe_dict_get(
            test_data["config"], "name", "default"
        )
        assert name_result == "test-service"

        timeout_value = test_data["config"]["settings"]["timeout"]

        # Convert timeout to int
        timeout_bool = FlextUtilities.Conversions.safe_bool(timeout_value != "0")
        assert timeout_bool is True


# Performance test for coverage of performance-critical paths
class TestPerformanceCoverage:
    """Test performance-critical code paths."""

    def test_bulk_operations(self) -> None:
        """Test bulk operations for performance paths."""
        # Test bulk validation
        items = [f"item_{i}" for i in range(100)]

        for item in items[:10]:  # Test first 10 for coverage
            result = FlextValidations.Fields.validate_string(item, min_length=1)
            assert result.is_success

    def test_large_data_handling(self) -> None:
        """Test handling of larger datasets."""
        # Create larger test data
        large_data = {f"key_{i}": f"value_{i}" for i in range(50)}

        # Test safe access to large data
        for i in range(5):  # Test first 5 keys for coverage
            key = f"key_{i}"
            result = FlextUtilities.Collections.safe_dict_get(
                large_data, key, "default"
            )
            assert result == f"value_{i}"
