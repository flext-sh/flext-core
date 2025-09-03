"""Targeted tests for 100% coverage on FlextUtilities module.

This file contains precise tests targeting the specific remaining uncovered lines
in utilities.py focusing on Configuration class, generators, and processing utils.
"""

from __future__ import annotations

import math
from typing import Literal, cast

from flext_core import FlextConstants, FlextTypes, FlextUtilities


class TestUtilitiesConfiguration100PercentCoverage:
    """Targeted tests for the Configuration class uncovered lines."""

    def test_create_default_config_all_environments(self) -> None:
        """Test lines 814-856: create_default_config method."""
        # Test all valid environments using proper literal types
        environments: list[FlextTypes.Config.Environment] = [
            "development",
            "staging",
            "production",
            "test",
            "local",
        ]

        for env in environments:
            result = FlextUtilities.Configuration.create_default_config(env)
            assert result.success
            config = result.unwrap()

            # Verify basic configuration structure
            assert isinstance(config, dict)
            assert len(config) > 0

    def test_create_default_config_invalid_environment(self) -> None:
        """Test lines 819-822: Invalid environment validation."""
        # Test with invalid environment
        result = FlextUtilities.Configuration.create_default_config(
            cast(
                "Literal['development', 'production', 'staging', 'test', 'local']",
                "invalid_env",
            )
        )
        assert result.failure
        assert "Invalid environment" in str(result.error)

    def test_create_default_config_exception_handling(self) -> None:
        """Test lines 855-858: Exception handling in create_default_config."""
        # Test with empty string
        result = FlextUtilities.Configuration.create_default_config(
            cast("Literal['development', 'production', 'staging', 'test', 'local']", "")
        )
        assert result.failure
        assert "Invalid environment" in str(result.error)

    def test_validate_configuration_with_types_missing_environment(self) -> None:
        """Test lines 882-885: Missing environment validation."""
        config = cast(
            "FlextTypes.Config.ConfigDict", {"log_level": "INFO"}
        )  # Missing environment

        result = FlextUtilities.Configuration.validate_configuration_with_types(config)
        assert result.failure
        assert "Required field 'environment' missing" in str(result.error)

    def test_validate_configuration_with_types_invalid_environment(self) -> None:
        """Test lines 887-895: Invalid environment validation."""
        config = cast(
            "FlextTypes.Config.ConfigDict",
            {"environment": "invalid_env", "log_level": "INFO"},
        )

        result = FlextUtilities.Configuration.validate_configuration_with_types(config)
        assert result.failure
        assert "Invalid environment 'invalid_env'" in str(result.error)

    def test_validate_configuration_with_types_log_level_validation(self) -> None:
        """Test lines 897-903: Log level validation."""
        # Test with invalid log level
        config = cast(
            "FlextTypes.Config.ConfigDict",
            {
                "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                "log_level": "INVALID_LEVEL",
            },
        )

        result = FlextUtilities.Configuration.validate_configuration_with_types(config)
        assert result.failure
        assert "Invalid log_level" in str(result.error)

    def test_validate_configuration_with_types_success(self) -> None:
        """Test successful configuration validation."""
        config = cast(
            "FlextTypes.Config.ConfigDict",
            {
                "environment": FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value,
                "log_level": FlextConstants.Config.LogLevel.INFO.value,
                "debug": True,
                "request_timeout": 30000,
                "max_retries": 3,
            },
        )

        result = FlextUtilities.Configuration.validate_configuration_with_types(config)
        assert result.success
        validated = result.unwrap()
        assert (
            validated["environment"]
            == FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
        )
        assert validated["log_level"] == FlextConstants.Config.LogLevel.INFO.value


class TestUtilitiesGenerators100PercentCoverage:
    """Test generator functions for uncovered lines."""

    def test_generate_session_id(self) -> None:
        """Test line 202: generate_session_id method."""
        session_id = FlextUtilities.Generators.generate_session_id()
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        assert session_id.startswith("sess_")

    def test_generate_request_id(self) -> None:
        """Test line 207: generate_request_id method."""
        request_id = FlextUtilities.Generators.generate_request_id()
        assert isinstance(request_id, str)
        assert len(request_id) > 0
        assert request_id.startswith("req_")

    def test_generators_uniqueness(self) -> None:
        """Test that generators produce unique values."""
        # Test session IDs are unique
        sessions = {FlextUtilities.Generators.generate_session_id() for _ in range(10)}
        assert len(sessions) == 10

        # Test request IDs are unique
        requests = {FlextUtilities.Generators.generate_request_id() for _ in range(10)}
        assert len(requests) == 10


class TestUtilitiesTextProcessor100PercentCoverage:
    """Test text processing methods for uncovered lines."""

    def test_safe_string_with_none(self) -> None:
        """Test line 238-239: safe_string with None and objects."""
        # Test with None
        result = FlextUtilities.TextProcessor.safe_string(None)
        assert result == ""

        # Test with default
        result = FlextUtilities.TextProcessor.safe_string(None, "default")
        assert result == "default"

        # Test with non-string objects
        result = FlextUtilities.TextProcessor.safe_string(123)
        assert result == "123"

        result = FlextUtilities.TextProcessor.safe_string([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_clean_text_edge_cases(self) -> None:
        """Test line 245: clean_text with various inputs."""
        # Test with whitespace
        result = FlextUtilities.TextProcessor.clean_text("  hello   world  ")
        assert result == "hello world"

        # Test with tabs and newlines
        result = FlextUtilities.TextProcessor.clean_text("hello\t\nworld")
        assert result == "hello world"

        # Test with empty string
        result = FlextUtilities.TextProcessor.clean_text("")
        assert result == ""


class TestUtilitiesProcessingUtils100PercentCoverage:
    """Test processing utilities for uncovered lines."""

    def test_safe_json_parse_edge_cases(self) -> None:
        """Test ProcessingUtils.safe_json_parse uncovered lines."""
        # Test with invalid JSON - returns default empty dict
        result = FlextUtilities.ProcessingUtils.safe_json_parse("invalid json")
        assert result == {}  # Default empty dict

        # Test with valid JSON
        valid_result = FlextUtilities.ProcessingUtils.safe_json_parse(
            '{"key": "value"}'
        )
        assert valid_result == {"key": "value"}

    def test_safe_json_stringify_edge_cases(self) -> None:
        """Test ProcessingUtils.safe_json_stringify uncovered lines."""

        # Test with non-serializable object
        class NonSerializable:
            pass

        obj = NonSerializable()
        result = FlextUtilities.ProcessingUtils.safe_json_stringify(obj)
        # It converts to string representation, not default JSON
        assert isinstance(result, str)

        # Test with custom default - should still work
        result = FlextUtilities.ProcessingUtils.safe_json_stringify(obj, "null")
        assert isinstance(result, str)


class TestUtilitiesPerformance100PercentCoverage:
    """Test performance tracking for uncovered lines."""

    def test_track_performance_decorator(self) -> None:
        """Test Performance.track_performance decorator."""

        @FlextUtilities.Performance.track_performance("test_function")
        def test_function(x: int) -> int:
            return x * 2

        result = test_function(5)
        assert result == 10

        # Check that metrics were recorded
        metrics = FlextUtilities.Performance.get_metrics("test_function")
        assert isinstance(metrics, dict)

    def test_record_metric(self) -> None:
        """Test Performance.record_metric method."""
        FlextUtilities.Performance.record_metric("test_operation", 1.5)

        # Verify metric was recorded
        metrics = FlextUtilities.Performance.get_metrics("test_operation")
        assert isinstance(metrics, dict)


class TestUtilitiesConversions100PercentCoverage:
    """Test conversion utilities for uncovered lines."""

    def test_conversion_edge_cases(self) -> None:
        """Test conversion methods with edge cases."""
        # Test safe_int with invalid input
        result = FlextUtilities.Conversions.safe_int("not_a_number")
        assert result == 0  # default

        result = FlextUtilities.Conversions.safe_int("not_a_number", 42)
        assert result == 42  # custom default

        # Test safe_float with invalid input
        float_result = FlextUtilities.Conversions.safe_float("not_a_float")
        assert float_result == 0.0  # default

        float_result = FlextUtilities.Conversions.safe_float("not_a_float", math.pi)
        assert float_result == math.pi  # custom default

        # Test safe_bool with various inputs
        assert FlextUtilities.Conversions.safe_bool("yes") is True
        assert FlextUtilities.Conversions.safe_bool("true") is True
        assert FlextUtilities.Conversions.safe_bool("1") is True
        assert FlextUtilities.Conversions.safe_bool("on") is True
        assert FlextUtilities.Conversions.safe_bool("no") is False
        assert FlextUtilities.Conversions.safe_bool("false") is False
        assert FlextUtilities.Conversions.safe_bool(1) is True
        assert FlextUtilities.Conversions.safe_bool(0) is False
        # "invalid" string returns False as it's not in the allowed true values
        assert FlextUtilities.Conversions.safe_bool("invalid") is False
        # None should use the default
        assert FlextUtilities.Conversions.safe_bool(None, default=True) is True
