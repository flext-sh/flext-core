"""Real functionality tests for utilities module without mocks.

Tests the actual FlextUtilities implementation with FlextTypes.Config integration,
StrEnum validation, and real execution paths.

Created to achieve comprehensive test coverage with actual functionality validation,
following the user's requirement for real tests without mocks.
"""

from __future__ import annotations

import time

import pytest

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.utilities import FlextUtilities

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextUtilitiesRealFunctionality:
    """Test real FlextUtilities functionality without mocks."""

    def test_generators_real_execution(self) -> None:
        """Test ID and timestamp generation with real execution."""
        # Test UUID generation
        uuid1 = FlextUtilities.Generators.generate_uuid()
        uuid2 = FlextUtilities.Generators.generate_uuid()

        assert len(uuid1) == 36  # Standard UUID length
        assert len(uuid2) == 36
        assert uuid1 != uuid2  # Should be unique
        assert "-" in uuid1  # UUID format
        assert "-" in uuid2

        # Test entity ID generation
        entity_id = FlextUtilities.Generators.generate_entity_id()
        assert entity_id.startswith("entity_")
        assert len(entity_id) > 7  # "entity_" + hex chars

        # Test correlation ID generation
        correlation_id = FlextUtilities.Generators.generate_correlation_id()
        assert correlation_id.startswith("corr_")
        assert len(correlation_id) > 5  # "corr_" + hex chars

        # Test session ID generation
        session_id = FlextUtilities.Generators.generate_session_id()
        assert session_id.startswith("sess_")
        assert len(session_id) > 5  # "sess_" + hex chars

        # Test request ID generation
        request_id = FlextUtilities.Generators.generate_request_id()
        assert request_id.startswith("req_")
        assert len(request_id) > 4  # "req_" + hex chars

        # Test ISO timestamp generation
        timestamp = FlextUtilities.Generators.generate_iso_timestamp()
        assert "T" in timestamp  # ISO format
        assert ":" in timestamp  # Time separator

    def test_text_processor_real_execution(self) -> None:
        """Test text processing utilities with real execution."""
        # Test text truncation
        long_text = "This is a very long text that needs to be truncated"
        truncated = FlextUtilities.TextProcessor.truncate(long_text, 20, "...")
        assert len(truncated) <= 20
        assert truncated.endswith("...")

        # Test safe string conversion
        safe_str = FlextUtilities.TextProcessor.safe_string(None, "default")
        assert safe_str == "default"

        safe_str = FlextUtilities.TextProcessor.safe_string(123)
        assert safe_str == "123"

        safe_str = FlextUtilities.TextProcessor.safe_string("test")
        assert safe_str == "test"

        # Test text cleaning
        dirty_text = "  Test\n\ttext\x00with\x01control  chars  "
        cleaned = FlextUtilities.TextProcessor.clean_text(dirty_text)
        # Control characters are removed and whitespace normalized
        assert "Test" in cleaned
        assert "text" in cleaned
        assert "control" in cleaned
        assert "chars" in cleaned

        # Test slugify
        text = "Hello World! Test 123"
        slug = FlextUtilities.TextProcessor.slugify(text)
        assert slug == "hello-world-test-123"

        # Test sensitive data masking
        sensitive = "1234567890"
        masked = FlextUtilities.TextProcessor.mask_sensitive(sensitive, visible_chars=4)
        assert masked.endswith("7890")
        assert len(masked) == len(sensitive)
        assert "*" in masked

    def test_performance_utilities_real_execution(self) -> None:
        """Test performance tracking with real execution."""
        # Record a performance metric
        FlextUtilities.Performance.record_metric("test_operation", 0.5, success=True)

        # Get the recorded metrics
        metrics = FlextUtilities.Performance.get_metrics("test_operation")
        assert metrics["total_calls"] == 1
        assert metrics["total_duration"] == 0.5
        assert metrics["success_count"] == 1
        assert metrics["error_count"] == 0

        # Record another metric with error
        FlextUtilities.Performance.record_metric(
            "test_operation", 0.3, success=False, error="Test error"
        )

        updated_metrics = FlextUtilities.Performance.get_metrics("test_operation")
        assert updated_metrics["total_calls"] == 2
        assert updated_metrics["error_count"] == 1
        assert updated_metrics["last_error"] == "Test error"

    def test_time_utils_real_execution(self) -> None:
        """Test time utilities with real execution."""
        # Test duration formatting
        duration_ms = FlextUtilities.TimeUtils.format_duration(0.5)
        assert duration_ms == "500.0ms"

        duration_s = FlextUtilities.TimeUtils.format_duration(5.0)
        assert duration_s == "5.0s"

        duration_m = FlextUtilities.TimeUtils.format_duration(120.0)
        assert duration_m == "2.0m"

        # Test UTC timestamp
        timestamp = FlextUtilities.TimeUtils.get_timestamp_utc()
        assert timestamp.tzinfo is not None  # Should be timezone-aware

    def test_conversions_real_execution(self) -> None:
        """Test safe type conversions with real execution."""
        # Test safe int conversion
        assert FlextUtilities.Conversions.safe_int("123", 0) == 123
        assert FlextUtilities.Conversions.safe_int("invalid", 0) == 0
        assert FlextUtilities.Conversions.safe_int(None, -1) == -1
        assert FlextUtilities.Conversions.safe_int(45.7) == 45

        # Test safe float conversion
        assert FlextUtilities.Conversions.safe_float("123.45", 0.0) == 123.45
        assert FlextUtilities.Conversions.safe_float("invalid", 1.0) == 1.0
        assert FlextUtilities.Conversions.safe_float(None, -1.0) == -1.0

        # Test safe bool conversion
        assert FlextUtilities.Conversions.safe_bool("true") is True
        assert FlextUtilities.Conversions.safe_bool("false") is False
        assert FlextUtilities.Conversions.safe_bool("1") is True
        assert FlextUtilities.Conversions.safe_bool("0") is False
        assert FlextUtilities.Conversions.safe_bool("yes") is True
        assert FlextUtilities.Conversions.safe_bool("invalid", default=False) is False

    def test_type_guards_real_execution(self) -> None:
        """Test type guard utilities with real execution."""
        # Test string checks
        assert FlextUtilities.TypeGuards.is_string_non_empty("test") is True
        assert FlextUtilities.TypeGuards.is_string_non_empty("") is False
        assert FlextUtilities.TypeGuards.is_string_non_empty(None) is False

        # Test dict checks
        assert FlextUtilities.TypeGuards.is_dict_non_empty({"key": "value"}) is True
        assert FlextUtilities.TypeGuards.is_dict_non_empty({}) is False
        assert FlextUtilities.TypeGuards.is_dict_non_empty(None) is False

        # Test list checks
        assert FlextUtilities.TypeGuards.is_list_non_empty(["item"]) is True
        assert FlextUtilities.TypeGuards.is_list_non_empty([]) is False
        assert FlextUtilities.TypeGuards.is_list_non_empty(None) is False

        # Test attribute checks
        assert FlextUtilities.TypeGuards.has_attribute("test", "upper") is True
        assert FlextUtilities.TypeGuards.has_attribute("test", "nonexistent") is False

        # Test None checks
        assert FlextUtilities.TypeGuards.is_not_none("value") is True
        assert FlextUtilities.TypeGuards.is_not_none(None) is False

    def test_formatters_real_execution(self) -> None:
        """Test data formatters with real execution."""
        # Test byte formatting
        assert FlextUtilities.Formatters.format_bytes(512) == "512 B"
        assert FlextUtilities.Formatters.format_bytes(1024) == "1.0 KB"
        assert FlextUtilities.Formatters.format_bytes(1048576) == "1.0 MB"
        assert FlextUtilities.Formatters.format_bytes(1073741824) == "1.0 GB"

        # Test percentage formatting
        assert FlextUtilities.Formatters.format_percentage(0.85) == "85.0%"
        assert (
            FlextUtilities.Formatters.format_percentage(0.123, precision=2) == "12.30%"
        )

    def test_processing_utils_real_execution(self) -> None:
        """Test data processing utilities with real execution."""
        # Test JSON parsing
        valid_json = '{"key": "value", "number": 42}'
        parsed = FlextUtilities.ProcessingUtils.safe_json_parse(valid_json)
        assert parsed["key"] == "value"
        assert parsed["number"] == 42

        invalid_json = '{"invalid": json}'
        parsed_invalid = FlextUtilities.ProcessingUtils.safe_json_parse(
            invalid_json, {"default": True}
        )
        assert parsed_invalid["default"] is True

        # Test JSON stringification
        data = {"key": "value", "number": 42}
        json_str = FlextUtilities.ProcessingUtils.safe_json_stringify(data)
        assert "key" in json_str
        assert "value" in json_str

        # Test model data extraction
        dict_data = {"field1": "value1", "field2": "value2"}
        extracted = FlextUtilities.ProcessingUtils.extract_model_data(dict_data)
        assert extracted == dict_data

    def test_result_utils_real_execution(self) -> None:
        """Test FlextResult utilities with real execution."""
        # Test result chaining
        result1 = FlextResult[str].ok("value1")
        result2 = FlextResult[str].ok("value2")
        result3 = FlextResult[str].ok("value3")

        chained = FlextUtilities.ResultUtils.chain_results(result1, result2, result3)
        assert chained.is_success
        assert chained.value == ["value1", "value2", "value3"]

        # Test chaining with failure
        result_fail = FlextResult[str].fail("error")
        chained_fail = FlextUtilities.ResultUtils.chain_results(
            result1, result_fail, result3
        )
        assert chained_fail.is_failure
        assert chained_fail.error == "error"

        # Test batch processing
        items = [1, 2, 3, 4, 5]

        def processor(item: int) -> FlextResult[str]:
            if item % 2 == 0:
                return FlextResult[str].ok(f"even_{item}")
            return FlextResult[str].fail(f"odd_{item}")

        successes, errors = FlextUtilities.ResultUtils.batch_process(items, processor)
        assert len(successes) == 2  # 2 and 4
        assert len(errors) == 3  # 1, 3, and 5
        assert "even_2" in successes
        assert "even_4" in successes


class TestFlextUtilitiesDelegatorMethods:
    """Test FlextUtilities main class delegator methods."""

    def test_class_method_delegation_real(self) -> None:
        """Test class method delegation works correctly."""
        # Test UUID generation delegation
        uuid = FlextUtilities.generate_uuid()
        assert len(uuid) == 36
        assert "-" in uuid

        # Test ID generation delegation
        flext_id = FlextUtilities.generate_id()
        assert flext_id.startswith("flext_")

        # Test entity ID delegation
        entity_id = FlextUtilities.generate_entity_id()
        assert entity_id.startswith("entity_")

        # Test text processing delegation
        truncated = FlextUtilities.truncate("long text here", 5, "...")
        assert len(truncated) <= 8  # 5 + len("...")

        # Test performance tracking delegation
        FlextUtilities.record_performance("test_op", 0.1, success=True)
        metrics = FlextUtilities.get_performance_metrics()
        assert "test_op" in metrics

        # Test JSON processing delegation
        data = {"test": "value"}
        json_str = FlextUtilities.safe_json_stringify(data)
        assert "test" in json_str

        parsed_back = FlextUtilities.safe_json_parse(json_str)
        assert parsed_back["test"] == "value"

    def test_legacy_compatibility_methods_real(self) -> None:
        """Test legacy compatibility methods work correctly."""
        # Test legacy int conversion methods
        assert FlextUtilities.safe_int_conversion("123") == 123
        assert FlextUtilities.safe_int_conversion("invalid") is None
        assert FlextUtilities.safe_int_conversion("invalid", 42) == 42

        # Test guaranteed default method
        assert FlextUtilities.safe_int_conversion_with_default("invalid", 99) == 99

        # Test bool conversion
        assert FlextUtilities.safe_bool_conversion("true") is True
        assert FlextUtilities.safe_bool_conversion("false") is False

        # Test additional utility methods
        assert FlextUtilities.is_non_empty_string("test") is True
        assert FlextUtilities.is_non_empty_string("") is False

        cleaned_text = FlextUtilities.clean_text("  test  ")
        assert cleaned_text == "test"

        timestamp = FlextUtilities.generate_timestamp()
        assert "T" in timestamp  # ISO format


class TestFlextUtilitiesPerformanceReal:
    """Test performance characteristics of utilities."""

    def test_id_generation_performance_real(self) -> None:
        """Test ID generation performance with real execution."""
        # Measure UUID generation performance
        start_time = time.perf_counter()

        uuids = []
        for _ in range(1000):
            uuid = FlextUtilities.generate_uuid()
            uuids.append(uuid)

        end_time = time.perf_counter()
        generation_time = end_time - start_time

        # Performance should be reasonable
        assert generation_time < 0.1  # Less than 100ms for 1000 UUIDs
        assert len(set(uuids)) == 1000  # All should be unique

    def test_text_processing_performance_real(self) -> None:
        """Test text processing performance with real execution."""
        long_text = "A" * 10000  # 10KB of text

        start_time = time.perf_counter()

        # Process text multiple times
        for _ in range(100):
            FlextUtilities.truncate(long_text, 1000)
            FlextUtilities.clean_text("  test  ")
            FlextUtilities.TextProcessor.slugify("Test Text")

        end_time = time.perf_counter()
        processing_time = end_time - start_time

        # Should process quickly
        assert processing_time < 0.1  # Less than 100ms

    def test_json_processing_performance_real(self) -> None:
        """Test JSON processing performance with real execution."""
        # Create large JSON data
        large_data = {"items": [{"id": i, "name": f"item_{i}"} for i in range(1000)]}

        start_time = time.perf_counter()

        # JSON operations
        for _ in range(10):
            json_str = FlextUtilities.safe_json_stringify(large_data)
            FlextUtilities.safe_json_parse(json_str)

        end_time = time.perf_counter()
        json_time = end_time - start_time

        # JSON processing should be reasonable
        assert json_time < 1.0  # Less than 1 second for 10 operations

    def test_performance_tracking_overhead_real(self) -> None:
        """Test performance tracking overhead is minimal."""

        # Define a simple function to track
        @FlextUtilities.Performance.track_performance("test_function")
        def simple_function() -> str:
            return "result"

        # Measure execution time with tracking
        start_time = time.perf_counter()

        for _ in range(1000):
            result = simple_function()
            assert result == "result"

        end_time = time.perf_counter()
        tracked_time = end_time - start_time

        # Tracking overhead should be minimal
        assert tracked_time < 0.1  # Less than 100ms for 1000 calls

        # Verify metrics were recorded
        metrics = FlextUtilities.Performance.get_metrics("test_function")
        assert metrics["total_calls"] == 1000
        assert metrics["success_count"] == 1000


class TestFlextUtilitiesConfigurationReal:
    """Test FlextUtilities.Configuration with FlextTypes.Config integration."""

    def test_create_default_config_real(self) -> None:
        """Test default configuration creation with FlextTypes.Config."""
        # Test development configuration
        dev_result = FlextUtilities.Configuration.create_default_config("development")
        assert dev_result.success is True

        dev_config = dev_result.unwrap()
        assert dev_config["environment"] == "development"
        assert dev_config["log_level"] == "DEBUG"
        assert dev_config["debug"] is True
        assert dev_config["validation_level"] == "normal"
        assert dev_config["request_timeout"] == 30000

        # Test production configuration
        prod_result = FlextUtilities.Configuration.create_default_config("production")
        assert prod_result.success is True

        prod_config = prod_result.unwrap()
        assert prod_config["environment"] == "production"
        assert prod_config["log_level"] == "ERROR"
        assert prod_config["debug"] is False
        assert prod_config["validation_level"] == "strict"
        assert prod_config["request_timeout"] == 60000

        # Test invalid environment
        invalid_result = FlextUtilities.Configuration.create_default_config("invalid")
        assert invalid_result.success is False
        assert "Invalid environment" in invalid_result.error

    def test_validate_configuration_with_types_real(self) -> None:
        """Test configuration validation with comprehensive StrEnum validation."""
        # Test valid configuration
        valid_config = {
            "environment": "staging",
            "log_level": "INFO",
            "validation_level": "normal",
            "config_source": "env",
            "debug": False,
            "performance_monitoring": True,
            "request_timeout": 45000,
            "max_retries": 5,
            "enable_caching": True,
        }

        result = FlextUtilities.Configuration.validate_configuration_with_types(
            valid_config
        )
        assert result.success is True

        validated = result.unwrap()
        assert validated["environment"] == "staging"
        assert validated["log_level"] == "INFO"
        assert validated["validation_level"] == "normal"

        # Test missing required field
        invalid_config = {"log_level": "INFO"}
        result = FlextUtilities.Configuration.validate_configuration_with_types(
            invalid_config
        )
        assert result.success is False
        assert "Required field 'environment' missing" in result.error

        # Test invalid environment
        invalid_env_config = {"environment": "invalid_env"}
        result = FlextUtilities.Configuration.validate_configuration_with_types(
            invalid_env_config
        )
        assert result.success is False
        assert "Invalid environment" in result.error

        # Test invalid log level
        invalid_log_config = {"environment": "development", "log_level": "INVALID"}
        result = FlextUtilities.Configuration.validate_configuration_with_types(
            invalid_log_config
        )
        assert result.success is False
        assert "Invalid log_level" in result.error

        # Test invalid validation level
        invalid_val_config = {
            "environment": "development",
            "validation_level": "invalid",
        }
        result = FlextUtilities.Configuration.validate_configuration_with_types(
            invalid_val_config
        )
        assert result.success is False
        assert "Invalid validation_level" in result.error

    def test_get_environment_configuration_real(self) -> None:
        """Test comprehensive environment configuration generation."""
        # Test development environment
        dev_result = FlextUtilities.Configuration.get_environment_configuration(
            "development"
        )
        assert dev_result.success is True

        dev_config = dev_result.unwrap()
        assert "base_configuration" in dev_config
        assert "environment_metadata" in dev_config
        assert "available_environments" in dev_config
        assert "performance_settings" in dev_config
        assert "security_settings" in dev_config

        # Verify environment metadata
        metadata = dev_config["environment_metadata"]
        assert metadata["name"] == "development"
        assert metadata["is_development"] is True
        assert metadata["is_production"] is False

        # Verify available options
        assert "development" in dev_config["available_environments"]
        assert "staging" in dev_config["available_environments"]
        assert "production" in dev_config["available_environments"]
        assert "INFO" in dev_config["available_log_levels"]
        assert "DEBUG" in dev_config["available_log_levels"]
        assert "strict" in dev_config["available_validation_levels"]
        assert "normal" in dev_config["available_validation_levels"]

        # Test production environment
        prod_result = FlextUtilities.Configuration.get_environment_configuration(
            "production"
        )
        assert prod_result.success is True

        prod_config = prod_result.unwrap()
        prod_metadata = prod_config["environment_metadata"]
        assert prod_metadata["is_production"] is True
        assert prod_metadata["is_development"] is False

        # Test invalid environment
        invalid_result = FlextUtilities.Configuration.get_environment_configuration(
            "invalid"
        )
        assert invalid_result.success is False

    def test_configuration_enum_integration_real(self) -> None:
        """Test StrEnum integration in configuration utilities."""
        # Test all environment values work
        for env_enum in FlextConstants.Config.ConfigEnvironment:
            result = FlextUtilities.Configuration.create_default_config(env_enum.value)
            assert result.success is True
            config = result.unwrap()
            assert config["environment"] == env_enum.value

        # Test all log levels work in validation
        for log_enum in FlextConstants.Config.LogLevel:
            test_config = {
                "environment": "development",
                "log_level": log_enum.value,
            }
            result = FlextUtilities.Configuration.validate_configuration_with_types(
                test_config
            )
            assert result.success is True
            validated = result.unwrap()
            assert validated["log_level"] == log_enum.value

        # Test all validation levels work
        for val_enum in FlextConstants.Config.ValidationLevel:
            test_config = {
                "environment": "development",
                "validation_level": val_enum.value,
            }
            result = FlextUtilities.Configuration.validate_configuration_with_types(
                test_config
            )
            assert result.success is True
            validated = result.unwrap()
            assert validated["validation_level"] == val_enum.value

    def test_configuration_performance_real(self) -> None:
        """Test configuration utilities performance."""
        # Test configuration creation performance
        start_time = time.perf_counter()

        for _ in range(100):
            result = FlextUtilities.Configuration.create_default_config("development")
            assert result.success is True

        creation_time = time.perf_counter() - start_time
        assert creation_time < 0.1  # Less than 100ms for 100 creations

        # Test configuration validation performance
        test_config = {
            "environment": "staging",
            "log_level": "INFO",
            "validation_level": "normal",
            "debug": True,
            "performance_monitoring": True,
            "request_timeout": 30000,
            "max_retries": 3,
            "enable_caching": True,
        }

        start_time = time.perf_counter()

        for _ in range(100):
            result = FlextUtilities.Configuration.validate_configuration_with_types(
                test_config
            )
            assert result.success is True

        validation_time = time.perf_counter() - start_time
        assert validation_time < 0.1  # Less than 100ms for 100 validations
