"""Complete test suite for FlextUtilities - Real Functionality Testing.

Tests all functionalities of the FlextUtilities class with real execution validation,
ensuring comprehensive coverage without external dependencies or mocking.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime

import pytest

from flext_core import FlextResult, FlextUtilities

pytestmark = [pytest.mark.unit, pytest.mark.core]


# ============================================================================
# GENERATORS TESTS
# ============================================================================


class TestFlextUtilitiesGenerators:
    """Test all Generator functionality with real execution."""

    def test_generate_uuid(self) -> None:
        """Test UUID generation functionality."""
        uuid_str = FlextUtilities.generate_uuid()

        # Test format
        assert isinstance(uuid_str, str)
        assert len(uuid_str) == 36
        assert uuid_str.count("-") == 4

        # Validate UUID format
        uuid_obj = uuid.UUID(uuid_str)
        assert str(uuid_obj) == uuid_str

        # Test uniqueness with multiple generations
        uuids = [FlextUtilities.generate_uuid() for _ in range(1000)]
        assert len(set(uuids)) == 1000, "All UUIDs should be unique"

    def test_generate_id(self) -> None:
        """Test flext ID generation."""
        id_str = FlextUtilities.generate_id()

        # Test format
        assert isinstance(id_str, str)
        assert id_str.startswith("flext_")
        assert len(id_str) == 14  # "flext_" + 8 hex chars

        # Test uniqueness
        ids = [FlextUtilities.generate_id() for _ in range(1000)]
        assert len(set(ids)) == 1000, "All IDs should be unique"

    def test_generate_entity_id(self) -> None:
        """Test entity ID generation."""
        entity_id = FlextUtilities.generate_entity_id()

        # Test format
        assert isinstance(entity_id, str)
        assert entity_id.startswith("entity_")
        assert len(entity_id) == 19  # "entity_" + 12 hex chars

        # Test uniqueness
        ids = [FlextUtilities.generate_entity_id() for _ in range(100)]
        assert len(set(ids)) == 100, "All entity IDs should be unique"

    def test_generate_correlation_id(self) -> None:
        """Test correlation ID generation."""
        corr_id = FlextUtilities.generate_correlation_id()

        # Test format
        assert isinstance(corr_id, str)
        assert corr_id.startswith("corr_")
        assert len(corr_id) == 21  # "corr_" + 16 hex chars

        # Test uniqueness
        ids = [FlextUtilities.generate_correlation_id() for _ in range(100)]
        assert len(set(ids)) == 100, "All correlation IDs should be unique"

    def test_generate_iso_timestamp(self) -> None:
        """Test ISO timestamp generation."""
        timestamp = FlextUtilities.generate_iso_timestamp()

        # Test format
        assert isinstance(timestamp, str)
        assert "T" in timestamp
        assert timestamp.endswith(("Z", "-00:00")) or "+" in timestamp

        # Test parseable as datetime
        parsed = datetime.fromisoformat(timestamp)
        assert isinstance(parsed, datetime)

        # Test multiple timestamps
        timestamps = [FlextUtilities.generate_iso_timestamp() for _ in range(10)]
        # Allow for some duplicates due to timing, but most should be unique
        unique_count = len(set(timestamps))
        assert unique_count >= 5, "Most timestamps should be unique"


# ============================================================================
# TEXT PROCESSOR TESTS
# ============================================================================


class TestFlextUtilitiesTextProcessor:
    """Test all TextProcessor functionality."""

    def test_truncate_normal_text(self) -> None:
        """Test normal text truncation."""
        text = "This is a long text that needs to be truncated"

        # Test normal truncation
        result = FlextUtilities.truncate(text, 20)
        assert len(result) == 20
        assert result.endswith("...")
        assert result == "This is a long te..."

        # Test text shorter than limit
        short_text = "short"
        result = FlextUtilities.truncate(short_text, 20)
        assert result == short_text

        # Test exact length
        exact_text = "exactly twenty chars"
        result = FlextUtilities.truncate(exact_text, 20)
        assert result == exact_text
        assert len(result) == 20

    def test_truncate_edge_cases(self) -> None:
        """Test truncate edge cases."""
        text = "Sample text for testing"

        # Test very short limits
        result = FlextUtilities.truncate(text, 1)
        assert len(result) == 1
        assert result == "S"

        result = FlextUtilities.truncate(text, 2)
        assert len(result) == 2
        assert result == "Sa"

        result = FlextUtilities.truncate(text, 3)
        assert len(result) == 3
        assert (
            result == "Sam"
        )  # When max_length <= len(suffix), return text without suffix

        # Test custom suffix
        result = FlextUtilities.truncate(text, 15, suffix="[more]")
        assert len(result) == 15
        assert result.endswith("[more]")

        # Test empty text
        result = FlextUtilities.truncate("", 10)
        assert result == ""

    def test_safe_string(self) -> None:
        """Test safe string conversion."""
        # Test normal strings
        assert FlextUtilities.TextProcessor.safe_string("test") == "test"

        # Test numbers
        assert FlextUtilities.TextProcessor.safe_string(123) == "123"
        assert FlextUtilities.TextProcessor.safe_string(45.67) == "45.67"

        # Test None
        assert FlextUtilities.TextProcessor.safe_string(None) == ""
        assert FlextUtilities.TextProcessor.safe_string(None, "default") == "default"

        # Test objects
        assert FlextUtilities.TextProcessor.safe_string([1, 2, 3]) == "[1, 2, 3]"
        assert (
            FlextUtilities.TextProcessor.safe_string({"key": "value"})
            == "{'key': 'value'}"
        )


# ============================================================================
# TIME UTILS TESTS
# ============================================================================


class TestFlextUtilitiesTimeUtils:
    """Test all TimeUtils functionality."""

    def test_format_duration(self) -> None:
        """Test duration formatting."""
        # Test milliseconds
        assert FlextUtilities.format_duration(0.001) == "1.0ms"
        assert FlextUtilities.format_duration(0.5) == "500.0ms"

        # Test seconds
        assert FlextUtilities.format_duration(1.5) == "1.5s"
        assert FlextUtilities.format_duration(30) == "30.0s"

        # Test minutes
        assert FlextUtilities.format_duration(90) == "1.5m"
        assert FlextUtilities.format_duration(180) == "3.0m"

        # Test hours
        assert FlextUtilities.format_duration(7200) == "2.0h"
        assert FlextUtilities.format_duration(5400) == "1.5h"

    def test_get_timestamp_utc(self) -> None:
        """Test UTC timestamp generation."""
        timestamp = FlextUtilities.TimeUtils.get_timestamp_utc()

        assert isinstance(timestamp, datetime)
        assert timestamp.tzinfo is not None

        # Test multiple calls
        timestamps = [FlextUtilities.TimeUtils.get_timestamp_utc() for _ in range(5)]
        assert all(isinstance(ts, datetime) for ts in timestamps)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestFlextUtilitiesPerformance:
    """Test Performance tracking functionality."""

    def test_record_metric(self) -> None:
        """Test metric recording."""
        # Clear existing metrics
        FlextUtilities.Performance.get_metrics().clear()

        # Record a successful operation
        FlextUtilities.Performance.record_metric("test_operation", 0.5, success=True)

        metrics = FlextUtilities.Performance.get_metrics("test_operation")
        assert metrics["total_calls"] == 1
        assert metrics["total_duration"] == 0.5
        assert metrics["avg_duration"] == 0.5
        assert metrics["success_count"] == 1
        assert metrics["error_count"] == 0

        # Record another operation
        FlextUtilities.Performance.record_metric(
            "test_operation", 1.0, success=False, error="Test error"
        )

        metrics = FlextUtilities.Performance.get_metrics("test_operation")
        assert metrics["total_calls"] == 2
        assert metrics["total_duration"] == 1.5
        assert metrics["avg_duration"] == 0.75
        assert metrics["success_count"] == 1
        assert metrics["error_count"] == 1
        assert metrics["last_error"] == "Test error"

    def test_track_performance_decorator(self) -> None:
        """Test performance tracking decorator."""
        # Clear existing metrics
        FlextUtilities.Performance.get_metrics().clear()

        @FlextUtilities.Performance.track_performance("decorated_function")
        def test_function(value: int) -> int:
            time.sleep(0.01)  # Small delay to measure
            return value * 2

        # Test successful execution
        result = test_function(5)
        assert result == 10

        metrics = FlextUtilities.Performance.get_metrics("decorated_function")
        assert metrics["total_calls"] == 1
        assert metrics["success_count"] == 1
        assert metrics["error_count"] == 0
        duration = metrics.get("total_duration", 0.0)
        assert isinstance(duration, (int, float))
        assert float(duration) > 0.005  # Should be > 0.01s

        @FlextUtilities.Performance.track_performance("failing_function")
        def failing_function() -> None:
            error_message = "Test error"
            raise ValueError(error_message)

        # Test error handling
        with pytest.raises(ValueError, match="Test error"):
            failing_function()

        metrics = FlextUtilities.Performance.get_metrics("failing_function")
        assert metrics["total_calls"] == 1
        assert metrics["success_count"] == 0
        assert metrics["error_count"] == 1
        assert "Test error" in str(metrics.get("last_error", ""))

    def test_get_metrics(self) -> None:
        """Test metrics retrieval."""
        # Clear existing metrics
        FlextUtilities.Performance.get_metrics().clear()

        # Record some metrics
        FlextUtilities.Performance.record_metric("op1", 0.1)
        FlextUtilities.Performance.record_metric("op2", 0.2)

        # Test getting specific metrics
        op1_metrics = FlextUtilities.Performance.get_metrics("op1")
        assert op1_metrics["total_calls"] == 1

        # Test getting all metrics (may contain metrics from other tests)
        all_metrics = FlextUtilities.Performance.get_metrics()
        assert "op1" in all_metrics
        assert "op2" in all_metrics
        assert (
            len(all_metrics) >= 2
        )  # At least our 2 metrics, may have more from other tests

        # Test non-existent operation
        empty_metrics = FlextUtilities.Performance.get_metrics("non_existent")
        assert empty_metrics == {}


# ============================================================================
# CONVERSIONS TESTS
# ============================================================================


class TestFlextUtilitiesConversions:
    """Test all Conversions functionality."""

    def test_safe_int(self) -> None:
        """Test safe integer conversion."""
        # Test valid conversions
        assert FlextUtilities.Conversions.safe_int("123") == 123
        assert FlextUtilities.Conversions.safe_int(45.7) == 45
        assert FlextUtilities.Conversions.safe_int(99) == 99

        # Test invalid conversions with default
        assert FlextUtilities.Conversions.safe_int("invalid") == 0
        assert FlextUtilities.Conversions.safe_int("invalid", 999) == 999
        assert FlextUtilities.Conversions.safe_int(None) == 0
        assert FlextUtilities.Conversions.safe_int(None, -1) == -1

        # Test edge cases
        assert FlextUtilities.Conversions.safe_int("") == 0
        assert FlextUtilities.Conversions.safe_int([1, 2, 3]) == 0

    def test_safe_float(self) -> None:
        """Test safe float conversion."""
        # Test valid conversions
        assert FlextUtilities.Conversions.safe_float("123.45") == 123.45
        assert FlextUtilities.Conversions.safe_float(67) == 67.0
        assert FlextUtilities.Conversions.safe_float(89.1) == 89.1

        # Test invalid conversions with default
        assert FlextUtilities.Conversions.safe_float("invalid") == 0.0
        assert FlextUtilities.Conversions.safe_float("invalid", 999.9) == 999.9
        assert FlextUtilities.Conversions.safe_float(None) == 0.0
        assert FlextUtilities.Conversions.safe_float(None, -1.5) == -1.5

        # Test edge cases
        assert FlextUtilities.Conversions.safe_float("") == 0.0
        assert FlextUtilities.Conversions.safe_float([1, 2, 3]) == 0.0

    def test_safe_bool(self) -> None:
        """Test safe boolean conversion."""
        # Test string conversions
        assert FlextUtilities.Conversions.safe_bool("true") is True
        assert FlextUtilities.Conversions.safe_bool("True") is True
        assert FlextUtilities.Conversions.safe_bool("TRUE") is True
        assert FlextUtilities.Conversions.safe_bool("1") is True
        assert FlextUtilities.Conversions.safe_bool("yes") is True
        assert FlextUtilities.Conversions.safe_bool("on") is True

        assert FlextUtilities.Conversions.safe_bool("false") is False
        assert FlextUtilities.Conversions.safe_bool("False") is False
        assert FlextUtilities.Conversions.safe_bool("0") is False
        assert FlextUtilities.Conversions.safe_bool("no") is False
        assert FlextUtilities.Conversions.safe_bool("off") is False
        assert FlextUtilities.Conversions.safe_bool("") is False

        # Test boolean conversions
        assert FlextUtilities.Conversions.safe_bool(True) is True
        assert FlextUtilities.Conversions.safe_bool(False) is False

        # Test numeric conversions
        assert FlextUtilities.Conversions.safe_bool(1) is True
        assert FlextUtilities.Conversions.safe_bool(0) is False
        assert FlextUtilities.Conversions.safe_bool(-1) is True

        # Test None with default
        assert FlextUtilities.Conversions.safe_bool(None) is False
        assert FlextUtilities.Conversions.safe_bool(None, default=True) is True

        # Test invalid with default - list with items is truthy in Python
        assert (
            FlextUtilities.Conversions.safe_bool([1, 2, 3]) is True
        )  # Non-empty list is truthy
        assert FlextUtilities.Conversions.safe_bool([]) is False  # Empty list is falsy
        assert (
            FlextUtilities.Conversions.safe_bool({"key": "value"}, default=False)
            is True
        )  # Non-empty dict is truthy


# ============================================================================
# TYPE GUARDS TESTS
# ============================================================================


class TestFlextUtilitiesTypeGuards:
    """Test all TypeGuards functionality."""

    def test_is_string_non_empty(self) -> None:
        """Test non-empty string checking."""
        assert FlextUtilities.TypeGuards.is_string_non_empty("hello") is True
        assert FlextUtilities.TypeGuards.is_string_non_empty("x") is True

        assert FlextUtilities.TypeGuards.is_string_non_empty("") is False
        assert FlextUtilities.TypeGuards.is_string_non_empty(123) is False
        assert FlextUtilities.TypeGuards.is_string_non_empty(None) is False
        assert FlextUtilities.TypeGuards.is_string_non_empty([]) is False

    def test_is_dict_non_empty(self) -> None:
        """Test non-empty dict checking."""
        assert FlextUtilities.TypeGuards.is_dict_non_empty({"key": "value"}) is True
        assert FlextUtilities.TypeGuards.is_dict_non_empty({"a": 1, "b": 2}) is True

        assert FlextUtilities.TypeGuards.is_dict_non_empty({}) is False
        assert FlextUtilities.TypeGuards.is_dict_non_empty("string") is False
        assert FlextUtilities.TypeGuards.is_dict_non_empty(None) is False
        assert FlextUtilities.TypeGuards.is_dict_non_empty([]) is False

    def test_is_list_non_empty(self) -> None:
        """Test non-empty list checking."""
        assert FlextUtilities.TypeGuards.is_list_non_empty([1, 2, 3]) is True
        assert FlextUtilities.TypeGuards.is_list_non_empty(["a"]) is True

        assert FlextUtilities.TypeGuards.is_list_non_empty([]) is False
        assert FlextUtilities.TypeGuards.is_list_non_empty("string") is False
        assert FlextUtilities.TypeGuards.is_list_non_empty(None) is False
        assert FlextUtilities.TypeGuards.is_list_non_empty({}) is False

    def test_has_attribute(self) -> None:
        """Test attribute existence checking."""

        class TestClass:
            def __init__(self) -> None:
                self.existing_attr = "value"

        obj = TestClass()

        assert FlextUtilities.TypeGuards.has_attribute(obj, "existing_attr") is True
        assert FlextUtilities.TypeGuards.has_attribute(obj, "__init__") is True

        assert FlextUtilities.TypeGuards.has_attribute(obj, "non_existent") is False
        assert (
            FlextUtilities.TypeGuards.has_attribute("string", "non_existent") is False
        )

    def test_is_not_none(self) -> None:
        """Test None checking."""
        assert FlextUtilities.TypeGuards.is_not_none("value") is True
        assert FlextUtilities.TypeGuards.is_not_none(0) is True
        assert FlextUtilities.TypeGuards.is_not_none(False) is True
        assert FlextUtilities.TypeGuards.is_not_none([]) is True
        assert FlextUtilities.TypeGuards.is_not_none({}) is True

        assert FlextUtilities.TypeGuards.is_not_none(None) is False


# ============================================================================
# FORMATTERS TESTS
# ============================================================================


class TestFlextUtilitiesFormatters:
    """Test all Formatters functionality."""

    def test_format_bytes(self) -> None:
        """Test byte formatting."""
        # Test bytes
        assert FlextUtilities.Formatters.format_bytes(500) == "500 B"
        assert FlextUtilities.Formatters.format_bytes(1000) == "1000 B"

        # Test kilobytes
        assert FlextUtilities.Formatters.format_bytes(1500) == "1.5 KB"
        assert FlextUtilities.Formatters.format_bytes(5120) == "5.0 KB"

        # Test megabytes
        assert FlextUtilities.Formatters.format_bytes(1572864) == "1.5 MB"
        assert FlextUtilities.Formatters.format_bytes(5242880) == "5.0 MB"

        # Test gigabytes
        assert FlextUtilities.Formatters.format_bytes(1610612736) == "1.5 GB"
        assert FlextUtilities.Formatters.format_bytes(5368709120) == "5.0 GB"

    def test_format_percentage(self) -> None:
        """Test percentage formatting."""
        assert FlextUtilities.Formatters.format_percentage(0.5) == "50.0%"
        assert FlextUtilities.Formatters.format_percentage(0.75) == "75.0%"
        assert FlextUtilities.Formatters.format_percentage(1.0) == "100.0%"
        assert FlextUtilities.Formatters.format_percentage(0.0) == "0.0%"

        # Test precision
        assert FlextUtilities.Formatters.format_percentage(0.1234, 2) == "12.34%"
        assert FlextUtilities.Formatters.format_percentage(0.1234, 0) == "12%"


# ============================================================================
# PROCESSING UTILS TESTS
# ============================================================================


class TestFlextUtilitiesProcessingUtils:
    """Test all ProcessingUtils functionality."""

    def test_safe_json_parse(self) -> None:
        """Test safe JSON parsing."""
        # Test valid JSON
        valid_json = '{"name": "test", "value": 123}'
        result = FlextUtilities.ProcessingUtils.safe_json_parse(valid_json)
        assert result == {"name": "test", "value": 123}

        # Test invalid JSON
        invalid_json = '{"name": "test", invalid}'
        result = FlextUtilities.ProcessingUtils.safe_json_parse(invalid_json)
        assert result == {}

        # Test invalid JSON with default
        default_value: dict[str, object] = {"error": "failed"}
        result = FlextUtilities.ProcessingUtils.safe_json_parse(
            invalid_json, default_value
        )
        assert result == default_value

        # Test non-dict JSON (should return default)
        list_json = "[1, 2, 3]"
        result = FlextUtilities.ProcessingUtils.safe_json_parse(list_json)
        assert result == {}

    def test_safe_json_stringify(self) -> None:
        """Test safe JSON stringification."""
        # Test normal objects
        obj = {"name": "test", "value": 123}
        result = FlextUtilities.ProcessingUtils.safe_json_stringify(obj)
        assert result == '{"name": "test", "value": 123}'

        # Test complex objects (using default=str)
        class CustomObject:
            def __str__(self) -> str:
                return "custom_object"

        obj_with_custom = {"normal": "value", "custom": CustomObject()}
        result = FlextUtilities.ProcessingUtils.safe_json_stringify(obj_with_custom)
        expected = '{"normal": "value", "custom": "custom_object"}'
        assert result == expected

        # Test unicode
        unicode_obj = {"text": "Hello 世界"}
        result = FlextUtilities.ProcessingUtils.safe_json_stringify(unicode_obj)
        assert "Hello 世界" in result

        # Test default fallback for non-serializable - object() with default=str gives string representation
        result = FlextUtilities.ProcessingUtils.safe_json_stringify(
            object(), "fallback"
        )
        # With default=str, object() gets converted to its string representation, not fallback
        assert isinstance(result, str)
        assert "object object at" in result

    def test_extract_model_data(self) -> None:
        """Test model data extraction."""
        # Test dict input
        dict_data = {"key": "value"}
        result = FlextUtilities.ProcessingUtils.extract_model_data(dict_data)
        assert result == {"key": "value"}

        # Test object with model_dump method (Pydantic v2 style)
        class MockPydanticV2:
            def model_dump(self) -> dict[str, object]:
                return {"from": "model_dump"}

        pydantic_v2 = MockPydanticV2()
        result = FlextUtilities.ProcessingUtils.extract_model_data(pydantic_v2)
        assert result == {"from": "model_dump"}

        # Test object with dict method (Pydantic v1 style)
        class MockPydanticV1:
            def dict(self) -> dict[str, object]:
                return {"from": "dict"}

        pydantic_v1 = MockPydanticV1()
        result = FlextUtilities.ProcessingUtils.extract_model_data(pydantic_v1)
        assert result == {"from": "dict"}

        # Test other objects (should return empty dict)
        result = FlextUtilities.ProcessingUtils.extract_model_data("string")
        assert result == {}

        result = FlextUtilities.ProcessingUtils.extract_model_data(123)
        assert result == {}

    def test_parse_json_to_model(self) -> None:
        """Test JSON parsing to model."""

        # Test with simple class that accepts kwargs
        class SimpleModel:
            def __init__(self, **kwargs: object) -> None:
                self.name = kwargs.get("name", "")
                self.value = kwargs.get("value", 0)

        json_text = '{"name": "test", "value": 123}'
        result = FlextUtilities.ProcessingUtils.parse_json_to_model(
            json_text, SimpleModel
        )

        assert result.success
        model = result.unwrap()
        assert model.name == "test"
        assert model.value == 123

        # Test with invalid JSON
        invalid_json = '{"name": invalid}'
        result = FlextUtilities.ProcessingUtils.parse_json_to_model(
            invalid_json, SimpleModel
        )

        assert result.is_failure
        assert result.error is not None
        assert "Invalid JSON" in result.error

        # Test with mock Pydantic model
        class MockPydanticModel:
            def __init__(self, name: str, value: int) -> None:
                self.name = name
                self.value = value

            @classmethod
            def model_validate(cls, data: dict[str, object]) -> MockPydanticModel:
                name_val = data["name"]
                value_val = data["value"]
                return cls(
                    str(name_val),
                    int(value_val) if isinstance(value_val, (int, str)) else 0,
                )

        result_pydantic = FlextUtilities.ProcessingUtils.parse_json_to_model(
            json_text, MockPydanticModel
        )
        assert result_pydantic.success
        model_pydantic = result_pydantic.unwrap()
        assert model_pydantic.name == "test"
        assert model_pydantic.value == 123


# ============================================================================
# RESULT UTILS TESTS
# ============================================================================


class TestFlextUtilitiesResultUtils:
    """Test all ResultUtils functionality."""

    def test_chain_results(self) -> None:
        """Test result chaining."""
        # Test all successful results
        results = [
            FlextResult[str].ok("first"),
            FlextResult[str].ok("second"),
            FlextResult[str].ok("third"),
        ]

        chained = FlextUtilities.ResultUtils.chain_results(*results)
        assert chained.success
        assert chained.unwrap() == ["first", "second", "third"]

        # Test with one failure
        results_with_failure = [
            FlextResult[str].ok("first"),
            FlextResult[str].fail("error occurred"),
            FlextResult[str].ok("third"),
        ]

        chained = FlextUtilities.ResultUtils.chain_results(*results_with_failure)
        assert chained.is_failure
        assert chained.error is not None
        assert "error occurred" in chained.error

        # Test empty results
        chained = FlextUtilities.ResultUtils.chain_results()
        assert chained.success
        assert chained.unwrap() == []

    def test_batch_process(self) -> None:
        """Test batch processing."""

        def square_if_positive(x: int) -> FlextResult[int]:
            if x < 0:
                return FlextResult[int].fail(f"Negative number: {x}")
            return FlextResult[int].ok(x * x)

        # Test mixed success/failure
        items = [1, 2, -3, 4, -5]
        successes, errors = FlextUtilities.ResultUtils.batch_process(
            items, square_if_positive
        )

        assert successes == [1, 4, 16]  # 1², 2², 4²
        assert len(errors) == 2
        assert "Negative number: -3" in errors
        assert "Negative number: -5" in errors

        # Test all successful
        items = [1, 2, 3, 4]
        successes, errors = FlextUtilities.ResultUtils.batch_process(
            items, square_if_positive
        )

        assert successes == [1, 4, 9, 16]
        assert errors == []

        # Test all failures
        items = [-1, -2, -3]
        successes, errors = FlextUtilities.ResultUtils.batch_process(
            items, square_if_positive
        )

        assert successes == []
        assert len(errors) == 3


# ============================================================================
# MAIN CLASS DELEGATION TESTS
# ============================================================================


class TestFlextUtilitiesMainClassDelegation:
    """Test main class method delegation."""

    def test_main_class_uuid_delegation(self) -> None:
        """Test that main class methods delegate correctly."""
        # Test UUID delegation
        uuid_str = FlextUtilities.generate_uuid()
        assert isinstance(uuid_str, str)
        assert len(uuid_str) == 36

        # Test ID delegation
        id_str = FlextUtilities.generate_id()
        assert isinstance(id_str, str)
        assert id_str.startswith("flext_")

        # Test entity ID delegation
        entity_id = FlextUtilities.generate_entity_id()
        assert isinstance(entity_id, str)
        assert entity_id.startswith("entity_")

        # Test correlation ID delegation
        corr_id = FlextUtilities.generate_correlation_id()
        assert isinstance(corr_id, str)
        assert corr_id.startswith("corr_")

    def test_truncate_delegation(self) -> None:
        """Test truncate method delegation."""
        text = "This is a long text that needs truncation"
        result = FlextUtilities.truncate(text, 20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_format_duration_delegation(self) -> None:
        """Test duration formatting delegation."""
        result = FlextUtilities.format_duration(1.5)
        assert result == "1.5s"

        result = FlextUtilities.format_duration(0.001)
        assert result == "1.0ms"

    def test_json_parsing_delegation(self) -> None:
        """Test JSON parsing delegation."""
        json_str = '{"key": "value", "number": 123}'
        result = FlextUtilities.safe_json_parse(json_str)
        assert result == {"key": "value", "number": 123}

        # Test with default
        invalid_json = "invalid json"
        default_value: dict[str, object] = {"error": "parsing failed"}
        result = FlextUtilities.safe_json_parse(invalid_json, default_value)
        assert result == default_value

    def test_conversion_delegation(self) -> None:
        """Test conversion methods delegation."""
        assert FlextUtilities.safe_int("123") == 123
        assert FlextUtilities.safe_int("invalid", 999) == 999

        assert FlextUtilities.safe_bool_conversion("true") is True
        assert FlextUtilities.safe_bool_conversion("false") is False
        assert FlextUtilities.safe_bool_conversion(None, default=True) is True

    def test_legacy_compatibility_methods(self) -> None:
        """Test legacy compatibility methods."""
        # Test safe_int_conversion
        assert FlextUtilities.safe_int_conversion("123") == 123
        assert FlextUtilities.safe_int_conversion("invalid") is None
        assert FlextUtilities.safe_int_conversion("invalid", 999) == 999

        # Test safe_int_conversion_with_default
        assert FlextUtilities.safe_int_conversion_with_default("123", 0) == 123
        assert FlextUtilities.safe_int_conversion_with_default("invalid", 999) == 999

        # Test performance metrics
        FlextUtilities.record_performance("test_op", 1.0, success=True)
        metrics = FlextUtilities.get_performance_metrics()
        assert "test_op" in metrics

    def test_batch_process_delegation(self) -> None:
        """Test batch processing delegation."""

        def double_positive(x: int) -> FlextResult[int]:
            if x <= 0:
                return FlextResult[int].fail(f"Non-positive: {x}")
            return FlextResult[int].ok(x * 2)

        items = [1, 2, -1, 3, 0]
        successes, errors = FlextUtilities.batch_process(items, double_positive)

        assert successes == [2, 4, 6]  # 1*2, 2*2, 3*2
        assert len(errors) == 2  # -1 and 0


# ============================================================================
# COMPREHENSIVE INTEGRATION TESTS
# ============================================================================


class TestFlextUtilitiesIntegration:
    """Integration tests combining multiple utilities."""

    def test_complete_workflow(self) -> None:
        """Test complete utility workflow."""
        # Clear performance metrics
        FlextUtilities.Performance.get_metrics().clear()

        # Generate IDs
        user_id = FlextUtilities.generate_id()
        correlation_id = FlextUtilities.generate_correlation_id()
        timestamp = FlextUtilities.generate_iso_timestamp()

        # Create data structure
        user_data = {
            "id": user_id,
            "correlation_id": correlation_id,
            "created_at": timestamp,
            "name": "John Doe with a very long name that needs truncation",
            "status": "active",
        }

        # Process data
        json_str = FlextUtilities.ProcessingUtils.safe_json_stringify(user_data)
        parsed_back = FlextUtilities.safe_json_parse(json_str)

        # Format data
        truncated_name = FlextUtilities.truncate(user_data["name"], 20)

        # Validate results
        assert parsed_back["id"] == user_id
        assert parsed_back["correlation_id"] == correlation_id
        assert len(truncated_name) == 20
        assert truncated_name.endswith("...")

        # Record performance
        FlextUtilities.record_performance("user_processing", 0.05, success=True)
        metrics = FlextUtilities.get_performance_metrics()
        assert "user_processing" in metrics

    def test_error_handling_integration(self) -> None:
        """Test integrated error handling."""

        # Test invalid JSON with model parsing
        class SimpleModel:
            def __init__(self, data: dict[str, object]) -> None:
                self.name = str(data.get("name", ""))

        invalid_json = '{"name": invalid json}'
        result = FlextUtilities.parse_json_to_model(invalid_json, SimpleModel)
        assert result.is_failure
        assert result.error is not None
        assert "Invalid JSON" in result.error

        # Test conversion with fallbacks
        invalid_data = "not a number"
        safe_number = FlextUtilities.safe_int(invalid_data, 0)
        assert safe_number == 0

        safe_bool = FlextUtilities.safe_bool_conversion(invalid_data, default=False)
        assert safe_bool is False

    def test_performance_tracking_integration(self) -> None:
        """Test performance tracking with real operations."""
        # Clear existing metrics
        FlextUtilities.Performance.get_metrics().clear()

        @FlextUtilities.track_performance("id_generation_batch")
        def generate_batch_ids(count: int) -> list[str]:
            return [FlextUtilities.generate_uuid() for _ in range(count)]

        @FlextUtilities.track_performance("json_processing_batch")
        def process_json_batch(items: list[dict[str, object]]) -> list[str]:
            return [
                FlextUtilities.ProcessingUtils.safe_json_stringify(item)
                for item in items
            ]

        # Generate test data
        ids = generate_batch_ids(100)
        test_data = [{"id": id_val, "index": i} for i, id_val in enumerate(ids)]
        json_results = process_json_batch(test_data)

        # Validate results
        assert len(ids) == 100
        assert len(json_results) == 100
        assert all('"id"' in json_str for json_str in json_results)

        # Check performance metrics
        all_metrics = FlextUtilities.get_performance_metrics()
        assert "id_generation_batch" in all_metrics
        assert "json_processing_batch" in all_metrics

        id_metrics = all_metrics.get("id_generation_batch", {})
        json_metrics = all_metrics.get("json_processing_batch", {})

        assert isinstance(id_metrics, dict)
        assert id_metrics.get("total_calls") == 1
        assert isinstance(id_metrics, dict)
        assert id_metrics.get("success_count") == 1
        assert isinstance(json_metrics, dict)
        assert json_metrics.get("total_calls") == 1
        assert isinstance(json_metrics, dict)
        assert json_metrics.get("success_count") == 1
