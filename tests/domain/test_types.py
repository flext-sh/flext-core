"""FLEXT Core Tests.

Copyright (c) 2025 Flext. All rights reserved.
SPDX-License-Identifier: MIT

Test suite for FLEXT Core framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.domain.shared_types import LogLevel
from flext_core.domain.shared_types import ServiceResult
if TYPE_CHECKING:
    from flext_core.domain.shared_types import ProjectName, Version


class TestServiceResult:
    """Test ServiceResult functionality."""

    def test_service_result_success(self) -> None:
        """Test successful ServiceResult creation."""
        result = ServiceResult.ok("test_data")

        assert result.success
        assert result.data == "test_data"
        assert result.error is None

    def test_service_result_failure(self) -> None:
        """Test failed ServiceResult creation."""
        result = ServiceResult.fail("test_error")

        assert not result.success
        assert result.data is None
        assert result.error == "test_error"

    def test_service_result_success_method(self) -> None:
        """Test ServiceResult.success method."""
        result = ServiceResult.ok("success_data")

        assert result.success
        assert result.data == "success_data"
        assert result.error is None

    def test_service_result_failure_method(self) -> None:
        """Test ServiceResult.failure method."""
        result = ServiceResult.fail("failure_message")

        assert not result.success
        assert result.data is None
        assert result.error == "failure_message"

    def test_service_result_none_data(self) -> None:
        """Test ServiceResult with None data."""
        result = ServiceResult.ok(None)

        assert result.success
        assert result.data is None
        assert result.error is None

    def test_service_result_complex_data(self) -> None:
        """Test ServiceResult with complex data types."""
        test_data = {"key": "value", "list": [1, 2, 3]}
        result = ServiceResult.ok(test_data)

        assert result.success
        assert result.data == test_data
        assert result.data["key"] == "value"
        assert result.data["list"] == [1, 2, 3]

    def test_service_result_string_representation(self) -> None:
        """Test ServiceResult string representation."""
        success_result = ServiceResult.ok("test")
        failure_result = ServiceResult.fail("error")

        # Should have meaningful string representations
        assert str(success_result) != ""
        assert str(failure_result) != ""

    def test_service_result_equality(self) -> None:
        """Test ServiceResult equality comparison."""
        result1 = ServiceResult.ok("test")
        result2 = ServiceResult.ok("test")
        result3 = ServiceResult.ok("different")

        # Same data should have same properties
        assert result1.data == result2.data
        assert result1.success == result2.success
        # Different data should have different values
        assert result1.data != result3.data

    def test_service_result_boolean_context(self) -> None:
        """Test ServiceResult in boolean context."""
        success_result = ServiceResult.ok("test")
        failure_result = ServiceResult.fail("error")

        # Success should be truthy
        assert success_result.success is True
        # Failure should be falsy
        assert failure_result.success is False


class TestTypedLiterals:
    """Test typed literal types."""

    def test_environment_literal_valid_values(self) -> None:
        """Test EnvironmentLiteral accepts valid values."""
        valid_environments = ["development", "staging", "production", "test"]

        for _env in valid_environments:
            # Should not raise any exception
            pass

    def test_log_level_enum(self) -> None:
        """Test LogLevel enum values."""
        # Test enum values are correctly set
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"

    def test_log_level_comparison(self) -> None:
        """Test LogLevel comparison operations."""
        # Test enum equality
        assert LogLevel.ERROR == LogLevel.ERROR
        # Test that different enums are different (comparing instances)
        debug_level = LogLevel.DEBUG
        info_level = LogLevel.INFO
        assert debug_level != info_level

        # Test string equality
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"


class TestCustomTypes:
    """Test custom type definitions."""

    def test_project_name_type(self) -> None:
        """Test ProjectName type alias."""
        project_name: ProjectName = "test-project"
        assert project_name == "test-project"
        assert isinstance(project_name, str)

    def test_version_type(self) -> None:
        """Test Version type alias."""
        version: Version = "1.0.0"
        assert version == "1.0.0"
        assert isinstance(version, str)


class TestServiceResultAdvanced:
    """Test advanced ServiceResult functionality."""

    def test_service_result_chaining(self) -> None:
        """Test ServiceResult can be used in chaining operations."""

        def process_data(data: str) -> ServiceResult[str]:
            if not data:
                return ServiceResult.fail("Empty data")
            return ServiceResult.ok(data.upper())

        def validate_data(data: str) -> ServiceResult[str]:
            if len(data) < 3:
                return ServiceResult.fail("Data too short")
            return ServiceResult.ok(data)

        # Test successful chain
        result1 = validate_data("test")
        if result1.success and result1.data is not None:
            result2 = process_data(result1.data)
            assert result2.success
            assert result2.data == "TEST"

        # Test failure chain
        result3 = validate_data("hi")
        assert not result3.success
        assert result3.error is not None
        assert "too short" in result3.error

    def test_service_result_error_handling(self) -> None:
        """Test ServiceResult error handling patterns."""

        def risky_operation(should_fail: bool) -> ServiceResult[str]:
            if should_fail:
                return ServiceResult.fail("Operation failed")
            return ServiceResult.ok(42)

        # Test success case
        success = risky_operation(False)
        assert success.success
        assert success.data == 42

        # Test failure case
        failure = risky_operation(True)
        assert not failure.success
        assert failure.error == "Operation failed"

    def test_service_result_type_safety(self) -> None:
        """Test ServiceResult type safety."""
        # Different types should work
        string_result = ServiceResult.ok("string")
        int_result = ServiceResult.ok(123)
        list_result = ServiceResult.ok([1, 2, 3])
        dict_result = ServiceResult.ok({"key": "value"})

        assert string_result.data == "string"
        assert int_result.data == 123
        assert list_result.data == [1, 2, 3]
        assert dict_result.data == {"key": "value"}

    def test_service_result_status_property(self) -> None:
        """Test ServiceResult status property."""
        success = ServiceResult.ok("test")
        failure = ServiceResult.fail("error")

        # Should have status property
        assert hasattr(success, "status")
        assert hasattr(failure, "status")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_service_result_empty_error(self) -> None:
        """Test ServiceResult with empty error message."""
        result = ServiceResult.fail("")
        assert not result.success
        assert result.error == ""

    def test_service_result_none_error(self) -> None:
        """Test ServiceResult with None error."""
        # This should create a failure with None error
        result = ServiceResult.fail(None)
        assert not result.success
        assert result.error is None

    def test_service_result_large_data(self) -> None:
        """Test ServiceResult with large data."""
        large_data = "x" * 10000  # Large string
        result = ServiceResult.ok(large_data)

        assert result.success
        assert result.data is not None
        assert len(result.data) == 10000
        assert result.data == large_data

    def test_service_result_nested_results(self) -> None:
        """Test ServiceResult containing other ServiceResults."""
        inner_result = ServiceResult.ok("inner")
        outer_result = ServiceResult.ok(inner_result)

        assert outer_result.success
        assert outer_result.data is not None
        assert outer_result.data.success
        assert outer_result.data is not None
        assert outer_result.data.data == "inner"
