"""Tests for _exceptions_base module."""

import time

from flext_core._exceptions_base import (
    _clear_exception_metrics,
    _FlextBaseError,
    _FlextOperationBaseError,
    _FlextTypeBaseError,
    _FlextValidationBaseError,
    _get_exception_metrics,
)


class TestFlextBaseError:
    """Test _FlextBaseError class."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        error = _FlextBaseError()
        assert error.message == "An error occurred"
        assert error.error_code == "GENERIC_ERROR"
        assert error.context == {}
        assert isinstance(error.timestamp, float)
        assert isinstance(error.traceback, str)

    def test_init_with_custom_values(self) -> None:
        """Test initialization with custom values."""
        context = {"key": "value", "number": 42}
        error = _FlextBaseError(
            message="Custom error",
            error_code="CUSTOM_ERROR",
            context=context,
        )
        assert error.message == "Custom error"
        assert error.error_code == "CUSTOM_ERROR"
        assert error.context == context

    def test_track_exception_metrics(self) -> None:
        """Test exception metrics tracking."""
        _clear_exception_metrics()

        _error1 = _FlextBaseError("Error 1")
        _error2 = _FlextBaseError("Error 2")
        _error3 = _FlextBaseError("Error 3")

        metrics = _get_exception_metrics()
        assert metrics["total_exceptions"] == 3
        assert metrics["exception_types"] == 1
        assert "_FlextBaseError" in metrics["exceptions"]

        error_data = metrics["exceptions"]["_FlextBaseError"]
        assert error_data["count"] == 3
        assert error_data["error_codes"]["GENERIC_ERROR"] == 3


class TestFlextValidationBaseError:
    """Test _FlextValidationBaseError class."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        error = _FlextValidationBaseError()
        assert error.message == "Validation failed"
        assert error.error_code == "VALIDATION_ERROR"
        assert error.field is None
        assert error.value is None

    def test_init_with_field_and_value(self) -> None:
        """Test initialization with field and value."""
        error = _FlextValidationBaseError(
            message="Invalid email",
            field="email",
            value="invalid-email",
        )
        assert error.field == "email"
        assert error.value == "invalid-email"
        assert error.context["field"] == "email"
        assert error.context["value"] == "invalid-email"

    def test_init_with_additional_context(self) -> None:
        """Test initialization with additional context."""
        error = _FlextValidationBaseError(
            message="Validation failed",
            field="age",
            value=15,
            min_age=18,
            max_age=65,
        )
        assert error.context["min_age"] == 18
        assert error.context["max_age"] == 65


class TestFlextTypeBaseError:
    """Test _FlextTypeBaseError class."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        error = _FlextTypeBaseError()
        assert error.message == "Type error"
        assert error.error_code == "TYPE_ERROR"
        assert error.expected_type is None
        assert error.actual_type is None

    def test_init_with_types(self) -> None:
        """Test initialization with type information."""
        error = _FlextTypeBaseError(
            message="Type mismatch",
            expected_type="int",
            actual_type="str",
        )
        assert error.expected_type == "int"
        assert error.actual_type == "str"
        assert error.context["expected_type"] == "int"
        assert error.context["actual_type"] == "str"


class TestFlextOperationBaseError:
    """Test _FlextOperationBaseError class."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        error = _FlextOperationBaseError()
        assert error.message == "Operation failed"
        assert error.error_code == "OPERATION_ERROR"
        assert error.operation is None

    def test_init_with_operation(self) -> None:
        """Test initialization with operation."""
        error = _FlextOperationBaseError(
            message="Database connection failed",
            operation="connect",
        )
        assert error.operation == "connect"
        assert error.context["operation"] == "connect"


class TestExceptionMetrics:
    """Test exception metrics functionality."""

    def test_get_exception_metrics_empty(self) -> None:
        """Test metrics when no exceptions have been raised."""
        _clear_exception_metrics()
        metrics = _get_exception_metrics()
        assert metrics["total_exceptions"] == 0
        assert metrics["exception_types"] == 0
        assert metrics["exceptions"] == {}

    def test_get_exception_metrics_with_exceptions(self) -> None:
        """Test metrics with multiple exceptions."""
        _clear_exception_metrics()

        # Raise different types of exceptions
        _FlextBaseError("Error 1")
        _FlextValidationBaseError("Validation failed")
        _FlextTypeBaseError("Type error")
        _FlextOperationBaseError("Operation failed")

        metrics = _get_exception_metrics()
        assert metrics["total_exceptions"] == 4
        assert metrics["exception_types"] == 4

        # Check each exception type
        assert "_FlextBaseError" in metrics["exceptions"]
        assert "_FlextValidationBaseError" in metrics["exceptions"]
        assert "_FlextTypeBaseError" in metrics["exceptions"]
        assert "_FlextOperationBaseError" in metrics["exceptions"]

    def test_clear_exception_metrics(self) -> None:
        """Test clearing exception metrics."""
        _clear_exception_metrics()

        # Add some exceptions
        _FlextBaseError("Test error")
        metrics_before = _get_exception_metrics()
        assert metrics_before["total_exceptions"] == 1

        # Clear metrics
        _clear_exception_metrics()
        metrics_after = _get_exception_metrics()
        assert metrics_after["total_exceptions"] == 0
        assert metrics_after["exception_types"] == 0

    def test_rate_per_minute_calculation(self) -> None:
        """Test rate per minute calculation."""
        _clear_exception_metrics()

        # Simulate exceptions with different timestamps
        _error1 = _FlextBaseError("Error 1")
        time.sleep(0.1)  # Small delay to ensure different timestamps
        _error2 = _FlextBaseError("Error 2")

        metrics = _get_exception_metrics()
        error_data = metrics["exceptions"]["_FlextBaseError"]

        # Should have a rate per minute > 0
        assert error_data["rate_per_minute"] > 0
        assert error_data["count"] == 2
