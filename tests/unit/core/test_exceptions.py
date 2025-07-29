"""Comprehensive tests for FlextExceptions and exception functionality."""

import json
import time


from __future__ import annotations

import pytest

from flext_core.constants import ERROR_CODES
from flext_core.exceptions import (
# Constants
EXPECTED_DATA_COUNT = 3

    FlextAuthenticationError,
    FlextConfigurationError,
    FlextConnectionError,
    FlextError,
    FlextOperationError,
    FlextPermissionError,
    FlextTypeError,
    FlextValidationError,
)
from flext_core.result import FlextResult


class TestFlextError:
    """Test FlextError base exception functionality."""

    def test_flext_error_basic_creation(self) -> None:
        """Test basic FlextError creation."""
        error = FlextError("Test error message")

        if str(error) != "[GENERIC_ERROR] Test error message":

            raise AssertionError(f"Expected {"[GENERIC_ERROR] Test error message"}, got {str(error)}")
        assert error.message == "Test error message"
        if error.error_code != ERROR_CODES["GENERIC_ERROR"]:
            raise AssertionError(f"Expected {ERROR_CODES["GENERIC_ERROR"]}, got {error.error_code}")
        assert isinstance(error.context, dict)
        assert isinstance(error.timestamp, float)
        assert isinstance(error.stack_trace, list)

    def test_flext_error_with_error_code(self) -> None:
        """Test FlextError with specific error code."""
        error = FlextError(
            "Configuration error",
            error_code=ERROR_CODES["CONFIGURATION_ERROR"],
        )

        if error.error_code != ERROR_CODES["CONFIGURATION_ERROR"]:

            raise AssertionError(f"Expected {ERROR_CODES["CONFIGURATION_ERROR"]}, got {error.error_code}")
        assert error.message == "Configuration error"

    def test_flext_error_with_context(self) -> None:
        """Test FlextError with custom context."""
        context: dict[str, object] = {
            "component": "test_component",
            "operation": "test_operation",
        }
        error = FlextError(
            "Context error",
            context=context,
        )

        if error.context["component"] != "test_component":

            raise AssertionError(f"Expected {"test_component"}, got {error.context["component"]}")
        assert error.context["operation"] == "test_operation"
        if error.message != "Context error":
            raise AssertionError(f"Expected {"Context error"}, got {error.message}")

    def test_flext_error_with_full_context(self) -> None:
        """Test FlextError with full context and error code."""
        context: dict[str, object] = {
            "component": "test_component",
            "operation": "test_operation",
            "user_id": "test_user",
        }

        error = FlextError(
            "Context error",
            error_code=ERROR_CODES["VALIDATION_ERROR"],
            context=context,
        )

        if error.context["component"] != "test_component":

            raise AssertionError(f"Expected {"test_component"}, got {error.context["component"]}")
        assert error.context["operation"] == "test_operation"
        if error.context["user_id"] != "test_user":
            raise AssertionError(f"Expected {"test_user"}, got {error.context["user_id"]}")
        assert error.error_code == ERROR_CODES["VALIDATION_ERROR"]

    def test_flext_error_context_enhancement(self) -> None:
        """Test automatic context enhancement."""
        error = FlextError("Test error")

        # Context should be available even if not explicitly set
        assert isinstance(error.context, dict)
        assert isinstance(error.timestamp, float)
        assert error.timestamp > 0

    def test_flext_error_serialization(self) -> None:
        """Test FlextError serialization."""
        error = FlextError(
            "Serialization test",
            error_code=ERROR_CODES["OPERATION_ERROR"],
            context={"test_key": "test_value"},
        )

        serialized = error.to_dict()

        assert isinstance(serialized, dict)
        if serialized["message"] != "Serialization test":
            raise AssertionError(f"Expected {"Serialization test"}, got {serialized["message"]}")
        assert serialized["error_code"] == ERROR_CODES["OPERATION_ERROR"]
        if "context" not in serialized:
            raise AssertionError(f"Expected {"context"} in {serialized}")
        context = serialized["context"]
        assert isinstance(context, dict)
        if context["test_key"] != "test_value":
            raise AssertionError(f"Expected {"test_value"}, got {context["test_key"]}")
        if "timestamp" not in serialized:
            raise AssertionError(f"Expected {"timestamp"} in {serialized}")
        assert isinstance(serialized["timestamp"], float)

    def test_flext_error_str_representation(self) -> None:
        """Test string representation of FlextError."""
        error = FlextError("String test error")
        str_repr = str(error)

        if "String test error" not in str_repr:

            raise AssertionError(f"Expected {"String test error"} in {str_repr}")
        assert isinstance(str_repr, str)
        if str_repr != "[GENERIC_ERROR] String test error":
            raise AssertionError(f"Expected {"[GENERIC_ERROR] String test error"}, got {str_repr}")

    def test_flext_error_repr_representation(self) -> None:
        """Test repr representation of FlextError."""
        error = FlextError("Repr test error")
        repr_str = repr(error)

        if "FlextError" not in repr_str:

            raise AssertionError(f"Expected {"FlextError"} in {repr_str}")
        assert "Repr test error" in repr_str

    def test_flext_error_inheritance(self) -> None:
        """Test FlextError inheritance from Exception."""
        error = FlextError("Inheritance test")

        assert isinstance(error, Exception)
        assert isinstance(error, FlextError)

    def test_flext_error_raise_and_catch(self) -> None:
        """Test raising and catching FlextError."""
        test_message = "Test exception"
        with pytest.raises(FlextError) as exc_info:
            raise FlextError(test_message)

        caught_error = exc_info.value
        if caught_error.message != test_message:
            raise AssertionError(f"Expected {test_message}, got {caught_error.message}")
        assert isinstance(caught_error, FlextError)


class TestFlextValidationError:
    """Test FlextValidationError functionality."""

    def test_validation_error_basic(self) -> None:
        """Test basic validation error creation."""
        error = FlextValidationError("Field validation failed")

        if error.message != "Field validation failed":

            raise AssertionError(f"Expected {"Field validation failed"}, got {error.message}")
        assert error.error_code == ERROR_CODES["VALIDATION_ERROR"]
        assert isinstance(error, FlextError)

    def test_validation_error_with_field_details(self) -> None:
        """Test validation error with field-specific context."""
        validation_details: dict[str, object] = {
            "field": "email",
            "value": "invalid-email",
            "rules": ["email_format"],
        }

        error = FlextValidationError(
            "Email validation failed",
            validation_details=validation_details,
        )

        if error.context["field"] != "email":

            raise AssertionError(f"Expected {"email"}, got {error.context["field"]}")
        assert error.context["value"] == "invalid-email"
        if error.field != "email":
            raise AssertionError(f"Expected {"email"}, got {error.field}")
        assert error.value == "invalid-email"

    def test_validation_error_multiple_fields(self) -> None:
        """Test validation error with multiple field failures."""
        validation_details: dict[str, object] = {
            "field": "user_data",
            "rules": ["required", "format", "length"],
        }

        error = FlextValidationError(
            "Multiple validation errors",
            validation_details=validation_details,
        )

        if error.field != "user_data":

            raise AssertionError(f"Expected {"user_data"}, got {error.field}")
        assert error.context["field"] == "user_data"
        if error.rules != ["required", "format", "length"]:
            raise AssertionError(f"Expected {["required", "format", "length"]}, got {error.rules}")

    def test_validation_error_serialization(self) -> None:
        """Test validation error serialization includes validation details."""
        validation_details: dict[str, object] = {"field": "username", "value": "ab"}
        error = FlextValidationError(
            "Username too short",
            validation_details=validation_details,
        )

        serialized = error.to_dict()

        if "context" not in serialized:

            raise AssertionError(f"Expected {"context"} in {serialized}")
        context = serialized["context"]
        assert isinstance(context, dict)
        if context["field"] != "username":
            raise AssertionError(f"Expected {"username"}, got {context["field"]}")
        assert context["value"] == "ab"


class TestFlextTypeError:
    """Test FlextTypeError functionality."""

    def test_type_error_basic(self) -> None:
        """Test basic type error creation."""
        error = FlextTypeError("Type conversion failed")

        if error.message != "Type conversion failed":

            raise AssertionError(f"Expected {"Type conversion failed"}, got {error.message}")
        assert error.error_code == ERROR_CODES["TYPE_ERROR"]
        assert isinstance(error, FlextError)

    def test_type_error_with_type_info(self) -> None:
        """Test type error with type information."""
        error = FlextTypeError(
            "Cannot convert string to int",
            expected_type=str,
            actual_type=int,
            context={
                "value": "not_a_number",
                "operation": "type_conversion",
            },
        )

        assert error.expected_type is str
        assert error.actual_type is int
        if error.context["expected_type"] != str(str):
            raise AssertionError(f"Expected {str(str)}, got {error.context["expected_type"]}")
        assert error.context["actual_type"] == str(int)

    def test_type_error_with_conversion_details(self) -> None:
        """Test type error with conversion context."""
        context: dict[str, object] = {
            "source": "user_input",
            "target": "database_field",
            "converter": "int()",
            "reason": "invalid_literal",
        }

        error = FlextTypeError(
            "Type conversion error in user input",
            context=context,
        )

        if error.context["source"] != "user_input":

            raise AssertionError(f"Expected {"user_input"}, got {error.context["source"]}")
        assert error.context["converter"] == "int()"
        if error.error_code != ERROR_CODES["TYPE_ERROR"]:
            raise AssertionError(f"Expected {ERROR_CODES["TYPE_ERROR"]}, got {error.error_code}")


class TestFlextOperationError:
    """Test FlextOperationError functionality."""

    def test_operation_error_basic(self) -> None:
        """Test basic operation error creation."""
        error = FlextOperationError("Operation failed")

        if error.message != "Operation failed":

            raise AssertionError(f"Expected {"Operation failed"}, got {error.message}")
        assert error.error_code == ERROR_CODES["OPERATION_ERROR"]
        assert isinstance(error, FlextError)

    def test_operation_error_with_operation_info(self) -> None:
        """Test operation error with operation details."""
        error = FlextOperationError(
            "File read operation failed",
            operation="file_read",
            stage="file_open",
            context={
                "file_path": "/path/to/file.txt",
                "attempt": 1,
            },
        )

        if error.operation != "file_read":

            raise AssertionError(f"Expected {"file_read"}, got {error.operation}")
        assert error.stage == "file_open"
        if error.context["operation"] != "file_read":
            raise AssertionError(f"Expected {"file_read"}, got {error.context["operation"]}")
        assert error.context["stage"] == "file_open"

    def test_operation_error_with_retry_info(self) -> None:
        """Test operation error with retry context."""
        error = FlextOperationError(
            "API call failed after retries",
            operation="api_call",
            context={
                "endpoint": "/api/users",
                "method": "POST",
                "attempt": 3,
                "max_retries": 3,
                "last_error": "connection_timeout",
            },
        )

        if error.context["attempt"] != EXPECTED_DATA_COUNT:

            raise AssertionError(f"Expected {3}, got {error.context["attempt"]}")
        assert error.context["max_retries"] == EXPECTED_DATA_COUNT
        if error.context["last_error"] != "connection_timeout":
            raise AssertionError(f"Expected {"connection_timeout"}, got {error.context["last_error"]}")
        assert error.error_code == ERROR_CODES["OPERATION_ERROR"]


class TestSpecificErrors:
    """Test specific domain error types."""

    def test_configuration_error(self) -> None:
        """Test FlextConfigurationError."""
        error = FlextConfigurationError("Invalid configuration")

        if error.message != "Invalid configuration":

            raise AssertionError(f"Expected {"Invalid configuration"}, got {error.message}")
        assert error.error_code == ERROR_CODES["CONFIG_ERROR"]
        assert isinstance(error, FlextError)

    def test_connection_error(self) -> None:
        """Test FlextConnectionError."""
        error = FlextConnectionError("Database connection failed")

        if error.message != "Database connection failed":

            raise AssertionError(f"Expected {"Database connection failed"}, got {error.message}")
        assert error.error_code == ERROR_CODES["CONNECTION_ERROR"]
        assert isinstance(error, FlextError)

    def test_authentication_error(self) -> None:
        """Test FlextAuthenticationError."""
        error = FlextAuthenticationError("Invalid credentials")

        if error.message != "Invalid credentials":

            raise AssertionError(f"Expected {"Invalid credentials"}, got {error.message}")
        assert error.error_code == ERROR_CODES["AUTH_ERROR"]
        assert isinstance(error, FlextError)

    def test_permission_error(self) -> None:
        """Test FlextPermissionError."""
        error = FlextPermissionError("Access denied")

        if error.message != "Access denied":

            raise AssertionError(f"Expected {"Access denied"}, got {error.message}")
        assert error.error_code == ERROR_CODES["PERMISSION_ERROR"]
        assert isinstance(error, FlextError)

    def test_specific_errors_with_context(self) -> None:
        """Test specific errors with additional context."""
        config_error = FlextConfigurationError(
            "Missing configuration key",
            config_file="/etc/app/config.yml",
            missing_key="database.host",
        )

        if config_error.context["config_file"] != "/etc/app/config.yml":

            raise AssertionError(f"Expected {"/etc/app/config.yml"}, got {config_error.context["config_file"]}")
        assert config_error.context["missing_key"] == "database.host"

    def test_error_code_consistency(self) -> None:
        """Test that error codes are consistent across error types."""
        validation_error = FlextValidationError("Test")
        type_error = FlextTypeError("Test")
        operation_error = FlextOperationError("Test")
        config_error = FlextConfigurationError("Test")

        # Each should have different error codes
        error_codes = {
            validation_error.error_code,
            type_error.error_code,
            operation_error.error_code,
            config_error.error_code,
        }

        if len(error_codes) != 4  # All different:

            raise AssertionError(f"Expected {4  # All different}, got {len(error_codes)}")


class TestExceptionIntegration:
    """Test exception integration with other FLEXT Core components."""

    def test_exception_with_result_pattern(self) -> None:
        """Test using exceptions with FlextResult pattern."""

        def risky_operation() -> FlextResult[str]:
            try:
                # Simulate an operation that might fail
                validation_error_message = "Validation failed"
                raise FlextValidationError(validation_error_message)
            except FlextError as e:
                return FlextResult.fail(str(e))

        result = risky_operation()

        assert result.is_failure
        assert result.error is not None
        if "Validation failed" not in result.error:
            raise AssertionError(f"Expected {"Validation failed"} in {result.error}")

    def test_exception_context_enhancement(self) -> None:
        """Test automatic context enhancement in exceptions."""
        error = FlextOperationError(
            "Test operation error",
            context={"test": "value"},
        )

        # Should have context and automatic fields
        if "test" not in error.context:
            raise AssertionError(f"Expected {"test"} in {error.context}")
        assert isinstance(error.timestamp, float)
        assert error.timestamp > 0

    def test_exception_serialization_roundtrip(self) -> None:
        """Test exception serialization and data preservation."""
        original_context: dict[str, object] = {
            "component": "user_service",
            "operation": "create_user",
            "user_data": {"name": "test", "email": "test@example.com"},
        }

        error = FlextValidationError(
            "User creation validation failed",
            validation_details={"field": "email", "value": "invalid_format"},
            context=original_context,
        )

        serialized = error.to_dict()

        # Check all data is preserved in serialization
        if serialized["message"] != "User creation validation failed":
            raise AssertionError(f"Expected {"User creation validation failed"}, got {serialized["message"]}")
        context = serialized["context"]
        assert isinstance(context, dict)
        if context["component"] != "user_service":
            raise AssertionError(f"Expected {"user_service"}, got {context["component"]}")
        assert context["field"] == "email"

    def test_exception_chaining_patterns(self) -> None:
        """Test exception chaining and cause tracking."""

        def _raise_original_error() -> None:
            original_message = "Original error"
            raise ValueError(original_message)

        def _chain_exceptions() -> None:
            try:
                _raise_original_error()
            except ValueError as e:
                operation_message = "Operation failed due to validation error"
                raise FlextOperationError(
                    operation_message,
                    context={"original_error": str(e)},
                ) from e

        with pytest.raises(FlextOperationError) as exc_info:
            _chain_exceptions()

        flext_error = exc_info.value
        original_error = flext_error.context["original_error"]
        assert isinstance(original_error, str)
        if "Original error" not in original_error:
            raise AssertionError(f"Expected {"Original error"} in {original_error}")
        assert flext_error.__cause__ is not None
        assert isinstance(flext_error.__cause__, ValueError)


class TestErrorCodeIntegration:
    """Test integration with error code system."""

    def test_error_code_enum_usage(self) -> None:
        """Test using error codes with exceptions."""
        error = FlextError(
            "Test error",
            error_code=ERROR_CODES["VALIDATION_ERROR"],
        )

        if error.error_code != ERROR_CODES["VALIDATION_ERROR"]:

            raise AssertionError(f"Expected {ERROR_CODES["VALIDATION_ERROR"]}, got {error.error_code}")
        assert isinstance(error.error_code, str)

    def test_error_context_variations(self) -> None:
        """Test error creation with different context variations."""
        low_error = FlextError("Basic error", context={"priority": "low"})
        high_error = FlextError("Important error", context={"priority": "high"})

        if low_error.context["priority"] != "low":

            raise AssertionError(f"Expected {"low"}, got {low_error.context["priority"]}")
        assert high_error.context["priority"] == "high"
        assert low_error.context != high_error.context

    def test_default_error_codes(self) -> None:
        """Test default error codes for each exception type."""
        # Test that each exception type has appropriate default error code
        errors = [
            (FlextValidationError("test"), ERROR_CODES["VALIDATION_ERROR"]),
            (FlextTypeError("test"), ERROR_CODES["TYPE_ERROR"]),
            (FlextOperationError("test"), ERROR_CODES["OPERATION_ERROR"]),
            (FlextConfigurationError("test"), ERROR_CODES["CONFIG_ERROR"]),
            (FlextConnectionError("test"), ERROR_CODES["CONNECTION_ERROR"]),
            (FlextAuthenticationError("test"), ERROR_CODES["AUTH_ERROR"]),
            (FlextPermissionError("test"), ERROR_CODES["PERMISSION_ERROR"]),
        ]

        for error, expected_code in errors:
            if error.error_code != expected_code:
                raise AssertionError(f"Expected {expected_code}, got {error.error_code}")


class TestExceptionEdgeCases:
    """Test edge cases and error conditions in exception handling."""

    def test_exception_with_none_message(self) -> None:
        """Test exception with empty message."""
        # Test with empty string instead of None since FlextError expects str
        error = FlextError("")
        if error.message != "":
            raise AssertionError(f"Expected {""}, got {error.message}")

    def test_exception_with_empty_context(self) -> None:
        """Test exception with empty context."""
        error = FlextError("Test", context={})

        # Should still have automatic fields
        assert isinstance(error.context, dict)
        assert isinstance(error.timestamp, float)
        assert error.timestamp > 0

    def test_exception_with_complex_context(self) -> None:
        """Test exception with complex nested context."""
        complex_context: dict[str, object] = {
            "nested": {
                "data": {"key": "value"},
                "list": [1, 2, 3],
                "tuple": (4, 5, 6),
            },
            "simple": "value",
        }

        error = FlextError("Complex context test", context=complex_context)

        nested_context = error.context["nested"]
        assert isinstance(nested_context, dict)
        data_context = nested_context["data"]
        assert isinstance(data_context, dict)
        if data_context["key"] != "value":
            raise AssertionError(f"Expected {"value"}, got {data_context["key"]}")
        assert nested_context["list"] == [1, 2, 3]
        if error.context["simple"] != "value":
            raise AssertionError(f"Expected {"value"}, got {error.context["simple"]}")

    def test_exception_serialization_with_non_serializable_context(self) -> None:
        """Test exception serialization with non-serializable context data."""
        # Objects that can't be easily serialized
        non_serializable_context = {
            "function": lambda x: x,
            "set": {1, 2, 3},
            "complex": complex(1, 2),
        }

        error = FlextError("Non-serializable context", context=non_serializable_context)

        # to_dict() doesn't serialize deeply, so it should succeed
        result = error.to_dict()
        if result["context"] != non_serializable_context:
            raise AssertionError(f"Expected {non_serializable_context}, got {result["context"]}")

        # But JSON serialization should fail


        with pytest.raises((TypeError, ValueError)):
            json.dumps(result)

    def test_exception_memory_efficiency(self) -> None:
        """Test exception memory efficiency with large contexts."""
        # Create many exceptions to test memory efficiency
        errors = []
        for i in range(100):
            error = FlextError(
                f"Error {i}",
                context={"index": i, "data": f"data_{i}"},
            )
            errors.append(error)

        if len(errors) != 100:

            raise AssertionError(f"Expected {100}, got {len(errors)}")

        # All should be properly constructed
        for i, error in enumerate(errors):
            if error.context["index"] != i:
                raise AssertionError(f"Expected {i}, got {error.context["index"]}")
            if f"Error {i}" not in error.message:
                raise AssertionError(f"Expected {f"Error {i}"} in {error.message}")

    def test_exception_thread_safety_basic(self) -> None:
        """Test basic thread safety of exception creation."""
        # Create exceptions in rapid succession
        errors = []
        for _ in range(50):
            error = FlextError("Thread safety test")
            errors.append(error)

        # All should have proper timestamps
        timestamps = [error.timestamp for error in errors]

        # All should have timestamps
        if all(ts is not None for ts not in timestamps):
            raise AssertionError(f"Expected {all(ts is not None for ts} in {timestamps)}")

        # Timestamps should be reasonable (all within last few seconds)


        current_time = time.time()
        for ts in timestamps:
            if ts is not None:
                assert abs(current_time - ts) < 10  # Within 10 seconds
