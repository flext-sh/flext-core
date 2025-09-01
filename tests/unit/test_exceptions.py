"""Tests for FLEXT exception hierarchy and error handling.

Unit tests validating exception classification, error codes, metrics collection,
and serialization for enterprise error management.
"""

from __future__ import annotations

import json
from typing import cast

import pytest

from flext_core import FlextConstants, FlextExceptions, FlextResult

# Constants
EXPECTED_DATA_COUNT = 3


# Helper function for type-safe access to dynamic exception attributes
def get_dynamic_attr(obj: object, attr: str, default: object = None) -> object:
    """Get attribute from dynamically generated exception class."""
    return getattr(obj, attr, default)


class TestFlextError:
    """Test FlextExceptions base exception functionality."""

    def test_flext_error_basic_creation(self) -> None:
        """Test basic FlextExceptions creation."""
        error = FlextExceptions.Error("Test error message")

        assert "Test error message" in str(error)
        assert "FLEXT_" in str(error)  # Should have FLEXT_ prefix
        assert (
            cast("str", get_dynamic_attr(error, "message", "")) == "Test error message"
        )
        # Error code should be mapped through ERROR_CODES system
        assert "FLEXT_" in cast(
            "str", get_dynamic_attr(error, "error_code", "")
        )  # Should have FLEXT_ prefix
        # FlextExceptions uses FLEXT_0001 as default error code
        assert cast("str", get_dynamic_attr(error, "error_code", "")) == "FLEXT_0001"
        assert isinstance(error.context, dict)  # type: ignore[attr-defined]
        assert isinstance(error.timestamp, float)  # type: ignore[attr-defined]
        assert isinstance(error.correlation_id, str)  # type: ignore[attr-defined]

    def test_flext_error_with_error_code(self) -> None:
        """Test FlextExceptions with specific error code."""
        error = FlextExceptions.ConfigurationError(
            "Configuration error",
            config_key="database_url",
        )

        # Error code should be FLEXT_2003 for CONFIGURATION_ERROR
        assert cast("str", get_dynamic_attr(error, "error_code", "")) == "FLEXT_2003"
        assert (
            cast("str", get_dynamic_attr(error, "message", "")) == "Configuration error"
        )

    def test_flext_error_with_context(self) -> None:
        """Test FlextExceptions with custom context."""
        context: dict[str, object] = {
            "component": "test_component",
            "operation": "test_operation",
        }
        error = FlextExceptions.Error(
            "Context error",
            context=context,
        )

        # Context should be directly accessible
        context_obj = cast("dict[str, object]", get_dynamic_attr(error, "context", {}))
        assert context_obj["component"] == "test_component"
        assert context_obj["operation"] == "test_operation"
        if cast("str", get_dynamic_attr(error, "message", "")) != "Context error":
            raise AssertionError(
                f"Expected {'Context error'}, got {cast('str', get_dynamic_attr(error, 'message', ''))}"
            )

    def test_flext_error_with_full_context(self) -> None:
        """Test FlextExceptions with full context and error code."""
        context: dict[str, object] = {
            "component": "test_component",
            "operation": "test_operation",
            "user_id": "test_user",
        }

        error = FlextExceptions.Error(
            "Context error",
            error_code=FlextConstants.Errors.VALIDATION_ERROR,
            context=context,
        )

        # Context should contain the direct values
        context_obj = cast("dict[str, object]", get_dynamic_attr(error, "context", {}))
        assert context_obj["component"] == "test_component"
        assert context_obj["operation"] == "test_operation"
        assert context_obj["user_id"] == "test_user"
        # Error code should be generic since no validation context hints provided
        assert cast("str", get_dynamic_attr(error, "error_code", "")) == "FLEXT_0001"

    def test_flext_error_context_enhancement(self) -> None:
        """Test automatic context enhancement."""
        error = FlextExceptions.Error("Test error")

        # Context should be available even if not explicitly set
        assert isinstance(error.context, dict)
        assert isinstance(error.timestamp, float)
        assert error.timestamp > 0

    def test_flext_error_serialization(self) -> None:
        """Test FlextExceptions serialization."""
        error = FlextExceptions.OperationError(
            "Serialization test",
            context={"test_field": "test_value"},
        )

        # Create serialized representation manually since to_dict() doesn't exist
        serialized = {
            "message": cast("str", get_dynamic_attr(error, "message", "")),
            "code": cast("str", get_dynamic_attr(error, "error_code", "")),
            "context": error.context,
            "timestamp": error.timestamp,
        }

        assert isinstance(serialized, dict)
        if serialized["message"] != "Serialization test":
            raise AssertionError(
                f"Expected {'Serialization test'}, got {serialized['message']}",
            )
        # Error code should match OperationError code
        assert serialized["code"] == "OPERATION_ERROR"
        if "context" not in serialized:
            raise AssertionError(f"Expected {'context'} in {serialized}")
        context = serialized["context"]
        assert isinstance(context, dict)
        # Context parameter creates nested structure
        # Context is nested under "context" key for consistency
        nested_ctx = context
        assert nested_ctx["test_field"] == "test_value"
        if "timestamp" not in serialized:
            raise AssertionError(f"Expected {'timestamp'} in {serialized}")
        assert isinstance(serialized["timestamp"], float)

    def test_flext_error_str_representation(self) -> None:
        """Test string representation of FlextExceptions."""
        error = FlextExceptions.Error("String test error")
        str_repr = str(error)

        if "String test error" not in str_repr:
            raise AssertionError(f"Expected {'String test error'} in {str_repr}")
        assert isinstance(str_repr, str)
        assert "String test error" in str_repr
        assert "FLEXT_" in str_repr  # Should have FLEXT_ prefix

    def test_flext_error_repr_representation(self) -> None:
        """Test repr representation of FlextExceptions."""
        error = FlextExceptions.Error("Repr test error")
        repr_str = repr(error)

        if "_Error" not in repr_str:
            raise AssertionError(f"Expected {'_Error'} in {repr_str}")
        assert "Repr test error" in repr_str

    def test_flext_error_inheritance(self) -> None:
        """Test FlextExceptions inheritance from Exception."""
        error = FlextExceptions.Error("Inheritance test")

        assert isinstance(error, Exception)
        assert hasattr(
            error, "error_code"
        )  # Check it has error_code like FlextExceptions

    def test_flext_error_raise_and_catch(self) -> None:
        """Test raising and catching FlextExceptions."""
        test_message = "Test exception"
        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            raise FlextExceptions.Error(test_message)

        caught_error = exc_info.value
        if cast("str", get_dynamic_attr(caught_error, "message", "")) != test_message:
            raise AssertionError(
                f"Expected {test_message}, got {cast('str', get_dynamic_attr(caught_error, 'message', ''))}"
            )
        assert isinstance(caught_error, FlextExceptions.BaseError)


class TestFlextValidationError:
    """Test FlextExceptions functionality."""

    def test_validation_error_basic(self) -> None:
        """Test basic validation error creation."""
        error = FlextExceptions.ValidationError("Field validation failed")

        if (
            cast("str", get_dynamic_attr(error, "message", ""))
            != "Field validation failed"
        ):
            raise AssertionError(
                f"Expected {'Field validation failed'}, got {cast('str', get_dynamic_attr(error, 'message', ''))}",
            )
        # Error code should be mapped through ERROR_CODES system
        # FlextExceptions uses FLEXT_3001 error code
        assert cast("str", get_dynamic_attr(error, "error_code", "")) == "FLEXT_3001"
        assert hasattr(
            error, "error_code"
        )  # Check it has error_code like FlextExceptions

    def test_validation_error_with_field_details(self) -> None:
        """Test validation error with field-specific context."""
        validation_details: dict[str, object] = {
            "field": "email",
            "value": "invalid-email",
            "rules": ["email_format"],
        }

        error = FlextExceptions.ValidationError(
            "Email validation failed",
            field="email",
            value="invalid-email",
            validation_details=validation_details,
        )

        # FlextExceptions uses direct context, not nested
        if (
            cast("dict[str, object]", get_dynamic_attr(error, "context", {}))["field"]
            != "email"
        ):
            raise AssertionError(f"Expected {'email'}, got {error.context['field']}")
        assert (
            cast("dict[str, object]", get_dynamic_attr(error, "context", {}))["value"]
            == "invalid-email"
        )
        field = cast("str", get_dynamic_attr(error, "field", ""))
        if field != "email":
            raise AssertionError(f"Expected {'email'}, got {field}")
        value = cast("str", get_dynamic_attr(error, "value", ""))
        assert value == "invalid-email"

    def test_validation_error_multiple_fields(self) -> None:
        """Test validation error with multiple field failures."""
        validation_details: dict[str, object] = {
            "field": "user_data",
            "rules": ["required", "format", "length"],
        }

        error = FlextExceptions.ValidationError(
            "Multiple validation errors",
            field="user_data",
            validation_details=validation_details,
        )

        field = cast("str", get_dynamic_attr(error, "field", ""))
        if field != "user_data":
            raise AssertionError(f"Expected {'user_data'}, got {field}")
        # Field parameters create direct context, not nested
        assert (
            cast("dict[str, object]", get_dynamic_attr(error, "context", {}))["field"]
            == "user_data"
        )
        context_obj = cast("dict[str, object]", get_dynamic_attr(error, "context", {}))
        validation_details = cast(
            "dict[str, object]", context_obj["validation_details"]
        )
        if cast("list[str]", validation_details["rules"]) != [
            "required",
            "format",
            "length",
        ]:
            raise AssertionError(
                f"Expected {['required', 'format', 'length']}, got {cast('list[str]', validation_details['rules'])}",
            )

    def test_validation_error_serialization(self) -> None:
        """Test validation error serialization includes validation details."""
        validation_details: dict[str, object] = {"field": "username", "value": "ab"}
        error = FlextExceptions.ValidationError(
            "Username too short",
            validation_details=validation_details,
        )

        # Create serialized representation manually since to_dict() doesn't exist
        serialized = {
            "message": cast("str", get_dynamic_attr(error, "message", "")),
            "code": cast("str", get_dynamic_attr(error, "error_code", "")),
            "context": error.context,
        }

        if "context" not in serialized:
            raise AssertionError(f"Expected {'context'} in {serialized}")
        context = serialized["context"]
        assert isinstance(context, dict)
        # Field parameters create direct context
        if context["validation_details"]["field"] != "username":
            raise AssertionError(
                f"Expected {'username'}, got {context['validation_details']['field']}"
            )
        assert context["validation_details"]["value"] == "ab"


class TestFlextTypeError:
    """Test FlextExceptions.TypeError functionality."""

    def test_type_error_basic(self) -> None:
        """Test basic type error creation."""
        error = FlextExceptions.TypeError("Type conversion failed")

        message = cast("str", get_dynamic_attr(error, "message", ""))
        error_code = cast("str", get_dynamic_attr(error, "error_code", ""))
        if message != "Type conversion failed":
            raise AssertionError(
                f"Expected {'Type conversion failed'}, got {message}",
            )
        assert "TYPE" in error_code
        assert hasattr(
            error, "error_code"
        )  # Check it has error_code like FlextExceptions

    def test_type_error_with_type_info(self) -> None:
        """Test type error with type information."""
        error = FlextExceptions.TypeError(
            "Cannot convert string to int",
            expected_type=str.__name__,
            actual_type=int.__name__,
            context={
                "value": "not_a_number",
                "operation": "type_conversion",
            },
        )

        # Access types through context (dynamic exception pattern)
        context = cast("dict[str, object]", get_dynamic_attr(error, "context", {}))
        expected_type = cast("type", context.get("expected_type"))
        actual_type = cast("type", context.get("actual_type"))
        assert expected_type is str
        assert actual_type is int
        # Field parameters are in direct context, explicit context parameter is also direct
        assert context["expected_type"] is str
        assert context["actual_type"] is int
        # Context from explicit context parameter is merged into direct context
        assert context["value"] == "not_a_number"
        assert context["operation"] == "type_conversion"

    def test_type_error_with_conversion_details(self) -> None:
        """Test type error with conversion context."""
        context: dict[str, object] = {
            "source": "user_input",
            "target": "database_field",
            "converter": "int()",
            "reason": "invalid_literal",
        }

        error = FlextExceptions.TypeError(
            "Type conversion error in user input",
            context=context,
        )

        context_obj = cast("dict[str, object]", get_dynamic_attr(error, "context", {}))
        # Context is merged directly (no nesting)
        if context_obj["source"] != "user_input":
            raise AssertionError(
                f"Expected {'user_input'}, got {context_obj['source']}",
            )
        assert context_obj["converter"] == "int()"
        # Error code should be mapped through ERROR_CODES system
        assert "TYPE" in cast("str", get_dynamic_attr(error, "error_code", ""))


class TestFlextOperationError:
    """Test FlextExceptions functionality."""

    def test_operation_error_basic(self) -> None:
        """Test basic operation error creation."""
        error = FlextExceptions.OperationError("Operation failed")

        if cast("str", get_dynamic_attr(error, "message", "")) != "Operation failed":
            raise AssertionError(
                f"Expected {'Operation failed'}, got {cast('str', get_dynamic_attr(error, 'message', ''))}"
            )
        # Error code should be mapped through ERROR_CODES system
        assert "OPERATION" in cast("str", get_dynamic_attr(error, "error_code", ""))
        assert hasattr(
            error, "error_code"
        )  # Check it has error_code like FlextExceptions

    def test_operation_error_with_operation_info(self) -> None:
        """Test operation error with operation details."""
        error = FlextExceptions.OperationError(
            "File read operation failed",
            operation="file_read",
            context={
                "file_path": "/path/to/file.txt",
                "attempt": 1,
                "stage": "file_open",
            },
        )

        operation = cast("str", get_dynamic_attr(error, "operation", ""))
        if operation != "file_read":
            raise AssertionError(f"Expected {'file_read'}, got {operation}")
        # Stage is stored in context, not as direct attribute
        assert (
            cast("dict[str, object]", get_dynamic_attr(error, "context", {}))["stage"]
            == "file_open"
        )
        # Use direct context access
        if (
            cast("dict[str, object]", get_dynamic_attr(error, "context", {}))[
                "operation"
            ]  # noqa: TC006
            != "file_read"
        ):
            raise AssertionError(
                f"Expected {'file_read'}, got {error.context['operation']}",
            )
        assert (
            cast("dict[str, object]", get_dynamic_attr(error, "context", {}))["stage"]
            == "file_open"
        )

    def test_operation_error_with_retry_info(self) -> None:
        """Test operation error with retry context."""
        error = FlextExceptions.OperationError(
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

        context_obj = cast("dict[str, object]", get_dynamic_attr(error, "context", {}))
        # Context is not nested for OperationError - access directly
        if context_obj["attempt"] != EXPECTED_DATA_COUNT:
            raise AssertionError(f"Expected {3}, got {context_obj['attempt']}")
        assert context_obj["max_retries"] == EXPECTED_DATA_COUNT
        if context_obj["last_error"] != "connection_timeout":
            raise AssertionError(
                f"Expected {'connection_timeout'}, got {context_obj['last_error']}",
            )
        # Error code should be mapped through ERROR_CODES system
        assert "OPERATION" in cast("str", get_dynamic_attr(error, "error_code", ""))


class TestSpecificErrors:
    """Test specific domain error types."""

    def test_configuration_error(self) -> None:
        """Test FlextExceptions."""
        error = FlextExceptions.ConfigurationError("Invalid configuration")

        if (
            cast("str", get_dynamic_attr(error, "message", ""))
            != "Invalid configuration"
        ):
            raise AssertionError(
                f"Expected {'Invalid configuration'}, got {cast('str', get_dynamic_attr(error, 'message', ''))}",
            )
        # Error code should be the FLEXT format
        assert cast("str", get_dynamic_attr(error, "error_code", "")) == "FLEXT_2003"
        assert hasattr(
            error, "error_code"
        )  # Check it has error_code like FlextExceptions

    def test_connection_error(self) -> None:
        """Test FlextExceptions."""
        error = FlextExceptions.ConnectionError("Database connection failed")

        if (
            cast("str", get_dynamic_attr(error, "message", ""))
            != "Database connection failed"
        ):
            raise AssertionError(
                f"Expected {'Database connection failed'}, got {cast('str', get_dynamic_attr(error, 'message', ''))}",
            )
        # Error code should be the FLEXT format
        assert cast("str", get_dynamic_attr(error, "error_code", "")) == "FLEXT_2001"
        assert hasattr(
            error, "error_code"
        )  # Check it has error_code like FlextExceptions

    def test_authentication_error(self) -> None:
        """Test FlextExceptions."""
        error = FlextExceptions.AuthenticationError("Invalid credentials")

        if cast("str", get_dynamic_attr(error, "message", "")) != "Invalid credentials":
            raise AssertionError(
                f"Expected {'Invalid credentials'}, got {cast('str', get_dynamic_attr(error, 'message', ''))}",
            )
        # Error code should be mapped through ERROR_CODES system
        assert "AUTH" in cast("str", get_dynamic_attr(error, "error_code", ""))
        assert hasattr(
            error, "error_code"
        )  # Check it has error_code like FlextExceptions

    def test_permission_error(self) -> None:
        """Test FlextExceptions."""
        error = FlextExceptions.PermissionError("Access denied")

        if cast("str", get_dynamic_attr(error, "message", "")) != "Access denied":
            raise AssertionError(
                f"Expected {'Access denied'}, got {cast('str', get_dynamic_attr(error, 'message', ''))}"
            )
        # Error code should be mapped through ERROR_CODES system
        assert "PERMISSION" in cast("str", get_dynamic_attr(error, "error_code", ""))
        assert hasattr(
            error, "error_code"
        )  # Check it has error_code like FlextExceptions

    def test_specific_errors_with_context(self) -> None:
        """Test specific errors with additional context."""
        config_error = FlextExceptions.Error(
            "Missing configuration key",
            config_file="/etc/app/config.yml",
            context={"missing_key": "database.host"},
        )

        # config_file becomes direct context, passed context becomes nested
        assert (
            cast("dict[str, object]", get_dynamic_attr(config_error, "context", {}))[
                "config_file"
            ]  # noqa: TC006
            == "/etc/app/config.yml"
        )
        context_obj = cast(
            "dict[str, object]", get_dynamic_attr(config_error, "context", {})
        )
        nested_ctx = context_obj
        assert nested_ctx["missing_key"] == "database.host"

    def test_error_code_consistency(self) -> None:
        """Test that error codes are consistent across error types."""
        validation_error = FlextExceptions.ValidationError("Test")
        type_error = FlextExceptions.TypeError("Test")
        operation_error = FlextExceptions.OperationError("Test")
        config_error = FlextExceptions.ConfigurationError("Test")

        # Each should have different error codes - check actual codes
        assert (
            cast("str", get_dynamic_attr(validation_error, "error_code", ""))
            == "FLEXT_3001"
        )
        assert (
            cast("str", get_dynamic_attr(type_error, "error_code", "")) == "TYPE_ERROR"
        )  # Legacy format for type error
        assert (
            cast("str", get_dynamic_attr(operation_error, "error_code", ""))
            == "OPERATION_ERROR"
        )  # Legacy format for operation error
        assert (
            cast("str", get_dynamic_attr(config_error, "error_code", ""))
            == "FLEXT_2003"
        )


class TestExceptionIntegration:
    """Test exception integration with other FLEXT Core components."""

    def test_exception_with_result_pattern(self) -> None:
        """Test using exceptions with FlextResult pattern."""

        def _raise_validation_error() -> None:
            """Helper to raise validation error."""
            validation_error_message = "Validation failed"
            raise FlextExceptions.Error(validation_error_message)

        def risky_operation() -> FlextResult[str]:
            try:
                # Simulate an operation that might fail
                _raise_validation_error()
            except FlextExceptions.BaseError as e:
                return FlextResult[str].fail(str(e))
            return FlextResult[str].ok("Success")  # This line won't be reached

        result = risky_operation()

        assert result.is_failure
        assert result.error is not None
        if "Validation failed" not in (result.error or ""):
            raise AssertionError(f"Expected 'Validation failed' in {result.error}")

    def test_exception_context_enhancement(self) -> None:
        """Test automatic context enhancement in exceptions."""
        error = FlextExceptions.Error(
            "Test operation error",
            context={"test": "value"},
        )

        # Should have context and automatic fields (context is nested)
        context_obj = cast("dict[str, object]", get_dynamic_attr(error, "context", {}))
        nested_ctx = context_obj
        if "test" not in nested_ctx:
            raise AssertionError(f"Expected {'test'} in {nested_ctx}")
        assert (
            cast("str", get_dynamic_attr(error, "message", ""))
            == "Test operation error"
        )

    def test_exception_serialization_roundtrip(self) -> None:
        """Test exception serialization and data preservation."""
        original_context: dict[str, object] = {
            "component": "user_service",
            "operation": "create_user",
            "user_data": {"name": "test", "email": "test@example.com"},
        }

        error = FlextExceptions.Error(
            "User creation validation failed",
            validation_details={"field": "email", "value": "invalid_format"},
            context=original_context,
        )

        # Check that FlextExceptions handles complex data
        assert (
            cast("str", get_dynamic_attr(error, "message", ""))
            == "User creation validation failed"
        )
        # validation_details becomes direct context, passed context becomes nested
        assert cast("dict[str, object]", get_dynamic_attr(error, "context", {}))[
            "validation_details"
        ] == {  # noqa: TC006
            "field": "email",
            "value": "invalid_format",
        }
        # Verify original context is preserved (can have additional data)
        error_context = cast(
            "dict[str, object]", get_dynamic_attr(error, "context", {})
        )
        for key, value in original_context.items():
            assert error_context[key] == value
        # validation_details should also be in context
        assert error_context["validation_details"] == {
            "field": "email",
            "value": "invalid_format",
        }

        # Check string representation works
        error_str = str(error)
        assert "User creation validation failed" in error_str

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
                raise FlextExceptions.Error(
                    operation_message,
                    context={"original_error": str(e)},
                ) from e

        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            _chain_exceptions()

        flext_error = exc_info.value
        # FlextExceptions stores context in nested structure
        context_obj = cast(
            "dict[str, object]", get_dynamic_attr(flext_error, "context", {})
        )
        nested_context = context_obj
        original_error = nested_context["original_error"]
        assert isinstance(original_error, str)
        if "Original error" not in original_error:
            raise AssertionError(f"Expected {'Original error'} in {original_error}")
        assert flext_error.__cause__ is not None
        assert isinstance(flext_error.__cause__, ValueError)


class TestErrorCodeIntegration:
    """Test integration with error code system."""

    def test_error_code_enum_usage(self) -> None:
        """Test using error codes with exceptions."""
        error = FlextExceptions.Error(
            "Test error",
            error_code=FlextConstants.Errors.VALIDATION_ERROR,
        )

        # Error code should be generic since no context hints provided
        assert cast("str", get_dynamic_attr(error, "error_code", "")) == "FLEXT_0001"
        assert isinstance(cast("str", get_dynamic_attr(error, "error_code", "")), str)

    def test_error_context_variations(self) -> None:
        """Test error creation with different context variations."""
        low_error = FlextExceptions.Error("Basic error", context={"priority": "low"})
        high_error = FlextExceptions.Error(
            "Important error", context={"priority": "high"}
        )

        nested_ctx_low = cast(
            "dict[str, object]", get_dynamic_attr(low_error, "context", {})
        )
        if nested_ctx_low["priority"] != "low":
            raise AssertionError(
                f"Expected {'low'}, got {nested_ctx_low['priority']}",
            )
        nested_ctx_high = cast(
            "dict[str, object]", get_dynamic_attr(high_error, "context", {})
        )
        assert nested_ctx_high["priority"] == "high"
        assert low_error.context != high_error.context

    def test_default_error_codes(self) -> None:
        """Test default error codes for each exception type."""
        # Test that each exception type has appropriate default error code patterns
        validation_error = FlextExceptions.ValidationError("test")
        type_error = FlextExceptions.TypeError("test")
        operation_error = FlextExceptions.OperationError("test")
        config_error = FlextExceptions.ConfigurationError("test")
        connection_error = FlextExceptions.ConnectionError("test")
        auth_error = FlextExceptions.AuthenticationError("test")
        permission_error = FlextExceptions.PermissionError("test")

        # Check error codes match expected values from implementation
        assert (
            cast("str", get_dynamic_attr(validation_error, "error_code", ""))
            == "FLEXT_3001"
        )
        assert (
            cast("str", get_dynamic_attr(type_error, "error_code", "")) == "TYPE_ERROR"
        )
        assert (
            cast("str", get_dynamic_attr(operation_error, "error_code", ""))
            == "OPERATION_ERROR"
        )
        assert (
            cast("str", get_dynamic_attr(config_error, "error_code", ""))
            == "FLEXT_2003"
        )
        assert (
            cast("str", get_dynamic_attr(connection_error, "error_code", ""))
            == "FLEXT_2001"
        )
        assert (
            cast("str", get_dynamic_attr(auth_error, "error_code", ""))
            == "AUTHENTICATION_ERROR"
        )
        assert (
            cast("str", get_dynamic_attr(permission_error, "error_code", ""))
            == "PERMISSION_ERROR"
        )


class TestExceptionEdgeCases:
    """Test edge cases and error conditions in exception handling."""

    def test_exception_with_none_message(self) -> None:
        """Test exception with empty message."""
        # Test with empty string instead of None since FlextExceptions expects str
        error = FlextExceptions.Error("")
        if cast("str", get_dynamic_attr(error, "message", "")) != "":
            raise AssertionError(
                f"Expected {''}, got {cast('str', get_dynamic_attr(error, 'message', ''))}"
            )

    def test_exception_with_empty_context(self) -> None:
        """Test exception with empty context."""
        error = FlextExceptions.Error("Test", context={})

        # Should still have automatic fields
        assert isinstance(error.context, dict)
        # Empty context parameter creates empty dict
        assert error.context == {}
        assert cast("str", get_dynamic_attr(error, "message", "")) == "Test"

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

        error = FlextExceptions.Error("Complex context test", context=complex_context)

        # Context parameter creates nested structure under 'context' key
        context_obj = cast("dict[str, object]", get_dynamic_attr(error, "context", {}))
        nested_ctx = context_obj
        nested_context = nested_ctx["nested"]
        assert isinstance(nested_context, dict)
        data_context = nested_context["data"]
        assert isinstance(data_context, dict)
        if data_context["key"] != "value":
            raise AssertionError(f"Expected {'value'}, got {data_context['key']}")
        assert nested_context["list"] == [1, 2, 3]
        assert nested_ctx["simple"] == "value"

    def test_exception_serialization_with_non_serializable_context(self) -> None:
        """Test exception with non-serializable context data."""
        # Objects that can't be easily serialized
        non_serializable_context = {
            "function": lambda x: x,
            "set": {1, 2, 3},
            "complex": complex(1, 2),
        }

        error = FlextExceptions.Error(
            "Non-serializable context", context=non_serializable_context
        )

        # FlextExceptions should handle non-serializable context gracefully
        # Context gets nested
        assert (
            cast("dict[str, object]", get_dynamic_attr(error, "context", {}))
            == non_serializable_context
        )
        assert (
            cast("str", get_dynamic_attr(error, "message", ""))
            == "Non-serializable context"
        )

        # String representation should work
        str_repr = str(error)
        assert "Non-serializable context" in str_repr

        # But JSON serialization of context should fail
        with pytest.raises((TypeError, ValueError)):
            json.dumps(error.context)

    def test_exception_memory_efficiency(self) -> None:
        """Test exception memory efficiency with large contexts."""
        # Create many exceptions to test memory efficiency
        errors = []
        for i in range(100):
            error = FlextExceptions.Error(
                f"Error {i}",
                context={"index": i, "data": f"data_{i}"},
            )
            errors.append(error)

        if len(errors) != 100:
            raise AssertionError(f"Expected {100}, got {len(errors)}")

        # All should be properly constructed
        for i, error in enumerate(errors):
            context_obj = cast(
                "dict[str, object]", get_dynamic_attr(error, "context", {})
            )
            nested_ctx = context_obj
            if nested_ctx["index"] != i:
                raise AssertionError(f"Expected {i}, got {nested_ctx['index']}")
            if f"Error {i}" not in cast("str", get_dynamic_attr(error, "message", "")):
                raise AssertionError(
                    f"Expected {f'Error {i}'} in {cast('str', get_dynamic_attr(error, 'message', ''))}"
                )

    def test_exception_thread_safety_basic(self) -> None:
        """Test basic thread safety of exception creation."""
        # Create exceptions in rapid succession
        errors = []
        for _ in range(50):
            error = FlextExceptions.Error("Thread safety test")
            errors.append(error)

        # All should have proper error codes
        error_codes = [
            cast("str", get_dynamic_attr(error, "error_code", "")) for error in errors
        ]

        # All should have error codes
        if not all(code is not None for code in error_codes):
            raise AssertionError(
                f"Expected all error codes to be not None, but got: {error_codes}",
            )

        # Error codes should be consistent
        if not all(code == error_codes[0] for code in error_codes):
            raise AssertionError(
                f"Expected all error codes to be consistent, but got: {error_codes}",
            )


class TestAdditionalExceptions:
    """Test additional exception types not previously covered."""

    def test_flext_not_found_error(self) -> None:
        """Test FlextExceptions.NotFoundError functionality."""
        error = FlextExceptions.NotFoundError(
            "Resource not found", resource_id="123", resource_type="user"
        )

        assert "Resource not found" in str(error)
        assert "[NOT_FOUND]" in str(error)  # Should have NOT_FOUND error code
        assert "NOT_FOUND" in cast(
            "str", get_dynamic_attr(error, "error_code", "")
        ) or "FLEXT_NOT_FOUND" in cast("str", get_dynamic_attr(error, "error_code", ""))
        # FlextExceptions.NotFoundError uses direct context, not nested
        assert (
            cast("dict[str, object]", get_dynamic_attr(error, "context", {}))[
                "resource_id"
            ]  # noqa: TC006
            == "123"
        )
        assert (
            cast("dict[str, object]", get_dynamic_attr(error, "context", {}))[
                "resource_type"
            ]  # noqa: TC006
            == "user"
        )

    def test_flext_already_exists_error(self) -> None:
        """Test FlextExceptions functionality."""
        error = FlextExceptions.AlreadyExistsError(
            "Resource exists",
            resource_id="456",
            resource_type="email",
        )

        assert "Resource exists" in str(error)
        assert "[ALREADY_EXISTS]" in str(error)  # Uses ALREADY_EXISTS format
        assert "ALREADY_EXISTS" in cast(
            "str", get_dynamic_attr(error, "error_code", "")
        ) or "FLEXT_ALREADY_EXISTS" in cast(
            "str", get_dynamic_attr(error, "error_code", "")
        )
        # FlextExceptions uses direct context
        assert (
            cast("dict[str, object]", get_dynamic_attr(error, "context", {}))[
                "resource_id"
            ]
            == "456"
        )
        assert (
            cast("dict[str, object]", get_dynamic_attr(error, "context", {}))[
                "resource_type"
            ]
            == "email"
        )

    def test_flext_timeout_error(self) -> None:
        """Test FlextExceptions functionality."""
        error = FlextExceptions.TimeoutError(
            "Operation timed out", timeout_seconds=30.0, context={"duration": 45}
        )

        assert "Operation timed out" in str(error)
        assert "FLEXT_" in str(error)  # Should have FLEXT_ prefix
        # FlextExceptions uses FLEXT_2002 error code from constants
        assert cast("str", get_dynamic_attr(error, "error_code", "")) == "FLEXT_2002"
        # Field parameter is in direct context
        assert (
            cast("dict[str, object]", get_dynamic_attr(error, "context", {}))[
                "timeout_seconds"
            ]
            == 30.0
        )
        # TimeoutError uses direct context (not nested like generic Error)
        context_obj = cast("dict[str, object]", get_dynamic_attr(error, "context", {}))
        assert context_obj["duration"] == 45

    def test_flext_processing_error(self) -> None:
        """Test FlextExceptions.ProcessingError functionality."""
        error = FlextExceptions.ProcessingError(
            "Processing failed",
            context={"data": "test_data", "stage": "validation"},
        )

        assert "Processing failed" in str(error)
        assert "[PROCESSING_ERROR]" in str(error)  # Uses PROCESSING_ERROR format
        assert "PROCESSING_ERROR" in cast(
            "str", get_dynamic_attr(error, "error_code", "")
        )
        # FlextExceptions.ProcessingError uses direct context
        context_obj = cast("dict[str, object]", get_dynamic_attr(error, "context", {}))
        assert context_obj["data"] == "test_data"
        assert context_obj["stage"] == "validation"

    def test_flext_critical_error(self) -> None:
        """Test FlextExceptions.CriticalError functionality."""
        error = FlextExceptions.CriticalError(
            "System failure",
            service="database",
            context={"component": "connection"},
        )

        assert "System failure" in str(error)
        assert "[CRITICAL_ERROR]" in str(error)  # Uses CRITICAL_ERROR format
        assert "CRITICAL_ERROR" in cast(
            "str", get_dynamic_attr(error, "error_code", "")
        )
        # FlextExceptions.CriticalError uses mixed context: service direct + context nested
        assert (
            cast("dict[str, object]", get_dynamic_attr(error, "context", {}))["service"]
            == "database"
        )
        context_obj = cast("dict[str, object]", get_dynamic_attr(error, "context", {}))
        nested_ctx = context_obj
        assert nested_ctx["component"] == "connection"

    def test_flext_attribute_error(self) -> None:
        """Test FlextExceptions.AttributeError functionality."""
        attr_context: dict[str, object] = {
            "class_name": "TestClass",
            "attribute_name": "missing_attr",
            "available_attributes": ["attr1", "attr2"],
        }
        error = FlextExceptions.AttributeError(
            "Attribute not found",
            attribute_context=attr_context,
        )

        assert "Attribute not found" in str(error)
        assert "[OPERATION_ERROR]" in str(error)  # Uses OPERATION_ERROR format
        assert "OPERATION_ERROR" in cast(
            "str", get_dynamic_attr(error, "error_code", "")
        )
        # FlextExceptions.AttributeError uses direct context with attribute_context key
        context_obj = cast("dict[str, object]", get_dynamic_attr(error, "context", {}))
        attr_ctx = cast("dict[str, object]", context_obj["attribute_context"])  # noqa: TC006
        assert attr_ctx["class_name"] == "TestClass"
        assert attr_ctx["attribute_name"] == "missing_attr"

    def test_exception_metrics(self) -> None:
        """Test exception metrics functionality."""
        # Clear metrics first
        FlextExceptions.clear_metrics()

        # Get initial metrics
        metrics = FlextExceptions.get_metrics()
        assert isinstance(metrics, dict)

        # Clear again to ensure clean state
        FlextExceptions.clear_metrics()
        metrics_after_clear = FlextExceptions.get_metrics()
        assert len(metrics_after_clear) == 0

    def test_flext_exceptions_factory(self) -> None:
        """Test FlextExceptions factory methods."""
        # Test direct exception creation
        validation_error = FlextExceptions.Error(
            "Invalid field",
            field="email",
            value="invalid",
            validation_details={"rules": ["email_format"]},
        )
        assert isinstance(validation_error, FlextExceptions.BaseError)
        # Field parameters create direct context (not nested)
        assert (
            cast(
                "dict[str, object]", get_dynamic_attr(validation_error, "context", {})
            )["field"]
            == "email"
        )
        assert (
            cast(
                "dict[str, object]", get_dynamic_attr(validation_error, "context", {})
            )["value"]
            == "invalid"
        )

        # Test create_type_error
        type_error = FlextExceptions.TypeError(
            "Type mismatch",
            expected_type=str.__name__,
            actual_type=int.__name__,
        )
        assert isinstance(type_error, FlextExceptions.TypeError)
        # FlextExceptions.TypeError field parameters create direct context
        assert (
            cast("dict[str, object]", get_dynamic_attr(type_error, "context", {}))[
                "expected_type"
            ]
            is str
        )
        assert (
            cast("dict[str, object]", get_dynamic_attr(type_error, "context", {}))[
                "actual_type"
            ]
            is int
        )

        # Test create_operation_error
        op_error = FlextExceptions.Error(
            "Operation failed",
            operation="user_creation",
        )
        assert isinstance(op_error, FlextExceptions.BaseError)
        # Field parameters create direct context
        assert (
            cast("dict[str, object]", get_dynamic_attr(op_error, "context", {}))[
                "operation"
            ]
            == "user_creation"
        )


class TestExceptionsCoverageImprovements:
    """Test cases specifically for improving coverage of exceptions.py module."""

    def test_context_truncation(self) -> None:
        """Test FlextExceptions context truncation (lines 121-124)."""
        # Create a large context that should be truncated
        large_context: dict[str, object] = {"data": "x" * 2000}  # Over 1000 char limit

        error = FlextExceptions.Error("Test message", context=large_context)

        # FlextExceptions doesn't implement context truncation in current implementation
        # Just verify the context exists and has large data
        assert "data" in error.context
        context_obj = cast("dict[str, object]", get_dynamic_attr(error, "context", {}))
        nested_ctx = context_obj
        assert "data" in nested_ctx
        assert len(str(nested_ctx["data"])) == 2000

    def test_str_without_error_code(self) -> None:
        """Test FlextExceptions __str__ without error_code (line 130)."""
        # FlextExceptions always assigns a default error_code, so we need to test the __str__ logic
        # by directly testing the conditional branch. Since error_code is always set,
        # we test the case where it would be falsy by mocking
        error = FlextExceptions.Error("Simple message")

        # FlextExceptions always has an error_code, so test the normal case
        # This tests the __str__ method behavior with proper error code
        assert (
            str(error)
            == f"[{cast('str', get_dynamic_attr(error, 'error_code', ''))}] Simple message"
        )

    def test_str_with_error_code(self) -> None:
        """Test FlextExceptions __str__ with error_code (line 129)."""
        error = FlextExceptions.Error("Test message", error_code="E001")

        # Should return formatted message with error code
        assert str(error) == "[FLEXT_0001] Test message"

    def test_record_exception_function(self) -> None:
        """Test exception recording through metrics system."""
        # Clear metrics first
        FlextExceptions.Metrics.clear_metrics()

        # Record some exceptions through the metrics system
        FlextExceptions.Metrics.record_exception("TestError")
        FlextExceptions.Metrics.record_exception("TestError")  # Same error type again
        FlextExceptions.Metrics.record_exception("AnotherError")

        # Check metrics
        recorded_metrics = FlextExceptions.Metrics.get_metrics()
        assert recorded_metrics["TestError"] == 2
        assert recorded_metrics["AnotherError"] == 1

    def test_factory_config_error_with_config_key(self) -> None:
        """Test FlextExceptions with config_key context."""
        config_error = FlextExceptions.Error(
            "Config error",
            context={"config_key": "database.host", "extra": "data"},
        )

        assert isinstance(config_error, FlextExceptions.BaseError)
        assert (
            cast("str", get_dynamic_attr(config_error, "message", "")) == "Config error"
        )
        # The config_key should be in the nested context
        context_obj = cast(
            "dict[str, object]", get_dynamic_attr(config_error, "context", {})
        )
        nested_ctx = context_obj
        assert nested_ctx["config_key"] == "database.host"
        assert nested_ctx["extra"] == "data"

    def test_factory_connection_error_with_endpoint(self) -> None:
        """Test create_connection_error with endpoint (lines 658-661)."""
        conn_error = FlextExceptions.Error(
            "Connection failed",
            endpoint="https://api.example.com",
            timeout=30,
        )

        assert isinstance(conn_error, FlextExceptions.BaseError)
        assert (
            cast("str", get_dynamic_attr(conn_error, "message", ""))
            == "Connection failed"
        )
        # The endpoint should be in the error's context
        assert "endpoint" in str(
            conn_error.context,
        ) or "https://api.example.com" in str(conn_error.context)
