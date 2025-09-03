"""Targeted tests for 100% coverage on FlextExceptions module."""

from __future__ import annotations

from typing import cast

from flext_core.exceptions import FlextExceptions
from flext_core.typings import FlextTypes


class TestExceptions100PercentCoverage:
    """Targeted tests for FlextExceptions uncovered lines."""

    def test_line_233_direct_call_method(self) -> None:
        """Test line 233: FlextExceptions.__call__ method."""
        # Test direct call to FlextExceptions instance
        exceptions = FlextExceptions()

        # This should trigger line 233 - direct call creates exception
        error = exceptions("Test error message")
        assert isinstance(error, FlextExceptions.BaseError)
        assert error.message == "Test error message"

    def test_direct_call_with_operation(self) -> None:
        """Test direct call with operation parameter."""
        exceptions = FlextExceptions()

        error = exceptions(
            "Operation failed", operation="user_creation", error_code="OP_001"
        )
        assert isinstance(error, FlextExceptions._OperationError)
        assert error.message == "Operation failed"

    def test_direct_call_with_field(self) -> None:
        """Test direct call with field parameter."""
        exceptions = FlextExceptions()

        error = exceptions(
            "Validation failed",
            field="email",
            value="invalid-email",
            error_code="VAL_001",
        )
        assert isinstance(error, FlextExceptions._ValidationError)
        assert error.message == "Validation failed"

    def test_direct_call_with_config_key(self) -> None:
        """Test direct call with config_key parameter."""
        exceptions = FlextExceptions()

        error = exceptions(
            "Config error",
            config_key="database_url",
            config_file="settings.yaml",
            error_code="CFG_001",
        )
        assert isinstance(error, FlextExceptions._ConfigurationError)
        assert error.message == "Config error"

    def test_create_method_lines_1043_1079(self) -> None:
        """Test lines 1043-1079: create method with different parameters."""
        # Test OperationError creation (lines 1046-1053)
        op_error = FlextExceptions.create(
            "Operation failed",
            operation="data_processing",
            error_code="OP_002",
            context={"user_id": 123},
            correlation_id="corr-123",
        )
        assert isinstance(op_error, FlextExceptions._OperationError)
        assert op_error.message == "Operation failed"

        # Test ValidationError creation (lines 1054-1065)
        val_error = FlextExceptions.create(
            "Field validation failed",
            field="username",
            value="invalid_user",
            validation_details={"min_length": 5},
            error_code="VAL_002",
            context={"form": "registration"},
        )
        assert isinstance(val_error, FlextExceptions._ValidationError)
        assert val_error.message == "Field validation failed"

        # Test ConfigurationError creation (lines 1066-1075)
        config_error = FlextExceptions.create(
            "Config missing",
            config_key="api_key",
            config_file="app.yaml",
            error_code="CFG_002",
            context={"environment": "production"},
        )
        assert isinstance(config_error, FlextExceptions._ConfigurationError)
        assert config_error.message == "Config missing"

        # Test default Error creation (lines 1077-1079)
        general_error = FlextExceptions.create(
            "General error",
            error_code="GEN_001",
            context={"module": "core"},
            correlation_id="corr-456",
        )
        assert isinstance(general_error, FlextExceptions._Error)
        assert general_error.message == "General error"

    def test_configuration_methods_lines_801_822(self) -> None:
        """Test configuration methods lines 801-822."""
        # Test configure_error_handling
        config = {
            "enable_metrics": True,
            "log_level": "ERROR",
            "context_tracking": True,
        }

        result = FlextExceptions.configure_error_handling(
            cast("FlextTypes.Config.ConfigDict", config)
        )
        assert result.success

        # Test get_error_handling_config
        config_result = FlextExceptions.get_error_handling_config()
        assert config_result.success
        config_dict = config_result.unwrap()
        assert isinstance(config_dict, dict)

    def test_create_environment_specific_config_lines_1115_1187(self) -> None:
        """Test create_environment_specific_config lines 1115-1187."""
        environments: list[FlextTypes.Config.Environment] = [
            "development",
            "production",
            "test",
        ]

        for env in environments:
            result = FlextExceptions.create_environment_specific_config(env)
            assert result.success
            config = result.unwrap()
            assert isinstance(config, dict)
            assert "environment" in config or len(config) > 0

    def test_invalid_environment_config(self) -> None:
        """Test invalid environment configuration."""
        # Use cast to test invalid environment handling
        result = FlextExceptions.create_environment_specific_config(
            cast("FlextTypes.Config.Environment", "invalid_env")
        )
        assert result.failure
        assert result.error is not None
        assert "Invalid environment" in result.error

    def test_exception_metrics_lines_854_855_899(self) -> None:
        """Test exception metrics collection."""
        # Record some exceptions to trigger metrics
        FlextExceptions.Metrics.record_exception("ValidationError")
        FlextExceptions.Metrics.record_exception("ConfigurationError")
        FlextExceptions.Metrics.record_exception("ValidationError")

        # Get metrics
        metrics = FlextExceptions.Metrics.get_metrics()
        assert isinstance(metrics, dict)
        assert "ValidationError" in metrics
        assert metrics["ValidationError"] >= 2

        # Clear metrics
        FlextExceptions.Metrics.clear_metrics()
        cleared_metrics = FlextExceptions.Metrics.get_metrics()
        assert cleared_metrics == {} or all(v == 0 for v in cleared_metrics.values())

    def test_error_codes_access_lines_1199_1216(self) -> None:
        """Test ErrorCodes access lines 1199-1216."""
        # Test actual error code constants
        error_codes = [
            FlextExceptions.ErrorCodes.VALIDATION_ERROR,
            FlextExceptions.ErrorCodes.CONFIGURATION_ERROR,
            FlextExceptions.ErrorCodes.CONNECTION_ERROR,
            FlextExceptions.ErrorCodes.PROCESSING_ERROR,
            FlextExceptions.ErrorCodes.TIMEOUT_ERROR,
            FlextExceptions.ErrorCodes.NOT_FOUND,
            FlextExceptions.ErrorCodes.ALREADY_EXISTS,
            FlextExceptions.ErrorCodes.PERMISSION_ERROR,
            FlextExceptions.ErrorCodes.AUTHENTICATION_ERROR,
            FlextExceptions.ErrorCodes.TYPE_ERROR,
            FlextExceptions.ErrorCodes.GENERIC_ERROR,
            FlextExceptions.ErrorCodes.CRITICAL_ERROR,
        ]

        for code in error_codes:
            assert isinstance(code, str)
            assert len(code) > 0

    def test_specialized_exception_creation_lines_1233_1288(self) -> None:
        """Test specialized exception creation lines 1233-1288."""
        # Test ValidationError
        val_error = FlextExceptions.ValidationError(
            "Invalid input", field="email", value="bad@email", error_code="VAL_003"
        )
        assert val_error.message == "Invalid input"

        # Test ConfigurationError
        config_error = FlextExceptions.ConfigurationError(
            "Missing config", config_key="database_url", error_code="CFG_003"
        )
        assert config_error.message == "Missing config"

        # Test ConnectionError
        conn_error = FlextExceptions.ConnectionError(
            "Connection failed", error_code="CONN_001"
        )
        assert conn_error.message == "Connection failed"

        # Test ProcessingError
        proc_error = FlextExceptions.ProcessingError(
            "Processing failed", error_code="PROC_001"
        )
        assert proc_error.message == "Processing failed"

        # Test TimeoutError
        timeout_error = FlextExceptions.TimeoutError(
            "Operation timeout", error_code="TIMEOUT_001"
        )
        assert timeout_error.message == "Operation timeout"


class TestExceptionsIntegration100PercentCoverage:
    """Integration tests for exception system functionality."""

    def test_complete_exception_workflow(self) -> None:
        """Test complete exception creation and handling workflow."""
        # Configure error handling
        config: dict[
            str, str | int | float | bool | list[object] | dict[str, object]
        ] = {"enable_metrics": True, "log_level": "DEBUG"}
        config_result = FlextExceptions.configure_error_handling(config)
        assert config_result.success

        # Create different types of exceptions
        exceptions = [
            FlextExceptions.create("Op failed", operation="test_op"),
            FlextExceptions.create("Val failed", field="test_field"),
            FlextExceptions.create("Config failed", config_key="test_key"),
            FlextExceptions.create("General failed"),
        ]

        for exc in exceptions:
            assert isinstance(exc, FlextExceptions.BaseError)
            assert len(exc.message) > 0

            # Record exception for metrics
            exc_type = type(exc).__name__
            FlextExceptions.Metrics.record_exception(exc_type)

        # Verify metrics were recorded
        metrics = FlextExceptions.Metrics.get_metrics()
        assert len(metrics) > 0

    def test_exception_inheritance_and_properties(self) -> None:
        """Test exception inheritance and property access."""
        # Test that specialized exceptions inherit from appropriate base classes
        val_error = FlextExceptions.ValidationError("Test validation error")

        # Should be instance of both specialized and base error
        assert isinstance(val_error, FlextExceptions.ValidationError)
        assert isinstance(val_error, FlextExceptions.BaseError)
        assert isinstance(val_error, Exception)

        # Test properties
        assert hasattr(val_error, "message")
        assert hasattr(val_error, "error_code")
        assert hasattr(val_error, "timestamp")
        assert hasattr(val_error, "context")

    def test_exception_string_representation(self) -> None:
        """Test exception string representation."""
        error = FlextExceptions.ValidationError(
            "Test error message",
            field="test_field",
            error_code="TEST_001",
            context={"user": "test_user"},
        )

        str_repr = str(error)
        assert "Test error message" in str_repr
        assert isinstance(str_repr, str)

    def test_context_and_correlation_tracking(self) -> None:
        """Test context and correlation ID tracking."""
        context = {"module": "test", "function": "test_func"}
        correlation_id = "test-corr-123"

        error = FlextExceptions.create(
            "Test with context", context=context, correlation_id=correlation_id
        )

        # Context may have additional fields like 'code', so check inclusion
        for key, value in context.items():
            assert key in error.context
            assert error.context[key] == value
        assert error.correlation_id == correlation_id

    def test_edge_cases_and_error_handling(self) -> None:
        """Test edge cases and error handling scenarios."""
        # Test with empty message
        error = FlextExceptions.create("")
        assert error.message == ""

        # Test with None context (should be handled gracefully)
        error = FlextExceptions.create("Test", context=None)
        assert isinstance(error, FlextExceptions.BaseError)

        # Test with complex context data
        complex_context = {
            "nested": {"data": [1, 2, 3]},
            "string": "test",
            "number": 42,
        }
        error = FlextExceptions.create("Complex context", context=complex_context)
        # Context may have additional fields, so check inclusion
        for key, value in complex_context.items():
            assert key in error.context
            assert error.context[key] == value
