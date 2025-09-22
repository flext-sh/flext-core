"""Targeted tests for 100% coverage on FlextExceptions module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextExceptions


class TestExceptions100PercentCoverage:
    """Targeted tests for FlextExceptions uncovered lines."""

    def test_attribute_error_with_context(self) -> None:
        """Test _AttributeError initialization with attribute_context (lines 128-138)."""
        # Test creating AttributeError with attribute_context to cover lines 128-138
        error = FlextExceptions.AttributeError(
            "Invalid attribute access",
            attribute_name="test_attr",
            attribute_context={
                "obj_type": "TestClass",
                "available_attrs": ["name", "value"],
            },
        )

        assert isinstance(error, FlextExceptions._AttributeError)
        assert error.message == "Invalid attribute access"
        assert error.attribute_name == "test_attr"
        assert "attribute_context" in error.context
        assert isinstance(error.context["attribute_context"], dict)
        assert error.context["attribute_context"]["obj_type"] == "TestClass"

    def test_not_found_error_with_resource_context(self) -> None:
        """Test _NotFoundError initialization with resource context (lines 305-311)."""
        # Test creating NotFoundError with resource context to cover lines 305-311
        error = FlextExceptions.NotFoundError(
            "Resource not found", resource_id="user_123", resource_type="User"
        )

        assert isinstance(error, FlextExceptions._NotFoundError)
        assert error.message == "Resource not found"
        assert error.resource_id == "user_123"
        assert error.resource_type == "User"
        assert error.context["resource_id"] == "user_123"
        assert error.context["resource_type"] == "User"

    def test_already_exists_error_with_resource_context(self) -> None:
        """Test _AlreadyExistsError initialization with resource context (lines 329-336)."""
        # Test creating AlreadyExistsError with resource context to cover lines 329-336
        error = FlextExceptions.AlreadyExistsError(
            "Resource already exists", resource_id="user_456", resource_type="User"
        )

        assert isinstance(error, FlextExceptions._AlreadyExistsError)
        assert error.message == "Resource already exists"
        assert error.resource_id == "user_456"
        assert error.resource_type == "User"
        assert error.context["resource_id"] == "user_456"
        assert error.context["resource_type"] == "User"

    def test_permission_error_with_required_permission(self) -> None:
        """Test _PermissionError initialization with required_permission (lines 353-358)."""
        # Test creating PermissionError with required_permission to cover lines 353-358
        error = FlextExceptions.PermissionError(
            "Permission denied", required_permission="admin_access"
        )

        assert isinstance(error, FlextExceptions._PermissionError)
        assert error.message == "Permission denied"
        assert error.required_permission == "admin_access"
        assert error.context["required_permission"] == "admin_access"

    def test_authentication_error_with_auth_method(self) -> None:
        """Test _AuthenticationError initialization with auth_method (lines 375-380)."""
        # Test creating AuthenticationError with auth_method to cover lines 375-380
        error = FlextExceptions.AuthenticationError(
            "Authentication failed", auth_method="oauth2"
        )

        assert isinstance(error, FlextExceptions._AuthenticationError)
        assert error.message == "Authentication failed"
        assert error.auth_method == "oauth2"
        assert error.context["auth_method"] == "oauth2"

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
            "Operation failed",
            operation="user_creation",
            error_code="OP_001",
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
        # Test metrics functionality
        FlextExceptions.clear_metrics()

        # Create some exceptions to generate metrics
        FlextExceptions.ValidationError("Test validation error")
        FlextExceptions.ConfigurationError("Test config error")

        # Test get_metrics
        metrics = FlextExceptions.get_metrics()
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

        # Test clear_metrics
        FlextExceptions.clear_metrics()
        cleared_metrics = FlextExceptions.get_metrics()
        assert len(cleared_metrics) == 0

    def test_create_environment_specific_config_lines_1115_1187(self) -> None:
        """Test create_module_exception_classes functionality."""
        # Test creating module-specific exception classes
        module_exceptions = FlextExceptions.create_module_exception_classes(
            "test_module",
        )

        assert isinstance(module_exceptions, dict)
        assert len(module_exceptions) > 0

        # Check that expected exception types are created
        expected_types = [
            "TEST_MODULEBaseError",
            "TEST_MODULEError",
            "TEST_MODULEConfigurationError",
            "TEST_MODULEConnectionError",
            "TEST_MODULEValidationError",
            "TEST_MODULEAuthenticationError",
            "TEST_MODULEProcessingError",
            "TEST_MODULETimeoutError",
        ]

        for expected_type in expected_types:
            assert expected_type in module_exceptions
            assert isinstance(module_exceptions[expected_type], type)

    def test_invalid_environment_config(self) -> None:
        """Test invalid module name handling."""
        # Test with invalid module name
        result = FlextExceptions.create_module_exception_classes("")
        assert isinstance(result, dict)
        assert len(result) > 0  # Should still create exceptions even with empty name

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
        # Test actual error code constants from FlextConstants.Errors
        from flext_core.constants import FlextConstants

        error_codes = [
            FlextConstants.Errors.VALIDATION_ERROR,
            FlextConstants.Errors.CONFIGURATION_ERROR,
            FlextConstants.Errors.CONNECTION_ERROR,
            FlextConstants.Errors.PROCESSING_ERROR,
            FlextConstants.Errors.TIMEOUT_ERROR,
            FlextConstants.Errors.NOT_FOUND,
            FlextConstants.Errors.ALREADY_EXISTS,
            FlextConstants.Errors.PERMISSION_ERROR,
            FlextConstants.Errors.AUTHENTICATION_ERROR,
            FlextConstants.Errors.TYPE_ERROR,
            FlextConstants.Errors.GENERIC_ERROR,
            FlextConstants.Errors.CRITICAL_ERROR,
        ]

        for code in error_codes:
            assert isinstance(code, str)
            assert len(code) > 0

    def test_specialized_exception_creation_lines_1233_1288(self) -> None:
        """Test specialized exception creation lines 1233-1288."""
        # Test ValidationError
        val_error = FlextExceptions.ValidationError(
            "Invalid input",
            field="email",
            value="bad@email",
            error_code="VAL_003",
        )
        assert val_error.message == "Invalid input"

        # Test ConfigurationError
        config_error = FlextExceptions.ConfigurationError(
            "Missing config",
            config_key="database_url",
            error_code="CFG_003",
        )
        assert config_error.message == "Missing config"

        # Test ConnectionError
        conn_error = FlextExceptions.ConnectionError(
            "Connection failed",
            error_code="CONN_001",
        )
        assert conn_error.message == "Connection failed"

        # Test ProcessingError
        proc_error = FlextExceptions.ProcessingError(
            "Processing failed",
            error_code="PROC_001",
        )
        assert proc_error.message == "Processing failed"

        # Test TimeoutError
        timeout_error = FlextExceptions.TimeoutError(
            "Operation timeout",
            error_code="TIMEOUT_001",
        )
        assert timeout_error.message == "Operation timeout"


class TestExceptionsIntegration100PercentCoverage:
    """Integration tests for exception system functionality."""

    def test_complete_exception_workflow(self) -> None:
        """Test complete exception creation and handling workflow."""
        # Clear metrics to start fresh
        FlextExceptions.clear_metrics()

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
            "Test with context",
            context=context,
            correlation_id=correlation_id,
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
        assert error.message is not None

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
