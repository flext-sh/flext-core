"""FlextExceptions comprehensive tests targeting specific uncovered functionality.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from flext_core import FlextExceptions


class TestFlextExceptionsComprehensiveTargeted:
    """Targeted tests for likely uncovered FlextExceptions functionality."""

    def test_flext_exceptions_callable_interface(self) -> None:
        """Test FlextExceptions.__call__ method for direct callable interface."""
        # Create FlextExceptions instance to test __call__ method
        exceptions_instance = FlextExceptions()

        # Test direct call with operation parameter
        operation_exc = exceptions_instance(
            "Operation failed",
            operation="data_processing",
            error_code="OP_001",  # Note: error_code is passed but not used by OperationError
        )
        assert isinstance(operation_exc, FlextExceptions.OperationError)
        assert operation_exc.message == "Operation failed"
        assert operation_exc.operation == "data_processing"
        # OperationError uses hardcoded OPERATION_ERROR code
        assert operation_exc.error_code == "OPERATION_ERROR"

    def test_flext_exceptions_callable_field_validation(self) -> None:
        """Test FlextExceptions.__call__ with field parameter."""
        exceptions_instance = FlextExceptions()

        # Test call with field parameter (creates ValidationError)
        validation_exc = exceptions_instance(
            "Invalid field value",
            field="email",
            value="invalid-email",
            validation_details={"pattern": "email_regex"},
        )
        assert isinstance(validation_exc, FlextExceptions.ValidationError)
        assert validation_exc.field == "email"
        assert validation_exc.value == "invalid-email"
        assert validation_exc.validation_details == {"pattern": "email_regex"}

    def test_flext_exceptions_callable_config_key(self) -> None:
        """Test FlextExceptions.__call__ with config_key parameter."""
        exceptions_instance = FlextExceptions()

        # Test call with config_key parameter (creates ConfigurationError)
        config_exc = exceptions_instance(
            "Configuration missing",
            config_key="database.host",
            config_file="/etc/app/config.yml",
        )
        assert isinstance(config_exc, FlextExceptions.ConfigurationError)
        assert config_exc.config_key == "database.host"
        assert config_exc.config_file == "/etc/app/config.yml"

    def test_create_method_automatic_type_selection(self) -> None:
        """Test FlextExceptions.create method with automatic exception type selection."""
        # Test operation parameter creates OperationError
        op_exc = FlextExceptions.create(
            "Operation failed",
            operation="batch_process",
            error_code="BATCH_001",  # Note: not used by OperationError constructor
        )
        assert isinstance(op_exc, FlextExceptions.OperationError)
        assert op_exc.operation == "batch_process"
        # Uses hardcoded OPERATION_ERROR code
        assert op_exc.error_code == "OPERATION_ERROR"

        # Test field parameter creates ValidationError
        val_exc = FlextExceptions.create(
            "Validation failed",
            field="username",
            value="invalid_user",
            validation_details={"min_length": 5},
        )
        assert isinstance(val_exc, FlextExceptions.ValidationError)
        assert val_exc.field == "username"
        assert val_exc.value == "invalid_user"

        # Test config_key parameter creates ConfigurationError
        config_exc = FlextExceptions.create(
            "Config error", config_key="api.timeout", config_file="app.ini"
        )
        assert isinstance(config_exc, FlextExceptions.ConfigurationError)
        assert config_exc.config_key == "api.timeout"

    def test_create_method_default_to_general_error(self) -> None:
        """Test FlextExceptions.create defaults to general Error when no specific params."""
        general_exc = FlextExceptions.create(
            "General error occurred",
            error_code="GEN_001",  # Note: not used by Error constructor
        )
        assert isinstance(general_exc, FlextExceptions.Error)
        # Uses hardcoded GENERIC_ERROR code
        assert general_exc.error_code == "GENERIC_ERROR"

    def test_type_error_type_conversion_functionality(self) -> None:
        """Test FlextExceptions._TypeError type conversion functionality."""
        # Test string type conversion
        type_exc1 = FlextExceptions.TypeError(
            "Expected string type", expected_type="str", actual_type="int"
        )
        assert type_exc1.expected_type == "str"
        assert type_exc1.actual_type == "int"
        assert type_exc1.context["expected_type"] is str  # Converted to actual type
        assert type_exc1.context["actual_type"] is int

        # Test int type conversion
        type_exc2 = FlextExceptions.TypeError(
            "Expected integer", expected_type="int", actual_type="str"
        )
        assert type_exc2.context["expected_type"] is int
        assert type_exc2.context["actual_type"] is str

        # Test float type conversion
        type_exc3 = FlextExceptions.TypeError(
            "Expected float", expected_type="float", actual_type="bool"
        )
        assert type_exc3.context["expected_type"] is float
        assert type_exc3.context["actual_type"] is bool

        # Test bool type conversion
        type_exc4 = FlextExceptions.TypeError(
            "Expected boolean", expected_type="bool", actual_type="list"
        )
        assert type_exc4.context["expected_type"] is bool
        assert type_exc4.context["actual_type"] is list

        # Test list type conversion
        type_exc5 = FlextExceptions.TypeError(
            "Expected list", expected_type="list", actual_type="dict"
        )
        assert type_exc5.context["expected_type"] is list
        assert type_exc5.context["actual_type"] is dict

        # Test dict type conversion
        type_exc6 = FlextExceptions.TypeError(
            "Expected dictionary", expected_type="dict", actual_type="str"
        )
        assert type_exc6.context["expected_type"] is dict
        assert type_exc6.context["actual_type"] is str

    def test_type_error_unknown_type_handling(self) -> None:
        """Test FlextExceptions._TypeError with unknown type strings."""
        type_exc = FlextExceptions.TypeError(
            "Unknown types",
            expected_type="custom_type",
            actual_type="another_custom_type",
        )
        # Unknown types should remain as strings
        assert type_exc.context["expected_type"] == "custom_type"
        assert type_exc.context["actual_type"] == "another_custom_type"

    def test_critical_error_kwargs_handling(self) -> None:
        """Test FlextExceptions._CriticalError kwargs handling."""
        # Test with existing context and additional kwargs
        existing_context = {"module": "core", "severity": "high"}
        critical_exc = FlextExceptions.CriticalError(
            "System failure",
            context=existing_context,
            correlation_id="crit_123",
            additional_info="system_overload",
            error_source="memory_leak",
        )

        # Context should include original context plus additional kwargs
        assert critical_exc.context["module"] == "core"
        assert critical_exc.context["severity"] == "high"
        assert critical_exc.context["additional_info"] == "system_overload"
        assert critical_exc.context["error_source"] == "memory_leak"
        assert critical_exc.correlation_id == "crit_123"

    def test_critical_error_no_context_kwargs_handling(self) -> None:
        """Test FlextExceptions._CriticalError with no context but kwargs."""
        critical_exc = FlextExceptions.CriticalError(
            "Critical failure",
            system_state="unstable",
            recovery_attempted=True,
            last_checkpoint="checkpoint_5",
        )

        # All kwargs should become context
        assert critical_exc.context["system_state"] == "unstable"
        assert critical_exc.context["recovery_attempted"] is True
        assert critical_exc.context["last_checkpoint"] == "checkpoint_5"

    def test_error_kwargs_handling(self) -> None:
        """Test FlextExceptions._Error kwargs handling."""
        # Test with existing context and additional kwargs
        existing_context = {"request_id": "req_123"}
        error_exc = FlextExceptions.Error(
            "General error",
            context=existing_context,
            correlation_id="err_456",
            user_action="data_upload",
            file_size=1024,
        )

        # Context should merge existing and new kwargs
        assert error_exc.context["request_id"] == "req_123"
        assert error_exc.context["user_action"] == "data_upload"
        assert error_exc.context["file_size"] == 1024
        assert error_exc.correlation_id == "err_456"

    def test_metrics_functionality(self) -> None:
        """Test FlextExceptions.Metrics tracking functionality."""
        # Clear metrics before test
        FlextExceptions.clear_metrics()

        # Create various exceptions to trigger metrics recording
        FlextExceptions.ValidationError("Test validation")
        FlextExceptions.OperationError("Test operation")
        FlextExceptions.ValidationError("Another validation")  # Same type again

        # Get metrics
        metrics = FlextExceptions.get_metrics()

        # Verify metrics tracking
        assert metrics["_ValidationError"] == 2  # Two ValidationErrors
        assert metrics["_OperationError"] == 1  # One OperationError

        # Test metrics clearing
        FlextExceptions.clear_metrics()
        cleared_metrics = FlextExceptions.get_metrics()
        assert cleared_metrics == {}

    def test_metrics_class_direct_usage(self) -> None:
        """Test FlextExceptions.Metrics class methods directly."""
        # Clear metrics
        FlextExceptions.Metrics.clear_metrics()

        # Record exceptions manually
        FlextExceptions.Metrics.record_exception("CustomException")
        FlextExceptions.Metrics.record_exception("CustomException")
        FlextExceptions.Metrics.record_exception("AnotherException")

        # Get metrics
        metrics = FlextExceptions.Metrics.get_metrics()
        assert metrics["CustomException"] == 2
        assert metrics["AnotherException"] == 1

        # Clear and verify
        FlextExceptions.Metrics.clear_metrics()
        assert FlextExceptions.Metrics.get_metrics() == {}

    def test_create_module_exception_classes_functionality(self) -> None:
        """Test create_module_exception_classes utility method."""
        # Test with typical module name
        module_exceptions = FlextExceptions.create_module_exception_classes(
            "flext_grpc"
        )

        # Verify expected exception classes are created
        expected_classes = [
            "FLEXT_GRPCBaseError",
            "FLEXT_GRPCError",
            "FLEXT_GRPCConfigurationError",
            "FLEXT_GRPCConnectionError",
            "FLEXT_GRPCValidationError",
            "FLEXT_GRPCAuthenticationError",
            "FLEXT_GRPCProcessingError",
            "FLEXT_GRPCTimeoutError",
        ]

        for class_name in expected_classes:
            assert class_name in module_exceptions
            exception_class = module_exceptions[class_name]
            assert issubclass(exception_class, FlextExceptions.BaseError)

    def test_create_module_exception_classes_name_normalization(self) -> None:
        """Test create_module_exception_classes with complex module names."""
        # Test with module name containing hyphens and dots
        complex_module = "flext-db.oracle-client"
        module_exceptions = FlextExceptions.create_module_exception_classes(
            complex_module
        )

        # Name should be normalized: FLEXT_DB_ORACLE_CLIENT
        assert "FLEXT_DB_ORACLE_CLIENTBaseError" in module_exceptions
        assert "FLEXT_DB_ORACLE_CLIENTError" in module_exceptions

        # Test creating and using a generated exception
        base_error_class = module_exceptions["FLEXT_DB_ORACLE_CLIENTBaseError"]
        exception_instance = base_error_class("Test module exception")
        assert isinstance(exception_instance, FlextExceptions.BaseError)
        assert exception_instance.message == "Test module exception"

    def test_attribute_error_with_attribute_context(self) -> None:
        """Test FlextExceptions._AttributeError with attribute_context parameter."""
        attr_context = {"object_type": "User", "available_attrs": ["name", "email"]}
        attr_exc = FlextExceptions.AttributeError(
            "Attribute not found",
            attribute_name="phone",
            attribute_context=attr_context,
        )

        assert attr_exc.attribute_name == "phone"
        assert attr_exc.context["attribute_name"] == "phone"
        assert attr_exc.context["attribute_context"] == attr_context

    def test_all_exception_types_inheritance(self) -> None:
        """Test that all exception types properly inherit from both BaseError and built-in exceptions."""
        # Test multiple inheritance patterns
        test_cases = [
            (FlextExceptions.ValidationError, ValueError),
            (FlextExceptions.ConfigurationError, ValueError),
            (FlextExceptions.ConnectionError, ConnectionError),
            (FlextExceptions.TimeoutError, TimeoutError),
            (FlextExceptions.NotFoundError, FileNotFoundError),
            (FlextExceptions.AlreadyExistsError, FileExistsError),
            (FlextExceptions.PermissionError, PermissionError),
            (FlextExceptions.AuthenticationError, PermissionError),
            (FlextExceptions.TypeError, TypeError),
            (FlextExceptions.AttributeError, AttributeError),
        ]

        for flext_exc_class, builtin_exc_class in test_cases:
            exception_instance = flext_exc_class("Test message")
            assert isinstance(exception_instance, FlextExceptions.BaseError)
            assert isinstance(exception_instance, builtin_exc_class)

    def test_legacy_api_aliases(self) -> None:
        """Test legacy API backward compatibility aliases."""
        # Test that legacy aliases point to correct classes
        assert FlextExceptions.FlextError is FlextExceptions.Error
        assert FlextExceptions.FlextValidationError is FlextExceptions.ValidationError
        assert (
            FlextExceptions.FlextConfigurationError
            is FlextExceptions.ConfigurationError
        )
        assert FlextExceptions.FlextConnectionError is FlextExceptions.ConnectionError

        # Test creating exceptions using legacy aliases
        legacy_exc = FlextExceptions.FlextValidationError("Legacy validation error")
        assert isinstance(legacy_exc, FlextExceptions.ValidationError)
        assert legacy_exc.message == "Legacy validation error"

    def test_error_codes_class_constants(self) -> None:
        """Test FlextExceptions.ErrorCodes class constants."""
        # Verify error codes are properly exposed
        assert hasattr(FlextExceptions.ErrorCodes, "GENERIC_ERROR")
        assert hasattr(FlextExceptions.ErrorCodes, "VALIDATION_ERROR")
        assert hasattr(FlextExceptions.ErrorCodes, "CONFIGURATION_ERROR")
        assert hasattr(FlextExceptions.ErrorCodes, "CONNECTION_ERROR")

        # Test using error codes
        validation_exc = FlextExceptions.ValidationError("Test", field="test")
        assert validation_exc.code == FlextExceptions.ErrorCodes.VALIDATION_ERROR

    def test_base_error_string_representation(self) -> None:
        """Test FlextExceptions.BaseError string representation."""
        base_exc = FlextExceptions.BaseError("Test base error", code="TEST_001")
        str_repr = str(base_exc)
        assert "[TEST_001]" in str_repr
        assert "Test base error" in str_repr

    def test_base_error_timestamp_and_correlation_generation(self) -> None:
        """Test BaseError automatic timestamp and correlation ID generation."""
        base_exc = FlextExceptions.BaseError("Test timestamp")

        # Should have automatic timestamp
        assert hasattr(base_exc, "timestamp")
        assert isinstance(base_exc.timestamp, float)

        # Should have automatic correlation ID
        assert base_exc.correlation_id.startswith("flext_")

        # Test with custom correlation ID
        custom_exc = FlextExceptions.BaseError(
            "Custom correlation", correlation_id="custom_123"
        )
        assert custom_exc.correlation_id == "custom_123"

    def test_complex_exception_context_integration(self) -> None:
        """Test complex exception scenarios with full context integration."""
        # Create complex ProcessingError with business context
        processing_exc = FlextExceptions.ProcessingError(
            "Business rule validation failed",
            business_rule="order_total_limit",
            operation="order_processing",
            context={"user_id": "user_123", "order_id": "order_456"},
            correlation_id="proc_789",
        )

        # Verify all context is properly integrated
        assert processing_exc.business_rule == "order_total_limit"
        assert processing_exc.operation == "order_processing"
        assert processing_exc.context["business_rule"] == "order_total_limit"
        assert processing_exc.context["operation"] == "order_processing"
        assert processing_exc.context["user_id"] == "user_123"
        assert processing_exc.context["order_id"] == "order_456"
        assert processing_exc.correlation_id == "proc_789"

    def test_integration_with_flext_tests_matchers(self) -> None:
        """Test integration scenarios with FlextTestsMatchers."""
        # Test that exceptions work properly with FlextTestsMatchers
        error_message = "Validation failed for integration test"
        with pytest.raises(FlextExceptions.ValidationError):
            raise FlextExceptions.ValidationError(
                error_message,
                field="integration_field",
                value="invalid_value",
            )

        # These assertions are unreachable because the exception is raised and caught
        # The test passes if the exception is raised with the correct type
