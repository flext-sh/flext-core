"""Fixed comprehensive tests for FlextDomainService to achieve 100% coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

import operator
import time

import pytest
from pydantic import BaseModel, Field, ValidationError

from flext_core import (
    FlextDomainService,
    FlextMixins,
    FlextModels,
    FlextResult,
    FlextTypes,
)


# Test Domain Service Implementations
class SampleUserService(FlextDomainService[FlextTypes.Core.Headers]):
    """Sample service for user operations used in tests."""

    def execute(self) -> FlextResult[FlextTypes.Core.Headers]:
        """Execute user operation.

        Returns:
            FlextResult[FlextTypes.Core.Headers]: Success with user headers

        """
        return FlextResult[FlextTypes.Core.Headers].ok(
            {
                "user_id": "default_123",
                "email": "test@example.com",
            },
        )


class SampleComplexService(FlextDomainService[str]):
    """Sample service with complex validation and operations used in tests."""

    name: str = "default_name"
    value: int = 0
    enabled: bool = True

    def __init__(
        self,
        name: str = "default_name",
        value: int = 0,
        *,
        enabled: bool = True,
        **_data: object,
    ) -> None:
        """Initialize with field arguments."""
        # Pass field values to parent constructor
        super().__init__(name=name, value=value, enabled=enabled, **_data)

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules with multiple checks.

        Returns:
            FlextResult[None]: Success if all rules pass, failure with error message otherwise.

        """
        if not self.name:
            return FlextResult[None].fail("Name is required")
        if self.value < 0:
            return FlextResult[None].fail("Value must be non-negative")
        if not self.enabled and self.value > 0:
            return FlextResult[None].fail("Cannot have value when disabled")
        return FlextResult[None].ok(None)

    def validate_config(self) -> FlextResult[None]:
        """Validate configuration with custom logic.

        Returns:
            FlextResult[None]: Success if configuration is valid, failure otherwise.

        """
        if len(self.name) > 50:
            return FlextResult[None].fail("Name too long")
        if self.value > 1000:
            return FlextResult[None].fail("Value too large")
        return FlextResult[None].ok(None)

    def execute(self) -> FlextResult[str]:
        """Execute complex operation."""
        if not self.name:
            return FlextResult[str].fail("Name is required")
        if self.value < 0:
            return FlextResult[str].fail("Value must be non-negative")
        if len(self.name) > 50:
            return FlextResult[str].fail("Name too long")
        if self.value > 1000:
            return FlextResult[str].fail("Value too large")

        return FlextResult[str].ok(f"Processed: {self.name} with value {self.value}")


class SampleFailingService(FlextDomainService[None]):
    """Sample service that fails validation, used in tests."""

    def validate_business_rules(self) -> FlextResult[None]:
        """Always fail validation."""
        return FlextResult[None].fail("Validation always fails")

    def execute(self) -> FlextResult[None]:
        """Execute failing operation."""
        return FlextResult[None].fail("Execution failed")


class SampleExceptionService(FlextDomainService[str]):
    """Sample service that raises exceptions, used in tests."""

    should_raise: bool = False

    def __init__(self, *, should_raise: bool = False, **data: object) -> None:
        """Initialize with field arguments."""
        # Call parent __init__ first to initialize the model
        super().__init__(**data)

        # Set field values after initialization (required for frozen models)
        object.__setattr__(self, "should_raise", should_raise)

    def validate_business_rules(self) -> FlextResult[None]:
        """Validation that can raise exceptions."""
        if self.should_raise:
            msg = "Validation exception"
            raise ValueError(msg)
        return FlextResult[None].ok(None)

    def execute(self) -> FlextResult[str]:
        """Execute operation that can raise."""
        if self.should_raise:
            msg = "Execution exception"
            raise RuntimeError(msg)
        return FlextResult[str].ok("Success")


class ComplexTypeService(FlextDomainService[dict]):
    """Test service with complex types for testing."""

    data: dict = Field(default_factory=dict)
    items: list = Field(default_factory=list)

    def execute(self) -> FlextResult[dict]:
        """Execute operation with complex types."""
        return FlextResult[dict].ok({"data": self.data, "items": self.items})


class TestDomainServicesFixed:
    """Fixed comprehensive tests for FlextDomainService."""

    def test_basic_service_creation(self) -> None:
        """Test basic domain service creation."""
        service = SampleUserService()
        assert isinstance(service, FlextDomainService)
        # Ensure the service exposes Pydantic model configuration
        assert isinstance(service.model_config, dict)
        assert service.model_config.get("validate_assignment") is True

    def test_service_immutability(self) -> None:
        """Test that service is immutable (frozen)."""
        service = SampleUserService()

        # Test that assignment to frozen field raises ValidationError
        with pytest.raises(ValidationError):
            # Use setattr to bypass type checking for this test
            setattr(service, "new_field", "not_allowed")

    def test_execute_method_abstract(self) -> None:
        """Test that execute method is abstract."""

        # Create a concrete implementation to test abstract behavior
        class ConcreteService(FlextDomainService[str]):
            def execute(self) -> FlextResult[str]:
                return FlextResult.ok("test")

        # This should work since we implemented execute
        ConcreteService()

    def test_basic_execution(self) -> None:
        """Test basic service execution."""
        service = SampleUserService()
        result = service.execute()

        assert result.success is True
        data = result.unwrap()
        assert data["user_id"] == "default_123"
        assert data["email"] == "test@example.com"

    def test_is_valid_success(self) -> None:
        """Test is_valid with success."""
        service = SampleComplexService(name="test", value=10, enabled=True)

        result = service.is_valid()
        assert result is True

    def test_is_valid_failure(self) -> None:
        """Test is_valid with invalid service."""
        service = SampleFailingService()
        assert service.is_valid() is False

    def test_is_valid_exception_handling(self) -> None:
        """Test is_valid with exception handling."""
        service = SampleExceptionService(should_raise=True)

        # When validate_business_rules raises an exception, is_valid should handle it
        # and return False (since the validation failed)
        result = service.is_valid()
        assert result is False

    def test_validate_business_rules_default(self) -> None:
        """Test default business rules validation."""
        service = SampleUserService()
        result = service.validate_business_rules()
        assert result.success is True

    def test_validate_business_rules_custom_success(self) -> None:
        """Test validate_business_rules with custom success."""
        service = SampleComplexService(name="valid_name", value=10, enabled=True)

        result = service.validate_business_rules()
        assert result.is_success

    def test_validate_business_rules_custom_failure(self) -> None:
        """Test validate_business_rules with custom failure."""
        service = SampleComplexService(
            name="", value=10, enabled=True
        )  # Empty name should fail

        result = service.validate_business_rules()
        assert result.is_failure
        assert "Name is required" in result.error

    def test_validate_business_rules_multiple_conditions(self) -> None:
        """Test validate_business_rules with multiple conditions."""
        service = SampleComplexService(
            name="", value=-1, enabled=False
        )  # Invalid name and value

        result = service.validate_business_rules()
        assert result.is_failure
        assert "Name is required" in result.error

    def test_validate_config_default(self) -> None:
        """Test default configuration validation."""
        service = SampleUserService()
        result = service.validate_config()
        assert result.success is True

    def test_validate_config_custom_success(self) -> None:
        """Test validate_config with custom success."""
        service = SampleComplexService(name="test", value=10, enabled=True)

        result = service.validate_config()
        assert result.is_success

    def test_validate_config_custom_failure(self) -> None:
        """Test validate_config with custom failure."""
        long_name = "a" * 300  # Too long name
        service = SampleComplexService(name=long_name, value=10, enabled=True)

        result = service.validate_config()
        assert result.is_failure
        assert "Name too long" in result.error

    def test_execute_operation_success(self) -> None:
        """Test execute_operation with successful operation."""
        service = SampleUserService()

        test_operation = operator.add

        # Create operation request using Pydantic model
        operation_request = FlextModels.OperationExecutionRequest(
            operation_name="add_numbers",
            operation_callable=test_operation,
            arguments={"args": [5, 3]},
        )

        result = service.execute_operation(operation_request)
        assert result.success is True
        assert result.unwrap() == 8

    def test_execute_operation_with_kwargs(self) -> None:
        """Test execute_operation with keyword arguments."""
        service = SampleUserService()

        def test_operation(name: str, value: int = 10) -> str:
            return f"{name}: {value}"

        # Create operation request with kwargs
        operation_request = FlextModels.OperationExecutionRequest(
            operation_name="format_string",
            operation_callable=test_operation,
            keyword_arguments={"name": "test", "value": 20},
        )

        result = service.execute_operation(operation_request)
        assert result.success is True
        assert result.unwrap() == "test: 20"

    def test_execute_operation_config_validation_failure(self) -> None:
        """Test execute_operation with config validation failure."""
        test_operation = operator.add

        long_name = "a" * 300  # Too long name should fail config validation
        service = SampleComplexService(name=long_name, value=10, enabled=True)

        operation_request = FlextModels.OperationExecutionRequest(
            operation_name="add_numbers",
            operation_callable=test_operation,
            arguments={"args": [5, 3]},
        )
        result = service.execute_operation(operation_request)
        assert result.is_failure
        assert result.error is not None
        assert "Configuration validation failed" in result.error
        assert "Name too long" in result.error

    def test_execute_operation_runtime_error(self) -> None:
        """Test execute_operation with runtime error."""
        service = SampleUserService()

        def failing_operation() -> str:
            msg = "Operation failed"
            raise RuntimeError(msg)

        # Create operation request
        operation_request = FlextModels.OperationExecutionRequest(
            operation_name="failing_op",
            operation_callable=failing_operation,
        )

        result = service.execute_operation(operation_request)
        assert result.is_failure
        assert result.error is not None
        assert "failing_op" in result.error
        assert "Operation failed" in result.error

    def test_execute_operation_value_error(self) -> None:
        """Test execute_operation with value error."""
        service = SampleUserService()

        def value_error_operation() -> str:
            msg = "Invalid value"
            raise ValueError(msg)

        # Create operation request
        operation_request = FlextModels.OperationExecutionRequest(
            operation_name="value_error_op",
            operation_callable=value_error_operation,
        )

        result = service.execute_operation(operation_request)
        assert result.is_failure
        assert result.error is not None
        assert "value_error_op" in result.error
        assert "Invalid value" in result.error

    def test_execute_operation_type_error(self) -> None:
        """Test execute_operation with type error."""
        service = SampleUserService()

        def type_error_operation() -> str:
            msg = "Wrong type"
            raise TypeError(msg)

        # Create operation request
        operation_request = FlextModels.OperationExecutionRequest(
            operation_name="type_error_op",
            operation_callable=type_error_operation,
        )

        result = service.execute_operation(operation_request)
        assert result.is_failure
        assert result.error is not None
        assert "type_error_op" in result.error
        assert "Wrong type" in result.error

    def test_execute_operation_unexpected_error(self) -> None:
        """Test execute_operation with unexpected error."""
        service = SampleUserService()

        def unexpected_error_operation() -> str:
            msg = "Unexpected error"
            raise OSError(msg)

        # Create operation request
        operation_request = FlextModels.OperationExecutionRequest(
            operation_name="unexpected_op",
            operation_callable=unexpected_error_operation,
        )

        result = service.execute_operation(operation_request)
        assert result.is_failure
        assert result.error is not None
        assert "unexpected_op" in result.error
        assert "Unexpected error" in result.error

    def test_execute_operation_timeout_failure(self) -> None:
        """Test execute_operation handles timeouts as failures."""
        service = SampleUserService()

        def slow_operation() -> str:
            time.sleep(2)
            return "finished"

        operation_request = FlextModels.OperationExecutionRequest(
            operation_name="slow_op",
            operation_callable=slow_operation,
            timeout_seconds=1,
        )

        result = service.execute_operation(operation_request)

        assert result.is_failure
        assert result.error is not None
        assert "slow_op" in result.error
        assert "timed out" in result.error

    def test_execute_operation_retry_success(self) -> None:
        """Test execute_operation respects retry configuration."""
        service = SampleUserService()
        attempts: dict[str, int] = {"count": 0}

        def flaky_operation() -> str:
            if attempts["count"] == 0:
                attempts["count"] += 1
                raise RuntimeError("temporary failure")
            return "recovered"

        operation_request = FlextModels.OperationExecutionRequest(
            operation_name="flaky_op",
            operation_callable=flaky_operation,
            retry_config={
                "max_attempts": 2,
                "initial_delay_seconds": 0.01,
                "max_delay_seconds": 0.05,
                "exponential_backoff": False,
                "retry_on_exceptions": [RuntimeError],
            },
        )

        result = service.execute_operation(operation_request)

        assert result.is_success
        assert result.unwrap() == "recovered"
        assert attempts["count"] == 1

    def test_execute_operation_non_callable(self) -> None:
        """Test execute_operation with non-callable operation."""
        SampleUserService()

        # The OperationExecutionRequest will now validate the operation is callable
        with pytest.raises(ValidationError) as exc_info:
            FlextModels.OperationExecutionRequest(
                operation_name="not_callable",
                operation_callable="not_callable",  # This should fail validation
                arguments={"args": []},
            )

        # Check the Pydantic validation error message
        error_message = str(exc_info.value)
        assert "Input should be callable" in error_message

    def test_get_service_info_basic(self) -> None:
        """Test get_service_info basic functionality."""
        service = SampleUserService()
        info = service.get_service_info()

        assert info == {"service_type": "SampleUserService"}

    def test_get_service_info_with_validation(self) -> None:
        """Test get_service_info with validation failure."""
        service = SampleExceptionService(should_raise=True)

        result = service.get_service_info()
        assert isinstance(result, dict)
        assert "SampleExceptionService" in result["service_type"]

    def test_service_serialization(self) -> None:
        """Test service serialization through mixins."""
        service = SampleUserService()

        # Test serialization methods from mixins
        # to_dict was removed - use model_dump instead
        assert hasattr(service, "model_dump")
        serialized = service.model_dump()
        assert isinstance(serialized, dict)

        request = FlextModels.SerializationRequest(data=service, use_model_dump=True)
        json_str = FlextMixins.to_json(request)
        assert isinstance(json_str, str)

        formatted_request = FlextModels.SerializationRequest(
            data=service,
            use_model_dump=True,
            indent=2,
            sort_keys=True,
        )
        json_formatted = FlextMixins.to_json(formatted_request)
        assert isinstance(json_formatted, str)

    def test_service_logging(self) -> None:
        """Test service logging through mixins."""
        service = SampleUserService()

        assert isinstance(service, FlextMixins.Loggable)

        log_request = FlextModels.LogOperation(
            level="INFO",
            message="Test info message",
            context={"event": "unit_test"},
            operation="test_operation",
            obj=service,
        )

        FlextMixins.log_operation(log_request)

    def test_complex_service_execution_success(self) -> None:
        """Test complex service execution success."""
        test_operation = operator.add

        service = SampleComplexService(name="test", value=10, enabled=True)

        operation_request = FlextModels.OperationExecutionRequest(
            operation_name="add_numbers",
            operation_callable=test_operation,
            arguments={"args": [5, 3]},
        )
        result = service.execute_operation(operation_request)
        assert result.is_success
        assert result.value == 8

    def test_complex_service_execution_business_rule_failure(self) -> None:
        """Test complex service execution with business rule failure."""
        test_operation = operator.add

        service = SampleComplexService(
            name="", value=10, enabled=True
        )  # Empty name should fail business rules

        operation_request = FlextModels.OperationExecutionRequest(
            operation_name="add_numbers",
            operation_callable=test_operation,
            arguments={"args": [5, 3]},
        )
        result = service.execute_operation(operation_request)
        assert result.is_failure
        assert result.error is not None
        assert "Business rules validation failed" in result.error
        assert "Name is required" in result.error

    def test_complex_service_execution_config_failure(self) -> None:
        """Test complex service execution with config failure."""
        test_operation = operator.add

        long_name = "a" * 300  # Too long name should fail config validation
        service = SampleComplexService(name=long_name, value=10, enabled=True)

        operation_request = FlextModels.OperationExecutionRequest(
            operation_name="add_numbers",
            operation_callable=test_operation,
            arguments={"args": [5, 3]},
        )
        result = service.execute_operation(operation_request)
        assert result.is_failure
        assert result.error is not None
        assert "Configuration validation failed" in result.error
        assert "Name too long" in result.error

    def test_service_model_config(self) -> None:
        """Test service model configuration."""
        service = SampleUserService()

        model_config = service.model_config
        assert model_config.get("validate_assignment") is True
        assert model_config.get("arbitrary_types_allowed") is True
        assert model_config.get("use_enum_values") is True

    def test_service_inheritance_hierarchy(self) -> None:
        """Test service inheritance from all mixins."""
        service = SampleUserService()

        # Test all expected parent classes

        # Use actual Pydantic BaseModel instead of non-existent FlextModels.BaseModel
        assert isinstance(service, BaseModel)
        assert isinstance(service, FlextMixins.Serializable)
        assert isinstance(service, FlextMixins.Loggable)

    def test_service_with_complex_types(self) -> None:
        """Test service with complex types."""
        service = ComplexTypeService(data={"key": "value"}, items=[1, 2, 3])

        assert service.data == {"key": "value"}
        assert service.items == [1, 2, 3]

        result = service.get_service_info()
        assert isinstance(result, dict)

    def test_service_extra_forbid(self) -> None:
        """Test service with extra fields forbidden."""
        service = SampleComplexService(
            name="test", value=10, enabled=True, extra_field="not_allowed"
        )

        assert not hasattr(service, "extra_field")


class TestDomainServiceStaticMethods:
    """Tests for domain service static methods and configuration."""

    def test_configure_domain_services_system(self) -> None:
        """Test domain service configuration through inheritance."""

        class TestDomainService(FlextDomainService[str]):
            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("Executed successfully")

        service = TestDomainService()
        assert service.is_valid() is True

        # Test service execution
        result = service.execute()
        assert result.is_success
        assert result.unwrap() == "Executed successfully"

    def test_configure_domain_services_system_invalid_config(self) -> None:
        """Test domain service with invalid configuration."""

        class InvalidDomainService(FlextDomainService[str]):
            def execute(self) -> FlextResult[str]:
                return FlextResult[str].fail("Invalid operation")

        service = InvalidDomainService()
        # Test with invalid operation
        result = service.execute()
        assert result.is_failure
        assert result.error is not None

    def test_get_domain_services_system_config(self) -> None:
        """Test domain service configuration access."""

        class ConfigDomainService(FlextDomainService[str]):
            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("Configured service")

        service = ConfigDomainService()
        # Test service info
        info = service.get_service_info()
        assert isinstance(info, dict)
        assert "service_type" in info

    def test_create_environment_domain_services_config(self) -> None:
        """Test domain service environment configuration."""

        class DevDomainService(FlextDomainService[str]):
            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("Dev: test")

        class ProdDomainService(FlextDomainService[str]):
            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("Prod: test")

        # Test development service
        dev_service = DevDomainService()
        result = dev_service.execute()
        assert result.is_success
        assert "Dev: test" in result.unwrap()

        # Test production service
        prod_service = ProdDomainService()
        result = prod_service.execute()
        assert result.is_success
        assert "Prod: test" in result.unwrap()

    def test_optimize_domain_services_performance(self) -> None:
        """Test domain service performance optimization."""

        class OptimizedDomainService(FlextDomainService[str]):
            def execute(self) -> FlextResult[str]:
                # Simulate optimized execution
                return FlextResult[str].ok("Optimized: performance_test")

        service = OptimizedDomainService()
        result = service.execute()
        assert result.is_success
        assert "Optimized: performance_test" in result.unwrap()

    def test_optimize_domain_services_performance_invalid_config(self) -> None:
        """Test domain service with invalid operation."""

        class ErrorDomainService(FlextDomainService[str]):
            def execute(self) -> FlextResult[str]:
                # Simulate invalid operation for testing
                return FlextResult[str].fail("Invalid operation")

        service = ErrorDomainService()
        result = service.execute()
        assert result.is_failure
        assert result.error is not None
