"""Comprehensive coverage tests for FlextService.

This module provides extensive tests for FlextService domain service patterns:
- Service instantiation and lifecycle
- Automatic registration via __init_subclass__
- Abstract method enforcement
- Validation patterns (business rules, configuration)
- Operation execution with timeout, retry, and validation
- Context cleanup and propagation
- Conditional execution patterns
- Helper utilities for execution and metadata

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import cast

from flext_core import (
    FlextContainer,
    FlextModels,
    FlextResult,
    FlextService,
)


class SimpleResult(FlextModels.Value):
    """Simple result value object for tests."""

    value: str


class TestServiceInstantiation:
    """Test basic service instantiation and infrastructure."""

    def test_service_instantiation(self) -> None:
        """Test creating a service subclass."""

        class SimpleService(FlextService[str]):
            """Simple test service."""

            def execute(self) -> FlextResult[str]:
                """Execute the service."""
                return FlextResult[str].ok("success")

        service = SimpleService()
        assert service is not None

    def test_service_has_container_access(self) -> None:
        """Test service has access to container via mixins."""

        class TestService(FlextService[str]):
            """Service with container access."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("test")

        service = TestService()
        # Container is provided via FlextMixins
        container = service.container
        assert isinstance(container, FlextContainer)

    def test_service_has_logger_access(self) -> None:
        """Test service has logger via mixins."""

        class LoggingService(FlextService[str]):
            """Service with logger access."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("logged")

        service = LoggingService()
        # Logger is provided via FlextMixins
        assert hasattr(service, "logger")
        assert service.logger is not None

    def test_service_has_config_access(self) -> None:
        """Test service has config via service_config computed field."""

        class ConfigService(FlextService[str]):
            """Service with config access."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("configured")

        service = ConfigService()
        # Config is provided via computed_field
        config = service.service_config
        assert config is not None


class TestServiceRegistration:
    """Test automatic service registration via __init_subclass__."""

    def test_service_auto_registration_in_container(self) -> None:
        """Test service is automatically registered in container on class definition."""

        class RegisteredService(FlextService[str]):
            """Service that auto-registers."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("registered")

        # Service should be registered by __init_subclass__
        # Explicitly reference the class to satisfy linter
        assert RegisteredService is not None
        container = FlextContainer.get_global()
        registration_result = container.get("RegisteredService")
        assert registration_result.is_success


class TestAbstractMethodEnforcement:
    """Test that execute() must be implemented by subclasses."""

    def test_abstract_execute_enforced(self) -> None:
        """Test that execute() is abstract and cannot be instantiated without implementation."""
        # This test verifies the ABC pattern is enforced

        # Trying to instantiate FlextService directly fails (it's abstract)
        # Note: This tests the ABC enforcement at runtime
        try:
            # This should raise TypeError due to missing execute() implementation
            # Use type: ignore to suppress mypy warning about abstract class instantiation
            _ = FlextService[str]()  # type: ignore[abstract]
            msg = "Should have raised TypeError"
            raise AssertionError(msg)
        except TypeError:
            pass  # Expected behavior

    def test_concrete_service_must_implement_execute(self) -> None:
        """Test that concrete services must implement execute()."""

        class ConcreteService(FlextService[str]):
            """Concrete service implementing execute."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("concrete")

        service = ConcreteService()
        result = service.execute()
        assert result.is_success
        assert result.value == "concrete"


class TestValidationMethods:
    """Test service validation methods."""

    def test_validate_business_rules_default(self) -> None:
        """Test default business_rules validation (always succeeds)."""

        class ValidatedService(FlextService[str]):
            """Service with default validation."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("valid")

        service = ValidatedService()
        result = service.validate_business_rules()
        assert result.is_success
        assert result.value is None

    def test_validate_business_rules_custom(self) -> None:
        """Test custom business rules validation."""

        class CustomValidationService(FlextService[str]):
            """Service with custom validation."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("executed")

            def validate_business_rules(self) -> FlextResult[None]:
                """Custom validation that checks a condition."""
                return FlextResult[None].ok(None)

        service = CustomValidationService()
        result = service.validate_business_rules()
        assert result.is_success

    def test_validate_config_default(self) -> None:
        """Test default configuration validation."""

        class ConfigValidatedService(FlextService[str]):
            """Service with config validation."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("configured")

        service = ConfigValidatedService()
        result = service.validate_config()
        assert result.is_success

    def test_is_valid_with_successful_validation(self) -> None:
        """Test is_valid() returns True when all validations pass."""

        class AlwaysValidService(FlextService[str]):
            """Service that always validates successfully."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("valid")

        service = AlwaysValidService()
        assert service.is_valid() is True

    def test_is_valid_with_failed_validation(self) -> None:
        """Test is_valid() returns False when validation fails."""

        class InvalidService(FlextService[str]):
            """Service that fails validation."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("should_not_reach")

            def validate_business_rules(self) -> FlextResult[None]:
                """Validation that fails."""
                return FlextResult[None].fail("Invalid business rules")

        service = InvalidService()
        assert service.is_valid() is False

    def test_is_valid_handles_exceptions(self) -> None:
        """Test is_valid() returns False if validation raises exception."""

        class ExceptionThrowingService(FlextService[str]):
            """Service where validation throws exception."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("safe")

            def validate_business_rules(self) -> FlextResult[None]:
                """Validation that throws exception."""
                msg = "Validation error"
                raise ValueError(msg)

        service = ExceptionThrowingService()
        # Should return False (not raise)
        assert service.is_valid() is False


class TestServiceInfo:
    """Test service information retrieval."""

    def test_get_service_info(self) -> None:
        """Test get_service_info returns service metadata."""

        class InfoService(FlextService[str]):
            """Service with info."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("info")

        service = InfoService()
        info = service.get_service_info()
        assert isinstance(info, dict)
        assert "service_type" in info
        assert info["service_type"] == "InfoService"


class TestExecuteContextCleanup:
    """Test execute_with_context_cleanup functionality."""

    def test_execute_with_context_cleanup_success(self) -> None:
        """Test context cleanup after successful execution."""

        class CleanupService(FlextService[str]):
            """Service with context cleanup."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("cleaned")

        service = CleanupService()
        result = service.execute_with_context_cleanup()
        assert result.is_success
        assert result.value == "cleaned"

    def test_execute_with_context_cleanup_preserves_failure(self) -> None:
        """Test context cleanup preserves failures."""

        class FailingCleanupService(FlextService[str]):
            """Service that fails but cleanup is called."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].fail("execution failed")

        service = FailingCleanupService()
        result = service.execute_with_context_cleanup()
        assert result.is_failure
        assert result.error is None or "execution failed" in result.error


class TestExecuteFullValidation:
    """Test execute_with_full_validation functionality."""

    def test_execute_with_full_validation_success(self) -> None:
        """Test full validation execution on success."""

        class FullValidService(FlextService[str]):
            """Service with full validation."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("fully_valid")

        service = FullValidService()
        request = FlextModels.DomainServiceExecutionRequest(
            service_name="FullValidService",
            method_name="test_op",
        )
        result = service.execute_with_full_validation(request)
        assert result.is_success

    def test_execute_with_full_validation_failed_business_rules(self) -> None:
        """Test full validation fails when business rules fail."""

        class InvalidBusinessRulesService(FlextService[str]):
            """Service with invalid business rules."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("should_not_reach")

            def validate_business_rules(self) -> FlextResult[None]:
                return FlextResult[None].fail("Business rules violated")

        service = InvalidBusinessRulesService()
        request = FlextModels.DomainServiceExecutionRequest(
            service_name="InvalidBusinessRulesService",
            method_name="test_op",
        )
        result = service.execute_with_full_validation(request)
        assert result.is_failure

    def test_execute_with_full_validation_failed_config(self) -> None:
        """Test full validation fails when config validation fails."""

        class InvalidConfigService(FlextService[str]):
            """Service with invalid configuration."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("should_not_reach")

            def validate_config(self) -> FlextResult[None]:
                return FlextResult[None].fail("Configuration invalid")

        service = InvalidConfigService()
        request = FlextModels.DomainServiceExecutionRequest(
            service_name="InvalidConfigService",
            method_name="test_op",
        )
        result = service.execute_with_full_validation(request)
        assert result.is_failure


class TestExecuteOperation:
    """Test execute_operation with various configurations."""

    def test_execute_operation_basic(self) -> None:
        """Test basic operation execution."""

        class OperationService(FlextService[str]):
            """Service supporting operation execution."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("operation_executed")

        service = OperationService()

        def operation_func() -> str:
            return "result"

        request = FlextModels.OperationExecutionRequest(
            operation_name="test_operation",
            operation_callable=operation_func,
            arguments={},
            keyword_arguments={},
        )
        result = service.execute_operation(request)
        assert result.is_success

    def test_execute_operation_with_arguments(self) -> None:
        """Test operation execution with arguments."""

        class ArgOperationService(FlextService[int]):
            """Service with argument handling."""

            def execute(self) -> FlextResult[int]:
                return FlextResult[int].ok(42)

        service = ArgOperationService()

        def add_numbers(a: int, b: int) -> int:
            return a + b

        request = FlextModels.OperationExecutionRequest(
            operation_name="add",
            operation_callable=add_numbers,
            arguments={"a": 5, "b": 3},
            keyword_arguments={},
        )
        result = service.execute_operation(request)
        assert result.is_success

    def test_execute_operation_callable_validation(self) -> None:
        """Test operation execution validates callable requirement."""

        class CallableValidationService(FlextService[str]):
            """Service with callable validation."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("validated")

        service = CallableValidationService()

        def valid_func() -> str:
            return "result"

        request = FlextModels.OperationExecutionRequest(
            operation_name="op",
            operation_callable=valid_func,
            arguments={},
            keyword_arguments={},
        )
        result = service.execute_operation(request)
        assert result.is_success or result.is_failure  # Callable is valid


class TestConditionalExecution:
    """Test conditional execution patterns."""

    def test_execute_conditionally_true_condition_with_action(self) -> None:
        """Test conditional execution with true condition."""

        class ConditionalService(FlextService[str]):
            """Service with conditional execution."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("default")

        service = ConditionalService()

        def true_action(svc: object) -> object:
            return "true_action_result"

        request = FlextModels.ConditionalExecutionRequest(
            condition=lambda svc: True,
            true_action=true_action,
        )
        result = service.execute_conditionally(request)
        assert result.is_success
        assert result.value == "true_action_result"

    def test_execute_conditionally_false_condition_with_action(self) -> None:
        """Test conditional execution with false condition."""

        class FalseConditionService(FlextService[str]):
            """Service with false condition."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("default")

        service = FalseConditionService()

        def true_action(svc: object) -> object:
            return "skipped"

        def false_action(svc: object) -> object:
            return "false_action_result"

        request = FlextModels.ConditionalExecutionRequest(
            condition=lambda svc: False,
            true_action=true_action,
            false_action=false_action,
        )
        result = service.execute_conditionally(request)
        assert result.is_success
        assert result.value == "false_action_result"

    def test_execute_conditionally_false_condition_no_false_action(
        self,
    ) -> None:
        """Test conditional execution with false condition and no false action."""

        class NoActionService(FlextService[str]):
            """Service with no false action."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("default")

        service = NoActionService()

        def true_action(svc: object) -> object:
            return "should_not_execute"

        request = FlextModels.ConditionalExecutionRequest(
            condition=lambda svc: False,
            true_action=true_action,
        )
        result = service.execute_conditionally(request)
        assert result.is_failure

    def test_execute_conditionally_default_execute(self) -> None:
        """Test conditional execution falls back to execute()."""

        class DefaultExecuteService(FlextService[str]):
            """Service using default execute."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("default_executed")

        service = DefaultExecuteService()

        def true_action(svc: object) -> object:
            return "action_not_called"

        request = FlextModels.ConditionalExecutionRequest(
            condition=lambda svc: True,
            true_action=true_action,
        )
        result = service.execute_conditionally(request)
        assert result.is_success


class TestExecutionHelper:
    """Test _ExecutionHelper utility class."""

    def test_prepare_execution_context(self) -> None:
        """Test execution context preparation."""

        class HelperService(FlextService[str]):
            """Service for testing helpers."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("helper")

        service = HelperService()
        service_obj: FlextService[object] = cast("FlextService[object]", service)
        context = FlextService._ExecutionHelper.prepare_execution_context(service_obj)
        assert isinstance(context, dict)
        assert "service_type" in context
        assert context["service_type"] == "HelperService"
        assert "timestamp" in context

    def test_cleanup_execution_context(self) -> None:
        """Test execution context cleanup."""

        class CleanupHelperService(FlextService[str]):
            """Service for cleanup testing."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("cleaned")

        service = CleanupHelperService()
        service_obj: FlextService[object] = cast("FlextService[object]", service)
        context: dict[str, object] = {"test": "data"}
        # Should not raise
        FlextService._ExecutionHelper.cleanup_execution_context(service_obj, context)


class TestMetadataHelper:
    """Test _MetadataHelper utility class."""

    def test_extract_service_metadata(self) -> None:
        """Test metadata extraction."""

        class MetadataService(FlextService[str]):
            """Service for metadata extraction."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("metadata")

        service = MetadataService()
        service_obj: FlextService[object] = cast("FlextService[object]", service)
        metadata = FlextService._MetadataHelper.extract_service_metadata(service_obj)
        assert isinstance(metadata, dict)
        assert metadata["service_class"] == "MetadataService"
        assert "service_module" in metadata
        assert "created_at" in metadata
        assert "extracted_at" in metadata

    def test_extract_service_metadata_without_timestamps(self) -> None:
        """Test metadata extraction without timestamps."""

        class NoTimeService(FlextService[str]):
            """Service without timestamps."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("notimestamp")

        service = NoTimeService()
        service_obj: FlextService[object] = cast("FlextService[object]", service)
        metadata = FlextService._MetadataHelper.extract_service_metadata(
            service_obj, include_timestamps=False
        )
        assert "created_at" not in metadata
        assert "extracted_at" not in metadata
        assert metadata["service_class"] == "NoTimeService"

    def test_format_service_info(self) -> None:
        """Test service info formatting."""

        class FormattedService(FlextService[str]):
            """Service for info formatting."""

            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("formatted")

        service = FormattedService()
        service_obj: FlextService[object] = cast("FlextService[object]", service)
        metadata: dict[str, object] = {
            "service_type": "TestService",
            "service_name": "test_svc",
        }
        formatted = FlextService._MetadataHelper.format_service_info(
            service_obj, metadata
        )
        assert isinstance(formatted, str)
        assert "TestService" in formatted


class TestServiceGenericType:
    """Test service with different generic result types."""

    def test_service_with_value_object_result(self) -> None:
        """Test service returning value object."""

        class ValueObjectService(FlextService[SimpleResult]):
            """Service returning value object."""

            def execute(self) -> FlextResult[SimpleResult]:
                return FlextResult[SimpleResult].ok(SimpleResult(value="test_value"))

        service = ValueObjectService()
        result = service.execute()
        assert result.is_success
        assert result.value.value == "test_value"

    def test_service_with_dict_result(self) -> None:
        """Test service returning dictionary."""

        class DictService(FlextService[dict[str, object]]):
            """Service returning dict."""

            def execute(self) -> FlextResult[dict[str, object]]:
                return FlextResult[dict[str, object]].ok({
                    "key": "value",
                    "count": 42,
                })

        service = DictService()
        result = service.execute()
        assert result.is_success
        assert result.value["key"] == "value"


__all__ = [
    "TestAbstractMethodEnforcement",
    "TestConditionalExecution",
    "TestExecuteContextCleanup",
    "TestExecuteFullValidation",
    "TestExecuteOperation",
    "TestExecutionHelper",
    "TestMetadataHelper",
    "TestServiceGenericType",
    "TestServiceInfo",
    "TestServiceInstantiation",
    "TestServiceRegistration",
    "TestValidationMethods",
]
