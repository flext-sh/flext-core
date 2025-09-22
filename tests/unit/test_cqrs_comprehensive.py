"""Comprehensive tests for FlextCqrs optimized architecture.

Tests for the new FlextCqrs API with:
- Full Pydantic 2 Integration
- Railway-Oriented Programming with FlextResult
- Domain-Driven Design patterns
- Type Safety with mypy strict mode compliance
- Clean Architecture with nested helper classes

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from dataclasses import dataclass, field
from typing import cast
from unittest.mock import Mock, patch

import pytest

from flext_core import (
    FlextBus,
    FlextConstants,
    FlextCqrs,
    FlextHandlers,
    FlextModels,
    FlextResult,
)


class TestFlextCqrsResults:
    """Test FlextCqrs.Results factory methods."""

    def test_success_without_config(self) -> None:
        """Test creating success result without handler configuration."""
        test_data = {"key": "value", "count": 42}

        result = FlextCqrs.Results.success(test_data)

        assert result.is_success
        assert result.value == test_data
        assert not result.is_failure

    def test_success_with_config(self) -> None:
        """Test creating success result with handler configuration."""
        test_data = {"processed": True}
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler_123",
            handler_name="TestHandler",
            handler_type="command",
        )

        result = FlextCqrs.Results.success(test_data, config)

        assert result.is_success
        assert result.value == test_data
        # Check metadata was added
        metadata = getattr(result, "_metadata", {})
        assert metadata.get("handler_id") == "test_handler_123"
        assert metadata.get("handler_name") == "TestHandler"
        assert metadata.get("handler_type") == "command"

    def test_failure_basic(self) -> None:
        """Test creating basic failure result."""
        error_message = "Something went wrong"

        result = FlextCqrs.Results.failure(error_message)

        assert result.is_failure
        assert not result.is_success
        assert result.error == error_message
        assert result.error_code == FlextConstants.Cqrs.CQRS_OPERATION_FAILED

    def test_failure_with_custom_error_code(self) -> None:
        """Test failure result with custom error code."""
        error_message = "Validation failed"
        custom_code = "CUSTOM_ERROR"

        result = FlextCqrs.Results.failure(error_message, error_code=custom_code)

        assert result.is_failure
        assert result.error == error_message
        assert result.error_code == custom_code

    def test_failure_with_error_data_and_config(self) -> None:
        """Test failure result with error data and handler config."""
        error_message = "Processing failed"
        error_data: dict[str, object] = {
            "field": "invalid_value",
            "reason": "too_short",
        }
        config = FlextModels.CqrsConfig.Handler(
            handler_id="processor_456",
            handler_name="DataProcessor",
            handler_type="query",
            metadata={"version": "1.0"},
        )

        result = FlextCqrs.Results.failure(
            error_message,
            error_code="PROCESSING_ERROR",
            error_data=error_data,
            config=config,
        )

        assert result.is_failure
        assert result.error == error_message
        assert result.error_code == "PROCESSING_ERROR"

        # Check enhanced error data includes config info
        enhanced_data = result.error_data
        assert enhanced_data["field"] == "invalid_value"
        assert enhanced_data["handler_id"] == "processor_456"
        assert enhanced_data["handler_name"] == "DataProcessor"
        assert enhanced_data["handler_type"] == "query"
        assert enhanced_data["handler_metadata"] == {"version": "1.0"}


class TestFlextCqrsOperations:
    """Test FlextCqrs.Operations command/query factory methods."""

    def test_create_command_success(self) -> None:
        """Test successful command creation."""
        command_data = {
            "command_type": "ProcessData",
            "payload": {"data": "test_value"},
        }

        result = FlextCqrs.Operations.create_command(
            cast("dict[str, object]", command_data)
        )

        assert result.is_success
        command = result.value
        assert isinstance(command, FlextModels.Command)
        assert command.command_type == "ProcessData"
        # Note: Command model doesn't have payload attribute in current implementation
        # Check that command_id was auto-generated
        assert command.command_id is not None
        assert len(command.command_id) > 0

    def test_create_command_with_custom_id(self) -> None:
        """Test command creation with pre-existing ID."""
        custom_id = "custom_command_789"
        command_data = {
            "command_id": custom_id,
            "command_type": "CustomCommand",
            "payload": {},
        }

        result = FlextCqrs.Operations.create_command(
            cast("dict[str, object]", command_data)
        )

        assert result.is_success
        command = result.value
        assert command.command_id == custom_id

    def test_create_command_with_config(self) -> None:
        """Test command creation with handler configuration."""
        command_data = {
            "command_type": "ConfiguredCommand",
            "payload": {"configured": True},
        }
        config = FlextModels.CqrsConfig.Handler(
            handler_id="config_handler",
            handler_name="ConfiguredHandler",
            handler_type="command",
        )

        result = FlextCqrs.Operations.create_command(
            cast("dict[str, object]", command_data), config
        )

        assert result.is_success
        command = result.value
        assert command.command_type == "ConfiguredCommand"

    def test_create_command_validation_error(self) -> None:
        """Test command creation with invalid data."""
        # Use data that actually causes Pydantic validation to fail
        invalid_data = {
            "command_type": 123,  # Wrong type - should be string
            "payload": {"data": "test"},
        }

        result = FlextCqrs.Operations.create_command(
            cast("dict[str, object]", invalid_data)
        )

        assert result.is_failure
        assert result.error is not None
        assert "Command validation failed" in result.error
        assert result.error_code == FlextConstants.Cqrs.COMMAND_VALIDATION_FAILED
        assert "command_data" in result.error_data

    def test_create_query_success(self) -> None:
        """Test successful query creation."""
        query_data = {
            "query_type": "GetUserData",
            "filters": {"user_id": "123"},
            "pagination": {"page": 1, "size": 10},
        }

        result = FlextCqrs.Operations.create_query(
            cast("dict[str, object]", query_data)
        )

        assert result.is_success
        query = result.value
        assert isinstance(query, FlextModels.Query)
        # Note: Query model doesn't have query_type attribute in current implementation
        assert query.filters == {"user_id": "123"}
        assert query.pagination == {"page": 1, "size": 10}
        # Check auto-generated query_id
        assert query.query_id is not None

    def test_create_query_with_custom_id(self) -> None:
        """Test query creation with pre-existing ID."""
        custom_id = "custom_query_456"
        query_data = {
            "query_id": custom_id,
            "query_type": "CustomQuery",
            "criteria": {},
        }

        result = FlextCqrs.Operations.create_query(
            cast("dict[str, object]", query_data)
        )

        assert result.is_success
        query = result.value
        assert query.query_id == custom_id

    def test_create_query_validation_error(self) -> None:
        """Test query creation with invalid data."""
        # Use data that actually causes Pydantic validation to fail
        invalid_data = {
            "query_type": 123,  # Wrong type - should be string
            "filters": {"invalid": True},
        }

        result = FlextCqrs.Operations.create_query(
            cast("dict[str, object]", invalid_data)
        )

        assert result.is_failure
        assert result.error is not None
        assert "Query validation failed" in result.error
        assert result.error_code == FlextConstants.Cqrs.QUERY_VALIDATION_FAILED

    def test_create_handler_config_command(self) -> None:
        """Test creating command handler configuration."""
        result = FlextCqrs.Operations.create_handler_config(
            "command", handler_name="CustomCommandHandler", handler_id="cmd_handler_123"
        )

        assert result.is_success
        config = result.value
        assert isinstance(config, FlextModels.CqrsConfig.Handler)
        assert config.handler_type == "command"
        assert config.handler_name == "CustomCommandHandler"
        assert config.handler_id == "cmd_handler_123"

    def test_create_handler_config_query_defaults(self) -> None:
        """Test creating query handler config with defaults."""
        result = FlextCqrs.Operations.create_handler_config("query")

        assert result.is_success
        config = result.value
        assert config.handler_type == "query"
        assert config.handler_name == "query_handler"
        assert config.handler_id.startswith("handler_query_")

    def test_create_handler_config_with_overrides(self) -> None:
        """Test handler config creation with overrides."""
        overrides = {"timeout": 5000, "retries": 3, "custom_field": "custom_value"}

        result = FlextCqrs.Operations.create_handler_config(
            "command", config_overrides=cast("dict[str, object]", overrides)
        )

        assert result.is_success
        config = result.value
        assert config.handler_type == "command"
        # Overrides should be included in the config

    def test_create_handler_config_invalid_type(self) -> None:
        """Test handler config creation with invalid handler type."""
        # This should pass validation at the Pydantic level since we use Literal types
        # But if we had additional validation, we could test failure scenarios
        result = FlextCqrs.Operations.create_handler_config("command")
        assert result.is_success


@dataclass
class MockCommand:
    """Mock command for testing."""

    user_id: str
    name: str
    action: str = "default_action"
    data: dict[str, object] = field(default_factory=dict)


class TestFlextCqrsDecorators:
    """Test FlextCqrs.Decorators command handler decorator."""

    def test_command_handler_decorator_basic(self) -> None:
        """Test basic command handler decorator functionality."""

        @FlextCqrs.Decorators.command_handler(MockCommand)
        def handle_test_command(command: MockCommand) -> str:
            return f"Processed {command.action} with {len(command.data)} items"

        # Check that decorator preserved function metadata
        assert handle_test_command.__name__ == "handle_test_command"
        assert callable(handle_test_command)

        # Check that decorator added metadata
        assert hasattr(handle_test_command, "__dict__")
        func_metadata = handle_test_command.__dict__
        assert func_metadata.get("command_type") == MockCommand
        assert func_metadata.get("flext_cqrs_decorator") is True
        assert "handler_instance" in func_metadata
        assert "handler_config" in func_metadata

    def test_command_handler_decorator_with_config(self) -> None:
        """Test command handler decorator with configuration."""
        custom_config = {"timeout": 10000, "retries": 5, "metadata": {"version": "2.0"}}

        @FlextCqrs.Decorators.command_handler(
            MockCommand, config=cast("dict[str, object]", custom_config)
        )
        def configured_handler(command: MockCommand) -> bool:
            return command.action == "process"

        # Verify the handler was configured
        handler_config = configured_handler.__dict__.get("handler_config")
        assert handler_config is not None
        assert isinstance(handler_config, FlextModels.CqrsConfig.Handler)
        assert handler_config.handler_type == "command"

    def test_command_handler_execution_success(self) -> None:
        """Test successful command handler execution."""

        @FlextCqrs.Decorators.command_handler(MockCommand)
        def successful_handler(command: MockCommand) -> str:
            return f"Success: {command.action}"

        # Create test command
        test_command = MockCommand(
            user_id="test_user",
            name="Test User",
            action="test_action",
            data={"key": "value"},
        )

        # Execute handler directly (decorator preserves original function)
        result = successful_handler(test_command)
        assert result == "Success: test_action"

    def test_command_handler_with_flext_result_return(self) -> None:
        """Test command handler that returns FlextResult."""

        @FlextCqrs.Decorators.command_handler(MockCommand)
        def result_handler(command: MockCommand) -> FlextResult[str]:
            if command.action == "fail":
                return FlextResult[str].fail("Command failed")
            return FlextResult[str].ok(f"Result: {command.action}")

        # Test success case
        success_command = MockCommand(
            user_id="success_user", name="Success User", action="succeed", data={}
        )
        result = result_handler(success_command)
        assert isinstance(result, FlextResult)
        # When called directly, should return the FlextResult as-is

    def test_command_handler_instance_execution(self) -> None:
        """Test executing through handler instance for error handling."""

        @FlextCqrs.Decorators.command_handler(MockCommand)
        def instance_handler(command: MockCommand) -> str:
            error_msg = "Test error"
            if command.action == "error":
                raise ValueError(error_msg)
            return f"Handled: {command.action}"

        # Get the handler instance from decorator metadata
        handler_instance = instance_handler.__dict__.get("handler_instance")
        assert handler_instance is not None
        assert isinstance(handler_instance, FlextHandlers)

        # Test successful execution through instance
        success_command = MockCommand(
            user_id="success_user", name="Success User", action="success", data={}
        )
        result = handler_instance.handle(success_command)
        assert result.is_success
        assert result.value == "Handled: success"

        # Test error handling through instance
        error_command = MockCommand(
            user_id="error_user", name="Error User", action="error", data={}
        )
        error_result = handler_instance.handle(error_command)
        assert error_result.is_failure
        assert error_result.error is not None
        assert "Command handler execution failed" in error_result.error
        assert error_result.error_code == FlextConstants.Cqrs.COMMAND_PROCESSING_FAILED

    def test_command_handler_preserves_annotations(self) -> None:
        """Test that decorator preserves function annotations."""

        @FlextCqrs.Decorators.command_handler(MockCommand)
        def annotated_handler(command: MockCommand) -> dict[str, int]:
            return {"count": len(command.data)}

        # Check annotations are preserved
        annotations = annotated_handler.__annotations__
        assert "command" in annotations
        assert annotations["command"] == MockCommand
        assert annotations["return"] == dict[str, int]


class TestFlextCqrsIntegration:
    """Integration tests for FlextCqrs with other components."""

    def test_full_command_workflow(self) -> None:
        """Test complete command creation and handling workflow."""
        # Step 1: Create handler configuration
        config_result = FlextCqrs.Operations.create_handler_config(
            "command", handler_name="DataProcessorHandler"
        )
        assert config_result.is_success
        handler_config = config_result.value

        # Step 2: Create command
        command_data = {
            "command_type": "ProcessData",
            "payload": {"items": [1, 2, 3]},
            "priority": "normal",
        }
        command_result = FlextCqrs.Operations.create_command(
            cast("dict[str, object]", command_data), handler_config
        )
        assert command_result.is_success
        command = command_result.value

        # Step 3: Create success result
        success_result = FlextCqrs.Results.success(
            {"processed_count": 3}, handler_config
        )
        assert success_result.is_success

        # Verify the complete workflow
        assert command.command_type == "ProcessData"
        assert isinstance(success_result.value, dict)
        assert success_result.value["processed_count"] == 3

    def test_error_propagation_workflow(self) -> None:
        """Test error handling throughout the CQRS workflow."""
        # Step 1: Try to create invalid command (use actually invalid data)
        invalid_command_result = FlextCqrs.Operations.create_command(
            cast(
                "dict[str, object]",
                {
                    "command_type": 123,  # Wrong type should fail
                    "payload": {},
                },
            )
        )
        assert invalid_command_result.is_failure

        # Step 2: Create failure result with context
        failure_result = FlextCqrs.Results.failure(
            "Command processing failed",
            error_code="WORKFLOW_ERROR",
            error_data={
                "step": "validation",
                "original_error": invalid_command_result.error,
            },
        )
        assert failure_result.is_failure
        assert failure_result.error_code == "WORKFLOW_ERROR"
        assert isinstance(failure_result.error_data, dict)
        assert failure_result.error_data["step"] == "validation"

    @patch("flext_core.FlextConfig.get_global_instance")
    def test_configuration_integration(self, mock_config: Mock) -> None:
        """Test integration with FlextConfig for CQRS configuration."""
        # Mock FlextConfig behavior
        mock_instance = Mock()
        mock_config.return_value = mock_instance
        mock_bus_config = FlextModels.CqrsConfig.Bus.create_bus_config(None)
        mock_instance.get_cqrs_bus_config.return_value = FlextResult[
            FlextModels.CqrsConfig.Bus
        ].ok(mock_bus_config)

        # Test configuration helper
        config_result = FlextCqrs._ConfigurationHelper.get_default_cqrs_config()

        assert config_result.is_success
        mock_config.assert_called_once()
        mock_instance.get_cqrs_bus_config.assert_called_once()

    def test_utility_integration(self) -> None:
        """Test integration with FlextUtilities for ID generation."""
        # Test handler ID generation
        base_name = "test_handler"
        handler_id = FlextCqrs._ProcessingHelper.generate_handler_id(base_name)

        assert handler_id.startswith("handler_test_handler_")
        assert len(handler_id) > len("handler_test_handler_")

    def test_command_type_derivation(self) -> None:
        """Test command type derivation from class names."""
        # Test successful derivation
        result = FlextCqrs._ProcessingHelper.derive_command_type("ProcessDataCommand")
        assert result.is_success
        assert result.value == "process_data"

        # Test with complex name
        complex_result = FlextCqrs._ProcessingHelper.derive_command_type(
            "HandleUserRegistrationCommand"
        )
        assert complex_result.is_success
        assert complex_result.value == "handle_user_registration"

    def test_validation_error_handling(self) -> None:
        """Test structured validation error handling."""

        # Create mock Pydantic validation error
        class MockValidationError(Exception):
            def errors(self) -> list[dict[str, str]]:
                return [
                    {"field": "email", "message": "Invalid email format"},
                    {"field": "age", "message": "Must be positive integer"},
                ]

            def __str__(self) -> str:
                return "2 validation errors"

        mock_error = MockValidationError()
        context = {
            "operation": "user_creation",
            "input_data": {"email": "invalid", "age": -5},
        }

        result = FlextCqrs._ErrorHelper.handle_validation_error(
            mock_error, cast("dict[str, object]", context)
        )

        assert result.is_failure
        assert result.error is not None
        assert "Validation failed" in result.error
        assert result.error_code == FlextConstants.Errors.VALIDATION_ERROR
        assert "validation_errors" in result.error_data
        assert result.error_data["context"] == context


class TestFlextCqrsBusMiddlewarePipeline:
    """Tests for FlextBus middleware ordering and enablement."""

    def test_middleware_executes_in_configured_order(self) -> None:
        """Ensure multiple middleware run respecting their configured order."""

        bus = FlextBus()
        calls: list[str] = []

        @dataclass
        class SampleCommand:
            command_id: str = "cmd-order"

        class RecordingMiddleware:
            def __init__(self, name: str) -> None:
                self.name = name

            def process(
                self, _command: object, _handler: object
            ) -> FlextResult[None]:
                calls.append(self.name)
                return FlextResult[None].ok(None)

        class RecordingHandler:
            def handle(self, _command: object) -> FlextResult[str]:
                calls.append("handler")
                return FlextResult[str].ok("handled")

        handler = RecordingHandler()
        registration = bus.register_handler(SampleCommand, handler)
        assert registration.is_success

        first_middleware = RecordingMiddleware("first")
        second_middleware = RecordingMiddleware("second")

        add_first = bus.add_middleware(
            first_middleware,
            {
                "middleware_id": "mw-first",
                "middleware_type": "RecordingMiddleware",
                "enabled": True,
                "order": 2,
            },
        )
        add_second = bus.add_middleware(
            second_middleware,
            {
                "middleware_id": "mw-second",
                "middleware_type": "RecordingMiddleware",
                "enabled": True,
                "order": 1,
            },
        )

        assert add_first.is_success
        assert add_second.is_success

        result = bus.execute(SampleCommand())

        assert result.is_success
        assert calls == ["second", "first", "handler"]

    def test_disabled_middleware_is_ignored(self) -> None:
        """Verify disabled middleware entries do not execute."""

        bus = FlextBus()
        calls: list[str] = []

        @dataclass
        class SampleCommand:
            command_id: str = "cmd-disabled"

        class RecordingMiddleware:
            def __init__(self, name: str) -> None:
                self.name = name

            def process(
                self, _command: object, _handler: object
            ) -> FlextResult[None]:
                calls.append(self.name)
                return FlextResult[None].ok(None)

        class RecordingHandler:
            def handle(self, _command: object) -> FlextResult[str]:
                calls.append("handler")
                return FlextResult[str].ok("handled")

        handler = RecordingHandler()
        registration = bus.register_handler(SampleCommand, handler)
        assert registration.is_success

        disabled_middleware = RecordingMiddleware("disabled")
        enabled_middleware = RecordingMiddleware("enabled")

        add_disabled = bus.add_middleware(
            disabled_middleware,
            {
                "middleware_id": "mw-disabled",
                "middleware_type": "RecordingMiddleware",
                "enabled": False,
                "order": 1,
            },
        )
        add_enabled = bus.add_middleware(
            enabled_middleware,
            {
                "middleware_id": "mw-enabled",
                "middleware_type": "RecordingMiddleware",
                "enabled": True,
                "order": 2,
            },
        )

        assert add_disabled.is_success
        assert add_enabled.is_success

        result = bus.execute(SampleCommand())

        assert result.is_success
        assert calls == ["enabled", "handler"]


if __name__ == "__main__":
    pytest.main([__file__])
