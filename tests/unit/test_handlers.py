"""Comprehensive tests for FlextHandlers - Handler Management.

Tests the actual FlextHandlers API with real functionality testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import math
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest

from flext_core import (
    FlextConstants,
    FlextExceptions,
    FlextHandlers,
    FlextMixins,
    FlextModels,
    FlextResult,
    FlextTypes,
)


class ConcreteTestHandler(FlextHandlers[str, str]):
    """Concrete implementation of FlextHandlers for testing."""

    def handle(self, message: str) -> FlextResult[str]:
        """Handle the message."""
        return FlextResult[str].ok(f"processed_{message}")


class FailingTestHandler(FlextHandlers[str, str]):
    """Concrete implementation that fails for testing error handling."""

    def handle(self, message: str) -> FlextResult[str]:
        """Handle the message with failure."""
        return FlextResult[str].fail(f"Handler failed for: {message}")


class TestFlextHandlers:
    """Test suite for FlextHandlers handler management."""

    def test_handlers_initialization(self) -> None:
        """Test handlers initialization."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_handler_1",
            handler_name="Test Handler 1",
        )
        handlers = ConcreteTestHandler(config=config)
        assert handlers is not None
        assert isinstance(handlers, FlextHandlers)

    def test_handlers_with_custom_config(self) -> None:
        """Test handlers initialization with custom configuration."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_handler_2",
            handler_name="Test Handler 2",
            handler_type=FlextConstants.Cqrs.HandlerType.QUERY,
            handler_mode=FlextConstants.Cqrs.HandlerType.QUERY,
        )
        handlers = ConcreteTestHandler(config=config)
        assert handlers is not None
        assert (
            handlers._config_model.handler_type == FlextConstants.Cqrs.HandlerType.QUERY
        )

    def test_handlers_handle_success(self) -> None:
        """Test successful handler execution."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_handler_3",
            handler_name="Test Handler 3",
        )
        handler = ConcreteTestHandler(config=config)

        result = handler.handle("test_message")
        assert result.is_success
        assert result.value == "processed_test_message"

    def test_handlers_handle_failure(self) -> None:
        """Test handler execution with failure."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_handler_4",
            handler_name="Test Handler 4",
        )
        handler = FailingTestHandler(config=config)

        result = handler.handle("test_message")
        assert result.is_failure
        assert result.error is not None
        assert "Handler failed for: test_message" in result.error

    def test_handlers_config_access(self) -> None:
        """Test access to handler configuration."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_handler_5",
            handler_name="Test Handler 5",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
            handler_mode=FlextConstants.Cqrs.HandlerType.COMMAND,
        )
        handler = ConcreteTestHandler(config=config)

        assert handler._config_model.handler_id == "test_handler_5"
        assert handler._config_model.handler_name == "Test Handler 5"
        assert (
            handler._config_model.handler_type
            == FlextConstants.Cqrs.HandlerType.COMMAND
        )
        assert (
            handler._config_model.handler_mode
            == FlextConstants.Cqrs.HandlerType.COMMAND
        )

    def test_handlers_execution_context(self) -> None:
        """Test handler execution context creation."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_handler_6",
            handler_name="Test Handler 6",
        )
        handler = ConcreteTestHandler(config=config)

        assert handler._execution_context is not None
        assert hasattr(handler._execution_context, "handler_name")
        assert hasattr(handler._execution_context, "handler_mode")

    def test_handlers_message_types(self) -> None:
        """Test accepted message types computation."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_handler_7",
            handler_name="Test Handler 7",
        )
        handler = ConcreteTestHandler(config=config)

        assert handler._accepted_message_types is not None
        assert isinstance(handler._accepted_message_types, (list, tuple, set))

    def test_handlers_revalidation_setting(self) -> None:
        """Test revalidation setting extraction."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_handler_8",
            handler_name="Test Handler 8",
        )
        handler = ConcreteTestHandler(config=config)

        assert isinstance(handler._revalidate_pydantic_messages, bool)

    def test_handlers_type_warning_tracking(self) -> None:
        """Test type warning emission tracking."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_handler_9",
            handler_name="Test Handler 9",
        )
        handler = ConcreteTestHandler(config=config)

        assert isinstance(handler._type_warning_emitted, bool)
        assert handler._type_warning_emitted is False

    def test_handlers_different_types(self) -> None:
        """Test handlers with different message and result types."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_handler_10",
            handler_name="Test Handler 10",
        )

        # Test with different types
        class IntHandler(FlextHandlers[int, str]):
            def handle(self, message: int) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed_{message}")

        handler = IntHandler(config=config)
        result = handler.handle(42)
        assert result.is_success
        assert result.value == "processed_42"

    def test_handlers_command_type(self) -> None:
        """Test handlers with command type."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_command_handler",
            handler_name="Test Command Handler",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
            handler_mode=FlextConstants.Cqrs.HandlerType.COMMAND,
        )
        handler = ConcreteTestHandler(config=config)

        assert (
            handler._config_model.handler_type
            == FlextConstants.Cqrs.HandlerType.COMMAND
        )
        assert (
            handler._config_model.handler_mode
            == FlextConstants.Cqrs.HandlerType.COMMAND
        )

    def test_handlers_query_type(self) -> None:
        """Test handlers with query type."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_query_handler",
            handler_name="Test Query Handler",
            handler_type=FlextConstants.Cqrs.HandlerType.QUERY,
            handler_mode=FlextConstants.Cqrs.HandlerType.QUERY,
        )
        handler = ConcreteTestHandler(config=config)

        assert (
            handler._config_model.handler_type == FlextConstants.Cqrs.HandlerType.QUERY
        )
        assert (
            handler._config_model.handler_mode == FlextConstants.Cqrs.HandlerType.QUERY
        )

    def test_handlers_event_type(self) -> None:
        """Test handlers with event type."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_event_handler",
            handler_name="Test Event Handler",
            handler_type=FlextConstants.Cqrs.HandlerType.EVENT,
            handler_mode=FlextConstants.Cqrs.HandlerType.EVENT,
        )
        handler = ConcreteTestHandler(config=config)

        assert (
            handler._config_model.handler_type == FlextConstants.Cqrs.HandlerType.EVENT
        )
        assert (
            handler._config_model.handler_mode == FlextConstants.Cqrs.HandlerType.EVENT
        )

    def test_handlers_saga_type(self) -> None:
        """Test handlers with saga type."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_saga_handler",
            handler_name="Test Saga Handler",
            handler_type=FlextConstants.Cqrs.HandlerType.SAGA,
            handler_mode=FlextConstants.Cqrs.HandlerType.SAGA,
        )
        handler = ConcreteTestHandler(config=config)

        assert (
            handler._config_model.handler_type == FlextConstants.Cqrs.HandlerType.SAGA
        )
        assert (
            handler._config_model.handler_mode == FlextConstants.Cqrs.HandlerType.SAGA
        )

    def test_handlers_with_metadata(self) -> None:
        """Test handlers with metadata configuration."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_handler_with_metadata",
            handler_name="Test Handler With Metadata",
            metadata={"test_key": "test_value", "priority": 1},
        )
        handler = ConcreteTestHandler(config=config)

        assert handler._config_model.metadata is not None
        assert handler._config_model.metadata["test_key"] == "test_value"
        assert handler._config_model.metadata["priority"] == 1

    def test_handlers_with_timeout(self) -> None:
        """Test handlers with timeout configuration."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_handler_with_timeout",
            handler_name="Test Handler With Timeout",
            command_timeout=60,
        )
        handler = ConcreteTestHandler(config=config)

        assert handler._config_model.command_timeout == 60

    def test_handlers_with_retry_config(self) -> None:
        """Test handlers with retry configuration."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_handler_with_retry",
            handler_name="Test Handler With Retry",
            max_command_retries=3,
        )
        handler = ConcreteTestHandler(config=config)

        assert handler._config_model.max_command_retries == 3

    def test_handlers_abstract_method_implementation(self) -> None:
        """Test that concrete handlers must implement handle method."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_abstract_handler",
            handler_name="Test Abstract Handler",
        )

        # This should work - concrete implementation
        handler = ConcreteTestHandler(config=config)
        assert hasattr(handler, "handle")
        assert callable(handler.handle)

    def test_handlers_inheritance_chain(self) -> None:
        """Test that handlers inherit from FlextMixins."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_inheritance_handler",
            handler_name="Test Inheritance Handler",
        )
        handler = ConcreteTestHandler(config=config)

        # Should inherit from FlextMixins
        assert isinstance(handler, FlextMixins)

    def test_handlers_config_model_type(self) -> None:
        """Test that config model is properly typed."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_config_type_handler",
            handler_name="Test Config Type Handler",
        )
        handler = ConcreteTestHandler(config=config)

        assert isinstance(handler._config_model, FlextModels.Cqrs.Handler)

    def test_handlers_execution_context_type(self) -> None:
        """Test that execution context is properly typed."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_context_type_handler",
            handler_name="Test Context Type Handler",
        )
        handler = ConcreteTestHandler(config=config)

        assert isinstance(
            handler._execution_context,
            FlextModels.HandlerExecutionContext,
        )

    def test_handlers_run_pipeline_with_dict_message_command_id(self) -> None:
        """Test _run_pipeline with dict[str, object] message having command_id."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_pipeline_dict_command_id",
            handler_name="Test Pipeline Dict Command ID",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
            handler_mode=FlextConstants.Cqrs.HandlerType.COMMAND,
        )

        # Create handler that accepts dict[str, object] messages
        class DictHandler(FlextHandlers[dict[str, object], str]):
            def __init__(self, config: FlextModels.Cqrs.Handler) -> None:
                super().__init__(config=config)

            def handle(self, message: dict[str, object]) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed_{message}")

        handler = DictHandler(config=config)
        dict_message: dict[str, object] = {
            "command_id": "cmd_123",
            "data": "test_data",
        }
        result = handler._run_pipeline(dict_message, operation="command")

        assert result.is_success
        assert (
            result.value == "processed_{'command_id': 'cmd_123', 'data': 'test_data'}"
        )

    def test_handlers_run_pipeline_with_dict_message_message_id(self) -> None:
        """Test _run_pipeline with dict[str, object] message having message_id."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_pipeline_dict_message_id",
            handler_name="Test Pipeline Dict Message ID",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
            handler_mode=FlextConstants.Cqrs.HandlerType.COMMAND,
        )

        # Create handler that accepts dict[str, object] messages
        class DictHandler(FlextHandlers[dict[str, object], str]):
            def __init__(self, config: FlextModels.Cqrs.Handler) -> None:
                super().__init__(config=config)

            def handle(self, message: dict[str, object]) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed_{message}")

        handler = DictHandler(config=config)
        dict_message: dict[str, object] = {
            "message_id": "msg_456",
            "data": "test_data",
        }
        result = handler._run_pipeline(dict_message, operation="command")

        assert result.is_success
        assert (
            result.value == "processed_{'message_id': 'msg_456', 'data': 'test_data'}"
        )

    def test_handlers_run_pipeline_with_object_message_command_id(
        self,
    ) -> None:
        """Test _run_pipeline with object message having command_id attribute."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_pipeline_object_command_id",
            handler_name="Test Pipeline Object Command ID",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
            handler_mode=FlextConstants.Cqrs.HandlerType.COMMAND,
        )

        # Create handler that accepts object messages
        class ObjectHandler(FlextHandlers[object, str]):
            def __init__(self, config: FlextModels.Cqrs.Handler) -> None:
                super().__init__(config=config)

            def handle(self, message: object) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed_{message}")

        handler = ObjectHandler(config=config)
        message_obj = SimpleNamespace(command_id="cmd_789", data="test_data")
        result = handler._run_pipeline(message_obj, operation="command")

        assert result.is_success
        assert "processed_" in result.value

    def test_handlers_run_pipeline_with_object_message_message_id(
        self,
    ) -> None:
        """Test _run_pipeline with object message having message_id attribute."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_pipeline_object_message_id",
            handler_name="Test Pipeline Object Message ID",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
            handler_mode=FlextConstants.Cqrs.HandlerType.COMMAND,
        )

        # Create handler that accepts object messages
        class ObjectHandler(FlextHandlers[object, str]):
            def __init__(self, config: FlextModels.Cqrs.Handler) -> None:
                super().__init__(config=config)

            def handle(self, message: object) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed_{message}")

        handler = ObjectHandler(config=config)
        message_obj = SimpleNamespace(message_id="msg_789", data="test_data")
        result = handler._run_pipeline(message_obj, operation="command")

        assert result.is_success
        assert "processed_" in result.value

    def test_handlers_run_pipeline_mode_validation_error(self) -> None:
        """Test _run_pipeline with mismatched operation and handler mode."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_pipeline_mode_error",
            handler_name="Test Pipeline Mode Error",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
            handler_mode=FlextConstants.Cqrs.HandlerType.COMMAND,
        )
        handler = ConcreteTestHandler(config=config)

        # Try to execute query operation on command handler
        result = handler._run_pipeline("test_message", operation="query")

        assert result.is_failure
        assert result.error is not None
        assert (
            "Handler with mode 'command' cannot execute query pipelines" in result.error
        )

    def test_handlers_run_pipeline_cannot_handle_message_type(self) -> None:
        """Test _run_pipeline when handler cannot handle message type."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_pipeline_cannot_handle",
            handler_name="Test Pipeline Cannot Handle",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
            handler_mode=FlextConstants.Cqrs.HandlerType.COMMAND,
        )

        # Create a handler that returns False for can_handle
        class RestrictiveHandler(FlextHandlers[str, str]):
            def __init__(self, config: FlextModels.Cqrs.Handler) -> None:
                super().__init__(config=config)

            def can_handle(self, message_type: object) -> bool:
                # Always return False to simulate cannot handle
                # Parameter is intentionally unused in this test
                _ = message_type  # Mark as intentionally unused
                return False

            def handle(self, message: str) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed_{message}")

        handler = RestrictiveHandler(config=config)
        result = handler._run_pipeline("test_message", operation="command")

        assert result.is_failure
        assert result.error is not None
        assert "Handler cannot handle message type str" in result.error

    def test_handlers_run_pipeline_validation_failure(self) -> None:
        """Test _run_pipeline when message validation fails."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_pipeline_validation_failure",
            handler_name="Test Pipeline Validation Failure",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
            handler_mode=FlextConstants.Cqrs.HandlerType.COMMAND,
        )

        # Create handler that fails validation
        class ValidationFailingHandler(FlextHandlers[str, str]):
            def __init__(self, config: FlextModels.Cqrs.Handler) -> None:
                super().__init__(config=config)

            def validate_command(self, command: object) -> FlextResult[None]:
                # Parameter is intentionally unused in this test
                _ = command  # Mark as intentionally unused
                return FlextResult[None].fail("Validation failed for test")

            def handle(self, message: str) -> FlextResult[str]:
                return FlextResult[str].ok(f"processed_{message}")

        handler = ValidationFailingHandler(config=config)
        result = handler._run_pipeline("test_message", operation="command")

        assert result.is_failure
        assert result.error is not None
        assert "Message validation failed: Validation failed for test" in result.error

    def test_handlers_run_pipeline_handler_exception(self) -> None:
        """Test _run_pipeline when handler.handle() raises exception."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_pipeline_exception",
            handler_name="Test Pipeline Exception",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
            handler_mode=FlextConstants.Cqrs.HandlerType.COMMAND,
        )

        # Create handler that raises exception
        class ExceptionHandler(FlextHandlers[str, str]):
            def __init__(self, config: FlextModels.Cqrs.Handler) -> None:
                super().__init__(config=config)

            def handle(self, message: str) -> FlextResult[str]:
                # Parameter is intentionally unused in this test
                _ = message  # Mark as intentionally unused
                error_message = "Test exception in handler"
                raise ValueError(error_message)

        handler = ExceptionHandler(config=config)
        result = handler._run_pipeline("test_message", operation="command")

        assert result.is_failure
        assert result.error is not None
        assert "Critical handler failure: Test exception in handler" in result.error

    def test_handlers_run_pipeline_query_operation(self) -> None:
        """Test _run_pipeline with query operation."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_pipeline_query",
            handler_name="Test Pipeline Query",
            handler_type=FlextConstants.Cqrs.HandlerType.QUERY,
            handler_mode=FlextConstants.Cqrs.HandlerType.QUERY,
        )

        # Create a query handler
        class QueryHandler(FlextHandlers[str, str]):
            def __init__(self, config: FlextModels.Cqrs.Handler) -> None:
                super().__init__(config=config)

            def handle(self, message: str) -> FlextResult[str]:
                return FlextResult[str].ok(f"queried_{message}")

        handler = QueryHandler(config=config)
        result = handler._run_pipeline("test_query", operation="query")

        assert result.is_success
        assert result.value == "queried_test_query"

    def test_handlers_run_pipeline_query_validation_failure(self) -> None:
        """Test _run_pipeline with query operation validation failure."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_pipeline_query_validation",
            handler_name="Test Pipeline Query Validation",
            handler_type=FlextConstants.Cqrs.HandlerType.QUERY,
            handler_mode=FlextConstants.Cqrs.HandlerType.QUERY,
        )

        # Create query handler that fails validation
        class QueryValidationHandler(FlextHandlers[str, str]):
            def __init__(self, config: FlextModels.Cqrs.Handler) -> None:
                super().__init__(config=config)

            def validate_query(self, query: object) -> FlextResult[None]:
                # Parameter is intentionally unused in this test
                _ = query  # Mark as intentionally unused
                return FlextResult[None].fail("Query validation failed")

            def handle(self, message: str) -> FlextResult[str]:
                return FlextResult[str].ok(f"queried_{message}")

        handler = QueryValidationHandler(config=config)
        result = handler._run_pipeline("test_query", operation="query")

        assert result.is_failure
        assert result.error is not None
        assert "Message validation failed: Query validation failed" in result.error

    def test_handlers_from_callable_basic(self) -> None:
        """Test from_callable with basic function."""

        def simple_handler(message: str) -> str:
            return f"handled_{message}"

        handler = FlextHandlers.from_callable(
            cast("Callable[[object], object]", simple_handler),
            handler_name="simple_handler",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
        )

        assert handler is not None
        assert handler.handler_name == "simple_handler"
        assert handler.mode == FlextConstants.Cqrs.HandlerType.COMMAND

        result = handler.handle("test")
        assert result.is_success
        assert result.value == "handled_test"

    def test_handlers_from_callable_with_flext_result(self) -> None:
        """Test from_callable with function returning FlextResult."""

        def result_handler(message: str) -> FlextResult[str]:
            return FlextResult[str].ok(f"result_{message}")

        handler = FlextHandlers.from_callable(
            cast("Callable[[object], object]", result_handler),
            handler_name="result_handler",
            handler_type=FlextConstants.Cqrs.HandlerType.QUERY,
        )

        assert handler.handler_name == "result_handler"
        assert handler.mode == FlextConstants.Cqrs.HandlerType.QUERY

        result = handler.handle("test")
        assert result.is_success
        assert result.value == "result_test"

    def test_handlers_from_callable_with_exception(self) -> None:
        """Test from_callable with function that raises exception."""

        def failing_handler(message: str) -> str:
            # Parameter is intentionally unused in this test
            _ = message  # Mark as intentionally unused
            error_message = "Handler failed"
            raise ValueError(error_message)

        handler = FlextHandlers.from_callable(
            cast("Callable[[object], object]", failing_handler),
            handler_name="failing_handler",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
        )

        result = handler.handle("test")
        assert result.is_failure
        assert result.error is not None
        assert "Handler failed" in result.error

    def test_handlers_from_callable_default_name(self) -> None:
        """Test from_callable with default handler name from function name."""

        def my_custom_function(message: str) -> str:
            return f"custom_{message}"

        handler = FlextHandlers.from_callable(
            cast("Callable[[object], object]", my_custom_function),
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
        )

        assert handler.handler_name == "my_custom_function"
        assert handler.mode == FlextConstants.Cqrs.HandlerType.COMMAND

    def test_handlers_from_callable_with_mode_parameter(self) -> None:
        """Test from_callable with mode parameter (compatibility)."""

        def mode_handler(message: str) -> str:
            return f"mode_{message}"

        handler = FlextHandlers.from_callable(
            cast("Callable[[object], object]", mode_handler),
            handler_name="mode_handler",
            mode=FlextConstants.Cqrs.HandlerType.QUERY,
        )

        assert handler.handler_name == "mode_handler"
        assert handler.mode == FlextConstants.Cqrs.HandlerType.QUERY

    def test_handlers_from_callable_invalid_mode(self) -> None:
        """Test from_callable with invalid mode."""

        def invalid_handler(message: str) -> str:
            return f"invalid_{message}"

        with pytest.raises(FlextExceptions.ValidationError) as exc_info:
            FlextHandlers.from_callable(
                cast("Callable[[object], object]", invalid_handler),
                handler_name="invalid_handler",
                mode=cast("FlextConstants.Cqrs.HandlerModeSimple", "invalid_mode"),
            )

        assert "Invalid handler mode: invalid_mode" in str(exc_info.value)

    def test_handlers_from_callable_with_model_config(self) -> None:
        """Test from_callable with FlextModels.Cqrs.Handler model config."""

        def model_config_handler(message: str) -> str:
            return f"model_config_{message}"

        handler_config = FlextModels.Cqrs.Handler(
            handler_id="custom_id",
            handler_name="Custom Name",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
            handler_mode=FlextConstants.Cqrs.HandlerType.COMMAND,
            metadata={"test": "value"},
        )

        handler = FlextHandlers.from_callable(
            cast("Callable[[object], object]", model_config_handler),
            handler_config=handler_config,
        )

        assert handler.handler_name == "Custom Name"
        assert handler._config_model.handler_id == "custom_id"
        assert handler._config_model.metadata == {"test": "value"}

    def test_handlers_from_callable_with_pydantic_config(self) -> None:
        """Test from_callable with FlextModels.Cqrs.Handler object."""

        def pydantic_config_handler(message: object) -> object:
            if isinstance(message, str):
                return f"pydantic_config_{message}"
            return f"pydantic_config_{message!s}"

        config = FlextModels.Cqrs.Handler(
            handler_id="pydantic_id",
            handler_name="Pydantic Handler",
            handler_type=FlextConstants.Cqrs.HandlerType.QUERY,
            handler_mode=FlextConstants.Cqrs.HandlerType.QUERY,
        )

        handler = FlextHandlers.from_callable(
            pydantic_config_handler,
            handler_config=config,
        )

        assert handler.handler_name == "Pydantic Handler"
        assert handler._config_model.handler_id == "pydantic_id"
        assert handler.mode == FlextConstants.Cqrs.HandlerType.QUERY

    def test_handlers_from_callable_anonymous_function(self) -> None:
        """Test from_callable with lambda (anonymous function)."""

        def process_message(message: str) -> str:
            return f"lambda_{message!s}"

        handler = FlextHandlers.from_callable(
            cast("FlextTypes.HandlerCallableType", process_message),
            handler_name="lambda_handler",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
        )

        assert handler.handler_name == "lambda_handler"
        result = handler.handle("test")
        assert result.is_success
        assert result.value == "lambda_test"

    def test_handlers_from_callable_function_without_name_attribute(
        self,
    ) -> None:
        """Test from_callable with function object without __name__ attribute."""

        # Create a callable object without __name__
        class CallableObject:
            def __call__(self, message: object) -> object:
                if isinstance(message, str):
                    return f"callable_object_{message}"
                return f"callable_object_{message!s}"

        callable_obj = CallableObject()

        handler = FlextHandlers.from_callable(
            callable_obj, handler_type=FlextConstants.Cqrs.HandlerType.COMMAND
        )

        # Should default to "unknown_handler" when no __name__ attribute
        assert handler.handler_name == "unknown_handler"
        result = handler.handle("test")
        assert result.is_success
        assert result.value == "callable_object_test"

    # =================================================================
    # COMPREHENSIVE MESSAGE VALIDATION TESTS - Fill coverage gaps
    # =================================================================

    def test_handlers_validate_command(self) -> None:
        """Test validate_command method."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_validate_command",
            handler_name="Test Validate Command",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
            handler_mode=FlextConstants.Cqrs.HandlerType.COMMAND,
        )
        handler = ConcreteTestHandler(config=config)

        result = handler.validate_command("test_message")
        assert result.is_success

    def test_handlers_validate_query(self) -> None:
        """Test validate_query method."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_validate_query",
            handler_name="Test Validate Query",
            handler_type=FlextConstants.Cqrs.HandlerType.QUERY,
            handler_mode=FlextConstants.Cqrs.HandlerType.QUERY,
        )
        handler = ConcreteTestHandler(config=config)

        result = handler.validate_query("test_query")
        assert result.is_success

    def test_handlers_execute_method(self) -> None:
        """Test execute method (Layer 1 interface)."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_execute",
            handler_name="Test Execute",
        )
        handler = ConcreteTestHandler(config=config)

        result = handler.execute("test_message")
        assert result.is_success
        assert result.value == "processed_test_message"

    def test_handlers_callable_interface(self) -> None:
        """Test __call__ method (callable interface)."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_callable",
            handler_name="Test Callable",
        )
        handler = ConcreteTestHandler(config=config)

        result = handler("test_message")
        assert result.is_success
        assert result.value == "processed_test_message"

    def test_handlers_can_handle_method(self) -> None:
        """Test can_handle method."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_can_handle",
            handler_name="Test Can Handle",
        )
        handler = ConcreteTestHandler(config=config)

        # Test with type object
        assert isinstance(handler.can_handle(str), bool)

    def test_handlers_mode_property(self) -> None:
        """Test mode property."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_mode_property",
            handler_name="Test Mode Property",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
            handler_mode=FlextConstants.Cqrs.HandlerType.COMMAND,
        )
        handler = ConcreteTestHandler(config=config)

        assert handler.mode == FlextConstants.Cqrs.HandlerType.COMMAND

    def test_handlers_validate_generic_command(self) -> None:
        """Test validate method for command handler."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_validate_generic_cmd",
            handler_name="Test Validate Generic Cmd",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
            handler_mode=FlextConstants.Cqrs.HandlerType.COMMAND,
        )
        handler = ConcreteTestHandler(config=config)

        result = handler.validate("test_message")
        assert result.is_success

    def test_handlers_validate_generic_query(self) -> None:
        """Test validate method for query handler."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_validate_generic_query",
            handler_name="Test Validate Generic Query",
            handler_type=FlextConstants.Cqrs.HandlerType.QUERY,
            handler_mode=FlextConstants.Cqrs.HandlerType.QUERY,
        )
        handler = ConcreteTestHandler(config=config)

        result = handler.validate("test_query")
        assert result.is_success

    def test_handlers_validate_generic_event(self) -> None:
        """Test validate method for event handler."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_validate_generic_event",
            handler_name="Test Validate Generic Event",
            handler_type=FlextConstants.Cqrs.HandlerType.EVENT,
            handler_mode=FlextConstants.Cqrs.HandlerType.EVENT,
        )
        handler = ConcreteTestHandler(config=config)

        result = handler.validate("test_event")
        assert result.is_success

    def test_handlers_validate_generic_saga(self) -> None:
        """Test validate method for saga handler."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_validate_generic_saga",
            handler_name="Test Validate Generic Saga",
            handler_type=FlextConstants.Cqrs.HandlerType.SAGA,
            handler_mode=FlextConstants.Cqrs.HandlerType.SAGA,
        )
        handler = ConcreteTestHandler(config=config)

        result = handler.validate("test_saga")
        assert result.is_success

    def test_handlers_validate_message_protocol(self) -> None:
        """Test validate_message protocol method."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_validate_message_protocol",
            handler_name="Test Validate Message Protocol",
        )
        handler = ConcreteTestHandler(config=config)

        result = handler.validate_message("test_message")
        assert result.is_success

    def test_handlers_record_metric(self) -> None:
        """Test record_metric protocol method."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_record_metric",
            handler_name="Test Record Metric",
        )
        handler = ConcreteTestHandler(config=config)

        result = handler.record_metric("test_metric", 42.0)
        assert result.is_success

    def test_handlers_get_metrics(self) -> None:
        """Test get_metrics protocol method."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_get_metrics",
            handler_name="Test Get Metrics",
        )
        handler = ConcreteTestHandler(config=config)

        # First record a metric
        handler.record_metric("test_metric", 42.0)

        # Then retrieve metrics
        result = handler.get_metrics()
        assert result.is_success
        metrics = result.value
        assert isinstance(metrics, dict)
        assert metrics.get("test_metric") == 42.0

    def test_handlers_push_context(self) -> None:
        """Test push_context protocol method."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_push_context",
            handler_name="Test Push Context",
        )
        handler = ConcreteTestHandler(config=config)

        context = {"user_id": "123", "operation": "test"}
        result = handler.push_context(context)
        assert result.is_success

    def test_handlers_pop_context(self) -> None:
        """Test pop_context protocol method."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_pop_context",
            handler_name="Test Pop Context",
        )
        handler = ConcreteTestHandler(config=config)

        # Push context first
        handler.push_context({"test": "data"})

        # Then pop it
        result = handler.pop_context()
        assert result.is_success

    def test_handlers_pop_context_empty_stack(self) -> None:
        """Test pop_context when stack is empty."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_pop_context_empty",
            handler_name="Test Pop Context Empty",
        )
        handler = ConcreteTestHandler(config=config)

        # Pop without pushing
        result = handler.pop_context()
        assert result.is_success

    def test_handlers_message_with_none_raises_validation_error(self) -> None:
        """Test message validation with None value."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_none_message",
            handler_name="Test None Message",
        )
        handler = ConcreteTestHandler(config=config)

        # None should fail serialization validation
        result = handler.validate(None)  # type: ignore[arg-type]
        assert result.is_failure

    def test_handlers_message_validation_with_custom_method(self) -> None:
        """Test message validation with custom validate_command method."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_custom_validation",
            handler_name="Test Custom Validation",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
            handler_mode=FlextConstants.Cqrs.HandlerType.COMMAND,
        )

        class CustomValidationHandler(FlextHandlers[object, str]):
            def __init__(self, config: FlextModels.Cqrs.Handler) -> None:
                super().__init__(config=config)

            def handle(self, message: object) -> FlextResult[str]:
                return FlextResult[str].ok("success")

        handler = CustomValidationHandler(config=config)

        # Create a message with custom validation
        class MessageWithValidation:
            def __init__(self, value: str) -> None:
                self.value = value

            def validate_command(self) -> FlextResult[None]:
                if len(self.value) > 0:
                    return FlextResult[None].ok(None)
                return FlextResult[None].fail("Value too short")

        msg = MessageWithValidation("test")
        result = handler.validate_command(msg)
        assert result.is_success

    def test_handlers_pydantic_model_validation(self) -> None:
        """Test Pydantic model validation."""
        from pydantic import BaseModel

        class TestMessage(BaseModel):
            value: str

        config = FlextModels.Cqrs.Handler(
            handler_id="test_pydantic_validation",
            handler_name="Test Pydantic Validation",
        )
        handler = ConcreteTestHandler(config=config)

        msg = TestMessage(value="test")
        result = handler.validate(msg)  # type: ignore[arg-type]
        assert result.is_success

    def test_handlers_dict_message_validation(self) -> None:
        """Test dict message validation."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_dict_message",
            handler_name="Test Dict Message",
        )
        handler = ConcreteTestHandler(config=config)

        msg_dict = {"key": "value", "number": 42}
        result = handler.validate(msg_dict)  # type: ignore[arg-type]
        assert result.is_success

    def test_handlers_int_message_validation(self) -> None:
        """Test int message validation."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_int_message",
            handler_name="Test Int Message",
        )
        handler = ConcreteTestHandler(config=config)

        result = handler.validate(42)  # type: ignore[arg-type]
        assert result.is_success

    def test_handlers_float_message_validation(self) -> None:
        """Test float message validation."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_float_message",
            handler_name="Test Float Message",
        )
        handler = ConcreteTestHandler(config=config)

        result = handler.validate(math.pi)  # type: ignore[arg-type]
        assert result.is_success

    def test_handlers_bool_message_validation(self) -> None:
        """Test bool message validation."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_bool_message",
            handler_name="Test Bool Message",
        )
        handler = ConcreteTestHandler(config=config)

        result = handler.validate(True)  # type: ignore[arg-type]
        assert result.is_success

    def test_handlers_dataclass_message_validation(self) -> None:
        """Test dataclass message validation."""
        from dataclasses import dataclass

        @dataclass
        class DataClassMessage:
            value: str
            number: int

        config = FlextModels.Cqrs.Handler(
            handler_id="test_dataclass_message",
            handler_name="Test Dataclass Message",
        )
        handler = ConcreteTestHandler(config=config)

        msg = DataClassMessage(value="test", number=42)
        result = handler.validate(msg)  # type: ignore[arg-type]
        assert result.is_success

    def test_handlers_slots_message_validation(self) -> None:
        """Test __slots__ message validation."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_slots_message",
            handler_name="Test Slots Message",
        )
        handler = ConcreteTestHandler(config=config)

        class SlotsMessage:
            __slots__ = ("number", "value")

            def __init__(self, value: str, number: int) -> None:
                self.value = value
                self.number = number

        msg = SlotsMessage(value="test", number=42)
        result = handler.validate(msg)  # type: ignore[arg-type]
        assert result.is_success

    def test_handlers_dict_with_method_message_validation(self) -> None:
        """Test message with dict() method validation."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_dict_method_message",
            handler_name="Test Dict Method Message",
        )
        handler = ConcreteTestHandler(config=config)

        class DictMethodMessage:
            def __init__(self, value: str) -> None:
                self.value = value

            def dict(self) -> dict[str, object]:
                return {"value": self.value}

        msg = DictMethodMessage(value="test")
        result = handler.validate(msg)  # type: ignore[arg-type]
        assert result.is_success

    def test_handlers_as_dict_method_message_validation(self) -> None:
        """Test message with as_dict() method validation."""
        config = FlextModels.Cqrs.Handler(
            handler_id="test_as_dict_message",
            handler_name="Test As Dict Message",
        )
        handler = ConcreteTestHandler(config=config)

        class AsDictMessage:
            def __init__(self, value: str) -> None:
                self.value = value

            def as_dict(self) -> dict[str, object]:
                return {"value": self.value}

        msg = AsDictMessage(value="test")
        result = handler.validate(msg)  # type: ignore[arg-type]
        assert result.is_success

    def test_handlers_from_callable_invalid_config_creation(self) -> None:
        """Test from_callable with invalid config creation."""

        def handler_func(message: str) -> str:
            return f"handled_{message}"

        # This should handle gracefully if config creation fails
        handler = FlextHandlers.from_callable(
            cast("Callable[[object], object]", handler_func),
            handler_name="test_func",
            handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
        )

        assert handler is not None
        assert handler.handler_name == "test_func"
