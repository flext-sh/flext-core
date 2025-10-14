"""Comprehensive tests for FlextCore.Handlers - Handler Management.

Tests the actual FlextCore.Handlers API with real functionality testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest

from flext_core import FlextCore
from flext_core.mixins import FlextMixins


class ConcreteTestHandler(FlextCore.Handlers[str, str]):
    """Concrete implementation of FlextCore.Handlers for testing."""

    def handle(self, message: str) -> FlextCore.Result[str]:
        """Handle the message."""
        return FlextCore.Result[str].ok(f"processed_{message}")


class FailingTestHandler(FlextCore.Handlers[str, str]):
    """Concrete implementation that fails for testing error handling."""

    def handle(self, message: str) -> FlextCore.Result[str]:
        """Handle the message with failure."""
        return FlextCore.Result[str].fail(f"Handler failed for: {message}")


class TestFlextHandlers:
    """Test suite for FlextCore.Handlers handler management."""

    def test_handlers_initialization(self) -> None:
        """Test handlers initialization."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_handler_1",
            handler_name="Test Handler 1",
        )
        handlers = ConcreteTestHandler(config=config)
        assert handlers is not None
        assert isinstance(handlers, FlextCore.Handlers)

    def test_handlers_with_custom_config(self) -> None:
        """Test handlers initialization with custom configuration."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_handler_2",
            handler_name="Test Handler 2",
            handler_type="query",
            handler_mode="query",
        )
        handlers = ConcreteTestHandler(config=config)
        assert handlers is not None
        assert handlers._config_model.handler_type == "query"

    def test_handlers_handle_success(self) -> None:
        """Test successful handler execution."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_handler_3",
            handler_name="Test Handler 3",
        )
        handler = ConcreteTestHandler(config=config)

        result = handler.handle("test_message")
        assert result.is_success
        assert result.value == "processed_test_message"

    def test_handlers_handle_failure(self) -> None:
        """Test handler execution with failure."""
        config = FlextCore.Models.Cqrs.Handler(
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
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_handler_5",
            handler_name="Test Handler 5",
            handler_type="command",
            handler_mode="command",
        )
        handler = ConcreteTestHandler(config=config)

        assert handler._config_model.handler_id == "test_handler_5"
        assert handler._config_model.handler_name == "Test Handler 5"
        assert handler._config_model.handler_type == "command"
        assert handler._config_model.handler_mode == "command"

    def test_handlers_execution_context(self) -> None:
        """Test handler execution context creation."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_handler_6",
            handler_name="Test Handler 6",
        )
        handler = ConcreteTestHandler(config=config)

        assert handler._execution_context is not None
        assert hasattr(handler._execution_context, "handler_name")
        assert hasattr(handler._execution_context, "handler_mode")

    def test_handlers_message_types(self) -> None:
        """Test accepted message types computation."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_handler_7",
            handler_name="Test Handler 7",
        )
        handler = ConcreteTestHandler(config=config)

        assert handler._accepted_message_types is not None
        assert isinstance(handler._accepted_message_types, (list, tuple, set))

    def test_handlers_revalidation_setting(self) -> None:
        """Test revalidation setting extraction."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_handler_8",
            handler_name="Test Handler 8",
        )
        handler = ConcreteTestHandler(config=config)

        assert isinstance(handler._revalidate_pydantic_messages, bool)

    def test_handlers_type_warning_tracking(self) -> None:
        """Test type warning emission tracking."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_handler_9",
            handler_name="Test Handler 9",
        )
        handler = ConcreteTestHandler(config=config)

        assert isinstance(handler._type_warning_emitted, bool)
        assert handler._type_warning_emitted is False

    def test_handlers_different_types(self) -> None:
        """Test handlers with different message and result types."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_handler_10",
            handler_name="Test Handler 10",
        )

        # Test with different types
        class IntHandler(FlextCore.Handlers[int, str]):
            def handle(self, message: int) -> FlextCore.Result[str]:
                return FlextCore.Result[str].ok(f"processed_{message}")

        handler = IntHandler(config=config)
        result = handler.handle(42)
        assert result.is_success
        assert result.value == "processed_42"

    def test_handlers_command_type(self) -> None:
        """Test handlers with command type."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_command_handler",
            handler_name="Test Command Handler",
            handler_type="command",
            handler_mode="command",
        )
        handler = ConcreteTestHandler(config=config)

        assert handler._config_model.handler_type == "command"
        assert handler._config_model.handler_mode == "command"

    def test_handlers_query_type(self) -> None:
        """Test handlers with query type."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_query_handler",
            handler_name="Test Query Handler",
            handler_type="query",
            handler_mode="query",
        )
        handler = ConcreteTestHandler(config=config)

        assert handler._config_model.handler_type == "query"
        assert handler._config_model.handler_mode == "query"

    def test_handlers_event_type(self) -> None:
        """Test handlers with event type."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_event_handler",
            handler_name="Test Event Handler",
            handler_type="event",
            handler_mode="event",
        )
        handler = ConcreteTestHandler(config=config)

        assert handler._config_model.handler_type == "event"
        assert handler._config_model.handler_mode == "event"

    def test_handlers_saga_type(self) -> None:
        """Test handlers with saga type."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_saga_handler",
            handler_name="Test Saga Handler",
            handler_type="saga",
            handler_mode="saga",
        )
        handler = ConcreteTestHandler(config=config)

        assert handler._config_model.handler_type == "saga"
        assert handler._config_model.handler_mode == "saga"

    def test_handlers_with_metadata(self) -> None:
        """Test handlers with metadata configuration."""
        config = FlextCore.Models.Cqrs.Handler(
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
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_handler_with_timeout",
            handler_name="Test Handler With Timeout",
            command_timeout=60,
        )
        handler = ConcreteTestHandler(config=config)

        assert handler._config_model.command_timeout == 60

    def test_handlers_with_retry_config(self) -> None:
        """Test handlers with retry configuration."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_handler_with_retry",
            handler_name="Test Handler With Retry",
            max_command_retries=3,
        )
        handler = ConcreteTestHandler(config=config)

        assert handler._config_model.max_command_retries == 3

    def test_handlers_abstract_method_implementation(self) -> None:
        """Test that concrete handlers must implement handle method."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_abstract_handler",
            handler_name="Test Abstract Handler",
        )

        # This should work - concrete implementation
        handler = ConcreteTestHandler(config=config)
        assert hasattr(handler, "handle")
        assert callable(handler.handle)

    def test_handlers_inheritance_chain(self) -> None:
        """Test that handlers inherit from FlextCore.Mixins."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_inheritance_handler",
            handler_name="Test Inheritance Handler",
        )
        handler = ConcreteTestHandler(config=config)

        # Should inherit from FlextMixins
        assert isinstance(handler, FlextMixins)

    def test_handlers_config_model_type(self) -> None:
        """Test that config model is properly typed."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_config_type_handler",
            handler_name="Test Config Type Handler",
        )
        handler = ConcreteTestHandler(config=config)

        assert isinstance(handler._config_model, FlextCore.Models.Cqrs.Handler)

    def test_handlers_execution_context_type(self) -> None:
        """Test that execution context is properly typed."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_context_type_handler",
            handler_name="Test Context Type Handler",
        )
        handler = ConcreteTestHandler(config=config)

        assert isinstance(
            handler._execution_context,
            FlextCore.Models.HandlerExecutionContext,
        )

    def test_handlers_run_pipeline_with_dict_message_command_id(self) -> None:
        """Test _run_pipeline with dict[str, object] message having command_id."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_pipeline_dict_command_id",
            handler_name="Test Pipeline Dict Command ID",
            handler_type="command",
            handler_mode="command",
        )

        # Create handler that accepts dict[str, object] messages
        class DictHandler(FlextCore.Handlers[FlextCore.Types.Dict, str]):
            def __init__(self, config: FlextCore.Models.Cqrs.Handler) -> None:
                super().__init__(config=config)

            def handle(self, message: FlextCore.Types.Dict) -> FlextCore.Result[str]:
                return FlextCore.Result[str].ok(f"processed_{message}")

        handler = DictHandler(config=config)
        dict_message: FlextCore.Types.Dict = {
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
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_pipeline_dict_message_id",
            handler_name="Test Pipeline Dict Message ID",
            handler_type="command",
            handler_mode="command",
        )

        # Create handler that accepts dict[str, object] messages
        class DictHandler(FlextCore.Handlers[FlextCore.Types.Dict, str]):
            def __init__(self, config: FlextCore.Models.Cqrs.Handler) -> None:
                super().__init__(config=config)

            def handle(self, message: FlextCore.Types.Dict) -> FlextCore.Result[str]:
                return FlextCore.Result[str].ok(f"processed_{message}")

        handler = DictHandler(config=config)
        dict_message: FlextCore.Types.Dict = {
            "message_id": "msg_456",
            "data": "test_data",
        }
        result = handler._run_pipeline(dict_message, operation="command")

        assert result.is_success
        assert (
            result.value == "processed_{'message_id': 'msg_456', 'data': 'test_data'}"
        )

    def test_handlers_run_pipeline_with_object_message_command_id(self) -> None:
        """Test _run_pipeline with object message having command_id attribute."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_pipeline_object_command_id",
            handler_name="Test Pipeline Object Command ID",
            handler_type="command",
            handler_mode="command",
        )

        # Create handler that accepts object messages
        class ObjectHandler(FlextCore.Handlers[object, str]):
            def __init__(self, config: FlextCore.Models.Cqrs.Handler) -> None:
                super().__init__(config=config)

            def handle(self, message: object) -> FlextCore.Result[str]:
                return FlextCore.Result[str].ok(f"processed_{message}")

        handler = ObjectHandler(config=config)
        message_obj = SimpleNamespace(command_id="cmd_789", data="test_data")
        result = handler._run_pipeline(message_obj, operation="command")

        assert result.is_success
        assert "processed_" in result.value

    def test_handlers_run_pipeline_with_object_message_message_id(self) -> None:
        """Test _run_pipeline with object message having message_id attribute."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_pipeline_object_message_id",
            handler_name="Test Pipeline Object Message ID",
            handler_type="command",
            handler_mode="command",
        )

        # Create handler that accepts object messages
        class ObjectHandler(FlextCore.Handlers[object, str]):
            def __init__(self, config: FlextCore.Models.Cqrs.Handler) -> None:
                super().__init__(config=config)

            def handle(self, message: object) -> FlextCore.Result[str]:
                return FlextCore.Result[str].ok(f"processed_{message}")

        handler = ObjectHandler(config=config)
        message_obj = SimpleNamespace(message_id="msg_789", data="test_data")
        result = handler._run_pipeline(message_obj, operation="command")

        assert result.is_success
        assert "processed_" in result.value

    def test_handlers_run_pipeline_mode_validation_error(self) -> None:
        """Test _run_pipeline with mismatched operation and handler mode."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_pipeline_mode_error",
            handler_name="Test Pipeline Mode Error",
            handler_type="command",
            handler_mode="command",
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
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_pipeline_cannot_handle",
            handler_name="Test Pipeline Cannot Handle",
            handler_type="command",
            handler_mode="command",
        )

        # Create a handler that returns False for can_handle
        class RestrictiveHandler(FlextCore.Handlers[str, str]):
            def __init__(self, config: FlextCore.Models.Cqrs.Handler) -> None:
                super().__init__(config=config)

            def can_handle(self, message_type: object) -> bool:
                # Always return False to simulate cannot handle
                # Parameter is intentionally unused in this test
                _ = message_type  # Mark as intentionally unused
                return False

            def handle(self, message: str) -> FlextCore.Result[str]:
                return FlextCore.Result[str].ok(f"processed_{message}")

        handler = RestrictiveHandler(config=config)
        result = handler._run_pipeline("test_message", operation="command")

        assert result.is_failure
        assert result.error is not None
        assert "Handler cannot handle message type str" in result.error

    def test_handlers_run_pipeline_validation_failure(self) -> None:
        """Test _run_pipeline when message validation fails."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_pipeline_validation_failure",
            handler_name="Test Pipeline Validation Failure",
            handler_type="command",
            handler_mode="command",
        )

        # Create handler that fails validation
        class ValidationFailingHandler(FlextCore.Handlers[str, str]):
            def __init__(self, config: FlextCore.Models.Cqrs.Handler) -> None:
                super().__init__(config=config)

            def validate_command(self, command: object) -> FlextCore.Result[None]:
                # Parameter is intentionally unused in this test
                _ = command  # Mark as intentionally unused
                return FlextCore.Result[None].fail("Validation failed for test")

            def handle(self, message: str) -> FlextCore.Result[str]:
                return FlextCore.Result[str].ok(f"processed_{message}")

        handler = ValidationFailingHandler(config=config)
        result = handler._run_pipeline("test_message", operation="command")

        assert result.is_failure
        assert result.error is not None
        assert "Message validation failed: Validation failed for test" in result.error

    def test_handlers_run_pipeline_handler_exception(self) -> None:
        """Test _run_pipeline when handler.handle() raises exception."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_pipeline_exception",
            handler_name="Test Pipeline Exception",
            handler_type="command",
            handler_mode="command",
        )

        # Create handler that raises exception
        class ExceptionHandler(FlextCore.Handlers[str, str]):
            def __init__(self, config: FlextCore.Models.Cqrs.Handler) -> None:
                super().__init__(config=config)

            def handle(self, message: str) -> FlextCore.Result[str]:
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
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_pipeline_query",
            handler_name="Test Pipeline Query",
            handler_type="query",
            handler_mode="query",
        )

        # Create a query handler
        class QueryHandler(FlextCore.Handlers[str, str]):
            def __init__(self, config: FlextCore.Models.Cqrs.Handler) -> None:
                super().__init__(config=config)

            def handle(self, message: str) -> FlextCore.Result[str]:
                return FlextCore.Result[str].ok(f"queried_{message}")

        handler = QueryHandler(config=config)
        result = handler._run_pipeline("test_query", operation="query")

        assert result.is_success
        assert result.value == "queried_test_query"

    def test_handlers_run_pipeline_query_validation_failure(self) -> None:
        """Test _run_pipeline with query operation validation failure."""
        config = FlextCore.Models.Cqrs.Handler(
            handler_id="test_pipeline_query_validation",
            handler_name="Test Pipeline Query Validation",
            handler_type="query",
            handler_mode="query",
        )

        # Create query handler that fails validation
        class QueryValidationHandler(FlextCore.Handlers[str, str]):
            def __init__(self, config: FlextCore.Models.Cqrs.Handler) -> None:
                super().__init__(config=config)

            def validate_query(self, query: object) -> FlextCore.Result[None]:
                # Parameter is intentionally unused in this test
                _ = query  # Mark as intentionally unused
                return FlextCore.Result[None].fail("Query validation failed")

            def handle(self, message: str) -> FlextCore.Result[str]:
                return FlextCore.Result[str].ok(f"queried_{message}")

        handler = QueryValidationHandler(config=config)
        result = handler._run_pipeline("test_query", operation="query")

        assert result.is_failure
        assert result.error is not None
        assert "Message validation failed: Query validation failed" in result.error

    def test_handlers_from_callable_basic(self) -> None:
        """Test from_callable with basic function."""

        def simple_handler(message: str) -> str:
            return f"handled_{message}"

        handler = FlextCore.Handlers.from_callable(
            cast("Callable[[object], object]", simple_handler),
            handler_name="simple_handler",
            handler_type="command",
        )

        assert handler is not None
        assert handler.handler_name == "simple_handler"
        assert handler.mode == "command"

        result = handler.handle("test")
        assert result.is_success
        assert result.value == "handled_test"

    def test_handlers_from_callable_with_flext_result(self) -> None:
        """Test from_callable with function returning FlextCore.Result."""

        def result_handler(message: str) -> FlextCore.Result[str]:
            return FlextCore.Result[str].ok(f"result_{message}")

        handler = FlextCore.Handlers.from_callable(
            cast("Callable[[object], object]", result_handler),
            handler_name="result_handler",
            handler_type="query",
        )

        assert handler.handler_name == "result_handler"
        assert handler.mode == "query"

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

        handler = FlextCore.Handlers.from_callable(
            cast("Callable[[object], object]", failing_handler),
            handler_name="failing_handler",
            handler_type="command",
        )

        result = handler.handle("test")
        assert result.is_failure
        assert result.error is not None
        assert "Handler failed" in result.error

    def test_handlers_from_callable_default_name(self) -> None:
        """Test from_callable with default handler name from function name."""

        def my_custom_function(message: str) -> str:
            return f"custom_{message}"

        handler = FlextCore.Handlers.from_callable(
            cast("Callable[[object], object]", my_custom_function),
            handler_type="command",
        )

        assert handler.handler_name == "my_custom_function"
        assert handler.mode == "command"

    def test_handlers_from_callable_with_mode_parameter(self) -> None:
        """Test from_callable with mode parameter (compatibility)."""

        def mode_handler(message: str) -> str:
            return f"mode_{message}"

        handler = FlextCore.Handlers.from_callable(
            cast("Callable[[object], object]", mode_handler),
            handler_name="mode_handler",
            mode="query",  # Using mode parameter instead of handler_type
        )

        assert handler.handler_name == "mode_handler"
        assert handler.mode == "query"

    def test_handlers_from_callable_invalid_mode(self) -> None:
        """Test from_callable with invalid mode."""

        def invalid_handler(message: str) -> str:
            return f"invalid_{message}"

        with pytest.raises(FlextCore.Exceptions.ValidationError) as exc_info:
            FlextCore.Handlers.from_callable(
                cast("Callable[[object], object]", invalid_handler),
                handler_name="invalid_handler",
                mode="invalid_mode",
            )

        assert "Invalid handler mode: invalid_mode" in str(exc_info.value)

    def test_handlers_from_callable_with_dict_config(self) -> None:
        """Test from_callable with dict[str, object] handler_config."""

        def dict_config_handler(message: str) -> str:
            return f"dict_config_{message}"

        handler_config: FlextCore.Types.Dict = {
            "handler_id": "custom_id",
            "handler_name": "Custom Name",
            "handler_type": "command",
            "handler_mode": "command",
            "metadata": {"test": "value"},
        }

        handler = FlextCore.Handlers.from_callable(
            cast("Callable[[object], object]", dict_config_handler),
            handler_config=handler_config,
        )

        assert handler.handler_name == "Custom Name"
        assert handler._config_model.handler_id == "custom_id"
        assert handler._config_model.metadata == {"test": "value"}

    def test_handlers_from_callable_with_invalid_dict_config(self) -> None:
        """Test from_callable with invalid dict[str, object] handler_config."""

        def invalid_config_handler(message: object) -> object:
            if isinstance(message, str):
                return f"invalid_config_{message}"
            return f"invalid_config_{message!s}"

        invalid_config: FlextCore.Types.Dict = {
            "handler_type": "invalid_type",  # Invalid value
        }

        with pytest.raises(FlextCore.Exceptions.ValidationError) as exc_info:
            FlextCore.Handlers.from_callable(
                invalid_config_handler,
                handler_config=invalid_config,
            )

        assert "Invalid handler config:" in str(exc_info.value)

    def test_handlers_from_callable_with_pydantic_config(self) -> None:
        """Test from_callable with FlextCore.Models.Cqrs.Handler object."""

        def pydantic_config_handler(message: object) -> object:
            if isinstance(message, str):
                return f"pydantic_config_{message}"
            return f"pydantic_config_{message!s}"

        config = FlextCore.Models.Cqrs.Handler(
            handler_id="pydantic_id",
            handler_name="Pydantic Handler",
            handler_type="query",
            handler_mode="query",
        )

        handler = FlextCore.Handlers.from_callable(
            pydantic_config_handler,
            handler_config=config,
        )

        assert handler.handler_name == "Pydantic Handler"
        assert handler._config_model.handler_id == "pydantic_id"
        assert handler.mode == "query"

    def test_handlers_from_callable_anonymous_function(self) -> None:
        """Test from_callable with lambda (anonymous function)."""

        def process_message(message: str) -> str:
            return f"lambda_{message!s}"

        handler = FlextCore.Handlers.from_callable(
            cast("Callable[[str], str]", process_message),
            handler_name="lambda_handler",
            handler_type="command",
        )

        assert handler.handler_name == "lambda_handler"
        result = handler.handle("test")
        assert result.is_success
        assert result.value == "lambda_test"

    def test_handlers_from_callable_function_without_name_attribute(self) -> None:
        """Test from_callable with function object without __name__ attribute."""

        # Create a callable object without __name__
        class CallableObject:
            def __call__(self, message: object) -> object:
                if isinstance(message, str):
                    return f"callable_object_{message}"
                return f"callable_object_{message!s}"

        callable_obj = CallableObject()

        handler = FlextCore.Handlers.from_callable(callable_obj, handler_type="command")

        # Should default to "unknown_handler" when no __name__ attribute
        assert handler.handler_name == "unknown_handler"
        result = handler.handle("test")
        assert result.is_success
        assert result.value == "callable_object_test"
