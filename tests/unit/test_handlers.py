"""Comprehensive tests for FlextHandlers - Handler Management.

Tests the actual FlextHandlers API with real functionality testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextHandlers, FlextModels, FlextResult
from flext_core.context import FlextContext
from flext_core.mixins import FlextMixins


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
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler_1", handler_name="Test Handler 1"
        )
        handlers = ConcreteTestHandler(config=config)
        assert handlers is not None
        assert isinstance(handlers, FlextHandlers)

    def test_handlers_with_custom_config(self) -> None:
        """Test handlers initialization with custom configuration."""
        config = FlextModels.CqrsConfig.Handler(
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
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler_3", handler_name="Test Handler 3"
        )
        handler = ConcreteTestHandler(config=config)

        result = handler.handle("test_message")
        assert result.is_success
        assert result.value == "processed_test_message"

    def test_handlers_handle_failure(self) -> None:
        """Test handler execution with failure."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler_4", handler_name="Test Handler 4"
        )
        handler = FailingTestHandler(config=config)

        result = handler.handle("test_message")
        assert result.is_failure
        assert "Handler failed for: test_message" in result.error

    def test_handlers_config_access(self) -> None:
        """Test access to handler configuration."""
        config = FlextModels.CqrsConfig.Handler(
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
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler_6", handler_name="Test Handler 6"
        )
        handler = ConcreteTestHandler(config=config)

        assert handler._execution_context is not None
        assert hasattr(handler._execution_context, "handler_name")
        assert hasattr(handler._execution_context, "handler_mode")

    def test_handlers_message_types(self) -> None:
        """Test accepted message types computation."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler_7", handler_name="Test Handler 7"
        )
        handler = ConcreteTestHandler(config=config)

        assert handler._accepted_message_types is not None
        assert isinstance(handler._accepted_message_types, (list, tuple, set))

    def test_handlers_revalidation_setting(self) -> None:
        """Test revalidation setting extraction."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler_8", handler_name="Test Handler 8"
        )
        handler = ConcreteTestHandler(config=config)

        assert isinstance(handler._revalidate_pydantic_messages, bool)

    def test_handlers_type_warning_tracking(self) -> None:
        """Test type warning emission tracking."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler_9", handler_name="Test Handler 9"
        )
        handler = ConcreteTestHandler(config=config)

        assert isinstance(handler._type_warning_emitted, bool)
        assert handler._type_warning_emitted is False

    def test_handlers_different_types(self) -> None:
        """Test handlers with different message and result types."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler_10", handler_name="Test Handler 10"
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
        config = FlextModels.CqrsConfig.Handler(
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
        config = FlextModels.CqrsConfig.Handler(
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
        config = FlextModels.CqrsConfig.Handler(
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
        config = FlextModels.CqrsConfig.Handler(
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
        config = FlextModels.CqrsConfig.Handler(
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
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler_with_timeout",
            handler_name="Test Handler With Timeout",
            command_timeout=60,
        )
        handler = ConcreteTestHandler(config=config)

        assert handler._config_model.command_timeout == 60

    def test_handlers_with_retry_config(self) -> None:
        """Test handlers with retry configuration."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_handler_with_retry",
            handler_name="Test Handler With Retry",
            max_command_retries=3,
        )
        handler = ConcreteTestHandler(config=config)

        assert handler._config_model.max_command_retries == 3

    def test_handlers_abstract_method_implementation(self) -> None:
        """Test that concrete handlers must implement handle method."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_abstract_handler", handler_name="Test Abstract Handler"
        )

        # This should work - concrete implementation
        handler = ConcreteTestHandler(config=config)
        assert hasattr(handler, "handle")
        assert callable(handler.handle)

    def test_handlers_inheritance_chain(self) -> None:
        """Test that handlers inherit from FlextMixins."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_inheritance_handler",
            handler_name="Test Inheritance Handler",
        )
        handler = ConcreteTestHandler(config=config)

        # Should inherit from FlextMixins
        assert isinstance(handler, FlextMixins)

    def test_handlers_config_model_type(self) -> None:
        """Test that config model is properly typed."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_config_type_handler",
            handler_name="Test Config Type Handler",
        )
        handler = ConcreteTestHandler(config=config)

        assert isinstance(handler._config_model, FlextModels.CqrsConfig.Handler)

    def test_handlers_execution_context_type(self) -> None:
        """Test that execution context is properly typed."""
        config = FlextModels.CqrsConfig.Handler(
            handler_id="test_context_type_handler",
            handler_name="Test Context Type Handler",
        )
        handler = ConcreteTestHandler(config=config)

        assert isinstance(
            handler._execution_context, FlextContext.HandlerExecutionContext
        )
