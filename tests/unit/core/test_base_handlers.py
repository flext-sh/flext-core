"""Comprehensive tests for base_handlers module - 0% to 100% coverage target."""

from __future__ import annotations

from abc import ABC

import pytest

from flext_core.base_handlers import (
    FlextAuthorizingHandler,
    FlextBaseHandler,
    FlextCommandHandler,
    FlextEventHandler,
    FlextHandlerChain,
    FlextHandlerRegistry,
    FlextHandlers,
    FlextMetricsHandler,
    FlextQueryHandler,
    FlextValidatingHandler,
)
from flext_core.handlers_base import FlextAbstractHandler
from flext_core.result import FlextResult

# Test markers
pytestmark = [pytest.mark.unit, pytest.mark.core]


class ConcreteBaseHandler(FlextAbstractHandler[object, object]):
    """Concrete implementation for testing abstract handler interface."""

    @property
    def handler_name(self) -> str:
        """Get handler name."""
        return "ConcreteBaseHandler"

    def can_handle(self, request: object) -> bool:
        """Check if handler can handle request."""
        return True

    def handle(self, request: object) -> FlextResult[object]:
        """Handle request."""
        if request is None:
            return FlextResult.fail("Request cannot be None")
        return FlextResult.ok(request)

    def validate_request(self, request: object) -> FlextResult[None]:
        """Validate request."""
        if request is None:
            return FlextResult.fail("Request validation failed: None request")
        return FlextResult.ok(None)


class TestFlextBaseHandler:
    """Test FlextBaseHandler coverage - lines 25-40."""

    def test_base_handler_initialization(self) -> None:
        """Test FlextBaseHandler initialization."""
        handler = FlextBaseHandler()
        assert isinstance(handler, FlextBaseHandler)

    def test_handler_name_property(self) -> None:
        """Test handler_name property (lines 29-31)."""
        handler = FlextBaseHandler()
        assert handler.handler_name == "FlextBaseHandler"

    def test_can_handle_method(self) -> None:
        """Test can_handle method (lines 33-35)."""
        handler = FlextBaseHandler()
        assert handler.can_handle("test_request") is True
        assert handler.can_handle(42) is True
        assert handler.can_handle(None) is True

    def test_handle_method(self) -> None:
        """Test handle method (lines 37-39)."""
        handler = FlextBaseHandler()
        test_request = {"data": "test"}
        result = handler.handle(test_request)

        assert isinstance(result, FlextResult)
        assert result.success
        assert result.data == test_request

    def test_validate_request_method(self) -> None:
        """Test validate_request method (lines 41-43)."""
        handler = FlextBaseHandler()
        result = handler.validate_request("any_request")

        assert isinstance(result, FlextResult)
        assert result.success

    def test_process_request_method(self) -> None:
        """Test process_request method (lines 37-39)."""
        handler = FlextBaseHandler()
        test_request = {"data": "test"}
        result = handler.process_request(test_request)

        assert isinstance(result, FlextResult)
        assert result.success
        assert result.data == test_request


class ConcreteCommandHandler(FlextCommandHandler[str, int]):
    """Concrete implementation for testing FlextCommandHandler."""

    def handle_command(self, command: str) -> FlextResult[int]:
        """Handle string command and return length."""
        if not command:
            return FlextResult.fail("Empty command")
        return FlextResult.ok(len(command))


class TestFlextCommandHandler:
    """Test FlextCommandHandler coverage - lines 42-64."""

    def test_command_handler_is_abstract(self) -> None:
        """Test FlextCommandHandler is abstract (lines 42-46)."""
        assert issubclass(FlextCommandHandler, ABC)

        # Cannot instantiate directly
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            FlextCommandHandler()  # type: ignore[type-arg]

    def test_concrete_command_handler_handle_command(self) -> None:
        """Test concrete command handler handle_command (lines 48-50)."""
        handler = ConcreteCommandHandler()
        result = handler.handle_command("test")

        assert isinstance(result, FlextResult)
        assert result.success
        assert result.data == 4  # Length of "test"

    def test_command_handler_can_handle(self) -> None:
        """Test can_handle method (lines 52-54)."""
        handler = ConcreteCommandHandler()
        assert handler.can_handle("any_command") is True

    def test_command_handler_process_request(self) -> None:
        """Test process_request delegates to handle_command (lines 56-58)."""
        handler = ConcreteCommandHandler()
        result = handler.process_request("hello")

        assert isinstance(result, FlextResult)
        assert result.success
        assert result.data == 5  # Length of "hello"

    def test_command_handler_name_property(self) -> None:
        """Test handler_name property (lines 60-63)."""
        handler = ConcreteCommandHandler()
        assert handler.handler_name == "ConcreteCommandHandler"

    def test_command_handler_failure_case(self) -> None:
        """Test command handler with failure case."""
        handler = ConcreteCommandHandler()
        result = handler.handle_command("")

        assert isinstance(result, FlextResult)
        assert result.is_failure
        assert "Empty command" in (result.error or "")


class ConcreteQueryHandler(FlextQueryHandler[dict[str, str], str]):
    """Concrete implementation for testing FlextQueryHandler."""

    def handle_query(self, query: dict[str, str]) -> FlextResult[str]:
        """Handle dict query and return formatted string."""
        if not query:
            return FlextResult.fail("Empty query")
        name = query.get("name", "Unknown")
        return FlextResult.ok(f"Hello, {name}!")


class TestFlextQueryHandler:
    """Test FlextQueryHandler coverage - lines 66-84."""

    def test_query_handler_is_abstract(self) -> None:
        """Test FlextQueryHandler is abstract (lines 66-68)."""
        assert issubclass(FlextQueryHandler, ABC)

        # Cannot instantiate directly
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            FlextQueryHandler()  # type: ignore[type-arg]

    def test_concrete_query_handler_handle_query(self) -> None:
        """Test concrete query handler handle_query (lines 69-71)."""
        handler = ConcreteQueryHandler()
        query = {"name": "Alice"}
        result = handler.handle_query(query)

        assert isinstance(result, FlextResult)
        assert result.success
        assert result.data == "Hello, Alice!"

    def test_query_handler_can_handle(self) -> None:
        """Test can_handle method (lines 73-75)."""
        handler = ConcreteQueryHandler()
        assert handler.can_handle({"any": "query"}) is True

    def test_query_handler_process_request(self) -> None:
        """Test process_request delegates to handle_query (lines 77-79)."""
        handler = ConcreteQueryHandler()
        query = {"name": "Bob"}
        result = handler.process_request(query)

        assert isinstance(result, FlextResult)
        assert result.success
        assert result.data == "Hello, Bob!"

    def test_query_handler_name_property(self) -> None:
        """Test handler_name property (lines 81-84)."""
        handler = ConcreteQueryHandler()
        assert handler.handler_name == "ConcreteQueryHandler"

    def test_query_handler_failure_case(self) -> None:
        """Test query handler with failure case."""
        handler = ConcreteQueryHandler()
        result = handler.handle_query({})

        assert isinstance(result, FlextResult)
        assert result.is_failure
        assert "Empty query" in (result.error or "")


class TestFlextCompatibilityHandlers:
    """Test compatibility handler classes - lines 88-106."""

    def test_validating_handler(self) -> None:
        """Test FlextValidatingHandler (lines 88-90)."""
        handler = FlextValidatingHandler()
        assert isinstance(handler, FlextBaseHandler)
        assert handler.handler_name == "FlextValidatingHandler"
        assert handler.can_handle("test") is True

        result = handler.process_request("test")
        assert result.success
        assert result.data == "test"

    def test_authorizing_handler(self) -> None:
        """Test FlextAuthorizingHandler (lines 92-94)."""
        handler = FlextAuthorizingHandler()
        assert isinstance(handler, FlextBaseHandler)
        assert handler.handler_name == "FlextAuthorizingHandler"

    def test_event_handler(self) -> None:
        """Test FlextEventHandler (lines 96-98)."""
        handler = FlextEventHandler()
        assert isinstance(handler, FlextBaseHandler)
        assert handler.handler_name == "FlextEventHandler"

    def test_metrics_handler(self) -> None:
        """Test FlextMetricsHandler (lines 100-102)."""
        handler = FlextMetricsHandler()
        assert isinstance(handler, FlextBaseHandler)
        assert handler.handler_name == "FlextMetricsHandler"

    def test_handler_chain(self) -> None:
        """Test FlextHandlerChain (lines 104-106)."""
        handler = FlextHandlerChain()
        assert isinstance(handler, FlextBaseHandler)
        assert handler.handler_name == "FlextHandlerChain"


class TestFlextHandlerRegistry:
    """Test FlextHandlerRegistry coverage - lines 108-122."""

    def test_registry_initialization(self) -> None:
        """Test FlextHandlerRegistry initialization (lines 111-113)."""
        registry = FlextHandlerRegistry()
        assert isinstance(registry._handlers, list)
        assert len(registry._handlers) == 0

    def test_register_handler(self) -> None:
        """Test register method (lines 115-117)."""
        registry = FlextHandlerRegistry()
        handler = FlextBaseHandler()

        registry.register(handler)
        assert len(registry._handlers) == 1
        assert registry._handlers[0] is handler

    def test_register_multiple_handlers(self) -> None:
        """Test registering multiple handlers."""
        registry = FlextHandlerRegistry()
        handler1 = FlextBaseHandler()
        handler2 = FlextValidatingHandler()

        registry.register(handler1)
        registry.register(handler2)

        assert len(registry._handlers) == 2
        assert registry._handlers[0] is handler1
        assert registry._handlers[1] is handler2

    def test_get_handlers(self) -> None:
        """Test get_handlers method (lines 119-121)."""
        registry = FlextHandlerRegistry()
        handler = FlextAuthorizingHandler()

        registry.register(handler)
        handlers = registry.get_handlers()

        assert len(handlers) == 1
        assert handlers[0] is handler
        assert handlers is not registry._handlers  # Should be a copy


class TestFlextLegacyAliases:
    """Test legacy aliases coverage - lines 124-126."""

    def test_flext_handlers_alias(self) -> None:
        """Test FlextHandlers alias (lines 124-125)."""
        assert FlextHandlers is FlextBaseHandler

        # Test instantiation through alias
        handler = FlextHandlers()
        assert isinstance(handler, FlextBaseHandler)
        assert handler.handler_name == "FlextBaseHandler"  # Alias points to same class


class TestFlextHandlersIntegration:
    """Integration tests for handler system."""

    def test_handler_registry_with_different_handlers(self) -> None:
        """Test registry with different handler types."""
        registry = FlextHandlerRegistry()

        # Register different types of handlers
        base_handler = FlextBaseHandler()
        validating_handler = FlextValidatingHandler()
        event_handler = FlextEventHandler()
        command_handler = ConcreteCommandHandler()
        query_handler = ConcreteQueryHandler()

        registry.register(base_handler)
        registry.register(validating_handler)
        registry.register(event_handler)
        registry.register(command_handler)
        registry.register(query_handler)

        handlers = registry.get_handlers()
        assert len(handlers) == 5

        # Test all handlers can process requests
        for handler in handlers:
            if isinstance(handler, ConcreteCommandHandler):
                result = handler.process_request("test")
                assert result.success
                assert result.data == 4
            elif isinstance(handler, ConcreteQueryHandler):
                result = handler.process_request({"name": "Test"})
                assert result.success
                assert result.data == "Hello, Test!"
            else:
                result = handler.process_request("any_request")
                assert result.success

    def test_handler_polymorphism(self) -> None:
        """Test handler polymorphism through base interface."""
        handlers: list[FlextBaseHandler] = [
            FlextBaseHandler(),
            FlextValidatingHandler(),
            FlextAuthorizingHandler(),
            FlextEventHandler(),
            FlextMetricsHandler(),
            FlextHandlerChain(),
        ]

        for handler in handlers:
            assert handler.can_handle("test")
            result = handler.process_request("test")
            assert result.success
            assert result.data == "test"
            assert isinstance(handler.handler_name, str)
            assert len(handler.handler_name) > 0
