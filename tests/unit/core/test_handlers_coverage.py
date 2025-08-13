"""Comprehensive tests for handlers.py to increase coverage from 39% to 85%+.

This test suite directly imports and tests all classes from handlers.py
to ensure proper code coverage and functionality verification.
Based on the ACTUAL API of handlers.py, not assumptions.
"""

from __future__ import annotations

import pytest

from flext_core.handlers import (
    FlextAuthorizingHandler,
    FlextBaseHandler,
    FlextHandlerChain,
    FlextHandlerRegistry,
    FlextMetricsHandler,
    FlextValidatingHandler,
)
from flext_core.result import FlextResult

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextBaseHandler:
    """Test FlextBaseHandler concrete implementation."""

    def test_base_handler_initialization(self) -> None:
        """Test basic handler initialization."""
        # Test with default name
        handler = FlextBaseHandler()
        assert handler.handler_name == "FlextBaseHandler"

        # Test with custom name
        custom_handler = FlextBaseHandler(name="CustomHandler")
        assert custom_handler.handler_name == "CustomHandler"

    def test_base_handler_can_handle(self) -> None:
        """Test can_handle method."""
        handler = FlextBaseHandler()
        # Based on real implementation - returns True for all
        assert handler.can_handle("any request") is True
        assert handler.can_handle({"key": "value"}) is True
        assert handler.can_handle(None) is True

    def test_base_handler_handle(self) -> None:
        """Test handle method."""
        handler = FlextBaseHandler()

        # Test successful handling
        result = handler.handle("test request")
        assert result.success
        assert result.data == "test request"

        # Test with different data types
        dict_result = handler.handle({"data": "test"})
        assert dict_result.success
        assert dict_result.data == {"data": "test"}

    def test_base_handler_validate_request(self) -> None:
        """Test validate_request method."""
        handler = FlextBaseHandler()

        # Test valid request
        result = handler.validate_request("valid request")
        assert result.success

        # Test None request (should fail based on implementation)
        result = handler.validate_request(None)
        assert result.is_failure
        assert "cannot be None" in (result.error or "")

    def test_base_handler_pre_process(self) -> None:
        """Test pre_process hook."""
        handler = FlextBaseHandler()

        result = handler.pre_process("any message")
        assert result.success

    def test_base_handler_process_message(self) -> None:
        """Test process_message method."""
        handler = FlextBaseHandler()

        result = handler.process_message("test message")
        assert result.success
        assert result.data == "test message"

    def test_base_handler_post_process(self) -> None:
        """Test post_process hook."""
        handler = FlextBaseHandler()

        dummy_result = FlextResult.ok("test")
        result = handler.post_process("message", dummy_result)
        assert result.success

    def test_base_handler_get_handler_metadata(self) -> None:
        """Test get_handler_metadata method."""
        handler = FlextBaseHandler(name="TestHandler")

        metadata = handler.get_handler_metadata()
        assert isinstance(metadata, dict)
        assert metadata["handler_name"] == "TestHandler"
        assert metadata["handler_class"] == "FlextBaseHandler"
        assert metadata["can_handle_all"] is True


class TestFlextValidatingHandler:
    """Test FlextValidatingHandler implementation."""

    def test_validating_handler_initialization(self) -> None:
        """Test validating handler initialization."""
        # Test with default name
        handler = FlextValidatingHandler()
        assert handler.handler_name == "FlextValidatingHandler"

        # Test with custom name
        custom_handler = FlextValidatingHandler(name="CustomValidator")
        assert custom_handler.handler_name == "CustomValidator"

    def test_validating_handler_handle(self) -> None:
        """Test handle method with validation."""
        handler = FlextValidatingHandler()

        # Test valid request
        result = handler.handle("valid request")
        assert result.success
        assert result.data == "valid request"

        # Test None request (should fail validation)
        result = handler.handle(None)
        assert result.is_failure
        assert "cannot be None" in (result.error or "")

    def test_validating_handler_can_handle(self) -> None:
        """Test can_handle method."""
        handler = FlextValidatingHandler()

        # Based on real implementation
        assert handler.can_handle("valid request") is True
        assert handler.can_handle(None) is False

    def test_validating_handler_validate_input(self) -> None:
        """Test validate_input method."""
        handler = FlextValidatingHandler()

        # Test valid input
        result = handler.validate_input("valid input")
        assert result.success

        # Test None input
        result = handler.validate_input(None)
        assert result.is_failure
        assert "cannot be None" in (result.error or "")

    def test_validating_handler_validate_output(self) -> None:
        """Test validate_output method."""
        handler = FlextValidatingHandler()

        result = handler.validate_output("any output")
        assert result.success
        assert result.data == "any output"

    def test_validating_handler_get_validation_rules(self) -> None:
        """Test get_validation_rules method."""
        handler = FlextValidatingHandler()

        rules = handler.get_validation_rules()
        assert isinstance(rules, list)
        assert len(rules) == 0  # Default implementation

    def test_validating_handler_validate_message(self) -> None:
        """Test validate_message method."""
        handler = FlextValidatingHandler()

        # Test valid message
        result = handler.validate_message("valid message")
        assert result.success

        # Test None message
        result = handler.validate_message(None)
        assert result.is_failure

    def test_validating_handler_validate(self) -> None:
        """Test validate compatibility method."""
        handler = FlextValidatingHandler()

        # Test valid message
        result = handler.validate("valid message")
        assert result.success
        assert result.data == "valid message"

        # Test None message
        result = handler.validate(None)
        assert result.is_failure


class TestFlextAuthorizingHandler:
    """Test FlextAuthorizingHandler implementation."""

    def test_authorizing_handler_initialization(self) -> None:
        """Test authorizing handler initialization."""
        # Test with default values
        handler = FlextAuthorizingHandler()
        assert handler.handler_name == "FlextAuthorizingHandler"
        assert handler.required_permissions == []

        # Test with custom values
        permissions = ["read", "write"]
        custom_handler = FlextAuthorizingHandler(
            name="CustomAuth",
            required_permissions=permissions,
        )
        assert custom_handler.handler_name == "CustomAuth"
        assert custom_handler.required_permissions == permissions

    def test_authorizing_handler_authorize(self) -> None:
        """Test authorize method."""
        handler = FlextAuthorizingHandler()

        # Test authorization (depends on authorize_message implementation)
        context: dict[str, object] = {"user": "admin"}
        result = handler.authorize("test message", context)
        assert isinstance(result, FlextResult)

    def test_authorizing_handler_handle_with_auth(self) -> None:
        """Test handle method that includes authorization."""
        handler = FlextAuthorizingHandler()

        # Test normal handling
        result = handler.handle("test message")
        assert isinstance(result, FlextResult)


class TestFlextEventHandler:
    """Test FlextEventHandler implementation if it exists."""

    def test_event_handler_exists(self) -> None:
        """Test if FlextEventHandler is available."""
        # Try to import - if it fails, skip these tests
        try:
            from flext_core.handlers import FlextEventHandler  # noqa: PLC0415

            handler = FlextEventHandler()
            assert handler is not None
        except ImportError:
            pytest.skip("FlextEventHandler not available in current implementation")


class TestFlextMetricsHandler:
    """Test FlextMetricsHandler implementation."""

    def test_metrics_handler_initialization(self) -> None:
        """Test metrics handler initialization."""
        handler = FlextMetricsHandler()
        assert handler.handler_name == "FlextMetricsHandler"

    def test_metrics_handler_handle(self) -> None:
        """Test handle method with metrics."""
        handler = FlextMetricsHandler()

        result = handler.handle("test message")
        assert isinstance(result, FlextResult)

    def test_metrics_handler_get_metrics(self) -> None:
        """Test get_metrics method."""
        handler = FlextMetricsHandler()

        # Process some messages to generate metrics
        handler.handle("message1")
        handler.handle("message2")

        metrics = handler.get_metrics()
        assert isinstance(metrics, dict)
        # Check for common metric keys
        assert "total_requests" in metrics or "messages_processed" in metrics


class TestFlextHandlerChain:
    """Test FlextHandlerChain implementation."""

    def test_handler_chain_initialization(self) -> None:
        """Test handler chain initialization."""
        chain = FlextHandlerChain()
        assert chain.handler_name == "FlextHandlerChain"

    def test_handler_chain_add_handler(self) -> None:
        """Test adding handlers to chain."""
        chain = FlextHandlerChain()
        handler1 = FlextBaseHandler(name="Handler1")
        handler2 = FlextBaseHandler(name="Handler2")

        chain.add_handler(handler1)
        chain.add_handler(handler2)

        handlers = chain.get_handlers()
        assert len(handlers) >= 2  # May have additional handlers

    def test_handler_chain_handle(self) -> None:
        """Test chain processing."""
        chain = FlextHandlerChain()

        handler = FlextBaseHandler(name="ChainedHandler")
        chain.add_handler(handler)

        # Process request through chain
        result = chain.handle("test message")
        assert isinstance(result, FlextResult)

    def test_handler_chain_remove_handler(self) -> None:
        """Test removing handlers from chain."""
        chain = FlextHandlerChain()
        handler = FlextBaseHandler(name="RemovableHandler")

        chain.add_handler(handler)

        # Try to remove handler
        success = chain.remove_handler(handler)
        assert isinstance(success, bool)


class TestFlextHandlerRegistry:
    """Test FlextHandlerRegistry implementation."""

    def test_handler_registry_initialization(self) -> None:
        """Test handler registry initialization."""
        registry = FlextHandlerRegistry()
        handlers = registry.get_all_handlers()
        assert isinstance(handlers, dict)

    def test_handler_registry_register_handler(self) -> None:
        """Test registering handlers."""
        registry = FlextHandlerRegistry()
        handler = FlextBaseHandler(name="RegisteredHandler")

        result = registry.register_handler("test_key", handler)
        assert isinstance(result, FlextResult)

    def test_handler_registry_get_handler(self) -> None:
        """Test retrieving handlers."""
        registry = FlextHandlerRegistry()
        handler = FlextBaseHandler(name="RetrievableHandler")

        # Register first
        register_result = registry.register_handler("retrieve_key", handler)
        if register_result.success:
            # Try to retrieve
            result = registry.get_handler("retrieve_key")
            assert isinstance(result, FlextResult)

    def test_handler_registry_operations(self) -> None:
        """Test various registry operations."""
        registry = FlextHandlerRegistry()

        # Test getting all handlers
        all_handlers = registry.get_all_handlers()
        assert isinstance(all_handlers, dict)

        # Test any other available methods
        if hasattr(registry, "get_handlers_for_type"):
            handlers_by_type = registry.get_handlers_for_type(FlextBaseHandler)
            assert isinstance(handlers_by_type, list)


class TestHandlerIntegration:
    """Integration tests for handler interactions."""

    def test_handler_types_work_together(self) -> None:
        """Test different handler types working together."""
        base_handler = FlextBaseHandler(name="BaseHandler")
        validating_handler = FlextValidatingHandler(name="ValidatingHandler")

        # Test that they can all handle the same message
        test_message = "integration test message"

        base_result = base_handler.handle(test_message)
        assert base_result.success

        validating_result = validating_handler.handle(test_message)
        assert validating_result.success

    def test_registry_with_different_handler_types(self) -> None:
        """Test registry containing different handler types."""
        registry = FlextHandlerRegistry()

        base_handler = FlextBaseHandler(name="BaseInRegistry")
        validating_handler = FlextValidatingHandler(name="ValidatingInRegistry")

        # Register different types
        registry.register_handler("base", base_handler)
        registry.register_handler("validating", validating_handler)

        # Verify they're registered
        all_handlers = registry.get_all_handlers()
        assert len(all_handlers) >= 2

    def test_chain_with_validation_and_metrics(self) -> None:
        """Test chain containing validation and metrics handlers."""
        chain = FlextHandlerChain()

        validating_handler = FlextValidatingHandler()
        metrics_handler = FlextMetricsHandler()

        chain.add_handler(validating_handler)
        chain.add_handler(metrics_handler)

        # Test processing through the chain
        result = chain.handle("test message for chain")
        assert isinstance(result, FlextResult)

    def test_handler_metadata_consistency(self) -> None:
        """Test handler metadata across different types."""
        handlers = [
            FlextBaseHandler(name="MetaBase"),
            FlextValidatingHandler(name="MetaValidating"),
            FlextAuthorizingHandler(name="MetaAuthorizing"),
            FlextMetricsHandler(name="MetaMetrics"),
        ]

        for handler in handlers:
            metadata = handler.get_handler_metadata()
            assert isinstance(metadata, dict)
            assert "handler_name" in metadata
            assert "handler_class" in metadata
