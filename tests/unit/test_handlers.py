"""Direct tests for handlers.py to force module import and increase coverage.

This test imports the handlers module directly to ensure coverage detection.
"""

from __future__ import annotations

import typing

import pytest

import flext_core.handlers as handlers_module
from flext_core import FlextResult

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestHandlersModuleDirect:
    """Test handlers module directly to ensure coverage."""

    def test_module_imports_correctly(self) -> None:
        """Test that the handlers module imports correctly."""
        assert handlers_module is not None
        assert hasattr(handlers_module, "FlextBaseHandler")

    def test_base_handler_direct(self) -> None:
        """Test FlextBaseHandler directly from module."""
        base_handler_cls = handlers_module.FlextBaseHandler

        handler = base_handler_cls()
        assert handler.handler_name == "Handler"

        # Test methods
        assert handler.can_handle("test") is True
        result = handler.handle("test")
        assert result.success
        assert result.value == "test"

    def test_validating_handler_direct(self) -> None:
        """Test FlextValidatingHandler directly from module."""
        validating_handler_cls = handlers_module.FlextValidatingHandler

        handler = validating_handler_cls()
        assert handler.handler_name == "ValidatingHandler"

        # Test validation - current implementation always succeeds
        result = handler.validate("valid")
        assert result.success

        result = handler.validate(None)
        assert result.success  # Current implementation doesn't validate None

    def test_authorizing_handler_direct(self) -> None:
        """Test FlextAuthorizingHandler directly from module."""
        authorizing_handler_cls = handlers_module.FlextAuthorizingHandler

        handler = authorizing_handler_cls()
        assert handler.handler_name == "AuthorizingHandler"

        # Test with authorization check
        def test_auth_check(request: object) -> bool:
            return str(request) == "authorized"

        handler_with_auth = authorizing_handler_cls(
            name="AuthHandler",
            authorization_check=test_auth_check,
        )
        assert handler_with_auth.handler_name == "AuthHandler"

        # Test authorization behavior
        result1 = handler_with_auth.handle("authorized")
        assert result1.success

        result2 = handler_with_auth.handle("unauthorized")
        assert result2.is_failure
        assert "Authorization failed" in str(result2.error)

    def test_metrics_handler_direct(self) -> None:
        """Test FlextMetricsHandler directly from module."""
        metrics_handler_cls = handlers_module.FlextMetricsHandler

        handler = metrics_handler_cls()
        assert handler.handler_name == "MetricsHandler"

        # Process messages to generate metrics
        handler.handle("message1")
        handler.handle("message2")

        # Get metrics
        metrics = handler.get_metrics()
        assert isinstance(metrics, dict)

    def test_handler_chain_direct(self) -> None:
        """Test FlextHandlerChain directly from module."""
        handler_chain_cls = handlers_module.FlextHandlerChain
        base_handler_cls = handlers_module.FlextBaseHandler

        chain = handler_chain_cls()
        handler = base_handler_cls(name="ChainTest")

        chain.add_handler(handler)
        # Test that chain has handlers by using internal attribute
        assert hasattr(chain, "_handlers")
        handlers = getattr(chain, "_handlers", [])
        assert len(handlers) >= 1

    def test_handler_registry_direct(self) -> None:
        """Test FlextHandlerRegistry directly from module."""
        handler_registry_cls = handlers_module.FlextHandlerRegistry
        base_handler_cls = handlers_module.FlextBaseHandler

        registry = handler_registry_cls()
        handler = base_handler_cls(name="RegistryTest")

        result = registry.register("test", handler)
        assert isinstance(result, FlextResult)

        # Get all handlers
        all_handlers = registry.get_all_handlers()
        assert isinstance(all_handlers, dict)
        assert "test" in all_handlers
        assert all_handlers["test"] == handler

    def test_event_handler_direct(self) -> None:
        """Test FlextEventHandler directly from module."""
        event_handler_cls = handlers_module.FlextEventHandler

        handler = event_handler_cls()
        assert handler is not None

        # Test basic event handling
        result = handler.handle_event("test event")
        assert isinstance(result, FlextResult)

    @typing.no_type_check  # Suppress type checking for dynamic attribute access
    def test_command_handler_direct(self) -> None:
        """Test concrete command handler from module."""
        # Use ValidatingHandler as a concrete implementation
        validating_handler_cls = handlers_module.FlextValidatingHandler

        handler: object = validating_handler_cls()
        assert handler is not None
        assert hasattr(handler, "handle")

    @typing.no_type_check  # Suppress type checking for dynamic attribute access
    def test_query_handler_direct(self) -> None:
        """Test concrete handler from module."""
        # Use FlextAuthorizingHandler as concrete implementation
        auth_handler_cls = handlers_module.FlextAuthorizingHandler

        handler: object = auth_handler_cls()
        assert handler is not None
        assert hasattr(handler, "handle")

        # Test basic functionality
        result = handler.handle("test query")
        assert isinstance(result, FlextResult)

    def test_all_handler_classes_exist(self) -> None:
        """Test that all expected handler classes exist in the module."""
        expected_classes = [
            "FlextBaseHandler",
            "FlextValidatingHandler",
            "FlextAuthorizingHandler",
            "FlextEventHandler",
            "FlextMetricsHandler",
            "FlextHandlerChain",
            "FlextHandlerRegistry",
            "FlextCommandHandler",
            "FlextQueryHandler",
        ]

        for class_name in expected_classes:
            assert hasattr(handlers_module, class_name), f"Missing class: {class_name}"

            # Try to instantiate each class
            cls = getattr(handlers_module, class_name)
            try:
                instance = cls()
                assert instance is not None
            except TypeError:
                # Some classes might require parameters
                pass

    def test_handlers_module_exports(self) -> None:
        """Test that the handlers module has proper exports."""
        # Check that __all__ exists and contains expected classes
        if hasattr(handlers_module, "__all__"):
            exports = handlers_module.__all__
            assert isinstance(exports, list)
            assert len(exports) > 0

            # Verify all exported names exist in the module
            for export in exports:
                assert hasattr(handlers_module, export), f"Missing export: {export}"
