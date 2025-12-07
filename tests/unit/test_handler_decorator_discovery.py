"""Comprehensive tests for h.handler() decorator and h.Discovery namespace.

Module: flext_core.handlers
Scope: Handler decorator metadata, discovery scanning, auto-registration

Tests:
- h.handler() decorator: metadata storage and retrieval
- h.Discovery.scan_class(): class method discovery with priority sorting
- h.Discovery.has_handlers(): quick handler presence check
- h.Discovery.scan_module(): module-level function discovery
- h.Discovery.has_handlers_module(): module handler presence check
- Integration with FlextService auto-setup and handler registration

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
import types
from typing import ClassVar

from flext_core import (
    FlextService,
    c,
    h,
    m,
    p,
    r,
    t,
)

# ============================================================================
# TEST FIXTURES - Sample Commands and Handlers
# ============================================================================


class UserCreateCommand:
    """Sample command for testing."""

    def __init__(self, name: str) -> None:
        self.name = name


class UserDeleteCommand:
    """Sample command for testing."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id


class UserQueryCommand:
    """Sample query command for testing."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id


class EventPublished:
    """Sample domain event for testing."""

    def __init__(self, event_id: str) -> None:
        self.event_id = event_id


# ============================================================================
# SERVICE WITH DECORATED HANDLERS - For Discovery Tests
# ============================================================================


class UserService:
    """Service with multiple handler decorators for discovery testing."""

    name: ClassVar[str] = "UserService"

    @h.handler(command=UserCreateCommand, priority=100)
    def handle_create_user(self, cmd: UserCreateCommand) -> r[str]:
        """Handler for creating users with highest priority."""
        return r[str].ok(f"created_{cmd.name}")

    @h.handler(command=UserDeleteCommand, priority=50)
    def handle_delete_user(self, cmd: UserDeleteCommand) -> r[str]:
        """Handler for deleting users with medium priority."""
        return r[str].ok(f"deleted_{cmd.user_id}")

    @h.handler(command=UserQueryCommand, priority=10)
    def handle_query_user(self, cmd: UserQueryCommand) -> r[str]:
        """Handler for querying users with low priority."""
        return r[str].ok(f"found_{cmd.user_id}")

    def non_handler_method(self) -> str:
        """Regular method without handler decorator."""
        return "not_a_handler"


class OrderService:
    """Service with single handler for testing."""

    @h.handler(command=EventPublished, priority=25)
    def handle_event(self, event: EventPublished) -> r[str]:
        """Handler for published events."""
        return r[str].ok(f"processed_{event.event_id}")


class ServiceWithoutHandlers:
    """Service with no handlers for negative testing."""

    def process(self) -> str:
        """Regular method."""
        return "no_handlers"


# ============================================================================
# MODULE-LEVEL HANDLER FUNCTIONS - For Module Scanning Tests
# ============================================================================


@h.handler(command=UserCreateCommand, priority=100)
def handle_user_create_globally(cmd: UserCreateCommand) -> r[str]:
    """Module-level handler for user creation."""
    return r[str].ok(f"global_create_{cmd.name}")


@h.handler(command=UserDeleteCommand, priority=50)
def handle_user_delete_globally(cmd: UserDeleteCommand) -> r[str]:
    """Module-level handler for user deletion."""
    return r[str].ok(f"global_delete_{cmd.user_id}")


def regular_module_function(value: str) -> str:
    """Regular module function without handler decoration."""
    return f"processed_{value}"


# ============================================================================
# TEST DECORATOR METADATA STORAGE
# ============================================================================


class TestHandlerDecoratorMetadata:
    """Test h.handler() decorator metadata storage and retrieval."""

    def test_decorator_stores_metadata_on_method(self) -> None:
        """handler() decorator should store config as method attribute."""

        class TestService:
            @h.handler(command=UserCreateCommand, priority=10)
            def handle_user(self, cmd: UserCreateCommand) -> r[str]:
                return r[str].ok("handled")

        method = TestService.handle_user
        assert hasattr(method, c.Discovery.HANDLER_ATTR)

    def test_decorator_metadata_contains_command_type(self) -> None:
        """Stored metadata should contain the command type."""

        class TestService:
            @h.handler(command=UserCreateCommand)
            def handle_user(self, cmd: UserCreateCommand) -> r[str]:
                return r[str].ok("handled")

        method = TestService.handle_user
        config: m.Handler.DecoratorConfig = getattr(method, c.Discovery.HANDLER_ATTR)
        assert config.command is UserCreateCommand

    def test_decorator_with_custom_priority(self) -> None:
        """Decorator should store custom priority value."""

        class TestService:
            @h.handler(command=UserCreateCommand, priority=42)
            def handle_user(self, cmd: UserCreateCommand) -> r[str]:
                return r[str].ok("handled")

        method = TestService.handle_user
        config: m.Handler.DecoratorConfig = getattr(method, c.Discovery.HANDLER_ATTR)
        assert config.priority == 42

    def test_decorator_default_priority(self) -> None:
        """Decorator should use default priority from constants."""

        class TestService:
            @h.handler(command=UserCreateCommand)
            def handle_user(self, cmd: UserCreateCommand) -> r[str]:
                return r[str].ok("handled")

        method = TestService.handle_user
        config: m.Handler.DecoratorConfig = getattr(method, c.Discovery.HANDLER_ATTR)
        assert config.priority == c.Discovery.DEFAULT_PRIORITY

    def test_decorator_with_timeout(self) -> None:
        """Decorator should store timeout value."""

        class TestService:
            @h.handler(
                command=UserCreateCommand,
                timeout=5.0,
            )
            def handle_user(self, cmd: UserCreateCommand) -> r[str]:
                return r[str].ok("handled")

        method = TestService.handle_user
        config: m.Handler.DecoratorConfig = getattr(method, c.Discovery.HANDLER_ATTR)
        assert config.timeout == 5.0

    def test_decorator_default_timeout(self) -> None:
        """Decorator should use default timeout from constants."""

        class TestService:
            @h.handler(command=UserCreateCommand)
            def handle_user(self, cmd: UserCreateCommand) -> r[str]:
                return r[str].ok("handled")

        method = TestService.handle_user
        config: m.Handler.DecoratorConfig = getattr(method, c.Discovery.HANDLER_ATTR)
        assert config.timeout == c.Discovery.DEFAULT_TIMEOUT

    def test_decorator_with_middleware_list(self) -> None:
        """Decorator should store middleware list."""
        middleware_types: list[type[p.Application.Middleware]] = []

        class TestService:
            @h.handler(
                command=UserCreateCommand,
                middleware=middleware_types,
            )
            def handle_user(self, cmd: UserCreateCommand) -> r[str]:
                return r[str].ok("handled")

        method = TestService.handle_user
        config: m.Handler.DecoratorConfig = getattr(method, c.Discovery.HANDLER_ATTR)
        assert config.middleware == middleware_types

    def test_decorator_default_middleware(self) -> None:
        """Decorator should use empty middleware list when none provided."""

        class TestService:
            @h.handler(command=UserCreateCommand)
            def handle_user(self, cmd: UserCreateCommand) -> r[str]:
                return r[str].ok("handled")

        method = TestService.handle_user
        config: m.Handler.DecoratorConfig = getattr(method, c.Discovery.HANDLER_ATTR)
        assert config.middleware == []

    def test_decorator_preserves_function_identity(self) -> None:
        """Decorator should return the same function (pass-through)."""

        def original_handler(self: object, cmd: UserCreateCommand) -> r[str]:
            return r[str].ok("handled")

        decorated = h.handler(command=UserCreateCommand)(original_handler)
        assert decorated is original_handler


# ============================================================================
# TEST CLASS DISCOVERY
# ============================================================================


class TestHandlerDiscoveryClass:
    """Test h.Discovery.scan_class() for class method discovery."""

    def test_scan_class_finds_all_handlers(self) -> None:
        """scan_class() should find all decorated handler methods."""
        handlers = h.Discovery.scan_class(UserService)
        assert len(handlers) == 3

    def test_scan_class_returns_tuples(self) -> None:
        """scan_class() should return list of (name, config) tuples."""
        handlers = h.Discovery.scan_class(UserService)
        for name, config in handlers:
            assert isinstance(name, str)
            assert isinstance(config, m.Handler.DecoratorConfig)

    def test_scan_class_sorts_by_priority_descending(self) -> None:
        """scan_class() should sort handlers by priority (highest first)."""
        handlers = h.Discovery.scan_class(UserService)
        priorities = [config.priority for _, config in handlers]
        # Verify sorted descending
        assert priorities == sorted(priorities, reverse=True)

    def test_scan_class_priority_order(self) -> None:
        """scan_class() should return handlers in correct priority order."""
        handlers = h.Discovery.scan_class(UserService)
        names = [name for name, _ in handlers]
        # Expected order: highest priority (100), then 50, then 10
        assert names[0] == "handle_create_user"  # priority 100
        assert names[1] == "handle_delete_user"  # priority 50
        assert names[2] == "handle_query_user"  # priority 10

    def test_scan_class_includes_handler_name(self) -> None:
        """scan_class() results should include method names."""
        handlers = h.Discovery.scan_class(UserService)
        names = [name for name, _ in handlers]
        assert "handle_create_user" in names
        assert "handle_delete_user" in names
        assert "handle_query_user" in names

    def test_scan_class_includes_handler_config(self) -> None:
        """scan_class() results should include DecoratorConfig."""
        handlers = h.Discovery.scan_class(UserService)
        for _, config in handlers:
            assert isinstance(config, m.Handler.DecoratorConfig)
            assert config.command is not None
            assert config.priority >= 0

    def test_scan_class_ignores_non_handler_methods(self) -> None:
        """scan_class() should ignore methods without @handler decorator."""
        handlers = h.Discovery.scan_class(UserService)
        names = [name for name, _ in handlers]
        assert "non_handler_method" not in names

    def test_scan_class_empty_for_service_without_handlers(self) -> None:
        """scan_class() should return empty list for classes with no handlers."""
        handlers = h.Discovery.scan_class(ServiceWithoutHandlers)
        assert len(handlers) == 0

    def test_scan_class_single_handler(self) -> None:
        """scan_class() should handle classes with single handler."""
        handlers = h.Discovery.scan_class(OrderService)
        assert len(handlers) == 1
        name, config = handlers[0]
        assert name == "handle_event"
        assert config.command is EventPublished
        assert config.priority == 25

    def test_has_handlers_true_when_handlers_exist(self) -> None:
        """has_handlers() should return True for classes with handlers."""
        assert h.Discovery.has_handlers(UserService) is True
        assert h.Discovery.has_handlers(OrderService) is True

    def test_has_handlers_false_when_no_handlers(self) -> None:
        """has_handlers() should return False for classes without handlers."""
        assert h.Discovery.has_handlers(ServiceWithoutHandlers) is False

    def test_has_handlers_efficient_check(self) -> None:
        """has_handlers() should check without scanning all methods."""
        # This is a positive test that the method returns correct result
        result = h.Discovery.has_handlers(UserService)
        assert result is True


# ============================================================================
# TEST MODULE DISCOVERY
# ============================================================================


class TestHandlerDiscoveryModule:
    """Test h.Discovery.scan_module() for module-level function discovery."""

    def test_scan_module_finds_decorated_functions(self) -> None:
        """scan_module() should find decorated module-level functions."""
        # Use this module as test module
        current_module = sys.modules[__name__]
        handlers = h.Discovery.scan_module(current_module)
        assert len(handlers) >= 2

    def test_scan_module_returns_tuples(self) -> None:
        """scan_module() should return (name, func, config) tuples."""
        current_module = sys.modules[__name__]
        handlers = h.Discovery.scan_module(current_module)
        for name, func, config in handlers:
            assert isinstance(name, str)
            assert callable(func)
            assert isinstance(config, m.Handler.DecoratorConfig)

    def test_scan_module_ignores_private_functions(self) -> None:
        """scan_module() should skip functions starting with underscore."""

        @h.handler(command=UserCreateCommand)
        def _private_handler(cmd: UserCreateCommand) -> r[str]:
            return r[str].ok("private")

        # Private function should not be in scanned results
        current_module = sys.modules[__name__]
        handlers = h.Discovery.scan_module(current_module)
        names = [name for name, _, _ in handlers]
        assert "_private_handler" not in names

    def test_scan_module_ignores_non_callable(self) -> None:
        """scan_module() should skip non-callable attributes."""
        current_module = sys.modules[__name__]
        handlers = h.Discovery.scan_module(current_module)
        # Only callables should be in results
        for _, func, _ in handlers:
            assert callable(func)

    def test_scan_module_sorts_by_priority(self) -> None:
        """scan_module() should sort results by priority (descending)."""
        current_module = sys.modules[__name__]
        handlers = h.Discovery.scan_module(current_module)
        priorities = [config.priority for _, _, config in handlers]
        # Verify sorted descending (if multiple handlers exist)
        if len(priorities) > 1:
            assert priorities == sorted(priorities, reverse=True)

    def test_has_handlers_module_true_when_handlers_exist(self) -> None:
        """has_handlers_module() should return True for modules with handlers."""
        current_module = sys.modules[__name__]
        assert h.Discovery.has_handlers_module(current_module) is True

    def test_has_handlers_module_false_when_no_handlers(self) -> None:
        """has_handlers_module() should return False for modules without handlers."""
        empty_module = types.ModuleType("empty_module")
        assert h.Discovery.has_handlers_module(empty_module) is False

    def test_has_handlers_module_efficient_check(self) -> None:
        """has_handlers_module() should check without scanning all items."""
        current_module = sys.modules[__name__]
        result = h.Discovery.has_handlers_module(current_module)
        assert result is True


# ============================================================================
# TEST INTEGRATION WITH HANDLER DISCOVERY
# ============================================================================


class TestHandlerDiscoveryIntegration:
    """Test integration of handler discovery with actual handler execution."""

    def test_discovered_handlers_are_callable(self) -> None:
        """Discovered handlers should be callable."""
        handlers = h.Discovery.scan_class(UserService)
        service = UserService()

        for method_name, _ in handlers:
            method = getattr(service, method_name)
            assert callable(method)

    def test_discovered_handlers_match_command_type(self) -> None:
        """Discovered handlers should contain correct command types."""
        handlers = h.Discovery.scan_class(UserService)
        command_types = {config.command for _, config in handlers}

        expected_types = {
            UserCreateCommand,
            UserDeleteCommand,
            UserQueryCommand,
        }
        assert command_types == expected_types

    def test_discovery_preserves_handler_metadata(self) -> None:
        """Discovery should preserve all handler metadata."""
        handlers = h.Discovery.scan_class(UserService)

        for _, config in handlers:
            assert config.command is not None
            assert isinstance(config.priority, int)
            assert config.priority >= 0
            assert isinstance(config.middleware, list)

    def test_scan_class_with_multiple_services(self) -> None:
        """Discovery should work independently for multiple service classes."""
        user_handlers = h.Discovery.scan_class(UserService)
        order_handlers = h.Discovery.scan_class(OrderService)

        assert len(user_handlers) == 3
        assert len(order_handlers) == 1

    def test_scan_class_returns_stable_order(self) -> None:
        """Discovery should return consistent order on multiple scans."""
        handlers1 = h.Discovery.scan_class(UserService)
        handlers2 = h.Discovery.scan_class(UserService)

        names1 = [name for name, _ in handlers1]
        names2 = [name for name, _ in handlers2]

        assert names1 == names2

    def test_discovery_with_priority_ties(self) -> None:
        """Discovery should handle handlers with same priority."""

        class ServiceWithEqualPriority:
            @h.handler(command=UserCreateCommand, priority=10)
            def handler_a(self, cmd: UserCreateCommand) -> r[str]:
                return r[str].ok("a")

            @h.handler(command=UserDeleteCommand, priority=10)
            def handler_b(self, cmd: UserDeleteCommand) -> r[str]:
                return r[str].ok("b")

        handlers = h.Discovery.scan_class(ServiceWithEqualPriority)
        assert len(handlers) == 2
        # Both should have same priority
        assert handlers[0][1].priority == 10
        assert handlers[1][1].priority == 10


# ============================================================================
# TEST EDGE CASES AND ERROR CONDITIONS
# ============================================================================


class TestHandlerDiscoveryEdgeCases:
    """Test edge cases and error conditions in handler discovery."""

    def test_scan_class_with_inherited_handlers(self) -> None:
        """scan_class() should find handlers from parent classes."""

        class BaseService:
            @h.handler(command=UserCreateCommand, priority=10)
            def handle_create(self, cmd: UserCreateCommand) -> r[str]:
                return r[str].ok("created")

        class DerivedService(BaseService):
            @h.handler(command=UserDeleteCommand, priority=5)
            def handle_delete(self, cmd: UserDeleteCommand) -> r[str]:
                return r[str].ok("deleted")

        handlers = h.Discovery.scan_class(DerivedService)
        names = [name for name, _ in handlers]
        # Should find both inherited and own handlers
        assert "handle_create" in names
        assert "handle_delete" in names

    def test_handler_decorator_with_none_timeout(self) -> None:
        """Decorator should handle None timeout explicitly."""

        class TestService:
            @h.handler(command=UserCreateCommand, timeout=None)
            def handle(self, cmd: UserCreateCommand) -> r[str]:
                return r[str].ok("ok")

        method = TestService.handle
        config: m.Handler.DecoratorConfig = getattr(method, c.Discovery.HANDLER_ATTR)
        assert config.timeout is None

    def test_scan_class_on_builtin_class(self) -> None:
        """scan_class() should safely handle built-in classes."""
        handlers = h.Discovery.scan_class(str)
        assert len(handlers) == 0

    def test_scan_module_with_objects_without_decorator(self) -> None:
        """scan_module() should handle various object types."""
        current_module = sys.modules[__name__]
        handlers = h.Discovery.scan_module(current_module)

        # Should only include decorated handler functions
        for _name, func, config in handlers:
            assert hasattr(func, c.Discovery.HANDLER_ATTR)
            assert config.command is not None

    def test_multiple_decorations_overwrites_previous(self) -> None:
        """Multiple @handler decorators should use the last one."""

        class TestService:
            @h.handler(command=UserCreateCommand, priority=10)
            @h.handler(command=UserDeleteCommand, priority=20)
            def handle(self, cmd: t.GeneralValueType) -> r[str]:
                return r[str].ok("ok")

        method = TestService.handle
        config: m.Handler.DecoratorConfig = getattr(method, c.Discovery.HANDLER_ATTR)
        # The innermost (first applied) decorator should take effect
        # Due to decorator order, UserDeleteCommand with priority 20 is applied last
        assert config.command is UserDeleteCommand
        assert config.priority == 20


# ============================================================================
# TEST DISCOVERY WITH FLEXT SERVICE INTEGRATION
# ============================================================================


class TestHandlerDiscoveryServiceIntegration:
    """Test handler discovery integration with FlextService."""

    def test_service_can_use_discovered_handlers(self) -> None:
        """FlextService should be able to discover and use handlers."""

        class TestService(FlextService[str]):
            @h.handler(command=UserCreateCommand, priority=10)
            def handle_user_create(
                self,
                cmd: UserCreateCommand,
            ) -> r[str]:
                return r[str].ok(f"created_{cmd.name}")

            def execute(self) -> r[str]:
                return r[str].ok("executed")

        # Service should be instantiable
        service = TestService()

        # Check that handler was discovered
        handlers = h.Discovery.scan_class(TestService)
        assert len(handlers) >= 1

        # Handler should be callable
        for method_name, _ in handlers:
            method = getattr(service, method_name)
            assert callable(method)

    def test_service_auto_discover_handlers_attribute(self) -> None:
        """Service should auto-discover handlers with @h.handler decorator."""

        class TestService(FlextService[str]):
            @h.handler(command=UserCreateCommand)
            def create_user(self, cmd: UserCreateCommand) -> r[str]:
                return r[str].ok(f"created_{cmd.name}")

            @h.handler(command=UserDeleteCommand)
            def delete_user(self, cmd: UserDeleteCommand) -> r[str]:
                return r[str].ok(f"deleted_{cmd.user_id}")

            def execute(self) -> r[str]:
                return r[str].ok("done")

        # Check discovery works
        has_handlers = h.Discovery.has_handlers(TestService)
        assert has_handlers is True

        handlers = h.Discovery.scan_class(TestService)
        assert len(handlers) == 2


__all__ = [
    "TestHandlerDecoratorMetadata",
    "TestHandlerDiscoveryClass",
    "TestHandlerDiscoveryEdgeCases",
    "TestHandlerDiscoveryIntegration",
    "TestHandlerDiscoveryModule",
    "TestHandlerDiscoveryServiceIntegration",
]
