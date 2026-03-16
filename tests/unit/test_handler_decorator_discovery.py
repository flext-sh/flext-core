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
from collections.abc import Callable
from typing import ClassVar, cast, override

from flext_tests import tm

from flext_core import FlextService, h, r
from tests.constants import c
from tests.models import m
from tests.protocols import p

type _TestCallable = Callable[..., r[str]]
type _TestDecorator = Callable[[_TestCallable], _TestCallable]


def _test_handler(
    command: type,
    *,
    priority: int = c.Discovery.DEFAULT_PRIORITY,
    timeout: float | None = c.Discovery.DEFAULT_TIMEOUT,
    middleware: list[type[p.Middleware]] | None = None,
) -> _TestDecorator:
    return cast(
        "_TestDecorator",
        h.handler(
            command=command,
            priority=priority,
            timeout=timeout,
            middleware=middleware,
        ),
    )


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


class UserService:
    """Service with multiple handler decorators for discovery testing."""

    name: ClassVar[str] = "UserService"

    @_test_handler(command=UserCreateCommand, priority=100)
    def handle_create_user(self, cmd: UserCreateCommand) -> r[str]:
        """Handler for creating users with highest priority."""
        return r[str].ok(f"created_{cmd.name}")

    @_test_handler(command=UserDeleteCommand, priority=50)
    def handle_delete_user(self, cmd: UserDeleteCommand) -> r[str]:
        """Handler for deleting users with medium priority."""
        return r[str].ok(f"deleted_{cmd.user_id}")

    @_test_handler(command=UserQueryCommand, priority=10)
    def handle_query_user(self, cmd: UserQueryCommand) -> r[str]:
        """Handler for querying users with low priority."""
        return r[str].ok(f"found_{cmd.user_id}")

    def non_handler_method(self) -> str:
        """Regular method without handler decorator."""
        return "not_a_handler"


class OrderService:
    """Service with single handler for testing."""

    @_test_handler(command=EventPublished, priority=25)
    def handle_event(self, event: EventPublished) -> r[str]:
        """Handler for published events."""
        return r[str].ok(f"processed_{event.event_id}")


class ServiceWithoutHandlers:
    """Service with no handlers for negative testing."""

    def process(self) -> str:
        """Regular method."""
        return "no_handlers"


@_test_handler(command=UserCreateCommand, priority=100)
def handle_user_create_globally(cmd: UserCreateCommand) -> r[str]:
    """Module-level handler for user creation."""
    return r[str].ok(f"global_create_{cmd.name}")


@_test_handler(command=UserDeleteCommand, priority=50)
def handle_user_delete_globally(cmd: UserDeleteCommand) -> r[str]:
    """Module-level handler for user deletion."""
    return r[str].ok(f"global_delete_{cmd.user_id}")


def regular_module_function(value: str) -> str:
    """Regular module function without handler decoration."""
    return f"processed_{value}"


class TestHandlerDecoratorMetadata:
    """Test h.handler() decorator metadata storage and retrieval."""

    def test_decorator_stores_metadata_on_method(self) -> None:
        """handler() decorator should store config as method attribute."""

        class TestService:
            @_test_handler(command=UserCreateCommand, priority=10)
            def handle_user(self, cmd: UserCreateCommand) -> r[str]:
                return r[str].ok("handled")

        method = TestService.handle_user
        tm.that(hasattr(method, c.Discovery.HANDLER_ATTR), eq=True)

    def test_decorator_metadata_contains_command_type(self) -> None:
        """Stored metadata should contain the command type."""

        class TestService:
            @_test_handler(command=UserCreateCommand)
            def handle_user(self, cmd: UserCreateCommand) -> r[str]:
                return r[str].ok("handled")

        method = TestService.handle_user
        config: m.DecoratorConfig = getattr(method, c.Discovery.HANDLER_ATTR)
        tm.that(config.command is UserCreateCommand, eq=True)

    def test_decorator_with_custom_priority(self) -> None:
        """Decorator should store custom priority value."""

        class TestService:
            @_test_handler(command=UserCreateCommand, priority=42)
            def handle_user(self, cmd: UserCreateCommand) -> r[str]:
                return r[str].ok("handled")

        method = TestService.handle_user
        config: m.DecoratorConfig = getattr(method, c.Discovery.HANDLER_ATTR)
        tm.that(config.priority == 42, eq=True)

    def test_decorator_default_priority(self) -> None:
        """Decorator should use default priority from constants."""

        class TestService:
            @_test_handler(command=UserCreateCommand)
            def handle_user(self, cmd: UserCreateCommand) -> r[str]:
                return r[str].ok("handled")

        method = TestService.handle_user
        config: m.DecoratorConfig = getattr(method, c.Discovery.HANDLER_ATTR)
        tm.that(config.priority == c.Discovery.DEFAULT_PRIORITY, eq=True)

    def test_decorator_with_timeout(self) -> None:
        """Decorator should store timeout value."""

        class TestService:
            @_test_handler(command=UserCreateCommand, timeout=5.0)
            def handle_user(self, cmd: UserCreateCommand) -> r[str]:
                return r[str].ok("handled")

        method = TestService.handle_user
        config: m.DecoratorConfig = getattr(method, c.Discovery.HANDLER_ATTR)
        timeout = config.timeout
        if timeout is None:
            tm.that(False, eq=True)
            return
        tm.that(abs(timeout - 5.0) < 1e-9, eq=True)

    def test_decorator_default_timeout(self) -> None:
        """Decorator should use default timeout from constants."""

        class TestService:
            @_test_handler(command=UserCreateCommand)
            def handle_user(self, cmd: UserCreateCommand) -> r[str]:
                return r[str].ok("handled")

        method = TestService.handle_user
        config: m.DecoratorConfig = getattr(method, c.Discovery.HANDLER_ATTR)
        tm.that(config.timeout == c.Discovery.DEFAULT_TIMEOUT, eq=True)

    def test_decorator_with_middleware_list(self) -> None:
        """Decorator should store middleware list."""
        middleware_types: list[type[p.Middleware]] = []

        class TestService:
            @_test_handler(command=UserCreateCommand, middleware=middleware_types)
            def handle_user(self, cmd: UserCreateCommand) -> r[str]:
                return r[str].ok("handled")

        method = TestService.handle_user
        config: m.DecoratorConfig = getattr(method, c.Discovery.HANDLER_ATTR)
        tm.that(config.middleware == middleware_types, eq=True)

    def test_decorator_default_middleware(self) -> None:
        """Decorator should use empty middleware list when none provided."""

        class TestService:
            @_test_handler(command=UserCreateCommand)
            def handle_user(self, cmd: UserCreateCommand) -> r[str]:
                return r[str].ok("handled")

        method = TestService.handle_user
        config: m.DecoratorConfig = getattr(method, c.Discovery.HANDLER_ATTR)
        tm.that(config.middleware == [], eq=True)

    def test_decorator_preserves_function_identity(self) -> None:
        """Decorator should return the same function (pass-through)."""

        def original_handler(self: FlextService, cmd: UserCreateCommand) -> r[str]:
            return r[str].ok("handled")

        decorated = _test_handler(command=UserCreateCommand)(original_handler)
        tm.that(decorated is original_handler, eq=True)


class TestHandlerDiscoveryClass:
    """Test h.Discovery.scan_class() for class method discovery."""

    def test_scan_class_finds_all_handlers(self) -> None:
        """scan_class() should find all decorated handler methods."""
        handlers = h.Discovery.scan_class(UserService)
        tm.that(len(handlers) == 3, eq=True)

    def test_scan_class_returns_tuples(self) -> None:
        """scan_class() should return list of (name, config) tuples."""
        handlers = h.Discovery.scan_class(UserService)
        for name, config in handlers:
            tm.that(isinstance(name, str), eq=True)
            tm.that(isinstance(config, m.DecoratorConfig), eq=True)

    def test_scan_class_sorts_by_priority_descending(self) -> None:
        """scan_class() should sort handlers by priority (highest first)."""
        handlers = h.Discovery.scan_class(UserService)
        priorities = [config.priority for _, config in handlers]
        tm.that(priorities == sorted(priorities, reverse=True), eq=True)

    def test_scan_class_priority_order(self) -> None:
        """scan_class() should return handlers in correct priority order."""
        handlers = h.Discovery.scan_class(UserService)
        names = [name for name, _ in handlers]
        tm.that(names[0] == "handle_create_user", eq=True)
        tm.that(names[1] == "handle_delete_user", eq=True)
        tm.that(names[2] == "handle_query_user", eq=True)

    def test_scan_class_includes_handler_name(self) -> None:
        """scan_class() results should include method names."""
        handlers = h.Discovery.scan_class(UserService)
        names = [name for name, _ in handlers]
        tm.that("handle_create_user" in names, eq=True)
        tm.that("handle_delete_user" in names, eq=True)
        tm.that("handle_query_user" in names, eq=True)

    def test_scan_class_includes_handler_config(self) -> None:
        """scan_class() results should include DecoratorConfig."""
        handlers = h.Discovery.scan_class(UserService)
        for _, config in handlers:
            tm.that(isinstance(config, m.DecoratorConfig), eq=True)
            tm.that(config.command is not None, eq=True)
            tm.that(config.priority >= 0, eq=True)

    def test_scan_class_ignores_non_handler_methods(self) -> None:
        """scan_class() should ignore methods without @handler decorator."""
        handlers = h.Discovery.scan_class(UserService)
        names = [name for name, _ in handlers]
        tm.that("non_handler_method" not in names, eq=True)

    def test_scan_class_empty_for_service_without_handlers(self) -> None:
        """scan_class() should return empty list for classes with no handlers."""
        handlers = h.Discovery.scan_class(ServiceWithoutHandlers)
        tm.that(len(handlers) == 0, eq=True)

    def test_scan_class_single_handler(self) -> None:
        """scan_class() should handle classes with single handler."""
        handlers = h.Discovery.scan_class(OrderService)
        tm.that(len(handlers) == 1, eq=True)
        name, config = handlers[0]
        tm.that(name == "handle_event", eq=True)
        tm.that(config.command is EventPublished, eq=True)
        tm.that(config.priority == 25, eq=True)

    def test_has_handlers_true_when_handlers_exist(self) -> None:
        """has_handlers() should return True for classes with handlers."""
        tm.that(h.Discovery.has_handlers(UserService) is True, eq=True)
        tm.that(h.Discovery.has_handlers(OrderService) is True, eq=True)

    def test_has_handlers_false_when_no_handlers(self) -> None:
        """has_handlers() should return False for classes without handlers."""
        tm.that(h.Discovery.has_handlers(ServiceWithoutHandlers) is False, eq=True)

    def test_has_handlers_efficient_check(self) -> None:
        """has_handlers() should check without scanning all methods."""
        result = h.Discovery.has_handlers(UserService)
        tm.that(result is True, eq=True)


class TestHandlerDiscoveryModule:
    """Test h.Discovery.scan_module() for module-level function discovery."""

    def test_scan_module_finds_decorated_functions(self) -> None:
        """scan_module() should find decorated module-level functions."""
        current_module = sys.modules[__name__]
        handlers = h.Discovery.scan_module(current_module)
        tm.that(len(handlers) >= 2, eq=True)

    def test_scan_module_returns_tuples(self) -> None:
        """scan_module() should return (name, func, config) tuples."""
        current_module = sys.modules[__name__]
        handlers = h.Discovery.scan_module(current_module)
        for name, func, config in handlers:
            tm.that(isinstance(name, str), eq=True)
            tm.that(callable(func), eq=True)
            tm.that(isinstance(config, m.DecoratorConfig), eq=True)

    def test_scan_module_ignores_private_functions(self) -> None:
        """scan_module() should skip functions starting with underscore."""

        @_test_handler(command=UserCreateCommand)
        def _private_handler(cmd: UserCreateCommand) -> r[str]:
            return r[str].ok("private")

        _ = _private_handler

        current_module = sys.modules[__name__]
        handlers = h.Discovery.scan_module(current_module)
        names = [name for name, _, _ in handlers]
        tm.that("_private_handler" not in names, eq=True)

    def test_scan_module_ignores_non_callable(self) -> None:
        """scan_module() should skip non-callable attributes."""
        current_module = sys.modules[__name__]
        handlers = h.Discovery.scan_module(current_module)
        for _, func, _ in handlers:
            tm.that(callable(func), eq=True)

    def test_scan_module_sorts_by_priority(self) -> None:
        """scan_module() should sort results by priority (descending)."""
        current_module = sys.modules[__name__]
        handlers = h.Discovery.scan_module(current_module)
        priorities = [config.priority for _, _, config in handlers]
        if len(priorities) > 1:
            tm.that(priorities == sorted(priorities, reverse=True), eq=True)

    def test_has_handlers_module_true_when_handlers_exist(self) -> None:
        """has_handlers_module() should return True for modules with handlers."""
        current_module = sys.modules[__name__]
        tm.that(h.Discovery.has_handlers_module(current_module) is True, eq=True)

    def test_has_handlers_module_false_when_no_handlers(self) -> None:
        """has_handlers_module() should return False for modules without handlers."""
        empty_module = types.ModuleType("empty_module")
        tm.that(h.Discovery.has_handlers_module(empty_module) is False, eq=True)

    def test_has_handlers_module_efficient_check(self) -> None:
        """has_handlers_module() should check without scanning all items."""
        current_module = sys.modules[__name__]
        result = h.Discovery.has_handlers_module(current_module)
        tm.that(result is True, eq=True)


class TestHandlerDiscoveryIntegration:
    """Test integration of handler discovery with actual handler execution."""

    def test_discovered_handlers_are_callable(self) -> None:
        """Discovered handlers should be callable."""
        handlers = h.Discovery.scan_class(UserService)
        service = UserService()
        for method_name, _ in handlers:
            method = getattr(service, method_name)
            tm.that(callable(method), eq=True)

    def test_discovered_handlers_match_command_type(self) -> None:
        """Discovered handlers should contain correct command types."""
        handlers = h.Discovery.scan_class(UserService)
        command_types = {config.command for _, config in handlers}
        expected_types = {UserCreateCommand, UserDeleteCommand, UserQueryCommand}
        tm.that(command_types == expected_types, eq=True)

    def test_discovery_preserves_handler_metadata(self) -> None:
        """Discovery should preserve all handler metadata."""
        handlers = h.Discovery.scan_class(UserService)
        for _, config in handlers:
            tm.that(config.command is not None, eq=True)
            tm.that(isinstance(config.priority, int), eq=True)
            tm.that(config.priority >= 0, eq=True)
            tm.that(isinstance(config.middleware, list), eq=True)

    def test_scan_class_with_multiple_services(self) -> None:
        """Discovery should work independently for multiple service classes."""
        user_handlers = h.Discovery.scan_class(UserService)
        order_handlers = h.Discovery.scan_class(OrderService)
        tm.that(len(user_handlers) == 3, eq=True)
        tm.that(len(order_handlers) == 1, eq=True)

    def test_scan_class_returns_stable_order(self) -> None:
        """Discovery should return consistent order on multiple scans."""
        handlers1 = h.Discovery.scan_class(UserService)
        handlers2 = h.Discovery.scan_class(UserService)
        names1 = [name for name, _ in handlers1]
        names2 = [name for name, _ in handlers2]
        tm.that(names1 == names2, eq=True)

    def test_discovery_with_priority_ties(self) -> None:
        """Discovery should handle handlers with same priority."""

        class ServiceWithEqualPriority:
            @_test_handler(command=UserCreateCommand, priority=10)
            def handler_a(self, cmd: UserCreateCommand) -> r[str]:
                return r[str].ok("a")

            @_test_handler(command=UserDeleteCommand, priority=10)
            def handler_b(self, cmd: UserDeleteCommand) -> r[str]:
                return r[str].ok("b")

        handlers = h.Discovery.scan_class(ServiceWithEqualPriority)
        tm.that(len(handlers) == 2, eq=True)
        tm.that(handlers[0][1].priority == 10, eq=True)
        tm.that(handlers[1][1].priority == 10, eq=True)


class TestHandlerDiscoveryEdgeCases:
    """Test edge cases and error conditions in handler discovery."""

    def test_scan_class_with_inherited_handlers(self) -> None:
        """scan_class() should find handlers from parent classes."""

        class BaseService:
            @_test_handler(command=UserCreateCommand, priority=10)
            def handle_create(self, cmd: UserCreateCommand) -> r[str]:
                return r[str].ok("created")

        class DerivedService(BaseService):
            @_test_handler(command=UserDeleteCommand, priority=5)
            def handle_delete(self, cmd: UserDeleteCommand) -> r[str]:
                return r[str].ok("deleted")

        handlers = h.Discovery.scan_class(DerivedService)
        names = [name for name, _ in handlers]
        tm.that("handle_create" in names, eq=True)
        tm.that("handle_delete" in names, eq=True)

    def test_handler_decorator_with_none_timeout(self) -> None:
        """Decorator should handle None timeout explicitly."""

        class TestService:
            @_test_handler(command=UserCreateCommand, timeout=None)
            def handle(self, cmd: UserCreateCommand) -> r[str]:
                return r[str].ok("ok")

        method = TestService.handle
        config: m.DecoratorConfig = getattr(method, c.Discovery.HANDLER_ATTR)
        tm.that(config.timeout is None, eq=True)

    def test_scan_class_on_builtin_class(self) -> None:
        """scan_class() should safely handle built-in classes."""
        handlers = h.Discovery.scan_class(str)
        tm.that(len(handlers) == 0, eq=True)

    def test_scan_module_with_objects_without_decorator(self) -> None:
        """scan_module() should handle various object types."""
        current_module = sys.modules[__name__]
        handlers = h.Discovery.scan_module(current_module)
        for _name, func, config in handlers:
            tm.that(hasattr(func, c.Discovery.HANDLER_ATTR), eq=True)
            tm.that(config.command is not None, eq=True)

    def test_multiple_decorations_overwrites_previous(self) -> None:
        """Multiple @handler decorators should use the last one."""

        class TestService:
            @_test_handler(command=UserCreateCommand, priority=10)
            @_test_handler(command=UserDeleteCommand, priority=20)
            def handle(self, cmd: UserDeleteCommand) -> r[str]:
                return r[str].ok("ok")

        method = TestService.handle
        config: m.DecoratorConfig = getattr(method, c.Discovery.HANDLER_ATTR)
        tm.that(config.command is UserDeleteCommand, eq=True)
        tm.that(config.priority == 20, eq=True)


class _TestServiceForDiscovery(FlextService[str]):
    """Test service class for handler discovery integration tests."""

    @_test_handler(command=UserCreateCommand, priority=10)
    def handle_user_create(self, cmd: UserCreateCommand) -> r[str]:
        return r[str].ok(f"created_{cmd.name}")

    @override
    def execute(self) -> r[str]:
        return r[str].ok("executed")


class _TestServiceWithMultipleHandlers(FlextService[str]):
    """Test service class with multiple handlers for discovery tests."""

    @_test_handler(command=UserCreateCommand)
    def create_user(self, cmd: UserCreateCommand) -> r[str]:
        return r[str].ok(f"created_{cmd.name}")

    @_test_handler(command=UserDeleteCommand)
    def delete_user(self, cmd: UserDeleteCommand) -> r[str]:
        return r[str].ok(f"deleted_{cmd.user_id}")

    @override
    def execute(self) -> r[str]:
        return r[str].ok("done")


class TestHandlerDiscoveryServiceIntegration:
    """Test handler discovery integration with FlextService."""

    def test_service_can_use_discovered_handlers(self) -> None:
        """FlextService should be able to discover handlers from decorated classes."""
        handlers = h.Discovery.scan_class(_TestServiceForDiscovery)
        tm.that(len(handlers) >= 1, eq=True)
        method_name, config = handlers[0]
        tm.that(method_name == "handle_user_create", eq=True)
        tm.that(config.command is UserCreateCommand, eq=True)
        tm.that(config.priority == 10, eq=True)

    def test_service_auto_discover_handlers_attribute(self) -> None:
        """Service should auto-discover handlers with @h.handler decorator."""
        has_handlers = h.Discovery.has_handlers(_TestServiceWithMultipleHandlers)
        tm.that(has_handlers is True, eq=True)
        handlers = h.Discovery.scan_class(_TestServiceWithMultipleHandlers)
        tm.that(len(handlers) == 2, eq=True)


__all__ = [
    "TestHandlerDecoratorMetadata",
    "TestHandlerDiscoveryClass",
    "TestHandlerDiscoveryEdgeCases",
    "TestHandlerDiscoveryIntegration",
    "TestHandlerDiscoveryModule",
    "TestHandlerDiscoveryServiceIntegration",
]
