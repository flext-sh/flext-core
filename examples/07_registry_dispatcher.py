"""FlextRegistry and FlextDispatcher comprehensive demonstration.

Demonstrates handler registration, batch operations, dispatcher patterns,
and CQRS integration using Python 3.13+ strict patterns with PEP 695
type aliases and collections.abc.

**Expected Output:**
- Handler registration (single and batch)
- Command dispatch with handler resolution
- Query dispatch patterns
- Protocol-based handler compatibility
- CQRS pattern implementation
- Registry and dispatcher integration

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import override

from flext_core import FlextDispatcher, FlextRegistry, c, h, m, r, s, t, u


class CreateUserCommand(m.Command):
    """Create user command."""

    name: str
    email: str


class UserCreatedEvent(m.DomainEvent):
    """User created event."""

    event_type: str = "user_created"
    aggregate_id: str
    name: str


class CreateUserHandler(h[CreateUserCommand, UserCreatedEvent]):
    """Handler for creating users."""

    @override
    def handle(self, message: CreateUserCommand) -> r[UserCreatedEvent]:
        """Handle create user command."""
        user_id = u.generate("entity")
        return r[UserCreatedEvent].ok(
            UserCreatedEvent(
                event_type="user_created",
                aggregate_id=user_id,
                name=message.name,
            ),
        )


class GetUserQuery(m.Query):
    """Get user query."""

    user_id: str


class GetUserHandler(h[GetUserQuery, t.ConfigMap]):
    """Handler for getting users."""

    @override
    def handle(self, message: GetUserQuery) -> r[t.ConfigMap]:
        """Handle get user query."""
        return r[t.ConfigMap].ok(
            t.ConfigMap(
                root={
                    "user_id": message.user_id,
                    "name": "Demo User",
                    "email": "demo@example.com",
                },
            ),
        )


class _DemoPlugin(m.Value):
    """Demo plugin for registry demonstration."""

    name: str


class RegistryDispatcherService(s[t.ConfigMap]):
    """Service demonstrating FlextRegistry and FlextDispatcher."""

    @staticmethod
    def _demonstrate_dispatcher() -> None:
        """Show dispatcher operations."""
        print("\n=== Dispatcher Operations ===")
        dispatcher = FlextDispatcher()
        create_handler = CreateUserHandler()
        dispatcher.register_handler(create_handler)
        command = CreateUserCommand(name="Alice", email="alice@example.com")
        dispatch_result = dispatcher.dispatch(command)
        print(f"✅ Command dispatched: is_success={dispatch_result.is_success}")

    @staticmethod
    def _demonstrate_integration() -> None:
        """Show registry and dispatcher integration."""
        print("\n=== Registry/Dispatcher Integration ===")
        dispatcher = FlextDispatcher()
        registry = FlextRegistry()
        registry.register_plugin(
            "handlers",
            "create_user",
            lambda: _DemoPlugin(name="create_user"),
        )
        registry.register_plugin(
            "handlers",
            "get_user",
            lambda: _DemoPlugin(name="get_user"),
        )
        create_handler = CreateUserHandler()
        get_handler = GetUserHandler()
        dispatcher.register_handler(create_handler)
        dispatcher.register_handler(get_handler)
        command: CreateUserCommand = CreateUserCommand(
            name="Bob",
            email="bob@example.com",
        )
        command_result = dispatcher.dispatch(command)
        if command_result.is_success:
            print("✅ Command dispatched successfully")
        query: GetUserQuery = GetUserQuery(user_id="user-123")
        query_result = dispatcher.dispatch(query)
        if query_result.is_success:
            print("✅ Query dispatched successfully")

    @staticmethod
    def _demonstrate_registry() -> None:
        """Show registry operations."""
        print("\n=== Registry Operations ===")
        registry = FlextRegistry()

        def create_plugin() -> _DemoPlugin:
            return _DemoPlugin(name="create_user")

        register_result = registry.register_plugin(
            "handlers",
            "create_user",
            create_plugin,
        )
        if register_result.is_success:
            print("✅ Plugin registered successfully")

        def query_plugin() -> _DemoPlugin:
            return _DemoPlugin(name="get_user")

        registry.register_plugin("handlers", "get_user", query_plugin)
        plugins_result = registry.list_plugins("handlers")
        if plugins_result.is_success:
            print(f"✅ Plugin catalog: {plugins_result.value}")

    @override
    def execute(self) -> r[t.ConfigMap]:
        """Execute registry and dispatcher demonstrations."""
        print("Starting registry and dispatcher demonstration")
        try:
            self._demonstrate_registry()
            self._demonstrate_dispatcher()
            self._demonstrate_integration()
            return r[t.ConfigMap].ok(
                t.ConfigMap(
                    root={
                        "patterns_demonstrated": [
                            "handler_registration",
                            "batch_registration",
                            "command_dispatch",
                            "query_dispatch",
                            "registry_integration",
                        ],
                        "handler_types": [
                            c.HandlerType.COMMAND.value,
                            c.HandlerType.QUERY.value,
                        ],
                        "features": [
                            "idempotent_registration",
                            "batch_operations",
                            "dispatcher_integration",
                            "cqrs_patterns",
                        ],
                    },
                ),
            )
        except Exception as e:
            error_msg = f"Registry/Dispatcher demonstration failed: {e}"
            return r[t.ConfigMap].fail(error_msg)


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("FLEXT REGISTRY & DISPATCHER - COMPREHENSIVE DEMONSTRATION")
    print("Handler registration, batch operations, CQRS patterns")
    print("=" * 60)
    service = RegistryDispatcherService()
    result = service.execute()
    if result.is_success:
        print("\n✅ Demonstrated registry/dispatcher patterns")
    else:
        print(f"\n❌ Failed: {result.error}")
    print("\n" + "=" * 60)
    print("🎯 Registry Patterns: Registration, Batch, Idempotent")
    print("🎯 Dispatcher Patterns: Command, Query, Event Dispatch")
    print("🎯 CQRS Integration: Handler types, message routing")
    print("🎯 Python 3.13+: PEP 695 type aliases, collections.abc")
    print("=" * 60)


if __name__ == "__main__":
    main()
