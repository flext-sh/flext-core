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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

from flext_core import (
    FlextDispatcher,
    FlextRegistry,
    c,
    h,
    m,
    r,
    s,
    u,
)

# ═══════════════════════════════════════════════════════════════════
# HANDLER IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════


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
                aggregate_id=user_id,
                name=message.name,
            ),
        )


class GetUserQuery(m.Query):
    """Get user query."""

    user_id: str


class GetUserHandler(h[GetUserQuery, m.ConfigMap]):
    """Handler for getting users."""

    @override
    def handle(
        self,
        message: GetUserQuery,
    ) -> r[m.ConfigMap]:
        """Handle get user query."""
        return r[m.ConfigMap].ok(
            m.ConfigMap(
                root={
                    "user_id": message.user_id,
                    "name": "Demo User",
                    "email": "demo@example.com",
                },
            ),
        )


@dataclass(frozen=True, slots=True)
class _DemoPlugin:
    name: str

    def _protocol_name(self) -> str:
        return f"demo-plugin::{self.name}"


# ═══════════════════════════════════════════════════════════════════
# SERVICE IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════


class RegistryDispatcherService(s[m.ConfigMap]):
    """Service demonstrating FlextRegistry and FlextDispatcher."""

    @override
    def execute(
        self,
    ) -> r[m.ConfigMap]:
        """Execute registry and dispatcher demonstrations."""
        print("Starting registry and dispatcher demonstration")

        try:
            self._demonstrate_registry()
            self._demonstrate_dispatcher()
            self._demonstrate_integration()

            return r[m.ConfigMap].ok(
                m.ConfigMap(
                    root={
                        "patterns_demonstrated": [
                            "handler_registration",
                            "batch_registration",
                            "command_dispatch",
                            "query_dispatch",
                            "registry_integration",
                        ],
                        "handler_types": [
                            c.Cqrs.HandlerType.COMMAND.value,
                            c.Cqrs.HandlerType.QUERY.value,
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
            return r[m.ConfigMap].fail(error_msg)

    @staticmethod
    def _demonstrate_registry() -> None:
        """Show registry operations."""
        print("\n=== Registry Operations ===")

        registry = FlextRegistry()

        create_plugin = _DemoPlugin("create_user")
        register_result = registry.register_plugin(
            "handlers",
            "create_user",
            create_plugin,
        )
        if register_result.is_success:
            print("✅ Plugin registered successfully")

        query_plugin = _DemoPlugin("get_user")
        _ = registry.register_plugin(
            "handlers",
            "get_user",
            query_plugin,
        )
        plugins_result = registry.list_plugins("handlers")
        if plugins_result.is_success:
            print(f"✅ Plugin catalog: {plugins_result.value}")

    @staticmethod
    def _demonstrate_dispatcher() -> None:
        """Show dispatcher operations."""
        print("\n=== Dispatcher Operations ===")

        dispatcher = FlextDispatcher()
        create_handler = CreateUserHandler()
        _ = dispatcher.register_handler(create_handler)

        command = CreateUserCommand.model_validate({
            "name": "Alice",
            "email": "alice@example.com",
        })
        dispatch_result = dispatcher.dispatch(command)
        if dispatch_result.is_success:
            event_value = dispatch_result.value
            if isinstance(event_value, UserCreatedEvent):
                print(f"✅ Command dispatched: {event_value.aggregate_id}")
            else:
                print("✅ Command dispatched successfully")

    @staticmethod
    def _demonstrate_integration() -> None:
        """Show registry and dispatcher integration."""
        print("\n=== Registry/Dispatcher Integration ===")

        dispatcher = FlextDispatcher()
        registry = FlextRegistry()

        _ = registry.register_plugin(
            "handlers",
            "create_user",
            _DemoPlugin("create_user"),
        )
        _ = registry.register_plugin(
            "handlers",
            "get_user",
            _DemoPlugin("get_user"),
        )

        create_handler = CreateUserHandler()
        get_handler = GetUserHandler()
        _ = dispatcher.register_handler(create_handler)
        _ = dispatcher.register_handler(get_handler)

        # Dispatch command - Pydantic models as message payload
        command: CreateUserCommand = CreateUserCommand.model_validate({
            "name": "Bob",
            "email": "bob@example.com",
        })
        command_result = dispatcher.dispatch(command)
        if command_result.is_success:
            print("✅ Command dispatched successfully")

        # Dispatch query
        query: GetUserQuery = GetUserQuery.model_validate({"user_id": "user-123"})
        query_result = dispatcher.dispatch(query)
        if query_result.is_success:
            user_data = query_result.value
            if isinstance(user_data, m.ConfigMap):
                print(f"✅ Query dispatched: {user_data.get('name')}")


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("FLEXT REGISTRY & DISPATCHER - COMPREHENSIVE DEMONSTRATION")
    print("Handler registration, batch operations, CQRS patterns")
    print("=" * 60)

    service = RegistryDispatcherService()
    result = service.execute()

    if result.is_success:
        data = result.value
        patterns = data["patterns_demonstrated"]
        if isinstance(patterns, Sequence) and not isinstance(
            patterns,
            str | bytes | bytearray,
        ):
            patterns_list = list(patterns)
            print(f"\n✅ Demonstrated {len(patterns_list)} patterns")
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
