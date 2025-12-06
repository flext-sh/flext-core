"""FlextRegistry and FlextDispatcher comprehensive demonstration.

Demonstrates handler registration, batch operations, dispatcher patterns,
and CQRS integration using Python 3.13+ strict patterns with PEP 695
type aliases and collections.abc.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from flext_core import (
    FlextConstants,
    FlextDispatcher,
    FlextModels,
    FlextRegistry,
    FlextResult,
    h,
    s,
    t,
    u,
)
from flext_core.protocols import p

# ═══════════════════════════════════════════════════════════════════
# HANDLER IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════


class CreateUserCommand(FlextModels.Cqrs.Command):
    """Create user command."""

    name: str
    email: str


class UserCreatedEvent(FlextModels.DomainEvent):
    """User created event."""

    event_type: str = "user_created"
    aggregate_id: str
    name: str


class CreateUserHandler(h[CreateUserCommand, UserCreatedEvent]):
    """Handler for creating users."""

    def handle(self, message: CreateUserCommand) -> FlextResult[UserCreatedEvent]:
        """Handle create user command."""
        user_id = u.Generators.generate_entity_id()
        return FlextResult[UserCreatedEvent].ok(
            UserCreatedEvent(
                aggregate_id=user_id,
                name=message.name,
            ),
        )


class GetUserQuery(FlextModels.Cqrs.Query):
    """Get user query."""

    user_id: str


class GetUserHandler(h[GetUserQuery, t.Types.ServiceMetadataMapping]):
    """Handler for getting users."""

    def handle(
        self,
        message: GetUserQuery,
    ) -> FlextResult[t.Types.ServiceMetadataMapping]:
        """Handle get user query."""
        return FlextResult[t.Types.ServiceMetadataMapping].ok({
            "user_id": message.user_id,
            "name": "Demo User",
            "email": "demo@example.com",
        })


# ═══════════════════════════════════════════════════════════════════
# SERVICE IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════


class RegistryDispatcherService(s[t.Types.ServiceMetadataMapping]):
    """Service demonstrating FlextRegistry and FlextDispatcher."""

    def execute(
        self,
    ) -> FlextResult[t.Types.ServiceMetadataMapping]:
        """Execute registry and dispatcher demonstrations."""
        print("Starting registry and dispatcher demonstration")

        try:
            self._demonstrate_registry()
            self._demonstrate_dispatcher()
            self._demonstrate_integration()

            return FlextResult[t.Types.ServiceMetadataMapping].ok({
                "patterns_demonstrated": [
                    "handler_registration",
                    "batch_registration",
                    "command_dispatch",
                    "query_dispatch",
                    "registry_integration",
                ],
                "handler_types": [
                    FlextConstants.Cqrs.HandlerType.COMMAND.value,
                    FlextConstants.Cqrs.HandlerType.QUERY.value,
                ],
                "features": [
                    "idempotent_registration",
                    "batch_operations",
                    "dispatcher_integration",
                    "cqrs_patterns",
                ],
            })

        except Exception as e:
            error_msg = f"Registry/Dispatcher demonstration failed: {e}"
            return FlextResult[t.Types.ServiceMetadataMapping].fail(error_msg)

    @staticmethod
    def _demonstrate_registry() -> None:
        """Show registry operations."""
        print("\n=== Registry Operations ===")

        registry = FlextRegistry()

        # Register single handler
        create_handler = CreateUserHandler()
        # Cast handler to compatible type for registry
        handler_for_registry = cast(
            "h[t.GeneralValueType, t.GeneralValueType]",
            create_handler,
        )
        register_result = registry.register_handler(handler_for_registry)
        if register_result.is_success:
            print("✅ Handler registered successfully")

        # Batch registration
        get_handler = GetUserHandler()
        # Business Rule: register_handlers accepts list of handlers compatible with Handler protocol
        # Handler types implement Handler protocol and are compatible at runtime
        # Cast needed because GetUserHandler[GetUserQuery, ServiceMetadataMapping] is compatible
        # with h[GeneralValueType, GeneralValueType] at runtime
        handler_for_registry = cast(
            "h[t.GeneralValueType, t.GeneralValueType]",
            get_handler,
        )
        batch_result = registry.register_handlers([handler_for_registry])
        if batch_result.is_success:
            summary = batch_result.unwrap()
            print(f"✅ Batch registration: {summary.successful_registrations} handlers")

    @staticmethod
    def _demonstrate_dispatcher() -> None:
        """Show dispatcher operations."""
        print("\n=== Dispatcher Operations ===")

        dispatcher = FlextDispatcher()
        registry = FlextRegistry(
            dispatcher=cast("p.Application.CommandBus | None", dispatcher)
        )

        # Register handlers
        # Business Rule: register_handler accepts handlers compatible with Handler protocol
        # Handler types implement Handler protocol and are compatible at runtime
        create_handler = CreateUserHandler()
        # Cast needed because CreateUserHandler[CreateUserCommand, UserCreatedEvent] is compatible
        # with h[GeneralValueType, GeneralValueType] at runtime
        handler_for_registry = cast(
            "h[t.GeneralValueType, t.GeneralValueType]",
            create_handler,
        )
        _ = registry.register_handler(handler_for_registry)

        # Dispatch command
        command = CreateUserCommand(name="Alice", email="alice@example.com")
        # Business Rule: dispatch accepts commands/queries compatible with GeneralValueType
        # Pydantic models implement model_dump() and are compatible at runtime
        # Cast needed because CreateUserCommand is compatible with GeneralValueType at runtime
        command_for_dispatch = cast("t.GeneralValueType", command)
        dispatch_result = dispatcher.dispatch(command_for_dispatch)
        if dispatch_result.is_success:
            event_value = dispatch_result.unwrap()
            if isinstance(event_value, UserCreatedEvent):
                print(f"✅ Command dispatched: {event_value.aggregate_id}")
            else:
                print("✅ Command dispatched successfully")

    @staticmethod
    def _demonstrate_integration() -> None:
        """Show registry and dispatcher integration."""
        print("\n=== Registry/Dispatcher Integration ===")

        dispatcher = FlextDispatcher()
        registry = FlextRegistry(
            dispatcher=cast("p.Application.CommandBus | None", dispatcher)
        )

        # Register handlers
        create_handler = CreateUserHandler()
        get_handler = GetUserHandler()

        # Business Rule: register_handler accepts handlers compatible with Handler protocol
        # Handler types implement Handler protocol and are compatible at runtime
        # Cast needed because handlers are compatible with h[GeneralValueType, GeneralValueType] at runtime
        create_handler_for_registry = cast(
            "h[t.GeneralValueType, t.GeneralValueType]",
            create_handler,
        )
        get_handler_for_registry = cast(
            "h[t.GeneralValueType, t.GeneralValueType]",
            get_handler,
        )
        _ = registry.register_handler(create_handler_for_registry)
        _ = registry.register_handler(get_handler_for_registry)

        # Dispatch command
        command = CreateUserCommand(name="Bob", email="bob@example.com")
        # Business Rule: dispatch accepts commands/queries compatible with GeneralValueType
        # Pydantic models implement model_dump() and are compatible at runtime
        # Cast needed because CreateUserCommand is compatible with GeneralValueType at runtime
        command_for_dispatch = cast("t.GeneralValueType", command)
        command_result = dispatcher.dispatch(command_for_dispatch)
        if command_result.is_success:
            print("✅ Command dispatched successfully")

        # Dispatch query
        query = GetUserQuery(user_id="user-123")
        # Business Rule: dispatch accepts commands/queries compatible with GeneralValueType
        # Pydantic models implement model_dump() and are compatible at runtime
        # Cast needed because GetUserQuery is compatible with GeneralValueType at runtime
        query_for_dispatch = cast("t.GeneralValueType", query)
        query_result = dispatcher.dispatch(query_for_dispatch)
        if query_result.is_success:
            user_data = query_result.unwrap()
            if isinstance(user_data, dict):
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
        data = result.unwrap()
        patterns = data["patterns_demonstrated"]
        if isinstance(patterns, Sequence):
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
