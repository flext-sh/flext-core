"""FlextRegistry and FlextDispatcher comprehensive demonstration.

Demonstrates handler registration, batch operations, dispatcher patterns,
and CQRS integration using Python 3.13+ strict patterns with PEP 695
type aliases and collections.abc.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence

from flext_core import (
    FlextConstants,
    FlextDispatcher,
    FlextHandlers,
    FlextModels,
    FlextRegistry,
    FlextResult,
    FlextService,
    FlextTypes,
    FlextUtilities,
)

# ═══════════════════════════════════════════════════════════════════
# HANDLER IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════


class CreateUserCommand(FlextModels.Cqrs.Command):  # type: ignore[misc,valid-type]
    """Create user command."""

    name: str
    email: str


class UserCreatedEvent(FlextModels.DomainEvent):  # type: ignore[misc,valid-type]
    """User created event."""

    event_type: str = "user_created"
    aggregate_id: str
    name: str


class CreateUserHandler(FlextHandlers[CreateUserCommand, UserCreatedEvent]):
    """Handler for creating users."""

    def handle(  # noqa: PLR6301  # Required by FlextHandlers interface
        self, message: CreateUserCommand
    ) -> FlextResult[UserCreatedEvent]:
        """Handle create user command."""
        user_id = FlextUtilities.Generators.generate_entity_id()
        return FlextResult[UserCreatedEvent].ok(
            UserCreatedEvent(
                aggregate_id=user_id,
                name=message.name,
            )
        )


class GetUserQuery(FlextModels.Cqrs.Query):  # type: ignore[misc,valid-type]
    """Get user query."""

    user_id: str


class GetUserHandler(
    FlextHandlers[GetUserQuery, FlextTypes.Types.ServiceMetadataMapping]
):
    """Handler for getting users."""

    def handle(  # noqa: PLR6301  # Required by FlextHandlers interface
        self, message: GetUserQuery
    ) -> FlextResult[FlextTypes.Types.ServiceMetadataMapping]:
        """Handle get user query."""
        return FlextResult[FlextTypes.Types.ServiceMetadataMapping].ok({
            "user_id": message.user_id,
            "name": "Demo User",
            "email": "demo@example.com",
        })


# ═══════════════════════════════════════════════════════════════════
# SERVICE IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════


class RegistryDispatcherService(FlextService[FlextTypes.Types.ServiceMetadataMapping]):
    """Service demonstrating FlextRegistry and FlextDispatcher."""

    def execute(
        self,
    ) -> FlextResult[FlextTypes.Types.ServiceMetadataMapping]:
        """Execute registry and dispatcher demonstrations."""
        print("Starting registry and dispatcher demonstration")

        try:
            self._demonstrate_registry()
            self._demonstrate_dispatcher()
            self._demonstrate_integration()

            return FlextResult[FlextTypes.Types.ServiceMetadataMapping].ok({
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
            return FlextResult[FlextTypes.Types.ServiceMetadataMapping].fail(error_msg)

    @staticmethod
    def _demonstrate_registry() -> None:
        """Show registry operations."""
        print("\n=== Registry Operations ===")

        registry = FlextRegistry()

        # Register single handler
        create_handler = CreateUserHandler()
        register_result = registry.register_handler(create_handler)  # type: ignore[arg-type]  # Handler types are more specific than object, object
        if register_result.is_success:
            print("✅ Handler registered successfully")

        # Batch registration
        get_handler = GetUserHandler()
        batch_result = registry.register_handlers([get_handler])  # type: ignore[arg-type,list-item]  # Handler types are more specific than object, object
        if batch_result.is_success:
            summary = batch_result.unwrap()
            print(f"✅ Batch registration: {summary.successful_registrations} handlers")

    @staticmethod
    def _demonstrate_dispatcher() -> None:
        """Show dispatcher operations."""
        print("\n=== Dispatcher Operations ===")

        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        # Register handlers
        create_handler = CreateUserHandler()
        registry.register_handler(create_handler)  # type: ignore[arg-type]  # Handler types are more specific than object, object

        # Dispatch command
        command = CreateUserCommand(name="Alice", email="alice@example.com")
        dispatch_result = dispatcher.dispatch(command)  # type: ignore[arg-type]  # Pydantic models are compatible with GeneralValueType at runtime
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
        registry = FlextRegistry(dispatcher=dispatcher)

        # Register handlers
        create_handler = CreateUserHandler()
        get_handler = GetUserHandler()

        registry.register_handler(create_handler)  # type: ignore[arg-type]  # Handler types are more specific than object, object
        registry.register_handler(get_handler)  # type: ignore[arg-type]  # Handler types are more specific than object, object

        # Dispatch command
        command = CreateUserCommand(name="Bob", email="bob@example.com")
        command_result = dispatcher.dispatch(command)  # type: ignore[arg-type]  # Pydantic models are compatible with GeneralValueType at runtime
        if command_result.is_success:
            print("✅ Command dispatched successfully")

        # Dispatch query
        query = GetUserQuery(user_id="user-123")
        query_result = dispatcher.dispatch(query)  # type: ignore[arg-type]  # Pydantic models are compatible with GeneralValueType at runtime
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
