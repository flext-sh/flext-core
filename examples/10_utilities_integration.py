#!/usr/bin/env python3
"""10 - FLEXT Core Utilities Integration: Manual Component Setup Patterns.

This comprehensive example demonstrates MANUAL integration patterns for flext-core
components WITHOUT using FlextMixins.Service inheritance. This contrasts with
Examples 01-09 which demonstrate inherited infrastructure patterns.

**IMPORTANT**: This example shows MANUAL setup for advanced integration scenarios.
For standard service development, use FlextMixins.Service inheritance as shown
in Examples 01-09 to get automatic infrastructure (logger, container, context, etc.).

Key Integration Patterns Demonstrated:
- FlextCore.Config: Manual configuration integration with computed fields and validators
- FlextCore.Container: Manual dependency injection without inherited container property
- FlextCore.Bus: Manual event-driven architecture setup
- FlextCore.Dispatcher: Manual message routing and dispatching
- FlextLogger: Manual logger instantiation vs inherited logger property
- Complete service integration with manual component wiring

**When to use Manual vs Inherited patterns**:
- Manual (this example): Advanced scenarios requiring fine-grained component control
- Inherited (Examples 01-09): Standard service development with automatic infrastructure

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import cast

from flext_core import FlextContainer, FlextCore, FlextLogger, FlextResult, FlextTypes


class MyService:
    """Example service demonstrating FlextCore.Config integration."""

    def __init__(self) -> None:
        """Initialize service with configuration."""
        super().__init__()
        self._config = FlextCore.create_config()

        # Use computed fields for enhanced configuration
        self._timeout = self._config.timeout_seconds
        self._max_retries = self._config.max_retry_attempts
        self._debug_mode = self._config.is_debug_enabled

        # Use component-specific configuration
        container_config = self._config.get_component_config("container")
        if container_config.is_success:
            self._container_settings = container_config.unwrap()

    def process_data(self, data: dict[str, object]) -> FlextResult[FlextTypes.Dict]:
        """Process data with configuration-driven validation."""
        # Use configuration-driven validation
        if len(data) > self._config.max_batch_size:
            return FlextResult[FlextTypes.Dict].fail("Data batch too large")

        # Process with configuration-driven settings
        return FlextResult[FlextTypes.Dict].ok({
            "processed": True,
            "config_used": self._debug_mode,
        })


class DatabaseService:
    """Example database service demonstrating FlextCore.Container integration."""

    def __init__(self) -> None:
        """Initialize database service."""
        super().__init__()
        self._connection: str | None = None

    def connect(self) -> FlextResult[None]:
        """Connect to database with proper error handling."""
        # Simulate connection logic
        self._connection = "connected"
        return FlextResult[None].ok(None)


class UserService:
    """Example user service demonstrating dependency injection."""

    def __init__(self) -> None:
        """Initialize user service with dependency injection."""
        super().__init__()
        self._container = FlextContainer.get_global()

        # Register services with proper error handling
        db_result = self._container.register("database", DatabaseService())
        if db_result.is_failure:
            msg = f"Failed to register database service: {db_result.error}"
            raise FlextCore.Exceptions.ConfigurationError(msg)

    def get_user(self, user_id: str) -> FlextResult[FlextTypes.Dict]:
        """Get user with dependency injection and error handling."""
        # Retrieve service with error handling
        db_result = self._container.get("database")
        if db_result.is_failure:
            return FlextResult[FlextTypes.Dict].fail(
                f"Database service unavailable: {db_result.error}"
            )

        db_service = db_result.unwrap()
        return FlextResult[FlextTypes.Dict].ok({
            "user_id": user_id,
            "service": type(db_service).__name__,
        })


class EventService:
    """Example event service demonstrating FlextCore.Bus integration."""

    def __init__(self) -> None:
        """Initialize event service with bus integration."""
        super().__init__()
        self._bus = FlextCore.Bus()
        self._events: list[dict[str, object]] = []

    def publish_event(
        self, event_type: str, data: dict[str, object]
    ) -> FlextResult[None]:
        """Publish event with proper error handling."""
        try:
            # FlextCore.Bus.publish_event takes a single event object
            event: dict[str, object] = {"type": event_type, **data}
            result = self._bus.publish_event(event)
            if result.is_failure:
                return result

            self._events.append({"type": event_type, "data": data, "timestamp": "now"})
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to publish event: {e}")

    def get_events(self) -> FlextResult[list[dict[str, object]]]:
        """Get events with railway pattern."""
        if not self._events:
            return FlextResult[list[dict[str, object]]].fail("No events available")

        return FlextResult[list[dict[str, object]]].ok(self._events)


class EventHandler:
    """Example event handler demonstrating event processing."""

    def __init__(self, event_service: EventService) -> None:
        """Initialize event handler."""
        super().__init__()
        self._event_service = event_service

    def handle_user_created(self, event_data: dict[str, object]) -> FlextResult[None]:
        """Handle user created event with validation."""
        user_id = event_data.get("user_id")
        if not user_id:
            return FlextResult[None].fail("Invalid event data: missing user_id")

        # Process event
        return self._event_service.publish_event("user_processed", {"user_id": user_id})


class MessageService:
    """Example message service demonstrating FlextCore.Dispatcher integration."""

    def __init__(self) -> None:
        """Initialize message service with dispatcher."""
        super().__init__()
        self._dispatcher = FlextCore.Dispatcher()
        self._messages: list[dict[str, object]] = []

    def send_message(self, message: dict[str, object]) -> FlextResult[str]:
        """Send message with proper error handling."""
        try:
            message_id = f"msg_{len(self._messages)}"
            enhanced_message = cast(
                "dict[str, object]", {**message, "id": message_id, "timestamp": "now"}
            )

            # Dispatch with error handling
            self._dispatcher.dispatch(enhanced_message)
            self._messages.append(enhanced_message)

            return FlextResult[str].ok(message_id)
        except Exception as e:
            return FlextResult[str].fail(f"Failed to send message: {e}")

    def get_messages(self) -> FlextResult[list[dict[str, object]]]:
        """Get messages with railway pattern."""
        if not self._messages:
            return FlextResult[list[dict[str, object]]].fail("No messages available")

        return FlextResult[list[dict[str, object]]].ok(self._messages)


class MessageHandler:
    """Example message handler demonstrating message processing."""

    def __init__(self, message_service: MessageService) -> None:
        """Initialize message handler."""
        super().__init__()
        self._message_service = message_service

    def process_message(
        self, message: dict[str, object]
    ) -> FlextResult[dict[str, object]]:
        """Process message with validation."""
        msg_type = message.get("type")
        if not msg_type:
            return FlextResult[dict[str, object]].fail("Message missing type")

        # Process based on type
        if msg_type == "greeting":
            return FlextResult[dict[str, object]].ok({
                "response": "Hello!",
                "original": message,
            })
        return FlextResult[dict[str, object]].ok({
            "response": "Processed",
            "type": msg_type,
        })


class IntegratedService:
    """Complete service integration demonstrating MANUAL component setup.

    This service demonstrates MANUAL integration of all flext-core components
    WITHOUT using FlextMixins.Service inheritance. This contrasts with the
    inherited infrastructure pattern shown in Examples 01-09.

    **IMPORTANT**: This manual setup is for advanced scenarios. For standard
    service development, use FlextMixins.Service to get automatic infrastructure:
    - Inherited self.logger (no manual FlextLogger instantiation)
    - Inherited self.container (no manual FlextContainer.get_global())
    - Inherited self.context (automatic request/correlation tracking)
    - Inherited self.config (automatic FlextCore.Config with settings)
    - Inherited self.metrics (automatic observability)

    See Examples 01-09 for the recommended FlextMixins.Service inheritance pattern.
    """

    def __init__(self) -> None:
        """Initialize with MANUAL component setup.

        This demonstrates manual wiring of all components. Compare with
        FlextMixins.Service inheritance which provides all these automatically
        via inherited properties (see Examples 01-09).
        """
        super().__init__()
        # MANUAL setup - compare with FlextMixins.Service automatic inheritance:
        self._config = FlextCore.create_config()  # vs inherited self.config
        self._container = FlextContainer.get_global()  # vs inherited self.container
        self._bus = FlextCore.Bus()  # vs inherited self.bus
        self._dispatcher = FlextCore.Dispatcher()  # vs inherited self.dispatcher
        self.logger = FlextCore.create_logger(__name__)  # vs inherited self.logger

        # Register this service in the container
        self._container.register("integrated_service", self)

    def process_request(
        self, request: dict[str, object]
    ) -> FlextResult[dict[str, object]]:
        """Process request with full flext-core integration."""
        # Log request with structured logging
        self.logger.info(
            "Processing request",
            extra={
                "request_size": len(str(request)),
                "request_type": request.get("type", "unknown"),
            },
        )

        # Validate using configuration
        if len(str(request)) > self._config.validation_timeout_ms:
            return FlextResult[dict[str, object]].fail("Request too large")

        # Publish processing event
        event = cast(
            "dict[str, object]",
            {
                "type": "request_processing",
                "request_id": request.get("id", "unknown"),
                "timestamp": "now",
            },
        )
        event_result = self._bus.publish_event(event)
        if event_result.is_failure:
            self.logger.warning(
                "Failed to publish processing event",
                extra={"error": event_result.error},
            )

        # Process with dispatcher
        response = cast(
            "dict[str, object]", {"processed": True, "request_id": request.get("id")}
        )

        # Send response through dispatcher
        dispatch_result = self._dispatcher.dispatch(
            cast(
                "dict[str, object]",
                {
                    "type": "response",
                    "data": response,
                    "original_request": request.get("id"),
                },
            )
        )

        if dispatch_result.is_failure:
            return FlextResult[dict[str, object]].fail(
                f"Failed to dispatch response: {dispatch_result.error}"
            )

        return FlextResult[dict[str, object]].ok(response)

    def get_service_status(self) -> FlextResult[dict[str, object]]:
        """Get comprehensive service status with flext-core integration."""
        # Get component status
        config_status = self._config.metadata_config
        container_status = "available" if self._container else "unavailable"
        bus_status = "available" if self._bus else "unavailable"
        dispatcher_status = "available" if self._dispatcher else "unavailable"

        return FlextResult[dict[str, object]].ok({
            "service_name": "IntegratedService",
            "status": "running",
            "components": {
                "config": config_status,
                "container": container_status,
                "bus": bus_status,
                "dispatcher": dispatcher_status,
                "logger": "available",
            },
            "integration_level": "complete",
        })


def run_integration_examples() -> None:
    """Run comprehensive integration examples."""
    print("🔧 Running FLEXT Core Integration Examples")
    print("=" * 50)

    # Example 1: Configuration integration
    print("\n1. Configuration Integration:")
    service = MyService()
    config_result = service.process_data({"test": "data"})
    print(f"   Config-driven processing: {config_result.is_success}")
    if config_result.is_success:
        print(f"   Debug mode used: {config_result.unwrap()['config_used']}")

    # Example 2: Dependency injection
    print("\n2. Dependency Injection:")
    user_service = UserService()
    user_result = user_service.get_user("user123")
    print(f"   DI pattern: {user_result.is_success}")
    if user_result.is_success:
        print(f"   Service used: {user_result.unwrap()['service']}")

    # Example 3: Event-driven architecture
    print("\n3. Event-Driven Architecture:")
    event_service = EventService()
    event_result = event_service.publish_event("user_created", {"user_id": "user123"})
    print(f"   Event publishing: {event_result.is_success}")

    # Example 4: Message routing
    print("\n4. Message Routing:")
    message_service = MessageService()
    msg_result = message_service.send_message({"type": "greeting", "content": "Hello"})
    print(f"   Message dispatching: {msg_result.is_success}")

    # Example 5: Complete integration
    print("\n5. Complete Integration:")
    integrated_service = IntegratedService()
    status_result = integrated_service.get_service_status()
    print(f"   Full integration status: {status_result.is_success}")
    if status_result.is_success:
        status = status_result.unwrap()
        print(f"   Integration level: {status['integration_level']}")
        components = status["components"]
        if isinstance(components, dict):
            print(
                f"   Components available: {len(cast('dict[str, object]', components))}"
            )
        else:
            print("   Components data unavailable")

    print("\n✅ All integration examples completed successfully!")
    print("\nThese examples demonstrate:")
    print("  • Configuration-driven behavior")
    print("  • Dependency injection patterns")
    print("  • Event-driven architecture")
    print("  • Message routing capabilities")
    print("  • Structured logging practices")
    print("  • Railway-oriented error handling")

    # Demonstrate new FlextResult methods (v0.9.9+)
    demonstrate_new_flextresult_methods()


def demonstrate_new_flextresult_methods() -> None:
    """Demonstrate new FlextResult methods (v0.9.9+)."""
    # 4. alt - Utility Provider Alternatives
    print("\n=== 4. alt: Utility Provider Alternatives ===")

    def get_custom_logger() -> FlextResult[FlextLogger]:
        """Try to get custom configured logger."""
        return FlextResult[FlextLogger].fail("Custom logger not configured")

    def get_default_logger() -> FlextResult[FlextLogger]:
        """Provide default logger as fallback."""
        logger = FlextCore.create_logger("utilities_integration")
        return FlextResult[FlextLogger].ok(logger)

    # Try custom, fall back to default
    logger_result = get_custom_logger().alt(get_default_logger())
    if logger_result.is_success:
        logger = logger_result.unwrap()
        print(f"✅ Logger acquired: {type(logger).__name__}")
        logger.info("Logger fallback test successful")
    else:
        print(f"❌ No logger available: {logger_result.error}")

    # 5. value_or_call - Lazy Utility Loading
    print("\n=== 5. value_or_call: Lazy Utility Loading ===")

    def create_expensive_container() -> FlextCore.Container:
        """Create and configure a new container (expensive operation)."""
        print("   ⚙️  Creating new container with full configuration...")
        container = FlextCore.Container()
        container.register("logger", FlextCore.create_logger("lazy_container"))
        container.register("config", FlextCore.create_config())
        return container

    # Try to get existing container, create new one if not available
    container_fail_result = FlextResult[FlextCore.Container].fail(
        "No existing container"
    )
    container = container_fail_result.value_or_call(create_expensive_container)
    print(f"✅ Container acquired: {type(container).__name__}")
    print(f"   Container has services: {hasattr(container, '_services')}")

    # Try again with successful result (lazy function NOT called)
    container_success_result = FlextResult[FlextContainer].ok(
        FlextContainer.get_global()
    )
    container_cached = container_success_result.value_or_call(
        create_expensive_container
    )
    print(f"✅ Existing container used: {type(container_cached).__name__}")
    print("   No expensive creation needed")

    print("\n" + "=" * 60)
    print("✅ NEW FlextResult METHODS UTILITIES INTEGRATION DEMO COMPLETE!")
    print("All 5 methods demonstrated with complete utility integration context")
    print("=" * 60)


if __name__ == "__main__":
    run_integration_examples()
