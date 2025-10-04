#!/usr/bin/env python3
"""FLEXT Core Utilities Integration Examples.

Comprehensive examples demonstrating how all flext-core components work together
in realistic scenarios. These examples were moved from utilities.py to maintain
the single-class-per-module pattern.

This file demonstrates:
- FlextConfig integration with computed fields and validators
- FlextContainer dependency injection patterns
- FlextBus event-driven architecture
- FlextDispatcher message routing
- FlextLogger structured logging
- Complete service integration patterns

Usage:
    python examples/utilities_integration_examples.py
"""

from __future__ import annotations

from flext_core import (
    FlextBus,
    FlextConfig,
    FlextContainer,
    FlextDispatcher,
    FlextExceptions,
    FlextLogger,
    FlextResult,
    FlextTypes,
)


class MyService:
    """Example service demonstrating FlextConfig integration."""

    def __init__(self) -> None:
        """Initialize service with configuration."""
        self._config = FlextConfig()

        # Use computed fields for enhanced configuration
        self._timeout = self._config.timeout_seconds
        self._max_retries = self._config.max_retry_attempts
        self._debug_mode = self._config.is_debug_enabled

        # Use component-specific configuration
        container_config = self._config.get_component_config("container")
        if container_config.is_success:
            self._container_settings = container_config.unwrap()

    def process_data(self, data: dict) -> FlextResult[FlextTypes.Dict]:
        """Process data with configuration-driven validation."""
        # Use configuration-driven validation
        if len(data) > self._config.batch_size:
            return FlextResult[FlextTypes.Dict].fail("Data batch too large")

        # Process with configuration-driven settings
        return FlextResult[FlextTypes.Dict].ok({
            "processed": True,
            "config_used": self._debug_mode,
        })


class DatabaseService:
    """Example database service demonstrating FlextContainer integration."""

    def __init__(self) -> None:
        """Initialize database service."""
        super().__init__()
        self._connection = None

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
            raise FlextExceptions.ConfigurationError(msg)

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
    """Example event service demonstrating FlextBus integration."""

    def __init__(self) -> None:
        """Initialize event service with bus integration."""
        super().__init__()
        self._bus = FlextBus()
        self._events = []

    def publish_event(self, event_type: str, data: dict) -> FlextResult[None]:
        """Publish event with proper error handling."""
        try:
            # FlextBus.publish_event takes a single event object
            event = {"type": event_type, **data}
            result = self._bus.publish_event(event)
            if result.is_failure:
                return result

            self._events.append({"type": event_type, "data": data, "timestamp": "now"})
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to publish event: {e}")

    def get_events(self) -> FlextResult[list]:
        """Get events with railway pattern."""
        if not self._events:
            return FlextResult[list].fail("No events available")

        return FlextResult[list].ok(self._events)


class EventHandler:
    """Example event handler demonstrating event processing."""

    def __init__(self, event_service: EventService) -> None:
        """Initialize event handler."""
        self._event_service = event_service

    def handle_user_created(self, event_data: dict) -> FlextResult[None]:
        """Handle user created event with validation."""
        user_id = event_data.get("user_id")
        if not user_id:
            return FlextResult[None].fail("Invalid event data: missing user_id")

        # Process event
        return self._event_service.publish_event("user_processed", {"user_id": user_id})


class MessageService:
    """Example message service demonstrating FlextDispatcher integration."""

    def __init__(self) -> None:
        """Initialize message service with dispatcher."""
        super().__init__()
        self._dispatcher = FlextDispatcher()
        self._messages = []

    def send_message(self, message: dict) -> FlextResult[str]:
        """Send message with proper error handling."""
        try:
            message_id = f"msg_{len(self._messages)}"
            enhanced_message = {**message, "id": message_id, "timestamp": "now"}

            # Dispatch with error handling
            self._dispatcher.dispatch(enhanced_message)
            self._messages.append(enhanced_message)

            return FlextResult[str].ok(message_id)
        except Exception as e:
            return FlextResult[str].fail(f"Failed to send message: {e}")

    def get_messages(self) -> FlextResult[list]:
        """Get messages with railway pattern."""
        if not self._messages:
            return FlextResult[list].fail("No messages available")

        return FlextResult[list].ok(self._messages)


class MessageHandler:
    """Example message handler demonstrating message processing."""

    def __init__(self, message_service: MessageService) -> None:
        """Initialize message handler."""
        self._message_service = message_service

    def process_message(self, message: dict) -> FlextResult[FlextTypes.Dict]:
        """Process message with validation."""
        msg_type = message.get("type")
        if not msg_type:
            return FlextResult[FlextTypes.Dict].fail("Message missing type")

        # Process based on type
        if msg_type == "greeting":
            return FlextResult[FlextTypes.Dict].ok({
                "response": "Hello!",
                "original": message,
            })
        return FlextResult[FlextTypes.Dict].ok({
            "response": "Processed",
            "type": msg_type,
        })


class IntegratedService:
    """Complete service integration demonstrating all flext-core components."""

    def __init__(self) -> None:
        """Initialize integrated service with all components."""
        super().__init__()
        self._config = FlextConfig()
        self._container = FlextContainer.get_global()
        self._bus = FlextBus()
        self._dispatcher = FlextDispatcher()
        self._logger = FlextLogger(__name__)

        # Register this service in the container
        self._container.register("integrated_service", self)

    async def process_request(self, request: dict) -> FlextResult[FlextTypes.Dict]:
        """Process request with full flext-core integration."""
        # Log request with structured logging
        self._logger.info(
            "Processing request",
            extra={
                "request_size": len(str(request)),
                "request_type": request.get("type", "unknown"),
            },
        )

        # Validate using configuration
        if len(str(request)) > self._config.validation_timeout_ms:
            return FlextResult[FlextTypes.Dict].fail("Request too large")

        # Publish processing event
        event = {
            "type": "request_processing",
            "request_id": request.get("id", "unknown"),
            "timestamp": "now",
        }
        event_result = self._bus.publish_event(event)
        if event_result.is_failure:
            self._logger.warning(
                "Failed to publish processing event",
                extra={"error": event_result.error},
            )

        # Process with dispatcher
        response = {"processed": True, "request_id": request.get("id")}

        # Send response through dispatcher
        dispatch_result = self._dispatcher.dispatch({
            "type": "response",
            "data": response,
            "original_request": request.get("id"),
        })

        if dispatch_result.is_failure:
            return FlextResult[FlextTypes.Dict].fail(
                f"Failed to dispatch response: {dispatch_result.error}"
            )

        return FlextResult[FlextTypes.Dict].ok(response)

    def get_service_status(self) -> FlextResult[FlextTypes.Dict]:
        """Get comprehensive service status with flext-core integration."""
        # Get component status
        config_status = self._config.metadata_config
        container_status = "available" if self._container else "unavailable"
        bus_status = "available" if self._bus else "unavailable"
        dispatcher_status = "available" if self._dispatcher else "unavailable"

        return FlextResult[FlextTypes.Dict].ok({
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
        print(f"   Components available: {len(status['components'])}")

    print("\n✅ All integration examples completed successfully!")
    print("\nThese examples demonstrate:")
    print("  • Configuration-driven behavior")
    print("  • Dependency injection patterns")
    print("  • Event-driven architecture")
    print("  • Message routing capabilities")
    print("  • Structured logging practices")
    print("  • Railway-oriented error handling")


if __name__ == "__main__":
    run_integration_examples()
