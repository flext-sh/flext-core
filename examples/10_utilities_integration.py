#!/usr/bin/env python3
"""10 - FLEXT Core Utilities Integration: Manual Component Setup Patterns.

This comprehensive example demonstrates MANUAL integration patterns for flext-core
components WITHOUT using FlextMixins.Service inheritance. This contrasts with
Examples 01-09 which demonstrate inherited infrastructure patterns.

**IMPORTANT**: This example shows MANUAL setup for advanced integration scenarios.
For standard service development, use FlextMixins.Service inheritance as shown
in Examples 01-09 to get automatic infrastructure (logger, container, context, etc.).

Key Integration Patterns Demonstrated:
- FlextConfig: Manual configuration integration with computed fields and validators
- FlextContainer: Manual dependency injection without inherited container property
- FlextBus: Manual event-driven architecture setup
- FlextDispatcher: Manual message routing and dispatching
- FlextLogger: Manual logger instantiation vs inherited logger property
- Complete service integration with manual component wiring

**When to use Manual vs Inherited patterns**:
- Manual (this example): Advanced scenarios requiring fine-grained component control
- Inherited (Examples 01-09): Standard service development with automatic infrastructure

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
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
    """Complete service integration demonstrating MANUAL component setup.

    This service demonstrates MANUAL integration of all flext-core components
    WITHOUT using FlextMixins.Service inheritance. This contrasts with the
    inherited infrastructure pattern shown in Examples 01-09.

    **IMPORTANT**: This manual setup is for advanced scenarios. For standard
    service development, use FlextMixins.Service to get automatic infrastructure:
    - Inherited self.logger (no manual FlextLogger instantiation)
    - Inherited self.container (no manual FlextContainer.get_global())
    - Inherited self.context (automatic request/correlation tracking)
    - Inherited self.config (automatic FlextConfig with settings)
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
        self._config = FlextConfig()                    # vs inherited self.config
        self._container = FlextContainer.get_global()   # vs inherited self.container
        self._bus = FlextBus()                          # vs inherited self.bus
        self._dispatcher = FlextDispatcher()            # vs inherited self.dispatcher
        self._logger = FlextLogger(__name__)            # vs inherited self.logger

        # Register this service in the container
        self._container.register("integrated_service", self)

    def process_request(self, request: dict) -> FlextResult[FlextTypes.Dict]:
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


def demonstrate_new_flextresult_methods() -> None:
    """Demonstrate the 5 new FlextResult methods in utilities integration context.
    
    Shows how the new v0.9.9+ methods work with complete utilities integration:
    - from_callable: Safe utility operations
    - flow_through: Utility pipeline composition
    - lash: Utility fallback recovery
    - alt: Utility provider alternatives
    - value_or_call: Lazy utility loading
    """
    print("\n" + "=" * 60)
    print("NEW FLEXTRESULT METHODS - UTILITIES INTEGRATION CONTEXT")
    print("Demonstrating v0.9.9+ methods with all flext-core utilities")
    print("=" * 60)

    # 1. from_callable - Safe Utility Operations
    print("\n=== 1. from_callable: Safe Utility Operations ===")

    def risky_config_operation() -> dict:
        """Configuration operation that might raise exceptions."""
        config = FlextConfig()
        if not hasattr(config, 'batch_size'):
            msg = "Configuration incomplete"
            raise FlextExceptions.ConfigurationError(msg)
        return {
            "batch_size": config.batch_size,
            "timeout": config.timeout_seconds,
            "debug": config.is_debug_enabled,
        }

    # Safe config access without try/except
    config_result = FlextResult.from_callable(risky_config_operation)
    if config_result.is_success:
        config_data = config_result.unwrap()
        print(f"✅ Configuration loaded safely: batch_size={config_data['batch_size']}")
    else:
        print(f"❌ Configuration loading failed: {config_result.error}")

    # 2. flow_through - Utility Pipeline Composition
    print("\n=== 2. flow_through: Utility Pipeline Composition ===")

    def validate_request_data(data: dict) -> FlextResult[dict]:
        """Validate incoming request data."""
        if not data:
            return FlextResult[dict].fail("Request data cannot be empty")
        if not data.get("id"):
            return FlextResult[dict].fail("Request ID required")
        return FlextResult[dict].ok(data)

    def enrich_with_config(data: dict) -> FlextResult[dict]:
        """Enrich request with configuration settings."""
        config = FlextConfig()
        enriched = {
            **data,
            "batch_size": config.batch_size,
            "timeout_ms": config.timeout_seconds * 1000,
        }
        return FlextResult[dict].ok(enriched)

    def register_in_container(data: dict) -> FlextResult[dict]:
        """Register request context in DI container."""
        container = FlextContainer.get_global()
        request_context = {"id": data["id"], "timestamp": "now"}
        reg_result = container.register(f"request_{data['id']}", request_context)
        if reg_result.is_failure:
            return FlextResult[dict].fail(f"Container registration failed: {reg_result.error}")
        enriched = {**data, "container_registered": True}
        return FlextResult[dict].ok(enriched)

    def publish_to_bus(data: dict) -> FlextResult[dict]:
        """Publish event to message bus."""
        bus = FlextBus()
        event = {
            "type": "request_processed",
            "request_id": data["id"],
            "enriched": True,
        }
        pub_result = bus.publish_event(event)
        if pub_result.is_failure:
            return FlextResult[dict].fail(f"Event publishing failed: {pub_result.error}")
        enriched = {**data, "event_published": True}
        return FlextResult[dict].ok(enriched)

    # Flow through complete utility pipeline
    request_data = {"id": "REQ-UTIL-001", "type": "integration_test"}
    pipeline_result = (
        FlextResult[dict]
        .ok(request_data)
        .flow_through(
            validate_request_data,
            enrich_with_config,
            register_in_container,
            publish_to_bus,
        )
    )

    if pipeline_result.is_success:
        final_data = pipeline_result.unwrap()
        print(f"✅ Utility pipeline complete: {final_data['id']}")
        print(f"   Batch size: {final_data.get('batch_size', 'N/A')}")
        print(f"   Container registered: {final_data.get('container_registered', False)}")
        print(f"   Event published: {final_data.get('event_published', False)}")
    else:
        print(f"❌ Pipeline failed: {pipeline_result.error}")

    # 3. lash - Utility Fallback Recovery
    print("\n=== 3. lash: Utility Fallback Recovery ===")

    def primary_dispatcher() -> FlextResult[dict]:
        """Primary dispatcher that might fail."""
        return FlextResult[dict].fail("Primary dispatcher unavailable")

    def fallback_dispatcher(error: str) -> FlextResult[dict]:
        """Fallback dispatcher when primary fails."""
        print(f"   ⚠️  Primary failed: {error}, using fallback dispatcher...")
        dispatcher = FlextDispatcher()
        message = {"type": "fallback_dispatch", "original_error": error}
        dispatch_result = dispatcher.dispatch(message)
        if dispatch_result.is_failure:
            return FlextResult[dict].fail(f"Fallback also failed: {dispatch_result.error}")
        return FlextResult[dict].ok({"dispatcher": "fallback", "message_sent": True})

    # Try primary, fall back on failure
    dispatch_result = primary_dispatcher().lash(fallback_dispatcher)
    if dispatch_result.is_success:
        dispatch_data = dispatch_result.unwrap()
        print(f"✅ Dispatch successful via: {dispatch_data['dispatcher']}")
        print(f"   Message sent: {dispatch_data.get('message_sent', False)}")
    else:
        print(f"❌ All dispatchers failed: {dispatch_result.error}")

    # 4. alt - Utility Provider Alternatives
    print("\n=== 4. alt: Utility Provider Alternatives ===")

    def get_custom_logger() -> FlextResult[FlextLogger]:
        """Try to get custom configured logger."""
        return FlextResult[FlextLogger].fail("Custom logger not configured")

    def get_default_logger() -> FlextResult[FlextLogger]:
        """Provide default logger as fallback."""
        logger = FlextLogger("utilities_integration")
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

    def create_expensive_container() -> FlextContainer:
        """Create and configure a new container (expensive operation)."""
        print("   ⚙️  Creating new container with full configuration...")
        container = FlextContainer()
        container.register("logger", FlextLogger("lazy_container"))
        container.register("config", FlextConfig())
        return container

    # Try to get existing container, create new one if not available
    container_fail_result = FlextResult[FlextContainer].fail("No existing container")
    container = container_fail_result.value_or_call(create_expensive_container)
    print(f"✅ Container acquired: {type(container).__name__}")
    print(f"   Container has services: {hasattr(container, '_services')}")

    # Try again with successful result (lazy function NOT called)
    container_success_result = FlextResult[FlextContainer].ok(FlextContainer.get_global())
    container_cached = container_success_result.value_or_call(create_expensive_container)
    print(f"✅ Existing container used: {type(container_cached).__name__}")
    print("   No expensive creation needed")

    print("\n" + "=" * 60)
    print("✅ NEW FLEXTRESULT METHODS UTILITIES INTEGRATION DEMO COMPLETE!")
    print("All 5 methods demonstrated with complete utility integration context")
    print("=" * 60)


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

    # Demonstrate new FlextResult methods (v0.9.9+)
    demonstrate_new_flextresult_methods()


if __name__ == "__main__":
    run_integration_examples()
