#!/usr/bin/env python3
"""FLEXT Payload Messaging and Events Example.

Comprehensive demonstration of FlextPayload system showing enterprise-grade
message and event patterns for structured data transport, validation, and
metadata management.

Features demonstrated:
    - Generic payload containers with type safety
    - Message payloads with level validation and source tracking
    - Domain event payloads with aggregate tracking and versioning
    - Metadata management and enrichment patterns
    - Payload validation and error handling
    - Enterprise messaging patterns for distributed systems
    - Event sourcing foundations for domain-driven design
    - Maximum type safety using flext_core.types

Key Components:
    - FlextPayload[T]: Generic type-safe payload container
    - FlextMessage: Specialized string message payload with levels
    - FlextEvent: Domain event payload with aggregate correlation
    - Payload factory methods with comprehensive validation
    - Metadata operations for transport context and debugging
    - Serialization support for cross-service communication

This example shows real-world enterprise messaging scenarios
demonstrating the power and flexibility of the FlextPayload system.
"""

import time
from collections.abc import Mapping

from flext_core import (
    FlextEvent,
    FlextMessage,
    FlextPayload,
    FlextResult,
    TErrorMessage,
    TLogMessage,
    TUserData,
)

# =============================================================================
# MESSAGING CONSTANTS - Event and messaging constraints
# =============================================================================

# Stock alert constants
LOW_STOCK_THRESHOLD = 10  # Minimum quantity before low stock alert


def demonstrate_generic_payloads() -> None:
    """Demonstrate generic payload containers with type safety."""
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print("📦 GENERIC PAYLOAD CONTAINERS")
    print("=" * 80)

    # 1. Basic payload creation
    log_message = "\n1. Basic payload creation:"
    print(log_message)

    # Simple string payload
    text_payload = FlextPayload(data="Hello, World!")
    log_message = f"✅ Text payload: {text_payload}"
    print(log_message)
    log_message = f"   Data: {text_payload.data}"
    print(log_message)
    log_message = f"   Metadata: {text_payload.metadata}"
    print(log_message)

    # Dictionary payload with metadata
    user_data: TUserData = {
        "id": "user123",
        "name": "John Doe",
        "email": "john@example.com",
    }
    user_payload = FlextPayload(
        data=user_data,
        metadata={
            "source": "user_service",
            "version": "1.0",
            "timestamp": time.time(),
        },
    )
    log_message = f"✅ User payload: {user_payload}"
    print(log_message)

    # 2. Type-safe payload creation with factory method
    log_message = "\n2. Type-safe payload creation:"
    print(log_message)

    # Create payload with validation
    order_data: TUserData = {
        "order_id": "ORD001",
        "customer_id": "CUST123",
        "items": [
            {"product": "laptop", "quantity": 1, "price": 999.99},
            {"product": "mouse", "quantity": 2, "price": 29.99},
        ],
        "total": 1059.97,
    }

    order_result = FlextPayload.create(
        order_data,
        source="order_service",
        correlation_id="req_456",
        processing_stage="created",
    )

    if order_result.success:
        order_payload = order_result.data
        if order_payload is not None:
            log_message = f"✅ Order payload created: {order_payload}"
            print(log_message)
            order_id = (
                order_payload.data.get("order_id") if order_payload.data else None
            )
            log_message = f"   Order ID: {order_id}"
            print(log_message)
            total = order_payload.data.get("total") if order_payload.data else None
            log_message = f"   Total: ${total}"
            print(log_message)
    else:
        error_message: TErrorMessage = (
            f"Order payload creation failed: {order_result.error}"
        )
        print(f"❌ {error_message}")

    # 3. Payload validation and error handling
    log_message = "\n3. Payload validation and error handling:"
    print(log_message)

    # Test invalid payload (None data)
    invalid_result = FlextPayload.create(
        None,
        source="test_service",
    )

    if invalid_result.success:
        log_message = f"✅ Invalid payload created: {invalid_result.data}"
        print(log_message)
    else:
        error_message = f"Invalid payload rejected (expected): {invalid_result.error}"
        print(f"❌ {error_message}")

    # 4. Metadata enrichment
    log_message = "\n4. Metadata enrichment:"
    print(log_message)

    base_payload = FlextPayload(
        data={"message": "Hello from service"},
        metadata={"source": "base_service"},
    )

    # Enrich with additional metadata
    enriched_payload = base_payload.enrich_metadata(
        {
            "timestamp": time.time(),
            "version": "2.0",
            "environment": "production",
        },
    )

    log_message = f"✅ Enriched payload: {enriched_payload}"
    print(log_message)
    log_message = f"   Metadata keys: {list(enriched_payload.metadata.keys())}"
    print(log_message)

    # 5. Payload transformation
    log_message = "\n5. Payload transformation:"
    print(log_message)

    # Transform payload data
    transform_result = base_payload.transform_data(
        lambda data: {"transformed_message": f"TRANSFORMED: {data.get('message', '')}"},
    )

    if transform_result.success:
        transformed_payload = transform_result.data
        if transformed_payload is not None:
            log_message = f"✅ Transformed payload: {transformed_payload}"
            print(log_message)
    else:
        error_message = f"Payload transformation failed: {transform_result.error}"
        print(f"❌ {error_message}")


def demonstrate_message_payloads() -> None:
    """Demonstrate message payloads with level validation using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print("💬 MESSAGE PAYLOADS")
    print("=" * 80)

    # 1. Basic message creation
    log_message = "\n1. Basic message creation:"
    print(log_message)

    # Create different message levels
    info_message = FlextMessage.create(
        "User login successful",
        level="info",
        source="auth_service",
    )

    warning_message = FlextMessage.create(
        "Database connection slow",
        level="warning",
        source="database_service",
    )

    error_message_result = FlextMessage.create(
        "Payment processing failed",
        level="error",
        source="payment_service",
    )

    if info_message.success:
        info_payload = info_message.data
        if info_payload is not None:
            log_message = f"✅ Info message: {info_payload}"
            print(log_message)
            log_message = f"   Level: {info_payload.level}"
            print(log_message)
            log_message = f"   Source: {info_payload.source}"
            print(log_message)

    if warning_message.success:
        warning_payload = warning_message.data
        if warning_payload is not None:
            log_message = f"✅ Warning message: {warning_payload}"
            print(log_message)

    if error_message_result.success:
        error_payload = error_message_result.data
        if error_payload is not None:
            log_message = f"✅ Error message: {error_payload}"
            print(log_message)

    # 2. Message validation
    log_message = "\n2. Message validation:"
    print(log_message)

    # Test invalid message level
    invalid_level_result = FlextMessage.create(
        "Test message",
        level="invalid_level",
        source="test_service",
    )

    if invalid_level_result.success:
        invalid_payload = invalid_level_result.data
        if invalid_payload is not None:
            log_message = f"✅ Invalid level message: {invalid_payload}"
            print(log_message)
    else:
        error_message = (
            f"Invalid level rejected (expected): {invalid_level_result.error}"
        )
        print(f"❌ {error_message}")

    # 3. Message enrichment
    log_message = "\n3. Message enrichment:"
    print(log_message)

    base_message_result = FlextMessage.create(
        "Base message",
        level="info",
        source="base_service",
    )

    if base_message_result.success:
        base_message = base_message_result.data
        if base_message is not None:
            # Enrich with additional context
            enriched_message = base_message.enrich_metadata(
                {
                    "user_id": "user_123",
                    "session_id": "sess_456",
                    "request_id": "req_789",
                },
            )

            log_message = f"✅ Enriched message: {enriched_message}"
            print(log_message)
            log_message = f"   User ID: {enriched_message.metadata.get('user_id')}"
            print(log_message)

    # 4. Message filtering and processing
    log_message = "\n4. Message filtering and processing:"
    print(log_message)

    # Create multiple messages for processing
    messages_data: list[TUserData] = [
        {"text": "System startup", "level": "info", "source": "system"},
        {"text": "Low memory warning", "level": "warning", "source": "monitoring"},
        {"text": "Database connection lost", "level": "error", "source": "database"},
        {"text": "User action completed", "level": "info", "source": "user_service"},
    ]

    created_messages = []

    for msg_data in messages_data:
        message_result = FlextMessage.create(
            str(msg_data["text"]),
            level=str(msg_data["level"]),
            source=str(msg_data["source"]),
        )
        if message_result.success:
            message = message_result.data
            if message is not None:
                created_messages.append(message)

    log_message = f"✅ Created {len(created_messages)} messages"
    print(log_message)

    # Filter error messages
    error_messages = [msg for msg in created_messages if msg.level == "error"]
    log_message = f"   Error messages: {len(error_messages)}"
    print(log_message)

    # Filter warning messages
    warning_messages = [msg for msg in created_messages if msg.level == "warning"]
    log_message = f"   Warning messages: {len(warning_messages)}"
    print(log_message)

    # 5. Message correlation
    log_message = "\n5. Message correlation:"
    print(log_message)

    # Create correlated messages
    correlation_id = "corr_123"

    request_message_result = FlextMessage.create(
        "Processing user request",
        level="info",
        source="api_gateway",
        correlation_id=correlation_id,
    )

    response_message_result = FlextMessage.create(
        "Request completed successfully",
        level="info",
        source="api_gateway",
        correlation_id=correlation_id,
    )

    if request_message_result.success and response_message_result.success:
        request_msg = request_message_result.data
        response_msg = response_message_result.data
        if request_msg is not None and response_msg is not None:
            log_message = f"✅ Request message: {request_msg.correlation_id}"
            print(log_message)
            log_message = f"✅ Response message: {response_msg.correlation_id}"
            print(log_message)
            log_message = f"   Correlation match: {request_msg.correlation_id == response_msg.correlation_id}"
            print(log_message)


def demonstrate_domain_events() -> None:
    """Demonstrate domain events with aggregate tracking using railway-oriented programming."""
    _print_domain_events_section_header("🎯 DOMAIN EVENTS")

    # Chain all domain event demonstrations using single-responsibility methods
    (
        _demonstrate_basic_domain_event_creation()
        .flat_map(lambda _: _demonstrate_order_lifecycle_events())
        .flat_map(lambda _: _demonstrate_inventory_management_events())
        .flat_map(lambda _: _demonstrate_event_correlation_and_tracing())
        .flat_map(lambda _: _demonstrate_event_validation_and_error_handling())
    )


def _print_domain_events_section_header(title: str) -> None:
    """Print formatted domain events section header."""
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print(title)
    print("=" * 80)


def _demonstrate_basic_domain_event_creation() -> FlextResult[None]:
    """Demonstrate basic domain event creation patterns."""
    print("\n1. Basic domain event creation:")

    # Create user registration event
    user_registration_data: TUserData = {
        "user_id": "user_456",
        "email": "alice@example.com",
        "registration_date": "2024-01-15T10:30:00Z",
        "source": "web_registration",
    }

    user_registration_event = FlextEvent.create(
        event_type="UserRegistered",
        aggregate_id="user_456",
        aggregate_type="User",
        data=user_registration_data,
        source="user_service",
    )

    return _display_user_registration_event(user_registration_event)


def _display_user_registration_event(
    user_registration_event: FlextResult[FlextPayload[Mapping[str, object]]],
) -> FlextResult[None]:
    """Display user registration event details."""
    if user_registration_event.success:
        event = user_registration_event.data
        if event is not None:
            print(f"✅ User registration event: {event}")
            print(f"   Event type: {event.event_type}")
            print(f"   Aggregate ID: {event.aggregate_id}")
            print(f"   Aggregate type: {event.aggregate_type}")

    return FlextResult.ok(None)


def _demonstrate_order_lifecycle_events() -> FlextResult[None]:
    """Demonstrate order lifecycle event patterns."""
    print("\n2. Order lifecycle events:")

    order_id = "order_789"

    # Create both order events
    order_created_event = _create_order_created_event(order_id)
    order_confirmed_event = _create_order_confirmed_event(order_id)

    return _display_order_lifecycle_events(order_created_event, order_confirmed_event)


def _create_order_created_event(
    order_id: str,
) -> FlextResult[FlextPayload[Mapping[str, object]]]:
    """Create order created event."""
    order_created_data: TUserData = {
        "order_id": order_id,
        "customer_id": "customer_123",
        "items": [
            {"product_id": "prod_1", "quantity": 2, "price": 29.99},
            {"product_id": "prod_2", "quantity": 1, "price": 99.99},
        ],
        "total_amount": 159.97,
        "created_at": "2024-01-15T14:20:00Z",
    }

    return FlextEvent.create(
        event_type="OrderCreated",
        aggregate_id=order_id,
        aggregate_type="Order",
        data=order_created_data,
        source="order_service",
        version=1,
    )


def _create_order_confirmed_event(
    order_id: str,
) -> FlextResult[FlextPayload[Mapping[str, object]]]:
    """Create order confirmed event."""
    order_confirmed_data: TUserData = {
        "order_id": order_id,
        "confirmed_at": "2024-01-15T14:25:00Z",
        "payment_method": "credit_card",
        "payment_id": "pay_456",
    }

    return FlextEvent.create(
        event_type="OrderConfirmed",
        aggregate_id=order_id,
        aggregate_type="Order",
        data=order_confirmed_data,
        source="order_service",
        version=2,
    )


def _display_order_lifecycle_events(
    order_created_event: FlextResult[FlextPayload[Mapping[str, object]]],
    order_confirmed_event: FlextResult[FlextPayload[Mapping[str, object]]],
) -> FlextResult[None]:
    """Display order lifecycle event details."""
    if order_created_event.success and order_confirmed_event.success:
        created_event = order_created_event.data
        confirmed_event = order_confirmed_event.data
        if (
            created_event is not None
            and confirmed_event is not None
            and hasattr(created_event, "version")
            and hasattr(confirmed_event, "version")
        ):
            print(
                f"✅ Order created event (v{created_event.version}): "
                f"{created_event.event_type}"
            )
            print(
                f"✅ Order confirmed event (v{confirmed_event.version}): "
                f"{confirmed_event.event_type}"
            )

    return FlextResult.ok(None)


def _demonstrate_inventory_management_events() -> FlextResult[None]:
    """Demonstrate inventory management event patterns."""
    print("\n3. Inventory management events:")

    product_id = "product_123"

    # Create stock updated event
    stock_updated_event = _create_stock_updated_event(product_id)

    return _handle_stock_updated_event(stock_updated_event, product_id)


def _create_stock_updated_event(
    product_id: str,
) -> FlextResult[FlextPayload[Mapping[str, object]]]:
    """Create stock updated event."""
    stock_updated_data: TUserData = {
        "product_id": product_id,
        "old_quantity": 50,
        "new_quantity": 35,
        "change_reason": "order_fulfillment",
        "updated_at": "2024-01-15T15:00:00Z",
    }

    return FlextEvent.create(
        event_type="StockUpdated",
        aggregate_id=product_id,
        aggregate_type="Product",
        data=stock_updated_data,
        source="inventory_service",
        version=1,
    )


def _handle_stock_updated_event(
    stock_updated_event: FlextResult[FlextPayload[Mapping[str, object]]],
    product_id: str,
) -> FlextResult[None]:
    """Handle stock updated event and create low stock alert if needed."""
    if stock_updated_event.success:
        stock_event = stock_updated_event.data
        if (
            stock_event is not None
            and hasattr(stock_event, "data")
            and stock_event.data is not None
        ):
            new_quantity = stock_event.data.get("new_quantity", 0)
            if isinstance(new_quantity, int) and new_quantity <= LOW_STOCK_THRESHOLD:
                return _create_and_display_low_stock_alert(product_id, new_quantity)

    return FlextResult.ok(None)


def _create_and_display_low_stock_alert(
    product_id: str, new_quantity: int
) -> FlextResult[None]:
    """Create and display low stock alert event."""
    low_stock_data: TUserData = {
        "product_id": product_id,
        "current_quantity": new_quantity,
        "threshold": LOW_STOCK_THRESHOLD,
        "alert_level": "warning",
        "alerted_at": "2024-01-15T15:00:00Z",
    }

    low_stock_event = FlextEvent.create(
        event_type="LowStockAlert",
        aggregate_id=product_id,
        aggregate_type="Product",
        data=low_stock_data,
        source="inventory_service",
        version=2,
    )

    if low_stock_event.success:
        alert_event = low_stock_event.data
        if alert_event is not None and hasattr(alert_event, "event_type"):
            print(f"✅ Low stock alert event: {alert_event.event_type}")
            if alert_event.data is not None:
                print(
                    f"   Current quantity: {alert_event.data.get('current_quantity')}"
                )

    return FlextResult.ok(None)


def _demonstrate_event_correlation_and_tracing() -> FlextResult[None]:
    """Demonstrate event correlation and tracing patterns."""
    print("\n4. Event correlation and tracing:")

    process_id = "process_123"

    # Create correlated events
    process_started_event = _create_process_started_event(process_id)
    step_completed_event = _create_step_completed_event(process_id)

    return _display_correlated_events(process_started_event, step_completed_event)


def _create_process_started_event(
    process_id: str,
) -> FlextResult[FlextPayload[Mapping[str, object]]]:
    """Create process started event."""
    return FlextEvent.create(
        event_type="ProcessStarted",
        aggregate_id=process_id,
        aggregate_type="BusinessProcess",
        data={"process_id": process_id, "started_at": "2024-01-15T16:00:00Z"},
        source="workflow_service",
        correlation_id=process_id,
    )


def _create_step_completed_event(
    process_id: str,
) -> FlextResult[FlextPayload[Mapping[str, object]]]:
    """Create process step completed event."""
    return FlextEvent.create(
        event_type="ProcessStepCompleted",
        aggregate_id=process_id,
        aggregate_type="BusinessProcess",
        data={
            "process_id": process_id,
            "step_name": "data_validation",
            "step_result": "success",
            "completed_at": "2024-01-15T16:05:00Z",
        },
        source="workflow_service",
        correlation_id=process_id,
    )


def _display_correlated_events(
    process_started_event: FlextResult[FlextPayload[Mapping[str, object]]],
    step_completed_event: FlextResult[FlextPayload[Mapping[str, object]]],
) -> FlextResult[None]:
    """Display correlated event details."""
    if process_started_event.success and step_completed_event.success:
        started_event = process_started_event.data
        completed_event = step_completed_event.data
        if (
            started_event is not None
            and completed_event is not None
            and hasattr(started_event, "correlation_id")
            and hasattr(completed_event, "correlation_id")
        ):
            print(f"✅ Process started: {started_event.correlation_id}")
            print(f"✅ Step completed: {completed_event.correlation_id}")
            print(
                f"   Correlation match: "
                f"{started_event.correlation_id == completed_event.correlation_id}"
            )

    return FlextResult.ok(None)


def _demonstrate_event_validation_and_error_handling() -> FlextResult[None]:
    """Demonstrate event validation and error handling patterns."""
    print("\n5. Event validation and error handling:")

    # Test invalid event (missing required fields)
    invalid_event_result = FlextEvent.create(
        event_type="",  # Empty event type
        aggregate_id="",  # Empty aggregate ID
        aggregate_type="",  # Empty aggregate type
        data={},
        source="test_service",
    )

    return _display_validation_results(invalid_event_result)


def _display_validation_results(
    invalid_event_result: FlextResult[FlextPayload[Mapping[str, object]]],
) -> FlextResult[None]:
    """Display event validation results."""
    if invalid_event_result.success:
        invalid_event = invalid_event_result.data
        if invalid_event is not None:
            print(f"✅ Invalid event created: {invalid_event}")
    else:
        error_message = (
            f"Invalid event rejected (expected): {invalid_event_result.error}"
        )
        print(f"❌ {error_message}")

    return FlextResult.ok(None)


def demonstrate_payload_serialization() -> None:
    """Demonstrate payload serialization for cross-service communication.

    Using flext_core.types for type safety.
    """
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print("🔄 PAYLOAD SERIALIZATION")
    print("=" * 80)

    # 1. Basic serialization
    log_message = "\n1. Basic serialization:"
    print(log_message)

    # Create payload for serialization
    serialization_data: TUserData = {
        "user_id": "user_789",
        "action": "profile_update",
        "changes": {
            "name": "Jane Smith",
            "email": "jane.smith@example.com",
        },
        "timestamp": "2024-01-15T17:00:00Z",
    }

    payload = FlextPayload(
        data=serialization_data,
        metadata={
            "source": "user_service",
            "version": "1.0",
            "correlation_id": "corr_789",
        },
    )

    # Serialize to dictionary
    serialized_dict = payload.to_dict()
    log_message = f"✅ Serialized to dict: {serialized_dict}"
    print(log_message)

    # 2. Message serialization
    log_message = "\n2. Message serialization:"
    print(log_message)

    message_result = FlextMessage.create(
        "User profile updated successfully",
        level="info",
        source="user_service",
        correlation_id="corr_789",
    )

    if message_result.success:
        message = message_result.data
        if message is not None:
            message_dict = message.to_dict()
            log_message = f"✅ Message serialized: {message_dict}"
            print(log_message)

    # 3. Event serialization
    log_message = "\n3. Event serialization:"
    print(log_message)

    event_result = FlextEvent.create(
        event_type="UserProfileUpdated",
        aggregate_id="user_789",
        aggregate_type="User",
        data=serialization_data,
        source="user_service",
        version=1,
        correlation_id="corr_789",
    )

    if event_result.success:
        event = event_result.data
        if event is not None:
            event_dict = event.to_dict()
            log_message = f"✅ Event serialized: {event_dict}"
            print(log_message)

    # 4. Cross-service payload transport
    log_message = "\n4. Cross-service payload transport:"
    print(log_message)

    # Simulate payload transport between services
    service_a_payload = FlextPayload(
        data={"request_id": "req_123", "user_id": "user_456"},
        metadata={"source": "service_a", "timestamp": time.time()},
    )

    # Serialize for transport
    transport_data = service_a_payload.to_dict()
    log_message = f"✅ Transport data: {transport_data}"
    print(log_message)

    # Simulate receiving service
    received_payload = FlextPayload.from_dict(transport_data)
    if received_payload.success:
        received_data = received_payload.data
        if received_data is not None:
            log_message = f"✅ Received payload: {received_data}"
            print(log_message)
            log_message = f"   Source: {received_data.metadata.get('source')}"
            print(log_message)
    else:
        error_message = f"Payload deserialization failed: {received_payload.error}"
        print(f"❌ {error_message}")

    # 5. Payload validation during serialization
    log_message = "\n5. Payload validation during serialization:"
    print(log_message)

    # Test serialization with complex data
    complex_data: TUserData = {
        "nested_object": {
            "level1": {
                "level2": {
                    "level3": "deep_value",
                },
            },
        },
        "array_data": [1, 2, 3, {"nested": "value"}],
        "boolean_flags": [True, False, True],
    }

    complex_payload = FlextPayload(
        data=complex_data,
        metadata={"complexity": "high", "validation": "passed"},
    )

    try:
        complex_dict = complex_payload.to_dict()
        log_message = "✅ Complex payload serialized successfully"
        print(log_message)
        log_message = f"   Keys: {list(complex_dict.keys())}"
        print(log_message)
    except (RuntimeError, ValueError, TypeError) as e:
        error_message = f"Complex serialization failed: {e}"
        print(f"❌ {error_message}")


def demonstrate_enterprise_messaging_patterns() -> None:
    """Demonstrate enterprise messaging patterns using flext_core.types."""
    log_message: TLogMessage = "\n" + "=" * 80
    print(log_message)
    print("🏭 ENTERPRISE MESSAGING PATTERNS")
    print("=" * 80)

    # 1. Request-Response pattern
    log_message = "\n1. Request-Response pattern:"
    print(log_message)

    # Create request payload
    request_data: TUserData = {
        "request_id": "req_456",
        "user_id": "user_123",
        "action": "get_user_profile",
        "parameters": {"include_preferences": True, "include_history": False},
    }

    request_payload = FlextPayload(
        data=request_data,
        metadata={
            "source": "api_gateway",
            "destination": "user_service",
            "request_type": "query",
            "timestamp": time.time(),
        },
    )

    # Simulate response
    response_data: TUserData = {
        "request_id": "req_456",
        "user_profile": {
            "user_id": "user_123",
            "name": "John Doe",
            "email": "john.doe@example.com",
            "preferences": {"theme": "dark", "language": "en"},
        },
        "status": "success",
        "processing_time_ms": 45,
    }

    response_payload = FlextPayload(
        data=response_data,
        metadata={
            "source": "user_service",
            "destination": "api_gateway",
            "response_type": "success",
            "timestamp": time.time(),
        },
    )

    if request_payload.data is not None:
        log_message = f"✅ Request payload: {request_payload.data.get('request_id')}"
        print(log_message)
    if response_payload.data is not None:
        log_message = f"✅ Response payload: {response_payload.data.get('request_id')}"
        print(log_message)

    # 2. Event-driven pattern
    log_message = "\n2. Event-driven pattern:"
    print(log_message)

    # Create domain events for order processing
    order_id = "order_999"

    # Order placed event
    order_placed_event = FlextEvent.create(
        event_type="OrderPlaced",
        aggregate_id=order_id,
        aggregate_type="Order",
        data={
            "order_id": order_id,
            "customer_id": "customer_456",
            "items": [{"product_id": "prod_1", "quantity": 1}],
            "total": 99.99,
        },
        source="order_service",
        version=1,
    )

    # Inventory reserved event
    inventory_reserved_event = FlextEvent.create(
        event_type="InventoryReserved",
        aggregate_id=order_id,
        aggregate_type="Order",
        data={
            "order_id": order_id,
            "reserved_items": [{"product_id": "prod_1", "quantity": 1}],
            "reservation_id": "res_789",
        },
        source="inventory_service",
        version=2,
    )

    # Payment processed event
    payment_processed_event = FlextEvent.create(
        event_type="PaymentProcessed",
        aggregate_id=order_id,
        aggregate_type="Order",
        data={
            "order_id": order_id,
            "payment_id": "pay_123",
            "amount": 99.99,
            "status": "completed",
        },
        source="payment_service",
        version=3,
    )

    events = []
    for event_result in [
        order_placed_event,
        inventory_reserved_event,
        payment_processed_event,
    ]:
        if event_result.success:
            event = event_result.data
            if event is not None:
                events.append(event)

    log_message = f"✅ Created {len(events)} events for order {order_id}"
    print(log_message)
    for event in events:
        log_message = f"   - {event.event_type} (v{event.version})"
        print(log_message)

    # 3. Message routing pattern
    log_message = "\n3. Message routing pattern:"
    print(log_message)

    # Create messages for different routing scenarios
    routing_messages = []

    # High priority message
    high_priority_result = FlextMessage.create(
        "Critical system alert",
        level="error",
        source="monitoring_service",
        metadata={"priority": "high", "route_to": "REDACTED_LDAP_BIND_PASSWORD_team"},
    )

    # Normal priority message
    normal_priority_result = FlextMessage.create(
        "User login successful",
        level="info",
        source="auth_service",
        metadata={"priority": "normal", "route_to": "log_aggregator"},
    )

    # Low priority message
    low_priority_result = FlextMessage.create(
        "Debug information",
        level="debug",
        source="debug_service",
        metadata={"priority": "low", "route_to": "debug_logs"},
    )

    for msg_result in [
        high_priority_result,
        normal_priority_result,
        low_priority_result,
    ]:
        if msg_result.success:
            message = msg_result.data
            if message is not None:
                routing_messages.append(message)

    log_message = f"✅ Created {len(routing_messages)} routing messages"
    print(log_message)

    # Route messages based on priority
    for message in routing_messages:
        priority = message.metadata.get("priority", "normal")
        route_to = message.metadata.get("route_to", "default")
        message_text = str(message.text)[:30] if message.text else "(empty)"
        log_message = f"   {priority} priority -> {route_to}: {message_text}..."
        print(log_message)

    # 4. Message correlation and tracing
    log_message = "\n4. Message correlation and tracing:"
    print(log_message)

    # Create correlated messages for a business transaction
    transaction_id = "txn_123"

    # Transaction start
    start_message = FlextMessage.create(
        "Transaction started",
        level="info",
        source="transaction_service",
        correlation_id=transaction_id,
        metadata={"step": "start", "transaction_type": "payment"},
    )

    # Transaction validation
    validation_message = FlextMessage.create(
        "Transaction validated",
        level="info",
        source="transaction_service",
        correlation_id=transaction_id,
        metadata={"step": "validation", "validation_result": "success"},
    )

    # Transaction completion
    completion_message = FlextMessage.create(
        "Transaction completed",
        level="info",
        source="transaction_service",
        correlation_id=transaction_id,
        metadata={"step": "completion", "final_status": "success"},
    )

    correlated_messages = []
    for msg_result in [start_message, validation_message, completion_message]:
        if msg_result.success:
            message = msg_result.data
            if message is not None:
                correlated_messages.append(message)

    log_message = f"✅ Created {len(correlated_messages)} correlated messages"
    print(log_message)
    log_message = f"   Transaction ID: {transaction_id}"
    print(log_message)

    # 5. Error handling and dead letter queues
    log_message = "\n5. Error handling and dead letter queues:"
    print(log_message)

    # Create problematic message
    problematic_message_result = FlextMessage.create(
        "Message with invalid data",
        level="error",
        source="problematic_service",
        metadata={"retry_count": 3, "max_retries": 3, "dead_letter": True},
    )

    if problematic_message_result.success:
        problematic_message = problematic_message_result.data
        if problematic_message is not None:
            retry_count_obj = problematic_message.metadata.get("retry_count", 0)
            max_retries_obj = problematic_message.metadata.get("max_retries", 0)
            dead_letter_obj = problematic_message.metadata.get("dead_letter", False)

            retry_count = (
                int(retry_count_obj) if isinstance(retry_count_obj, (int, str)) else 0
            )
            max_retries = (
                int(max_retries_obj) if isinstance(max_retries_obj, (int, str)) else 0
            )
            dead_letter = bool(dead_letter_obj)

            log_message = "✅ Problematic message created"
            print(log_message)
            log_message = f"   Retry count: {retry_count}/{max_retries}"
            print(log_message)
            log_message = f"   Dead letter: {dead_letter}"
            print(log_message)

            if retry_count >= max_retries and dead_letter:
                log_message = "   → Routing to dead letter queue"
                print(log_message)


def main() -> None:
    """Run comprehensive FlextPayload demonstration with maximum type safety."""
    print("=" * 80)
    print("🚀 FLEXT PAYLOAD - MESSAGING AND EVENTS DEMONSTRATION")
    print("=" * 80)

    # Run all demonstrations
    demonstrate_generic_payloads()
    demonstrate_message_payloads()
    demonstrate_domain_events()
    demonstrate_payload_serialization()
    demonstrate_enterprise_messaging_patterns()

    print("\n" + "=" * 80)
    print("🎉 FLEXT PAYLOAD DEMONSTRATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
