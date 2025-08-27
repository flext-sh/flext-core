#!/usr/bin/env python3
"""Messaging and events using FlextPayload.

Demonstrates structured data transport, validation,
and metadata management for message patterns.
    - Domain event payloads with aggregate tracking and versioning
    - Metadata management and enrichment patterns
    - Payload validation and error handling
    - Enterprise messaging patterns for distributed systems
    - Event sourcing foundations for domain-driven design
    - Maximum type safety using flext_core.typings

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

import contextlib
import time
from collections.abc import Mapping

from flext_core import (
    FlextPayload,
    FlextResult,
)

# =============================================================================
# MESSAGING CONSTANTS - Event and messaging constraints
# =============================================================================

# Stock alert constants
LOW_STOCK_THRESHOLD = 10  # Minimum quantity before low stock alert


def demonstrate_generic_payloads() -> None:
    """Demonstrate generic payload containers with type safety."""
    _print_generic_header()
    _demo_basic_payloads()
    _demo_type_safe_payload()
    _demo_invalid_payload()
    _demo_metadata_enrichment()
    _demo_payload_transformation()


def _print_generic_header() -> None:
    _separator = "\n" + "=" * 80


def _demo_basic_payloads() -> None:
    FlextPayload(data="Hello, World!")

    user_data: dict[str, object] = {
        "id": "user123",
        "name": "John Doe",
        "email": "john@example.com",
    }
    FlextPayload(
        data=user_data,
        metadata={"source": "user_service", "version": "1.0", "timestamp": time.time()},
    )


def _demo_type_safe_payload() -> None:
    order_data: dict[str, object] = {
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
    # Modern pattern: Check success and use value directly
    if order_result.success:
        order_payload = order_result.value
        order_payload.value.get("order_id") if order_payload.value else None
        order_payload.value.get("total") if order_payload.value else None


def _demo_invalid_payload() -> None:
    invalid_result = FlextPayload.create(None, source="test_service")
    if invalid_result.success:
        pass


def _demo_metadata_enrichment() -> None:
    base_payload = FlextPayload(
        data={"message": "Hello from service"},
        metadata={"source": "base_service"},
    )
    base_payload.enrich_metadata(
        {"timestamp": time.time(), "version": "2.0", "environment": "production"},
    )


def _demo_payload_transformation() -> None:
    base_payload = FlextPayload(
        data={"message": "Hello from service"},
        metadata={"source": "base_service"},
    )
    transform_result = base_payload.transform_data(
        lambda data: {
            "transformed_message": f"TRANSFORMED: {data.get('message', '') if isinstance(data, dict) else str(data)}"
        },
    )
    # Modern pattern: Check success and use value directly
    if transform_result.success:
        # Process the transformed payload - payload available at transform_result.value
        pass  # Placeholder for transformation processing


def demonstrate_message_payloads() -> None:
    """Demonstrate message payloads with level validation using flext_core.typings."""
    _print_message_header()
    info_res, warning_res, error_res = _create_basic_messages()
    _display_basic_messages(info_res, warning_res, error_res)
    _validate_invalid_level_message()
    _demonstrate_message_enrichment()
    _process_and_filter_messages()
    _demonstrate_message_correlation()


def _print_message_header() -> None:
    _separator = "\n" + "=" * 80


def _create_basic_messages() -> tuple[
    FlextResult[FlextPayload[str]],
    FlextResult[FlextPayload[str]],
    FlextResult[FlextPayload[str]],
]:
    info_message = FlextResult[FlextPayload[str]].ok(
        FlextPayload[str](
            data="User login successful",
            metadata={"level": "info", "source": "auth_service"},
        )
    )

    warning_message = FlextResult[FlextPayload[str]].ok(
        FlextPayload[str](
            data="Database connection slow",
            metadata={"level": "warning", "source": "database_service"},
        )
    )

    error_message_result = FlextResult[FlextPayload[str]].ok(
        FlextPayload[str](
            data="Payment processing failed",
            metadata={"level": "error", "source": "payment_service"},
        )
    )
    return info_message, warning_message, error_message_result


def _display_basic_messages(
    info_message: FlextResult[FlextPayload[str]],
    warning_message: FlextResult[FlextPayload[str]],
    error_message_result: FlextResult[FlextPayload[str]],
) -> None:
    if info_message.success:
        info_payload = info_message.value
        if info_payload is not None:
            pass

    if warning_message.success:
        warning_payload = warning_message.value
        if warning_payload is not None:
            pass

    if error_message_result.success:
        error_payload = error_message_result.value
        if error_payload is not None:
            pass


def _validate_invalid_level_message() -> None:
    invalid_level_result = FlextPayload[str].create(
        "Test message",
        level="invalid_level",
        source="test_service",
    )
    if invalid_level_result.success:
        invalid_payload = invalid_level_result.value
        if invalid_payload is not None:
            pass


def _demonstrate_message_enrichment() -> None:
    base_message_result = FlextPayload[str].create(
        "Base message",
        level="info",
        source="base_service",
    )
    if base_message_result.success:
        base_message = base_message_result.value
        if base_message is not None:
            base_message.enrich_metadata(
                {
                    "user_id": "user_123",
                    "session_id": "sess_456",
                    "request_id": "req_789",
                },
            )


def _process_and_filter_messages() -> None:
    messages_data: list[dict[str, object]] = [
        {"text": "System startup", "level": "info", "source": "system"},
        {"text": "Low memory warning", "level": "warning", "source": "monitoring"},
        {"text": "Database connection lost", "level": "error", "source": "database"},
        {"text": "User action completed", "level": "info", "source": "user_service"},
    ]

    created_messages: list[FlextPayload[str]] = []
    for msg_data in messages_data:
        message_result = FlextPayload[str].create(
            str(msg_data["text"]),
            level=str(msg_data["level"]),
            source=str(msg_data["source"]),
        )
        if message_result.success:
            message = message_result.value
            if message is not None:
                created_messages.append(message)

    [msg for msg in created_messages if msg.level == "error"]
    [msg for msg in created_messages if msg.level == "warning"]


def _demonstrate_message_correlation() -> None:
    correlation_id = "corr_123"
    request_message_result = FlextPayload[str].create(
        "Processing user request",
        level="info",
        source="api_gateway",
        correlation_id=correlation_id,
    )
    response_message_result = FlextPayload[str].create(
        "Request completed successfully",
        level="info",
        source="api_gateway",
        correlation_id=correlation_id,
    )
    if request_message_result.success and response_message_result.success:
        request_msg = request_message_result.value
        response_msg = response_message_result.value
        if request_msg is not None and response_msg is not None:
            pass


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


def _print_domain_events_section_header(title: str) -> None:  # noqa: ARG001
    """Print formatted domain events section header."""
    _separator = "\n" + "=" * 80


def _demonstrate_basic_domain_event_creation() -> FlextResult[None]:
    """Demonstrate basic domain event creation patterns."""
    # Create user registration event
    user_registration_data: dict[str, object] = {
        "user_id": "user_456",
        "email": "alice@example.com",
        "registration_date": "2024-01-15T10:30:00Z",
        "source": "web_registration",
    }

    user_registration_event = FlextPayload[Mapping[str, object]].create(
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
        event = user_registration_event.value
        if event is not None:
            pass

    return FlextResult.ok(None)


def _demonstrate_order_lifecycle_events() -> FlextResult[None]:
    """Demonstrate order lifecycle event patterns."""
    order_id = "order_789"

    # Create both order events
    order_created_event = _create_order_created_event(order_id)
    order_confirmed_event = _create_order_confirmed_event(order_id)

    return _display_order_lifecycle_events(order_created_event, order_confirmed_event)


def _create_order_created_event(
    order_id: str,
) -> FlextResult[FlextPayload[Mapping[str, object]]]:
    """Create order created event."""
    order_created_data: dict[str, object] = {
        "order_id": order_id,
        "customer_id": "customer_123",
        "items": [
            {"product_id": "prod_1", "quantity": 2, "price": 29.99},
            {"product_id": "prod_2", "quantity": 1, "price": 99.99},
        ],
        "total_amount": 159.97,
        "created_at": "2024-01-15T14:20:00Z",
    }

    return FlextPayload[Mapping[str, object]].create(
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
    order_confirmed_data: dict[str, object] = {
        "order_id": order_id,
        "confirmed_at": "2024-01-15T14:25:00Z",
        "payment_method": "credit_card",
        "payment_id": "pay_456",
    }

    return FlextPayload[Mapping[str, object]].create(
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
        created_event = order_created_event.value
        confirmed_event = order_confirmed_event.value
        if (
            created_event is not None
            and confirmed_event is not None
            and hasattr(created_event, "version")
            and hasattr(confirmed_event, "version")
        ):
            pass

    return FlextResult.ok(None)


def _demonstrate_inventory_management_events() -> FlextResult[None]:
    """Demonstrate inventory management event patterns."""
    product_id = "product_123"

    # Create stock updated event
    stock_updated_event = _create_stock_updated_event(product_id)

    return _handle_stock_updated_event(stock_updated_event, product_id)


def _create_stock_updated_event(
    product_id: str,
) -> FlextResult[FlextPayload[Mapping[str, object]]]:
    """Create stock updated event."""
    stock_updated_data: dict[str, object] = {
        "product_id": product_id,
        "old_quantity": 50,
        "new_quantity": 35,
        "change_reason": "order_fulfillment",
        "updated_at": "2024-01-15T15:00:00Z",
    }

    return FlextPayload[Mapping[str, object]].create(
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
        stock_event = stock_updated_event.value
        if (
            stock_event is not None
            and hasattr(stock_event, "data")
            and stock_event.value is not None
        ):
            event_data = stock_event.value
            new_quantity = (
                event_data.get("new_quantity", 0) if isinstance(event_data, dict) else 0
            )
            if isinstance(new_quantity, int) and new_quantity <= LOW_STOCK_THRESHOLD:
                return _create_and_display_low_stock_alert(product_id, new_quantity)

    return FlextResult.ok(None)


def _create_and_display_low_stock_alert(
    product_id: str,
    new_quantity: int,
) -> FlextResult[None]:
    """Create and display low stock alert event."""
    low_stock_data: dict[str, object] = {
        "product_id": product_id,
        "current_quantity": new_quantity,
        "threshold": LOW_STOCK_THRESHOLD,
        "alert_level": "warning",
        "alerted_at": "2024-01-15T15:00:00Z",
    }

    low_stock_event = FlextPayload[Mapping[str, object]].create(
        event_type="LowStockAlert",
        aggregate_id=product_id,
        aggregate_type="Product",
        data=low_stock_data,
        source="inventory_service",
        version=2,
    )

    if (
        low_stock_event.success
        and low_stock_event.value is not None
        and hasattr(low_stock_event.value, "event_type")
        and getattr(low_stock_event.value, "data", None) is not None
    ):
        pass

    return FlextResult.ok(None)


def _demonstrate_event_correlation_and_tracing() -> FlextResult[None]:
    """Demonstrate event correlation and tracing patterns."""
    process_id = "process_123"

    # Create correlated events
    process_started_event = _create_process_started_event(process_id)
    step_completed_event = _create_step_completed_event(process_id)

    return _display_correlated_events(process_started_event, step_completed_event)


def _create_process_started_event(
    process_id: str,
) -> FlextResult[FlextPayload[Mapping[str, object]]]:
    """Create process started event."""
    return FlextPayload[Mapping[str, object]].create(
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
    return FlextPayload[Mapping[str, object]].create(
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
        started_event = process_started_event.value
        completed_event = step_completed_event.value
        if (
            started_event is not None
            and completed_event is not None
            and hasattr(started_event, "correlation_id")
            and hasattr(completed_event, "correlation_id")
        ):
            pass

    return FlextResult.ok(None)


def _demonstrate_event_validation_and_error_handling() -> FlextResult[None]:
    """Demonstrate event validation and error handling patterns."""
    # Test invalid event (missing required fields)
    invalid_event_result = FlextPayload[Mapping[str, object]].create(
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
        invalid_event = invalid_event_result.value
        if invalid_event is not None:
            pass

    return FlextResult.ok(None)


def demonstrate_payload_serialization() -> None:
    """Demonstrate payload serialization for cross-service communication.

    Using flext_core.typings for type safety.
    """
    _print_serialization_header()
    serialization_data = _basic_serialization_demo()
    _message_serialization_demo()
    _event_serialization_demo(serialization_data)
    _cross_service_transport_demo()
    _payload_validation_during_serialization()


def _print_serialization_header() -> None:
    _separator = "\n" + "=" * 80


def _basic_serialization_demo() -> dict[str, object]:
    serialization_data: dict[str, object] = {
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
    payload.to_dict()
    return serialization_data


def _message_serialization_demo() -> None:
    message_result = FlextPayload[str].create(
        "User profile updated successfully",
        level="info",
        source="user_service",
        correlation_id="corr_789",
    )
    if message_result.success:
        message = message_result.value
        if message is not None:
            message.to_dict()


def _event_serialization_demo(serialization_data: dict[str, object]) -> None:
    event_result = FlextPayload[Mapping[str, object]].create(
        event_type="UserProfileUpdated",
        aggregate_id="user_789",
        aggregate_type="User",
        data=serialization_data,
        source="user_service",
        version=1,
        correlation_id="corr_789",
    )
    if event_result.success:
        event = event_result.value
        if event is not None:
            event.to_dict()


def _cross_service_transport_demo() -> None:
    service_a_payload = FlextPayload(
        data={"request_id": "req_123", "user_id": "user_456"},
        metadata={"source": "service_a", "timestamp": time.time()},
    )
    transport_data = service_a_payload.to_dict()
    received_payload = FlextPayload.from_dict(transport_data)
    if received_payload.success:
        received_data = received_payload.value
        if received_data is not None:
            pass


def _payload_validation_during_serialization() -> None:
    complex_data: dict[str, object] = {
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
    with contextlib.suppress(RuntimeError, ValueError, TypeError):
        complex_payload.to_dict()


def demonstrate_enterprise_messaging_patterns() -> None:
    """Demonstrate enterprise messaging patterns using flext_core.typings."""
    _print_enterprise_header()
    _request_response_pattern_demo()
    _event_driven_pattern_demo()
    _message_routing_pattern_demo()
    _message_correlation_tracing_demo()
    _dead_letter_queue_demo()


def _print_enterprise_header() -> None:
    _separator = "\n" + "=" * 80


def _request_response_pattern_demo() -> None:
    request_data: dict[str, object] = {
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
    response_data: dict[str, object] = {
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
    if request_payload.value is not None:
        pass
    if response_payload.value is not None:
        pass


def _event_driven_pattern_demo() -> None:
    order_id = "order_999"
    order_placed_event = FlextPayload[Mapping[str, object]].create(
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
    inventory_reserved_event = FlextPayload[Mapping[str, object]].create(
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
    payment_processed_event = FlextPayload[Mapping[str, object]].create(
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
    events: list[FlextPayload[Mapping[str, object]]] = []
    for event_result in [
        order_placed_event,
        inventory_reserved_event,
        payment_processed_event,
    ]:
        if event_result.success:
            event = event_result.value
            if event is not None:
                events.append(event)
    for _event in events:
        pass


def _message_routing_pattern_demo() -> None:
    routing_messages: list[FlextPayload[str]] = []
    high_priority_result = FlextPayload[str].create(
        "Critical system alert",
        level="error",
        source="monitoring_service",
        metadata={"priority": "high", "route_to": "admin_team"},
    )
    normal_priority_result = FlextPayload[str].create(
        "User login successful",
        level="info",
        source="auth_service",
        metadata={"priority": "normal", "route_to": "log_aggregator"},
    )
    low_priority_result = FlextPayload[str].create(
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
            message = msg_result.value
            if message is not None:
                routing_messages.append(message)
    for message in routing_messages:
        message.metadata.get("priority", "normal")
        message.metadata.get("route_to", "default")
        str(message.data)[:30] if message.data else "(empty)"


def _message_correlation_tracing_demo() -> None:
    transaction_id = "txn_123"
    start_message = FlextPayload[str].create(
        "Transaction started",
        level="info",
        source="transaction_service",
        correlation_id=transaction_id,
        metadata={"step": "start", "transaction_type": "payment"},
    )
    validation_message = FlextPayload[str].create(
        "Transaction validated",
        level="info",
        source="transaction_service",
        correlation_id=transaction_id,
        metadata={"step": "validation", "validation_result": "success"},
    )
    completion_message = FlextPayload[str].create(
        "Transaction completed",
        level="info",
        source="transaction_service",
        correlation_id=transaction_id,
        metadata={"step": "completion", "final_status": "success"},
    )
    correlated_messages: list[FlextPayload[str]] = []
    for msg_result in [start_message, validation_message, completion_message]:
        if msg_result.success:
            message = msg_result.value
            if message is not None:
                correlated_messages.append(message)


def _dead_letter_queue_demo() -> None:
    problematic_message_result = FlextPayload[str].create(
        "Message with invalid data",
        level="error",
        source="problematic_service",
        metadata={"retry_count": 3, "max_retries": 3, "dead_letter": True},
    )
    if problematic_message_result.success:
        problematic_message = problematic_message_result.value
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
            if retry_count >= max_retries and dead_letter:
                pass


def main() -> None:
    """Run comprehensive FlextPayload demonstration with maximum type safety."""
    # Run all demonstrations
    demonstrate_generic_payloads()
    demonstrate_message_payloads()
    demonstrate_domain_events()
    demonstrate_payload_serialization()
    demonstrate_enterprise_messaging_patterns()


if __name__ == "__main__":
    main()
