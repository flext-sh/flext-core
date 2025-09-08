#!/usr/bin/env python3
"""FlextCore Payload and Messaging Events - Simplified Example.

This example demonstrates the correct usage of FlextModels for messaging
and event handling patterns using the current API.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

from flext_core import FlextModels, FlextTypes


def demonstrate_basic_payload_creation() -> None:
    """Show basic payload creation with required fields."""
    print("=== Basic Payload Creation ===")

    # Simple string payload
    greeting_payload = FlextModels.Payload[str](
        data="Hello, World!",
        source_service="demo_service",
        message_type="greeting",
    )
    print(f"Greeting: {greeting_payload.data}")
    print(f"From: {greeting_payload.source_service}")
    print(f"Type: {greeting_payload.message_type}")

    # Structured data payload
    user_data: FlextTypes.Core.Dict = {
        "id": "user123",
        "name": "John Doe",
        "email": "john@example.com",
    }

    user_payload = FlextModels.Payload[FlextTypes.Core.Dict](
        data=user_data,
        source_service="user_service",
        message_type="user_registered",
        headers={"version": "1.0", "timestamp": time.time()},
    )

    print(f"User ID: {user_payload.data['id']}")
    print(f"Headers: {user_payload.headers}")
    print(f"Priority: {user_payload.priority}")


def demonstrate_message_routing() -> None:
    """Show message routing with target services."""
    print("\n=== Message Routing ===")

    # Order processing message
    order_payload = FlextModels.Payload[FlextTypes.Core.Dict](
        data={"order_id": "ORD001", "amount": 99.99},
        source_service="order_service",
        target_service="payment_service",
        message_type="process_payment",
        priority=8,  # High priority
        headers={"correlation_id": "req_456"},
    )

    print(f"Order: {order_payload.data}")
    print(f"Route: {order_payload.source_service} → {order_payload.target_service}")
    print(f"Priority: {order_payload.priority}")
    print(f"Created: {order_payload.timestamp}")


def demonstrate_event_messaging() -> None:
    """Show event-driven messaging patterns."""
    print("\n=== Event Messaging ===")

    # Domain event
    event_data: FlextTypes.Core.JsonObject = {
        "user_id": "user123",
        "action": "profile_updated",
        "fields_changed": ["email", "phone"],
        "timestamp": time.time(),
    }

    domain_event = FlextModels.Event(
        data=event_data,
        source_service="user_service",
        message_type="domain_event",
        aggregate_id="user123",
        aggregate_type="User",
        headers={
            "event_type": "UserProfileUpdated",
            "version": "1",
        },
    )

    print(f"Event: {domain_event.headers['event_type']}")
    print(f"User: {domain_event.data['user_id']}")
    print(f"Changes: {domain_event.data['fields_changed']}")


def demonstrate_payload_properties() -> None:
    """Show payload properties and metadata."""
    print("\n=== Payload Properties ===")

    # Message with expiration

    expire_time = datetime.now(UTC) + timedelta(minutes=5)

    temp_message = FlextModels.Message(
        data={"content": "Temporary notification", "type": "info"},
        source_service="notification_service",
        message_type="temp_notification",
        expires_at=expire_time,
        retry_count=0,
    )

    print(f"Message: {temp_message.data}")
    print(f"Created: {temp_message.timestamp}")
    print(f"Expires: {temp_message.expires_at}")
    print(f"Retries: {temp_message.retry_count}")


def demonstrate_typed_payloads() -> None:
    """Show type-safe payload handling."""
    print("\n=== Typed Payloads ===")

    # String payload
    string_payload = FlextModels.Payload[str](
        data="Hello typed world",
        source_service="demo_service",
        message_type="typed_greeting",
    )

    # The data is properly typed as str
    message_length = len(string_payload.data)  # Type-safe
    print(f"String payload: {string_payload.data} (length: {message_length})")

    # Dict payload
    dict_payload = FlextModels.Payload[FlextTypes.Core.Dict](
        data={"key": "value", "number": 42},
        source_service="demo_service",
        message_type="typed_dict",
    )

    # The data is properly typed as dict
    keys = list(dict_payload.data.keys())  # Type-safe
    print(f"Dict payload keys: {keys}")


def demonstrate_message_specialization() -> None:
    """Show the specialized Message class."""
    print("\n=== Specialized Message Class ===")

    # Message is a specialized Payload for JSON data
    json_message = FlextModels.Message(
        data={"content": "This is a JSON message", "level": "info"},
        source_service="logger_service",
        message_type="log_message",
    )

    print(f"JSON Message: {json_message.data}")
    print(f"Content: {json_message.data['content']}")
    print(f"Level: {json_message.data['level']}")


def main() -> None:
    """Run all payload messaging examples."""
    print("FlextCore Payload & Messaging Examples")
    print("=" * 40)

    demonstrate_basic_payload_creation()
    demonstrate_message_routing()
    demonstrate_event_messaging()
    demonstrate_payload_properties()
    demonstrate_typed_payloads()
    demonstrate_message_specialization()

    print("\n" + "=" * 40)
    print("✅ All examples completed successfully!")


if __name__ == "__main__":
    main()
