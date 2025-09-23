#!/usr/bin/env python3
"""06 - Messaging and Event Patterns: Payloads and Domain Events.

This example demonstrates intermediate messaging patterns using FlextModels
for event-driven architectures, message passing, and domain event handling.
These patterns form the foundation for CQRS, event sourcing, and microservices.

Key Concepts Demonstrated:
- Payload: Message container with metadata
- DomainEvent: Domain-driven design event patterns
- Message routing: Source and target service patterns
- Correlation tracking: Request/response correlation
- Event metadata: Timestamps, versions, priorities
- Event aggregation: Collecting and processing events
- Command patterns: Using payloads for commands
- Query patterns: Using payloads for queries

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from flext_core import (
    FlextConfig,
    FlextContainer,
    FlextLogger,
    FlextModels,
    FlextResult,
    FlextService,
    FlextTypes,
)

# ========== MESSAGING SERVICE ==========


class MessagingPatternsService(FlextService[FlextTypes.Core.Dict]):
    """Service demonstrating messaging and event patterns."""

    def __init__(self) -> None:
        """Initialize with dependencies."""
        super().__init__()
        self._container = FlextContainer.get_global()
        self._logger = FlextLogger(__name__)
        self._config = FlextConfig.get_global_instance()
        self._event_store: list[FlextModels.DomainEvent] = []
        self._message_queue: list[FlextModels.Payload[FlextTypes.Core.Dict]] = []

    def execute(self) -> FlextResult[FlextTypes.Core.Dict]:
        """Execute method required by FlextService."""
        self._logger.info("Executing messaging patterns demo")
        return FlextResult[FlextTypes.Core.Dict].ok({
            "status": "completed",
            "events_processed": len(self._event_store),
            "messages_queued": len(self._message_queue),
        })

    def get_event_count(self) -> int:
        """Get the number of events stored."""
        return len(self._event_store)

    def get_message_count(self) -> int:
        """Get the number of messages queued."""
        return len(self._message_queue)

    # ========== PAYLOAD PATTERNS ==========

    def demonstrate_basic_payload(self) -> None:
        """Show basic payload creation and usage."""
        print("\n=== Basic Payload Patterns ===")

        # Simple payload with string data wrapped in dict
        greeting_payload = FlextModels.Payload[FlextTypes.Core.Dict](
            data={
                "message": "Hello, FLEXT!",
                "source_service": "greeting_service",
                "message_type": "greeting",
                "message_id": str(uuid4()),
            },
            metadata={
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        print(f"✅ Greeting: {greeting_payload.data.get('message')}")
        print(f"Source: {greeting_payload.data.get('source_service')}")
        print(f"Type: {greeting_payload.data.get('message_type')}")
        print(f"ID: {greeting_payload.data.get('message_id')}")
        print(f"Timestamp: {greeting_payload.metadata.get('timestamp')}")

        # Structured data payload
        user_data: FlextTypes.Core.Dict = {
            "id": str(uuid4()),
            "username": "john_doe",
            "email": "john@example.com",
            "roles": ["user", "admin"],
        }

        user_payload = FlextModels.Payload[FlextTypes.Core.Dict](
            data=user_data,
            metadata={
                "source_service": "user_service",
                "message_type": "user_created",
                "version": "1.0",
                "correlation_id": str(uuid4()),
                "priority": 5,
            },
        )
        print("\n✅ User payload created")
        print(f"User ID: {user_payload.data['id']}")
        print(f"Username: {user_payload.data['username']}")
        print(f"Metadata: {user_payload.metadata}")

        # Add to message queue
        self._message_queue.append(user_payload)

    # ========== COMMAND PATTERNS ==========

    def demonstrate_command_patterns(self) -> None:
        """Show command message patterns for CQRS."""
        print("\n=== Command Patterns ===")

        # Create order command
        create_order_command = FlextModels.Payload[FlextTypes.Core.Dict](
            data={
                "command": "CreateOrder",
                "order_id": str(uuid4()),
                "customer_id": "CUST-123",
                "items": [
                    {"sku": "PROD-A", "quantity": 2, "price": 29.99},
                    {"sku": "PROD-B", "quantity": 1, "price": 49.99},
                ],
                "total": 109.97,
            },
            metadata={
                "source_service": "web_api",
                "message_type": "command",
                "command_type": "CreateOrder",
                "correlation_id": str(uuid4()),
                "user_id": "user_456",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        print(f"✅ Command: {create_order_command.data['command']}")
        print(f"Order ID: {create_order_command.data['order_id']}")
        print(f"Total: ${create_order_command.data['total']}")

        # Process payment command
        process_payment_command = FlextModels.Payload[FlextTypes.Core.Dict](
            data={
                "command": "ProcessPayment",
                "order_id": create_order_command.data["order_id"],
                "amount": create_order_command.data["total"],
                "payment_method": "credit_card",
                "card_token": "tok_xxxxx",
            },
            metadata={
                "source_service": "order_service",
                "message_type": "command",
                "command_type": "ProcessPayment",
                "correlation_id": create_order_command.metadata["correlation_id"],
                "retry_count": 0,
                "max_retries": 3,
            },
        )
        print("\n✅ Payment command created")
        print(f"Correlation: {process_payment_command.metadata['correlation_id']}")

        self._message_queue.extend([create_order_command, process_payment_command])

    # ========== QUERY PATTERNS ==========

    def demonstrate_query_patterns(self) -> None:
        """Show query message patterns for CQRS."""
        print("\n=== Query Patterns ===")

        # Query for user orders
        get_orders_query = FlextModels.Payload[FlextTypes.Core.Dict](
            data={
                "query": "GetUserOrders",
                "user_id": "user_456",
                "filters": {
                    "status": ["pending", "completed"],
                    "date_from": "2025-01-01",
                    "date_to": "2025-12-31",
                },
                "pagination": FlextModels.Pagination(page=1, size=20).model_dump(),
            },
            metadata={
                "source_service": "web_api",
                "message_type": "query",
                "query_type": "GetUserOrders",
                "correlation_id": str(uuid4()),
                "cache_key": "user_456_orders_2025",
                "cache_ttl": 300,  # 5 minutes
            },
        )
        print(f"✅ Query: {get_orders_query.data['query']}")
        print(f"User: {get_orders_query.data['user_id']}")
        print(f"Filters: {get_orders_query.data['filters']}")

        # Aggregation query
        sales_report_query = FlextModels.Payload[FlextTypes.Core.Dict](
            data={
                "query": "GenerateSalesReport",
                "period": "monthly",
                "year": 2025,
                "metrics": ["total_sales", "order_count", "avg_order_value"],
                "group_by": ["product_category", "region"],
            },
            metadata={
                "source_service": "analytics_service",
                "message_type": "query",
                "query_type": "AggregationQuery",
                "priority": 3,  # Lower priority
                "timeout_seconds": 30,
            },
        )
        print(f"\n✅ Report query: {sales_report_query.data['query']}")
        print(f"Metrics: {sales_report_query.data['metrics']}")

        self._message_queue.extend([get_orders_query, sales_report_query])

    # ========== DOMAIN EVENT PATTERNS ==========

    def demonstrate_domain_events(self) -> None:
        """Show domain event patterns for event sourcing."""
        print("\n=== Domain Event Patterns ===")

        # Order created event
        order_created_event = FlextModels.DomainEvent(
            aggregate_id=str(uuid4()),
            event_type="OrderCreated",
            data={
                "order_id": str(uuid4()),
                "customer_id": "CUST-123",
                "total": str(Decimal("109.97")),  # Convert Decimal to string for JSON
                "status": "pending",
                "created_at": datetime.now(UTC).isoformat(),
            },
            metadata={
                "sequence_number": 1,
                "aggregate_version": 1,
            },
        )
        print(f"✅ Event: {order_created_event.event_type}")
        print(f"Aggregate: {order_created_event.aggregate_id}")
        print(f"Sequence: {order_created_event.metadata.get('sequence_number')}")
        print(f"Version: {order_created_event.metadata.get('aggregate_version')}")

        # Payment processed event
        payment_processed_event = FlextModels.DomainEvent(
            aggregate_id=order_created_event.aggregate_id,
            event_type="PaymentProcessed",
            data={
                "payment_id": str(uuid4()),
                "amount": str(Decimal("109.97")),  # Convert Decimal to string for JSON
                "method": "credit_card",
                "status": "success",
                "processed_at": datetime.now(UTC).isoformat(),
            },
            metadata={
                "sequence_number": 2,
                "aggregate_version": 2,
            },
        )
        print(f"\n✅ Payment event: {payment_processed_event.event_type}")
        print(f"Payment ID: {payment_processed_event.data['payment_id']}")

        # Order shipped event
        order_shipped_event = FlextModels.DomainEvent(
            aggregate_id=order_created_event.aggregate_id,
            event_type="OrderShipped",
            data={
                "shipment_id": str(uuid4()),
                "carrier": "FedEx",
                "tracking_number": "1234567890",
                "estimated_delivery": (
                    datetime.now(UTC) + timedelta(days=3)
                ).isoformat(),
            },
            metadata={
                "sequence_number": 3,
                "aggregate_version": 3,
            },
        )
        print(f"\n✅ Shipment event: {order_shipped_event.event_type}")
        print(f"Tracking: {order_shipped_event.data['tracking_number']}")

        # Store events
        self._event_store.extend([
            order_created_event,
            payment_processed_event,
            order_shipped_event,
        ])

    # ========== EVENT AGGREGATION ==========

    def demonstrate_event_aggregation(self) -> None:
        """Show event aggregation and projection patterns."""
        print("\n=== Event Aggregation ===")

        # Create multiple events for the same aggregate
        user_aggregate_id = str(uuid4())

        events = [
            FlextModels.DomainEvent(
                aggregate_id=user_aggregate_id,
                event_type="UserRegistered",
                data={"username": "jane_doe", "email": "jane@example.com"},
                metadata={
                    "sequence_number": 1,
                    "aggregate_version": 1,
                },
            ),
            FlextModels.DomainEvent(
                aggregate_id=user_aggregate_id,
                event_type="EmailVerified",
                data={"verified_at": datetime.now(UTC).isoformat()},
                metadata={
                    "sequence_number": 2,
                    "aggregate_version": 2,
                },
            ),
            FlextModels.DomainEvent(
                aggregate_id=user_aggregate_id,
                event_type="ProfileUpdated",
                data={"name": "Jane Smith", "bio": "Software Engineer"},
                metadata={
                    "sequence_number": 3,
                    "aggregate_version": 3,
                },
            ),
            FlextModels.DomainEvent(
                aggregate_id=user_aggregate_id,
                event_type="RoleGranted",
                data={"role": "premium_user", "granted_by": "admin"},
                metadata={
                    "sequence_number": 4,
                    "aggregate_version": 4,
                },
            ),
        ]

        # Build aggregate state from events
        user_state: FlextTypes.Core.Dict = {}
        for event in events:
            version = event.metadata.get("aggregate_version", 0)
            print(f"Processing: {event.event_type} (v{version})")

            if event.event_type == "UserRegistered":
                user_state.update(event.data)
                user_state["roles"] = []
                user_state["verified"] = False
            elif event.event_type == "EmailVerified":
                user_state["verified"] = True
                user_state["verified_at"] = event.data["verified_at"]
            elif event.event_type == "ProfileUpdated":
                user_state.update(event.data)
            elif event.event_type == "RoleGranted":
                roles_raw = user_state.get("roles", [])
                # Use Python 3.13+ type narrowing with explicit type annotation
                roles: list[str] = (
                    roles_raw
                    if isinstance(roles_raw, list)
                    and all(isinstance(r, str) for r in roles_raw)
                    else []
                )
                role = event.data["role"]
                if isinstance(role, str):
                    roles.append(role)
                    user_state["roles"] = roles

        print("\n✅ Final user state:")
        print(f"Username: {user_state.get('username')}")
        print(f"Name: {user_state.get('name')}")
        print(f"Verified: {user_state.get('verified')}")
        print(f"Roles: {user_state.get('roles')}")

        self._event_store.extend(events)

    # ========== MESSAGE ROUTING ==========

    def demonstrate_message_routing(self) -> None:
        """Show message routing and handling patterns."""
        print("\n=== Message Routing ===")

        # Create messages with routing information
        messages = [
            FlextModels.Payload[FlextTypes.Core.Dict](
                data={"action": "send_email", "to": "user@example.com"},
                metadata={
                    "source_service": "order_service",
                    "message_type": "notification",
                    "target_service": "email_service",
                    "routing_key": "notifications.email",
                    "priority": 5,
                },
            ),
            FlextModels.Payload[FlextTypes.Core.Dict](
                data={"action": "update_inventory", "sku": "PROD-A", "quantity": -2},
                metadata={
                    "source_service": "order_service",
                    "message_type": "command",
                    "target_service": "inventory_service",
                    "routing_key": "inventory.update",
                    "priority": 8,
                },
            ),
            FlextModels.Payload[FlextTypes.Core.Dict](
                data={
                    "action": "calculate_shipping",
                    "weight": 2.5,
                    "destination": "NY",
                },
                metadata={
                    "source_service": "order_service",
                    "message_type": "query",
                    "target_service": "shipping_service",
                    "routing_key": "shipping.calculate",
                    "priority": 3,
                },
            ),
        ]

        # Route messages based on metadata
        routes: dict[str, list[FlextModels.Payload[FlextTypes.Core.Dict]]] = {}
        for msg in messages:
            target = msg.metadata.get("target_service", "unknown")
            if isinstance(target, str):
                if target not in routes:
                    routes[target] = []
                routes[target].append(msg)

        print("✅ Message routing:")
        for service, service_messages in routes.items():
            print(f"\n{service}:")
            for msg in service_messages:
                print(
                    f"  - {msg.data['action']} (priority: {msg.metadata['priority']})"
                )

        self._message_queue.extend(messages)

    # ========== CORRELATION TRACKING ==========

    def demonstrate_correlation_tracking(self) -> None:
        """Show request/response correlation patterns."""
        print("\n=== Correlation Tracking ===")

        # Original request
        correlation_id = str(uuid4())

        request_payload = FlextModels.Payload[FlextTypes.Core.Dict](
            data={
                "operation": "process_order",
                "order_id": str(uuid4()),
                "amount": 150.00,
                "message_id": str(uuid4()),  # Add message_id to data
            },
            metadata={
                "source_service": "api_gateway",
                "message_type": "request",
                "correlation_id": correlation_id,
                "session_id": str(uuid4()),
                "user_id": "user_789",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        print("✅ Request sent")
        print(f"Correlation ID: {correlation_id}")
        print(f"Operation: {request_payload.data['operation']}")

        # Chain of internal messages with same correlation
        internal_messages: list[FlextModels.Payload[FlextTypes.Core.Dict]] = []

        # Validation message
        validation_msg = FlextModels.Payload[FlextTypes.Core.Dict](
            data={"validate": "order", "order_id": request_payload.data["order_id"]},
            metadata={
                "source_service": "order_service",
                "message_type": "internal",
                "correlation_id": correlation_id,
                "parent_message_id": request_payload.data.get("message_id", ""),
                "step": "validation",
            },
        )
        internal_messages.append(validation_msg)

        # Payment message
        payment_msg = FlextModels.Payload[FlextTypes.Core.Dict](
            data={"process": "payment", "amount": request_payload.data["amount"]},
            metadata={
                "source_service": "order_service",
                "message_type": "internal",
                "correlation_id": correlation_id,
                "parent_message_id": request_payload.data.get("message_id", ""),
                "step": "payment",
            },
        )
        internal_messages.append(payment_msg)

        # Response message
        response_payload = FlextModels.Payload[FlextTypes.Core.Dict](
            data={
                "status": "success",
                "order_id": request_payload.data["order_id"],
                "transaction_id": str(uuid4()),
                "message": "Order processed successfully",
            },
            metadata={
                "source_service": "order_service",
                "message_type": "response",
                "correlation_id": correlation_id,
                "request_message_id": request_payload.data.get("message_id", ""),
                "duration_ms": 234,
            },
        )

        print("\n✅ Response received")
        print(f"Status: {response_payload.data['status']}")
        print(
            f"Same correlation: {response_payload.metadata['correlation_id'] == correlation_id}"
        )
        print(f"Processing time: {response_payload.metadata['duration_ms']}ms")

        self._message_queue.extend([
            request_payload,
            *internal_messages,
            response_payload,
        ])

    # ========== EVENT REPLAY ==========

    def demonstrate_event_replay(self) -> None:
        """Show event replay for rebuilding state."""
        print("\n=== Event Replay ===")

        # Events representing account transactions
        account_id = str(uuid4())
        events = [
            FlextModels.DomainEvent(
                aggregate_id=account_id,
                event_type="AccountOpened",
                data={"initial_balance": 0, "currency": "USD"},
                metadata={
                    "sequence_number": 1,
                    "aggregate_version": 1,
                },
            ),
            FlextModels.DomainEvent(
                aggregate_id=account_id,
                event_type="MoneyDeposited",
                data={"amount": 1000, "source": "direct_deposit"},
                metadata={
                    "sequence_number": 2,
                    "aggregate_version": 2,
                },
            ),
            FlextModels.DomainEvent(
                aggregate_id=account_id,
                event_type="MoneyWithdrawn",
                data={"amount": 250, "reason": "ATM"},
                metadata={
                    "sequence_number": 3,
                    "aggregate_version": 3,
                },
            ),
            FlextModels.DomainEvent(
                aggregate_id=account_id,
                event_type="MoneyDeposited",
                data={"amount": 500, "source": "transfer"},
                metadata={
                    "sequence_number": 4,
                    "aggregate_version": 4,
                },
            ),
            FlextModels.DomainEvent(
                aggregate_id=account_id,
                event_type="MoneyWithdrawn",
                data={"amount": 100, "reason": "purchase"},
                metadata={
                    "sequence_number": 5,
                    "aggregate_version": 5,
                },
            ),
        ]

        # Replay events to rebuild balance
        balance: float = 0.0
        print("Replaying account events:")
        for event in events:
            if event.event_type == "AccountOpened":
                initial_balance = event.data["initial_balance"]
                if isinstance(initial_balance, (int, float, str)):
                    balance = float(initial_balance)
                print(f"  Account opened: ${balance}")
            elif event.event_type == "MoneyDeposited":
                amount_value = event.data["amount"]
                if isinstance(amount_value, (int, float, str)):
                    amount = float(amount_value)
                    balance += amount
                    print(f"  + Deposited ${amount}: balance = ${balance}")
            elif event.event_type == "MoneyWithdrawn":
                amount_value = event.data["amount"]
                if isinstance(amount_value, (int, float, str)):
                    amount = float(amount_value)
                    balance -= amount
                    print(f"  - Withdrew ${amount}: balance = ${balance}")

        print(f"\n✅ Final balance after replay: ${balance}")
        print(f"Total events: {len(events)}")
        final_version = events[-1].metadata.get("aggregate_version", 0)
        print(f"Final version: {final_version}")

        self._event_store.extend(events)

    # ========== DEPRECATED PATTERNS ==========

    def demonstrate_deprecated_patterns(self) -> None:
        """Show deprecated messaging patterns."""
        print("\n=== ⚠️ DEPRECATED PATTERNS ===")

        # OLD: Untyped dictionaries for messages (DEPRECATED)
        warnings.warn(
            "Untyped dictionaries are DEPRECATED! Use FlextModels.Payload.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("❌ OLD WAY (untyped dict):")
        print("message = {'type': 'order', 'data': {...}}")

        print("\n✅ CORRECT WAY (typed Payload):")
        print("message = FlextModels.Payload(")
        print("    data={...},")
        print("    metadata={'source_service': 'service', 'message_type': 'order'}")
        print(")")

        # OLD: Manual event timestamps (DEPRECATED)
        warnings.warn(
            "Manual timestamps are DEPRECATED! FlextModels add them automatically.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (manual timestamp):")
        print("event = {'timestamp': datetime.now(), 'type': 'event'}")

        print("\n✅ CORRECT WAY (auto timestamp):")
        print("event = FlextModels.DomainEvent(")
        print("    event_type='OrderCreated', event_data={...}")
        print(")  # timestamp added automatically")

        # OLD: String-based routing (DEPRECATED)
        warnings.warn(
            "String routing is DEPRECATED! Use structured metadata.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (string routing):")
        print("send_message('order_service', message)")

        print("\n✅ CORRECT WAY (metadata routing):")
        print("payload = FlextModels.Payload(")
        print("    data={...},")
        print("    metadata={")
        print("        'target_service': 'order_service',")
        print("        'routing_key': 'orders.create'")
        print("    }")
        print(")")


def main() -> None:
    """Main entry point demonstrating messaging patterns."""
    service = MessagingPatternsService()

    print("=" * 60)
    print("MESSAGING AND EVENT PATTERNS")
    print("Payloads, Domain Events, and Message-Driven Architecture")
    print("=" * 60)

    # Basic patterns
    service.demonstrate_basic_payload()
    service.demonstrate_command_patterns()
    service.demonstrate_query_patterns()

    # Event patterns
    service.demonstrate_domain_events()
    service.demonstrate_event_aggregation()
    service.demonstrate_event_replay()

    # Routing and correlation
    service.demonstrate_message_routing()
    service.demonstrate_correlation_tracking()

    # Deprecation warnings
    service.demonstrate_deprecated_patterns()

    # Summary
    print("\n" + "=" * 60)
    print("✅ ALL messaging patterns demonstrated!")
    print(f"📊 Events stored: {service.get_event_count()}")
    print(f"📬 Messages queued: {service.get_message_count()}")
    print("🎯 Next: See advanced examples for complex patterns")
    print("=" * 60)


if __name__ == "__main__":
    main()
