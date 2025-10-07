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
from typing import cast
from uuid import uuid4

from flext_core import (
    FlextConstants,
    FlextCore,
    FlextExceptions,
    FlextLogger,
    FlextResult,
    FlextRuntime,
    FlextService,
    FlextTypes,
)

from .example_scenarios import ExampleScenarios


class MessagingPatternsService(FlextService[FlextTypes.Dict]):
    """Service demonstrating ALL messaging and event patterns with FlextMixins.Service infrastructure.

    This service inherits from FlextCore.Service to demonstrate:
    - Inherited container property (FlextContainer singleton)
    - Inherited logger property (FlextLogger with service context - MESSAGING FOCUS!)
    - Inherited context property (FlextContext for request/correlation tracking)
    - Inherited config property (FlextConfig with message processing settings)
    - Inherited metrics property (FlextMetrics for message observability)

    The focus is on demonstrating messaging patterns (Payloads, DomainEvents, CQRS)
    with structured logging, correlation tracking, and event sourcing patterns,
    while leveraging complete FlextMixins.Service infrastructure for orchestration.
    """

    def __init__(self) -> None:
        """Initialize with inherited FlextMixins.Service infrastructure.

        Note: No manual logger initialization needed!
        All infrastructure is inherited from FlextCore.Service base class:
        - self.logger: FlextLogger with service context (ALREADY CONFIGURED!)
        - self.container: FlextContainer global singleton
        - self.context: FlextContext for correlation and request tracking
        - self.config: FlextConfig with message processing configuration
        - self.metrics: FlextMetrics for message observability
        """
        super().__init__()
        # Use self.logger from FlextMixins.Logging, not logger
        self._event_store: list[FlextCore.Models.DomainEvent] = []
        self._message_queue: list[FlextCore.Models.Payload[FlextTypes.Dict]] = []
        self._scenarios = ExampleScenarios()
        self._metadata = self._scenarios.metadata(tags=["messaging", "demo"])
        # Filter out incompatible metadata fields for Payload models and ensure type compatibility
        self._safe_metadata: dict[str, str | int | float] = {
            k: v
            for k, v in self._metadata.items()
            if k != "tags" and isinstance(v, (str, int, float))
        }
        self._user = self._scenarios.user()
        self._users = self._scenarios.users()
        self._order = self._scenarios.realistic_data()["order"]
        self._api_payload = self._scenarios.payload()

        # Demonstrate inherited logger (no manual instantiation needed!)
        self.logger.info(
            "MessagingPatternsService initialized with inherited infrastructure",
            extra={
                "service_type": "Messaging & Event Patterns demonstration",
                "event_store_size": len(self._event_store),
                "message_queue_size": len(self._message_queue),
                "correlation_tracking": True,
                "event_sourcing": True,
            },
        )

    def execute(self) -> FlextResult[FlextTypes.Dict]:
        """Execute all messaging and event pattern demonstrations.

        Demonstrates inherited infrastructure alongside messaging patterns:
        - Inherited logger for structured messaging logs
        - Inherited context for correlation tracking
        - Complete event sourcing and CQRS patterns
        - Foundation layer integration (FlextConstants, FlextExceptions, FlextRuntime)

        Returns:
            FlextResult[Dict] with demonstration summary including infrastructure details

        """
        self.logger.info("Starting comprehensive messaging patterns demonstration")

        try:
            # Basic patterns
            self.demonstrate_basic_payload()
            self.demonstrate_command_patterns()
            self.demonstrate_query_patterns()

            # Event patterns
            self.demonstrate_domain_events()
            self.demonstrate_event_aggregation()
            self.demonstrate_event_replay()

            # Routing and correlation
            self.demonstrate_message_routing()
            self.demonstrate_correlation_tracking()

            # NEW: FlextResult v0.9.9+ methods for messaging
            self.demonstrate_from_callable_messaging()
            self.demonstrate_flow_through_messaging()
            self.demonstrate_lash_messaging()
            self.demonstrate_alt_messaging()
            self.demonstrate_value_or_call_messaging()

            # Foundation layer integration
            self.demonstrate_flext_constants_messages()
            self.demonstrate_flext_exceptions_messaging()
            self.demonstrate_flext_runtime_messaging()

            # Deprecation warnings
            self.demonstrate_deprecated_patterns()

            summary: dict[str, object] = {
                "demonstrations_completed": 17,
                "events_stored": len(self._event_store),
                "messages_queued": len(self._message_queue),
                "status": "completed",
                "infrastructure": {
                    "logger": type(self.logger).__name__,
                    "container": type(self.container).__name__,
                    "context": type(self.context).__name__,
                    "config": type(self.config).__name__,
                },
                "messaging_features": {
                    "event_sourcing": True,
                    "correlation_tracking": True,
                    "cqrs_patterns": True,
                    "domain_events": True,
                },
            }

            self.logger.info(
                "Messaging patterns demonstration completed successfully", extra=summary
            )

            return FlextResult[FlextTypes.Dict].ok(summary)

        except Exception as e:
            error_msg = f"Messaging patterns demonstration failed: {e}"
            self.logger.exception(error_msg)
            return FlextResult[FlextTypes.Dict].fail(
                error_msg, error_code=FlextConstants.Errors.VALIDATION_ERROR
            )

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

        greeting_data: FlextTypes.Dict = {
            "message": "Hello, FLEXT!",
            "source_service": "greeting_service",
            "message_type": "greeting",
            "user": self._user,
        }
        greeting_payload = FlextCore.Models.Payload[FlextTypes.Dict](
            data=greeting_data,
            metadata={
                **self._safe_metadata,
                "timestamp": datetime.now(UTC).isoformat(),
                "correlation_id": str(
                    self._api_payload.get("request_id", str(uuid4()))
                ),
            },
        )
        data_dict = greeting_payload.data
        print(f"✅ Greeting: {data_dict.get('message')}")
        print(f"Source: {data_dict.get('source_service')}")
        print(f"Type: {data_dict.get('message_type')}")
        user_data = cast("FlextTypes.Dict", data_dict.get("user", {}))
        print(f"User email: {user_data.get('email')}")
        print(f"Timestamp: {greeting_payload.metadata.get('timestamp')}")

        user_payload = FlextCore.Models.Payload[FlextTypes.Dict](
            data=self._user,
            metadata={
                **self._safe_metadata,
                "source_service": "user_service",
                "message_type": "user_created",
                "version": "1.0",
                "correlation_id": str(uuid4()),
            },
        )
        print("\n✅ User payload created")
        print(f"User ID: {user_payload.data['id']}")
        print(f"Username: {user_payload.data.get('name')}")
        print(f"Metadata: {user_payload.metadata}")

        self._message_queue.append(user_payload)

    # ========== COMMAND PATTERNS ==========

    def demonstrate_command_patterns(self) -> None:
        """Show command message patterns for CQRS."""
        print("\n=== Command Patterns ===")

        order_id = self._order["order_id"]
        total = float(self._order["total"])

        create_order_command = FlextCore.Models.Payload[FlextTypes.Dict](
            data={
                "command": "CreateOrder",
                "order_id": order_id,
                "customer_id": self._order["customer_id"],
                "items": self._order["items"],
                "total": total,
            },
            metadata={
                **self._safe_metadata,
                "source_service": "web_api",
                "message_type": "command",
                "command_type": "CreateOrder",
                "correlation_id": str(uuid4()),
                "user_id": str(self._user["id"]),
            },
        )
        command_data = create_order_command.data
        print(f"✅ Command: {command_data['command']}")
        print(f"Order ID: {command_data['order_id']}")
        print(f"Total: ${command_data['total']}")

        process_payment_command = FlextCore.Models.Payload[FlextTypes.Dict](
            data={
                "command": "ProcessPayment",
                "order_id": command_data["order_id"],
                "amount": command_data["total"],
                "payment_method": "credit_card",
                "card_token": "tok_xxxxx",
            },
            metadata={
                **self._safe_metadata,
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

        user_id = self._user["id"]

        get_orders_query = FlextCore.Models.Payload[FlextTypes.Dict](
            data={
                "query": "GetUserOrders",
                "user_id": user_id,
                "filters": {
                    "status": ["pending", "completed"],
                    "date_from": datetime.now(UTC).date().replace(day=1).isoformat(),
                    "date_to": datetime.now(UTC).date().isoformat(),
                },
                "pagination": FlextCore.Models.Pagination(page=1, size=20).model_dump(),
            },
            metadata={
                **self._safe_metadata,
                "source_service": "web_api",
                "message_type": "query",
                "query_type": "GetUserOrders",
                "correlation_id": str(uuid4()),
                "cache_key": f"{user_id}_orders",
                "cache_ttl": 300,
            },
        )
        query_data = get_orders_query.data
        print(f"✅ Query: {query_data['query']}")
        print(f"User: {query_data['user_id']}")
        print(f"Filters: {query_data['filters']}")

        sales_report_query = FlextCore.Models.Payload[FlextTypes.Dict](
            data={
                "query": "GenerateSalesReport",
                "period": "monthly",
                "year": datetime.now(UTC).year,
                "metrics": ["total_sales", "order_count", "avg_order_value"],
                "group_by": ["product_category", "region"],
            },
            metadata={
                **self._safe_metadata,
                "source_service": "analytics_service",
                "message_type": "query",
                "query_type": "AggregationQuery",
                "priority": 3,
                "timeout_seconds": 30,
            },
        )
        report_data = sales_report_query.data
        print(f"\n✅ Report query: {report_data['query']}")
        print(f"Metrics: {report_data['metrics']}")

        self._message_queue.extend([get_orders_query, sales_report_query])

    # ========== DOMAIN EVENT PATTERNS ==========

    def demonstrate_domain_events(self) -> None:
        """Show domain event patterns for event sourcing."""
        print("\n=== Domain Event Patterns ===")

        aggregate_id = self._order["order_id"]
        order_total = str(self._order["total"])

        order_created_event = FlextCore.Models.DomainEvent(
            aggregate_id=aggregate_id,
            event_type="OrderCreated",
            data={
                "order_id": aggregate_id,
                "customer_id": self._order["customer_id"],
                "total": order_total,
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

        payment_processed_event = FlextCore.Models.DomainEvent(
            aggregate_id=aggregate_id,
            event_type="PaymentProcessed",
            data={
                "payment_id": str(uuid4()),
                "amount": order_total,
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

        order_shipped_event = FlextCore.Models.DomainEvent(
            aggregate_id=aggregate_id,
            event_type="OrderShipped",
            data={
                "shipment_id": str(uuid4()),
                "carrier": "FedEx",
                "tracking_number": str(
                    self._api_payload.get("request_id", "tracking-123")
                ),
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

        self._event_store.extend([
            order_created_event,
            payment_processed_event,
            order_shipped_event,
        ])

    # ========== EVENT AGGREGATION ==========

    def demonstrate_event_aggregation(self) -> None:
        """Show event aggregation and projection patterns."""
        print("\n=== Event Aggregation ===")

        user_record = self._users[0]
        user_aggregate_id = str(user_record["id"])

        events = [
            FlextCore.Models.DomainEvent(
                aggregate_id=user_aggregate_id,
                event_type="UserRegistered",
                data={
                    "username": user_record["name"],
                    "email": user_record["email"],
                },
                metadata={
                    "sequence_number": 1,
                    "aggregate_version": 1,
                },
            ),
            FlextCore.Models.DomainEvent(
                aggregate_id=user_aggregate_id,
                event_type="EmailVerified",
                data={"verified_at": datetime.now(UTC).isoformat()},
                metadata={
                    "sequence_number": 2,
                    "aggregate_version": 2,
                },
            ),
            FlextCore.Models.DomainEvent(
                aggregate_id=user_aggregate_id,
                event_type="ProfileUpdated",
                data={
                    "name": user_record["name"],
                    "bio": "Scenario generated profile",
                },
                metadata={
                    "sequence_number": 3,
                    "aggregate_version": 3,
                },
            ),
            FlextCore.Models.DomainEvent(
                aggregate_id=user_aggregate_id,
                event_type="RoleGranted",
                data={"role": "premium_user", "granted_by": "admin"},
                metadata={
                    "sequence_number": 4,
                    "aggregate_version": 4,
                },
            ),
        ]

        user_state: FlextTypes.Dict = {}
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
                roles: FlextCore.Types.StringList
                if isinstance(roles_raw, list) and all(
                    isinstance(role, str) for role in roles_raw
                ):
                    roles = list(roles_raw)
                else:
                    roles = []
                role: str = event.data["role"]
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

        messages = [
            FlextCore.Models.Payload[FlextTypes.Dict](
                data={
                    "action": "send_email",
                    "to": self._user["email"],
                    "template": "welcome",
                },
                metadata={
                    **self._safe_metadata,
                    "source_service": "order_service",
                    "message_type": "notification",
                    "target_service": "email_service",
                    "routing_key": "notifications.email",
                    "priority": 5,
                },
            ),
            FlextCore.Models.Payload[FlextTypes.Dict](
                data={
                    "action": "update_inventory",
                    "items": self._order["items"],
                },
                metadata={
                    **self._safe_metadata,
                    "source_service": "order_service",
                    "message_type": "command",
                    "target_service": "inventory_service",
                    "routing_key": "inventory.update",
                    "priority": 8,
                },
            ),
            FlextCore.Models.Payload[FlextTypes.Dict](
                data={
                    "action": "calculate_shipping",
                    "weight": 2.5,
                    "destination": self._metadata.get("component", "shipping"),
                },
                metadata={
                    **self._safe_metadata,
                    "source_service": "order_service",
                    "message_type": "query",
                    "target_service": "shipping_service",
                    "routing_key": "shipping.calculate",
                    "priority": 3,
                },
            ),
        ]

        routes: dict[str, list[FlextCore.Models.Payload[FlextTypes.Dict]]] = {}
        for msg in messages:
            target = msg.metadata.get("target_service", "unknown")
            if isinstance(target, str):
                routes.setdefault(target, []).append(msg)

        print("✅ Message routing:")
        for service, service_messages in routes.items():
            print(f"\n{service}:")
            for msg in service_messages:
                msg_data = msg.data
                msg_metadata = msg.metadata
                print(
                    f"  - {msg_data['action']} (priority: {msg_metadata['priority']})",
                )

        self._message_queue.extend(messages)

    # ========== CORRELATION TRACKING ==========

    def demonstrate_correlation_tracking(self) -> None:
        """Show request/response correlation patterns."""
        print("\n=== Correlation Tracking ===")

        correlation_id = str(uuid4())
        order_id = self._order["order_id"]

        request_payload = FlextCore.Models.Payload[FlextTypes.Dict](
            data={
                "operation": "process_order",
                "order_id": order_id,
                "amount": float(self._order["total"]),
                "message_id": str(uuid4()),
            },
            metadata={
                **self._safe_metadata,
                "source_service": "api_gateway",
                "message_type": "request",
                "correlation_id": correlation_id,
                "session_id": str(uuid4()),
                "user_id": str(self._user["id"]),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        print("✅ Request sent")
        print(f"Correlation ID: {correlation_id}")
        request_data = request_payload.data
        print(f"Operation: {request_data['operation']}")

        internal_messages: list[FlextCore.Models.Payload[FlextTypes.Dict]] = []

        validation_msg = FlextCore.Models.Payload[FlextTypes.Dict](
            data={"validate": "order", "order_id": order_id},
            metadata={
                **self._safe_metadata,
                "source_service": "order_service",
                "message_type": "internal",
                "correlation_id": correlation_id,
                "parent_message_id": str(request_data["message_id"]),
                "step": "validation",
            },
        )
        internal_messages.append(validation_msg)

        payment_msg = FlextCore.Models.Payload[FlextTypes.Dict](
            data={"process": "payment", "amount": request_data["amount"]},
            metadata={
                **self._safe_metadata,
                "source_service": "order_service",
                "message_type": "internal",
                "correlation_id": correlation_id,
                "parent_message_id": str(request_data["message_id"]),
                "step": "payment",
            },
        )
        internal_messages.append(payment_msg)

        response_payload = FlextCore.Models.Payload[FlextTypes.Dict](
            data={
                "status": "success",
                "order_id": order_id,
                "transaction_id": str(uuid4()),
            },
            metadata={
                **self._safe_metadata,
                "source_service": "order_service",
                "message_type": "response",
                "correlation_id": correlation_id,
                "duration_ms": 120,
            },
        )

        for msg in [request_payload, *internal_messages, response_payload]:
            self._message_queue.append(msg)

        print("\n✅ Response sent")
        print(f"Status: {response_payload.data['status']}")

    # ========== EVENT REPLAY ==========

    def demonstrate_event_replay(self) -> None:
        """Show event replay for rebuilding state."""
        print("\n=== Event Replay ===")

        account_id = str(uuid4())
        order_total = float(self._order["total"])
        first_item = self._order["items"][0]
        second_item = (
            self._order["items"][0]
            if len(self._order["items"]) == 1
            else self._order["items"][1]
        )

        events = [
            FlextCore.Models.DomainEvent(
                aggregate_id=account_id,
                event_type="AccountOpened",
                data={"initial_balance": 0, "currency": "USD"},
                metadata={
                    "sequence_number": 1,
                    "aggregate_version": 1,
                },
            ),
            FlextCore.Models.DomainEvent(
                aggregate_id=account_id,
                event_type="MoneyDeposited",
                data={"amount": order_total, "source": "order_payment"},
                metadata={
                    "sequence_number": 2,
                    "aggregate_version": 2,
                },
            ),
            FlextCore.Models.DomainEvent(
                aggregate_id=account_id,
                event_type="MoneyWithdrawn",
                data={
                    "amount": float(first_item["price"]),
                    "reason": "inventory_purchase",
                },
                metadata={
                    "sequence_number": 3,
                    "aggregate_version": 3,
                },
            ),
            FlextCore.Models.DomainEvent(
                aggregate_id=account_id,
                event_type="MoneyDeposited",
                data={
                    "amount": float(second_item["price"]),
                    "source": "upsell_revenue",
                },
                metadata={
                    "sequence_number": 4,
                    "aggregate_version": 4,
                },
            ),
            FlextCore.Models.DomainEvent(
                aggregate_id=account_id,
                event_type="MoneyWithdrawn",
                data={"amount": 100, "reason": "operational_cost"},
                metadata={
                    "sequence_number": 5,
                    "aggregate_version": 5,
                },
            ),
        ]

        balance: float = 0.0
        print("Replaying account events:")
        for event in events:
            event_data = event.data
            if event.event_type == "AccountOpened":
                initial_balance = event_data["initial_balance"]
                try:
                    balance = (
                        float(initial_balance)
                        if isinstance(initial_balance, (int, float, str))
                        else 0.0
                    )
                except (TypeError, ValueError):
                    balance = 0.0
                print(f"  Account opened: ${balance}")
            elif event.event_type == "MoneyDeposited":
                amount = event_data["amount"]
                try:
                    balance += (
                        float(amount) if isinstance(amount, (int, float, str)) else 0.0
                    )
                except (TypeError, ValueError):
                    balance += 0.0
                print(f"  + Deposited ${amount}: balance = ${balance}")
            elif event.event_type == "MoneyWithdrawn":
                amount = event_data["amount"]
                try:
                    balance -= (
                        float(amount) if isinstance(amount, (int, float, str)) else 0.0
                    )
                except (TypeError, ValueError):
                    balance -= 0.0
                print(f"  - Withdrew ${amount}: balance = ${balance}")

        print(f"\n✅ Final balance after replay: ${balance}")
        print(f"Total events: {len(events)}")
        final_version = events[-1].metadata.get("aggregate_version", 0)
        print(f"Final version: {final_version}")

        self._event_store.extend(events)

    # ========== FOUNDATION LAYER INTEGRATION ==========

    # ========== NEW FLEXTRESULT METHODS (v0.9.9+) ==========

    def demonstrate_from_callable_messaging(self) -> None:
        """Show from_callable for safe message processing."""
        print("\n=== from_callable: Safe Message Processing ===")

        def risky_message_parse(
            data: dict[str, object],
        ) -> FlextCore.Models.Payload[FlextTypes.Dict]:
            """Message parsing that might raise exceptions."""
            if not data.get("message_type"):
                msg = "Message type required"
                raise FlextExceptions.ValidationError(
                    msg, field="message_type", value=None
                )
            return FlextCore.Models.Payload[FlextTypes.Dict](
                data=data,
                metadata={
                    **self._safe_metadata,
                    "message_type": str(data["message_type"]),
                },
            )

        # Safe message parsing without try/except
        result: FlextResult[FlextCore.Models.Payload[FlextTypes.Dict]] = (
            FlextResult.from_callable(
                lambda: risky_message_parse({
                    "message_type": "order",
                    "data": self._order,
                }),
            )
        )
        if result.is_success:
            payload: FlextCore.Models.Payload[FlextTypes.Dict] = result.unwrap()
            print(f"✅ Message parsed: {payload.metadata.get('message_type')}")

        # Failed parsing captured as Result
        failed: FlextResult[FlextCore.Models.Payload[FlextTypes.Dict]] = (
            FlextResult.from_callable(
                lambda: risky_message_parse({}),
            )
        )
        if failed.is_failure:
            print(f"❌ Parse failed: {failed.error}")

    def demonstrate_flow_through_messaging(self) -> None:
        """Show flow_through for message pipeline composition."""
        print("\n=== flow_through: Message Pipeline ===")

        def validate_message(
            msg: FlextCore.Models.Payload[FlextTypes.Dict],
        ) -> FlextResult[FlextCore.Models.Payload[FlextTypes.Dict]]:
            """Validate message structure."""
            if not msg.data.get("command"):
                return FlextResult[FlextCore.Models.Payload[FlextTypes.Dict]].fail(
                    "Missing command"
                )
            return FlextResult[FlextCore.Models.Payload[FlextTypes.Dict]].ok(msg)

        def enrich_message(
            msg: FlextCore.Models.Payload[FlextTypes.Dict],
        ) -> FlextResult[FlextCore.Models.Payload[FlextTypes.Dict]]:
            """Add metadata to message."""
            enriched = FlextCore.Models.Payload[FlextTypes.Dict](
                data=msg.data,
                metadata={
                    **msg.metadata,
                    "enriched": True,
                    "enriched_at": datetime.now(UTC).isoformat(),
                },
            )
            return FlextResult[FlextCore.Models.Payload[FlextTypes.Dict]].ok(enriched)

        def route_message(
            msg: FlextCore.Models.Payload[FlextTypes.Dict],
        ) -> FlextResult[FlextCore.Models.Payload[FlextTypes.Dict]]:
            """Add routing information."""
            routed = FlextCore.Models.Payload[FlextTypes.Dict](
                data=msg.data,
                metadata={
                    **msg.metadata,
                    "target_service": "order_processor",
                    "routing_key": "orders.process",
                },
            )
            return FlextResult[FlextCore.Models.Payload[FlextTypes.Dict]].ok(routed)

        # Create initial message
        initial_msg = FlextCore.Models.Payload[FlextTypes.Dict](
            data={"command": "ProcessOrder", "order_id": self._order["order_id"]},
            metadata={**self._safe_metadata},
        )

        # Flow through message pipeline
        result = (
            FlextResult[FlextCore.Models.Payload[FlextTypes.Dict]]
            .ok(initial_msg)
            .flow_through(
                validate_message,
                enrich_message,
                route_message,
            )
        )

        if result.is_success:
            final_msg = result.unwrap()
            print(f"✅ Message pipeline complete: {final_msg.data['command']}")
            print(f"   Enriched: {final_msg.metadata.get('enriched')}")
            print(f"   Routed to: {final_msg.metadata.get('target_service')}")

    def demonstrate_lash_messaging(self) -> None:
        """Show lash for message retry and fallback."""
        print("\n=== lash: Message Retry with Fallback ===")

        attempt_count = {"count": 0}

        def send_to_primary_queue(
            msg: FlextCore.Models.Payload[FlextTypes.Dict],
        ) -> FlextResult[str]:
            """Try sending to primary message queue (might fail)."""
            attempt_count["count"] += 1
            if attempt_count["count"] < 3:
                return FlextResult[str].fail("Primary queue unavailable")
            return FlextResult[str].ok(f"Sent to primary: {msg.data.get('command')}")

        def send_to_backup_queue(error: str) -> FlextResult[str]:
            """Fallback to backup queue on failure."""
            print(f"   Retrying via backup queue after: {error}")
            return FlextResult[str].ok("Sent to backup queue")

        message = FlextCore.Models.Payload[FlextTypes.Dict](
            data={"command": "ProcessPayment", "amount": 100},
            metadata={**self._safe_metadata},
        )

        # Try primary, fall back to backup on failure
        result = send_to_primary_queue(message).lash(send_to_backup_queue)
        if result.is_success:
            print(f"✅ Message sent: {result.unwrap()}")

    def demonstrate_alt_messaging(self) -> None:
        """Show alt for event handler fallback."""
        print("\n=== alt: Event Handler Fallback ===")

        def get_specialized_handler(event_type: str) -> FlextResult[str]:
            """Try to get specialized event handler."""
            handlers = {
                "OrderCreated": "OrderCreatedHandler",
                "PaymentProcessed": "PaymentHandler",
            }
            if event_type in handlers:
                return FlextResult[str].ok(handlers[event_type])
            return FlextResult[str].fail(f"No specialized handler for {event_type}")

        def get_default_handler() -> FlextResult[str]:
            """Fallback to default event handler."""
            return FlextResult[str].ok("DefaultEventHandler")

        # Try specialized, use default as fallback
        event_type = "UserRegistered"
        handler = get_specialized_handler(event_type).alt(get_default_handler())

        if handler.is_success:
            print(f"✅ Event handler: {handler.unwrap()}")
            print(f"   Event type: {event_type}")

    def demonstrate_value_or_call_messaging(self) -> None:
        """Show value_or_call for lazy message defaults."""
        print("\n=== value_or_call: Lazy Message Defaults ===")

        def create_default_payload() -> FlextCore.Models.Payload[FlextTypes.Dict]:
            """Create default payload (only called if needed)."""
            print("   Generating default payload...")
            return FlextCore.Models.Payload[FlextTypes.Dict](
                data={"type": "system", "message": "default"},
                metadata={**self._safe_metadata, "is_default": True},
            )

        # Success case - default not called
        msg_result = FlextResult[FlextCore.Models.Payload[FlextTypes.Dict]].ok(
            FlextCore.Models.Payload[FlextTypes.Dict](
                data={"type": "user", "message": "Hello"},
                metadata={**self._safe_metadata},
            ),
        )
        payload = msg_result.value_or_call(create_default_payload)
        print(f"✅ Got user payload: {payload.data['message']}")

        # Failure case - default called lazily
        failed_result = FlextResult[FlextCore.Models.Payload[FlextTypes.Dict]].fail(
            "No message"
        )
        default_payload = failed_result.value_or_call(create_default_payload)
        print(f"✅ Got default payload: {default_payload.data['message']}")

    def demonstrate_flext_constants_messages(self) -> None:
        """Show FlextConstants.Messages integration with messaging patterns."""
        print("\n=== FlextConstants.Messages Integration (Layer 1) ===")

        logger = FlextLogger(__name__)

        # Message constants for validation
        print("FlextConstants.Messages validation:")
        print(f"  TYPE_MISMATCH: {FlextConstants.Messages.TYPE_MISMATCH}")
        print(f"  INVALID_INPUT: {FlextConstants.Messages.INVALID_INPUT}")
        print(f"  VALIDATION_FAILED: {FlextConstants.Messages.VALIDATION_FAILED}")

        # Error codes for messaging
        print("\nFlextConstants.Errors for messaging:")
        print(f"  VALIDATION_ERROR: {FlextConstants.Errors.VALIDATION_ERROR}")
        print(f"  NOT_FOUND_ERROR: {FlextConstants.Errors.NOT_FOUND_ERROR}")
        print(f"  CONFIG_ERROR: {FlextConstants.Errors.CONFIG_ERROR}")

        # Use constants in message validation
        message_data: FlextTypes.Dict = {
            "type": "UserCommand",
            "payload": {"user_id": "123", "action": "update"},
        }

        if not message_data.get("type"):
            logger.error(
                FlextConstants.Messages.INVALID_INPUT,
                extra={"error_code": FlextConstants.Errors.VALIDATION_ERROR},
            )

        logger.info(
            "Message validated successfully",
            extra={"message_type": message_data["type"], "validation_status": "passed"},
        )

        print("✅ FlextConstants.Messages integration demonstrated")

    def demonstrate_flext_exceptions_messaging(self) -> None:
        """Show FlextExceptions integration with messaging error handling."""
        print("\n=== FlextExceptions Integration (Layer 2) ===")

        logger = FlextLogger(__name__)

        # Message validation error
        try:
            message_type = ""
            if not message_type:
                error_message = "Message type is required"
                raise FlextExceptions.ValidationError(
                    error_message,
                    field="message_type",
                    value=message_type,
                )
        except FlextExceptions.ValidationError as e:
            logger.exception(
                "Message validation failed",
                extra={
                    "error_code": e.error_code,
                    "field": e.field,
                    "correlation_id": e.correlation_id,
                },
            )
            print(f"✅ ValidationError logged: {e.error_code}")
            print(f"   Field: {e.field}")

        # Event processing error
        try:
            error_message = "Event handler not found"
            raise FlextExceptions.NotFoundError(
                error_message,
                resource_type="event_handler",
                resource_id="OrderCreatedHandler",
            )
        except FlextExceptions.NotFoundError as e:
            logger.exception(
                "Event handler not found",
                extra={
                    "error_code": e.error_code,
                    "resource_type": e.resource_type,
                    "resource_id": e.resource_id,
                    "correlation_id": e.correlation_id,
                },
            )
            print(f"✅ NotFoundError logged: {e.error_code}")
            print(f"   Resource: {e.resource_type}/{e.resource_id}")

        # Message routing error
        try:
            error_message = "Message routing configuration invalid"
            raise FlextExceptions.ConfigurationError(
                error_message,
                config_key="message_routing",
                config_source="messaging_config.yaml",
            )
        except FlextExceptions.ConfigurationError as e:
            logger.exception(
                "Message routing configuration error",
                extra={
                    "error_code": e.error_code,
                    "config_key": e.config_key,
                    "config_source": e.config_source,
                    "correlation_id": e.correlation_id,
                },
            )
            print(f"✅ ConfigurationError logged: {e.error_code}")
            print(f"   Config key: {e.config_key}, Source: {e.config_source}")

        print("✅ FlextExceptions messaging integration demonstrated")

    def demonstrate_flext_runtime_messaging(self) -> None:
        """Show FlextRuntime integration with messaging defaults."""
        print("\n=== FlextRuntime Integration (Layer 0.5) ===")

        # FlextRuntime configuration defaults for messaging
        print("FlextRuntime messaging defaults:")
        print(f"  DEFAULT_TIMEOUT: {FlextConstants.Config.DEFAULT_TIMEOUT}")
        print(f"  DEFAULT_MAX_WORKERS: {FlextConstants.Processing.DEFAULT_MAX_WORKERS}")
        print(f"  DEFAULT_BATCH_SIZE: {FlextConstants.Processing.DEFAULT_BATCH_SIZE}")
        print(
            f"  DEFAULT_RETRY_ATTEMPTS: {FlextConstants.Reliability.MAX_RETRY_ATTEMPTS}"
        )

        # Message processing configuration
        processing_config: FlextTypes.Dict = {
            "timeout": FlextConstants.Config.DEFAULT_TIMEOUT,
            "max_workers": FlextConstants.Processing.DEFAULT_MAX_WORKERS,
            "batch_size": FlextConstants.Processing.DEFAULT_BATCH_SIZE,
            "retry_attempts": FlextConstants.Reliability.MAX_RETRY_ATTEMPTS,
        }

        print(f"\nMessage processing config: {processing_config}")

        # Type guards for message validation
        message_data = {"user_id": "test@example.com", "type": "UserCommand"}
        email = message_data.get("user_id", "")

        if FlextRuntime.is_valid_email(email):
            print(f"✅ Valid email in message: {email}")
        else:
            print(f"❌ Invalid email in message: {email}")

        print("✅ FlextRuntime messaging integration demonstrated")

    # ========== DEPRECATED PATTERNS ==========

    def demonstrate_deprecated_patterns(self) -> None:
        """Show deprecated messaging patterns."""
        print("\n=== ⚠️ DEPRECATED PATTERNS ===")

        # OLD: Untyped dictionaries for messages (DEPRECATED)
        warnings.warn(
            "Untyped dictionaries are DEPRECATED! Use FlextCore.Models.Payload.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("❌ OLD WAY (untyped dict[str, object]):")
        print("message = {'type': 'order', 'data': {...}}")

        print("\n✅ CORRECT WAY (typed Payload):")
        print("message = FlextCore.Models.Payload(")
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
        print("event = FlextCore.Models.DomainEvent(")
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
        print("payload = FlextCore.Models.Payload(")
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

    # NEW: FlextResult v0.9.9+ methods for messaging
    service.demonstrate_from_callable_messaging()
    service.demonstrate_flow_through_messaging()
    service.demonstrate_lash_messaging()
    service.demonstrate_alt_messaging()
    service.demonstrate_value_or_call_messaging()

    # Foundation layer integration (NEW in Phase 2)
    service.demonstrate_flext_constants_messages()
    service.demonstrate_flext_exceptions_messaging()
    service.demonstrate_flext_runtime_messaging()

    # Deprecation warnings
    service.demonstrate_deprecated_patterns()

    # Summary
    print("\n" + "=" * 60)
    print("✅ ALL messaging patterns demonstrated!")
    print(f"📊 Events stored: {service.get_event_count()}")
    print(f"📬 Messages queued: {service.get_message_count()}")
    print("🎯 Next: See advanced examples for complex patterns")
    print(
        "✨ Including new v0.9.9+ methods: from_callable, flow_through, lash, alt, value_or_call"
    )
    print(
        "🔧 Including foundation integration: FlextConstants.Messages, FlextRuntime (Layer 0.5), FlextExceptions (Layer 2)"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
