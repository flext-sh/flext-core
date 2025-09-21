#!/usr/bin/env python3
"""11 - FlextBus: Message Bus and Command/Query Handling.

This example demonstrates the COMPLETE FlextBus API for implementing
message bus patterns with commands, queries, handlers, and middleware.

Key Concepts Demonstrated:
- Command Bus: Execute commands with single handlers
- Query Bus: Execute queries and return results
- Handler Registration: Register and discover handlers
- Middleware: Pre/post processing of messages
- Auto-discovery: Automatic handler registration
- Error Handling: Bus-level error management
- Performance Metrics: Execution tracking

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from flext_core import (
    FlextBus,
    FlextConstants,
    FlextContainer,
    FlextDomainService,
    FlextLogger,
    FlextModels,
    FlextResult,
)

# ========== BUS SERVICE ==========


class BusMessagingService(FlextDomainService[dict[str, object]]):
    """Service demonstrating ALL FlextBus patterns."""

    def __init__(self) -> None:
        """Initialize with dependencies."""
        super().__init__()
        self._logger = FlextLogger(__name__)
        self._container = FlextContainer.get_global()

    def execute(self) -> FlextResult[dict[str, object]]:
        """Execute method required by FlextDomainService."""
        self._logger.info(
            "Executing bus demo", extra={"data": {"demo": "bus_messaging"}}
        )
        return FlextResult[dict[str, object]].ok({
            "status": "processed",
            "bus_executed": True,
        })

    # ========== BASIC BUS USAGE ==========

    def demonstrate_basic_bus(self) -> None:
        """Show basic bus creation and usage."""
        print("\n=== Basic Bus Usage ===")

        # Create bus with configuration
        config = FlextModels.CqrsConfig.Bus(
            enable_middleware=True,
            enable_metrics=True,
            execution_timeout=int(FlextConstants.Defaults.TIMEOUT),
            max_cache_size=FlextConstants.Performance.DEFAULT_BATCH_SIZE,
        )

        bus = FlextBus(bus_config=config)
        print(f"✅ Bus created with config: {bus.config}")

        # Define commands and handlers
        @dataclass
        class CreateUserCommand:
            """Command to create a user."""

            name: str
            email: str

        @dataclass
        class GetUserQuery:
            """Query to get a user."""

            user_id: str

        # Create handlers
        def handle_create_user(cmd: CreateUserCommand) -> FlextResult[str]:
            """Handle user creation."""
            user_id = f"USER-{uuid4().hex[:8]}"
            print(f"  Creating user: {cmd.name} with ID {user_id}")
            return FlextResult[str].ok(user_id)

        def handle_get_user(query: GetUserQuery) -> FlextResult[dict[str, object]]:
            """Handle user query."""
            print(f"  Getting user: {query.user_id}")
            return FlextResult[dict[str, object]].ok({
                "id": query.user_id,
                "name": "John Doe",
                "email": "john@example.com",
            })

        # Register handlers
        bus.register_handler(CreateUserCommand, handle_create_user)
        bus.register_handler(GetUserQuery, handle_get_user)
        print("✅ Handlers registered")

        # Execute command
        print("\n1. Execute Command:")
        cmd = CreateUserCommand(name="Alice", email="alice@example.com")
        result = bus.execute(cmd)
        if result.is_success:
            print(f"  ✅ User created: {result.unwrap()}")

        # Execute query
        print("\n2. Execute Query:")
        if result.is_success:
            query = GetUserQuery(user_id=str(result.unwrap()))
            query_result = bus.execute(query)
            if query_result.is_success:
                print(f"  ✅ User found: {query_result.unwrap()}")

    # ========== COMMAND BUS PATTERN ==========

    def demonstrate_command_bus(self) -> None:
        """Show command bus pattern."""
        print("\n=== Command Bus Pattern ===")

        # Create dedicated command bus
        bus = FlextBus.create_command_bus()
        print("✅ Command bus created")

        # Define domain commands
        @dataclass
        class PlaceOrderCommand:
            """Command to place an order."""

            customer_id: str
            items: list[dict[str, object]]
            payment_method: str

        @dataclass
        class CancelOrderCommand:
            """Command to cancel an order."""

            order_id: str
            reason: str

        @dataclass
        class ShipOrderCommand:
            """Command to ship an order."""

            order_id: str
            carrier: str
            tracking_number: str

        # Command handlers with business logic
        class OrderCommandHandlers:
            """Handles order commands."""

            def __init__(self) -> None:
                self._orders: dict[str, dict[str, object]] = {}
                self._logger = FlextLogger(__name__)

            def handle_place_order(self, cmd: PlaceOrderCommand) -> FlextResult[str]:
                """Place a new order."""
                order_id = f"ORD-{uuid4().hex[:8]}"

                # Validate
                if not cmd.items:
                    return FlextResult[str].fail("No items in order")

                # Create order
                self._orders[order_id] = {
                    "id": order_id,
                    "customer_id": cmd.customer_id,
                    "items": cmd.items,
                    "status": "placed",
                    "payment_method": cmd.payment_method,
                    "created_at": time.time(),
                }

                self._logger.info(f"Order placed: {order_id}")
                return FlextResult[str].ok(order_id)

            def handle_cancel_order(self, cmd: CancelOrderCommand) -> FlextResult[None]:
                """Cancel an order."""
                if cmd.order_id not in self._orders:
                    return FlextResult[None].fail("Order not found")

                order = self._orders[cmd.order_id]
                if order["status"] != "placed":
                    return FlextResult[None].fail(
                        f"Cannot cancel order in {order['status']} status"
                    )

                order["status"] = "cancelled"
                order["cancel_reason"] = cmd.reason
                order["cancelled_at"] = time.time()

                self._logger.info(f"Order cancelled: {cmd.order_id}")
                return FlextResult[None].ok(None)

            def handle_ship_order(self, cmd: ShipOrderCommand) -> FlextResult[None]:
                """Ship an order."""
                if cmd.order_id not in self._orders:
                    return FlextResult[None].fail("Order not found")

                order = self._orders[cmd.order_id]
                if order["status"] != "placed":
                    return FlextResult[None].fail(
                        f"Cannot ship order in {order['status']} status"
                    )

                order["status"] = "shipped"
                order["carrier"] = cmd.carrier
                order["tracking_number"] = cmd.tracking_number
                order["shipped_at"] = time.time()

                self._logger.info(f"Order shipped: {cmd.order_id}")
                return FlextResult[None].ok(None)

        # Register command handlers
        handlers = OrderCommandHandlers()
        bus.register_handler(PlaceOrderCommand, handlers.handle_place_order)
        bus.register_handler(CancelOrderCommand, handlers.handle_cancel_order)
        bus.register_handler(ShipOrderCommand, handlers.handle_ship_order)

        # Execute command workflow
        print("\n1. Place Order:")
        place_cmd = PlaceOrderCommand(
            customer_id="CUST-123",
            items=[{"product": "Widget", "qty": 2}],
            payment_method="credit_card",
        )
        result = bus.send_command(place_cmd)
        if result.is_success:
            order_id = result.unwrap()
            print(f"  ✅ Order placed: {order_id}")

            print("\n2. Ship Order:")
            ship_cmd = ShipOrderCommand(
                order_id=str(order_id), carrier="FedEx", tracking_number="1234567890"
            )
            ship_result = bus.send_command(ship_cmd)
            print(f"  {'✅' if ship_result.is_success else '❌'} Order shipped")

            print("\n3. Try to Cancel Shipped Order:")
            cancel_cmd = CancelOrderCommand(
                order_id=str(order_id), reason="Customer request"
            )
            cancel_result = bus.send_command(cancel_cmd)
            if cancel_result.is_failure:
                print(f"  ✅ Correctly rejected: {cancel_result.error}")

    # ========== MIDDLEWARE PATTERN ==========

    def demonstrate_middleware(self) -> None:
        """Show middleware pattern."""
        print("\n=== Middleware Pattern ===")

        bus = FlextBus()

        # Logging middleware
        class LoggingMiddleware:
            """Logs all bus operations."""

            def __init__(self) -> None:
                self._logger = FlextLogger(__name__)

            def __call__(
                self,
                message: object,
                next_handler: Callable[[object], FlextResult[object]],
            ) -> FlextResult[object]:
                """Log before and after."""
                message_type = type(message).__name__
                self._logger.info(f"Processing: {message_type}")

                start = time.time()
                result = next_handler(message)
                duration = time.time() - start

                self._logger.info(
                    f"Completed: {message_type}",
                    extra={"duration": duration, "success": bool(result.is_success)},
                )
                return result

        # Validation middleware
        class ValidationMiddleware:
            """Validates messages."""

            def __call__(
                self,
                message: object,
                next_handler: Callable[[object], FlextResult[object]],
            ) -> FlextResult[object]:
                """Validate message."""
                # Check for required attributes
                if hasattr(message, "__dataclass_fields__"):
                    fields = getattr(message, "__dataclass_fields__", {})
                    for field_name, field_info in fields.items():
                        value = getattr(message, field_name, None)
                        default_value = getattr(field_info, "default", None)
                        if value is None and default_value is None:
                            return FlextResult[object].fail(
                                f"Field {field_name} is required"
                            )

                return next_handler(message)

        # Performance middleware
        class PerformanceMiddleware:
            """Tracks performance metrics."""

            def __init__(self) -> None:
                self._metrics: dict[str, list[float]] = {}

            def __call__(
                self,
                message: object,
                next_handler: Callable[[object], FlextResult[object]],
            ) -> FlextResult[object]:
                """Track performance."""
                message_type = type(message).__name__

                start = time.time()
                result = next_handler(message)
                duration = time.time() - start

                if message_type not in self._metrics:
                    self._metrics[message_type] = []
                self._metrics[message_type].append(duration)

                # Log slow operations
                if duration > 0.1:
                    print(f"  ⚠️ Slow operation: {message_type} took {duration:.3f}s")

                return result

            def get_stats(self) -> dict[str, dict[str, float]]:
                """Get performance statistics."""
                stats: dict[str, dict[str, float]] = {}
                for msg_type, durations in self._metrics.items():
                    stats[msg_type] = {
                        "count": float(len(durations)),
                        "avg": sum(durations) / len(durations),
                        "max": max(durations),
                        "min": min(durations),
                    }
                return stats

        # Add middleware to bus
        logging_middleware = LoggingMiddleware()
        validation_middleware = ValidationMiddleware()
        performance_middleware = PerformanceMiddleware()

        bus.add_middleware(logging_middleware)
        bus.add_middleware(validation_middleware)
        bus.add_middleware(performance_middleware)
        print("✅ Middleware added: Logging, Validation, Performance")

        # Test with messages
        @dataclass
        class TestCommand:
            """Test command."""

            value: str

        def handle_test(cmd: TestCommand) -> FlextResult[str]:
            """Handle test command."""
            # Simulate some work
            time.sleep(0.05)
            return FlextResult[str].ok(f"Processed: {cmd.value}")

        bus.register_handler(TestCommand, handle_test)

        print("\n1. Execute with middleware:")
        cmd = TestCommand(value="test-value")
        result = bus.execute(cmd)
        print(
            f"  Result: {'✅' if result.is_success else '❌'} {result.unwrap() if result.is_success else result.error}"
        )

        print("\n2. Performance stats:")
        stats = performance_middleware.get_stats()
        for msg_type, metrics in stats.items():
            print(f"  {msg_type}:")
            print(f"    Count: {metrics['count']}")
            print(f"    Avg: {metrics['avg']:.3f}s")

    # ========== HANDLER DISCOVERY ==========

    def demonstrate_handler_discovery(self) -> None:
        """Show handler discovery and management."""
        print("\n=== Handler Discovery ===")

        bus = FlextBus()

        # Register multiple handlers
        @dataclass
        class Command1:
            value: int

        @dataclass
        class Command2:
            value: str

        @dataclass
        class Query1:
            id: str

        def handle_command1(cmd: Command1) -> FlextResult[int]:
            return FlextResult[int].ok(cmd.value * 2)

        def handle_command2(cmd: Command2) -> FlextResult[str]:
            return FlextResult[str].ok(cmd.value.upper())

        def handle_query1(query: Query1) -> FlextResult[dict[str, object]]:
            return FlextResult[dict[str, object]].ok({"id": query.id, "found": True})

        bus.register_handler(Command1, handle_command1)
        bus.register_handler(Command2, handle_command2)
        bus.register_handler(Query1, handle_query1)

        # Get all handlers
        all_handlers = bus.get_all_handlers()
        print(f"✅ Registered handlers: {len(all_handlers)}")
        for handler_info in all_handlers:
            print(f"  - {handler_info}")

        # Find specific handler
        handler = bus.find_handler(Command1)
        print(f"\nFound handler for Command1: {handler is not None}")

        # Get registered handler types
        registered = bus.get_registered_handlers()
        print(f"\nRegistered message types: {registered}")

        # Unregister handler
        bus.unregister_handler(Command2)
        print(f"\nAfter unregistering Command2: {len(bus.get_all_handlers())} handlers")

    # ========== AUTO HANDLERS ==========

    def demonstrate_auto_handlers(self) -> None:
        """Show automatic handler creation."""
        print("\n=== Auto Handlers ===")

        bus = FlextBus()

        # Create simple handler
        @dataclass
        class CalculateCommand:
            """Calculate something."""

            a: int
            b: int
            operation: str

        def calculate(a: int, b: int, operation: str) -> int:
            """Perform calculation."""
            if operation == "add":
                return a + b
            if operation == "multiply":
                return a * b
            return 0

        # Create handler automatically
        def calculate_wrapper(cmd: object) -> object:
            if isinstance(cmd, CalculateCommand):
                return calculate(cmd.a, cmd.b, cmd.operation)
            return 0

        handler = bus.create_simple_handler(handler_func=calculate_wrapper)

        bus.register_handler(CalculateCommand, handler)

        # Execute
        print("\n1. Simple auto handler:")
        cmd = CalculateCommand(a=10, b=5, operation="add")
        result = bus.execute(cmd)
        if result.is_success:
            print(f"  ✅ Result: {result.unwrap()}")

        # Create query handler
        @dataclass
        class SearchQuery:
            """Search for items."""

            keyword: str
            limit: int = 10

        def search(keyword: str, limit: int = 10) -> list[str]:
            """Perform search."""
            # Simulate search
            return [f"{keyword}-{i}" for i in range(limit)]

        # Create query handler automatically
        def search_wrapper(query: object) -> object:
            if isinstance(query, SearchQuery):
                return search(query.keyword, query.limit)
            return []

        query_handler = bus.create_query_handler(handler_func=search_wrapper)

        bus.register_handler(SearchQuery, query_handler)

        print("\n2. Auto query handler:")
        query = SearchQuery(keyword="product", limit=3)
        result = bus.execute(query)
        if result.is_success:
            print(f"  ✅ Results: {result.unwrap()}")

    # ========== ERROR HANDLING ==========

    def demonstrate_error_handling(self) -> None:
        """Show bus error handling."""
        print("\n=== Error Handling ===")

        # Configure bus with error handling
        config = FlextModels.CqrsConfig.Bus(
            enable_middleware=True, enable_metrics=True, execution_timeout=1
        )
        bus = FlextBus(bus_config=config)

        # Handler that fails
        @dataclass
        class FailingCommand:
            """Command that will fail."""

            fail_type: str

        def failing_handler(cmd: FailingCommand) -> FlextResult[None]:
            """Handler that fails in different ways."""
            if cmd.fail_type == "validation":
                return FlextResult[None].fail("Validation error")
            if cmd.fail_type == "timeout":
                time.sleep(2)  # Exceed timeout
                return FlextResult[None].ok(None)
            if cmd.fail_type == "exception":
                msg = "Unexpected error"
                raise ValueError(msg)
            return FlextResult[None].ok(None)

        bus.register_handler(FailingCommand, failing_handler)

        # Test different failure modes
        print("\n1. Validation failure:")
        cmd = FailingCommand(fail_type="validation")
        result = bus.execute(cmd)
        print(
            f"  Result: {'✅' if result.is_success else '❌'} {result.error if result.is_failure else ''}"
        )

        print("\n2. Success case:")
        cmd = FailingCommand(fail_type="ok")
        result = bus.execute(cmd)
        print(f"  Result: {'✅' if result.is_success else '❌'} Success")

    # ========== DEPRECATED PATTERNS ==========

    def demonstrate_deprecated_patterns(self) -> None:
        """Show deprecated bus patterns."""
        print("\n=== ⚠️ DEPRECATED PATTERNS ===")

        # OLD: Direct handler invocation (DEPRECATED)
        warnings.warn(
            "Direct handler invocation is DEPRECATED! Use bus.execute().",
            DeprecationWarning,
            stacklevel=2,
        )
        print("❌ OLD WAY (direct call):")
        print("handler = get_handler()")
        print("result = handler(command)")

        print("\n✅ CORRECT WAY (bus):")
        print("bus.register_handler(CommandType, handler)")
        print("result = bus.execute(command)")

        # OLD: Global handler registry (DEPRECATED)
        warnings.warn(
            "Global handler registry is DEPRECATED! Use FlextBus instance.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (global):")
        print("HANDLERS = {}")
        print("HANDLERS[CommandType] = handler")

        print("\n✅ CORRECT WAY (bus instance):")
        print("bus = FlextBus()")
        print("bus.register_handler(CommandType, handler)")

        # OLD: No middleware support (DEPRECATED)
        warnings.warn(
            "Handlers without middleware are DEPRECATED! Use bus middleware.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (no middleware):")
        print("def handle(cmd):")
        print("    # Direct processing")

        print("\n✅ CORRECT WAY (middleware):")
        print("bus.add_middleware(LoggingMiddleware())")
        print("bus.add_middleware(ValidationMiddleware())")


def main() -> None:
    """Main entry point demonstrating all FlextBus capabilities."""
    service = BusMessagingService()

    print("=" * 60)
    print("FLEXTBUS COMPLETE API DEMONSTRATION")
    print("Message Bus and Command/Query Handling")
    print("=" * 60)

    # Core patterns
    service.demonstrate_basic_bus()
    service.demonstrate_command_bus()

    # Advanced patterns
    service.demonstrate_middleware()
    service.demonstrate_handler_discovery()

    # Auto handlers
    service.demonstrate_auto_handlers()
    service.demonstrate_error_handling()

    # Deprecation warnings
    service.demonstrate_deprecated_patterns()

    print("\n" + "=" * 60)
    print("✅ ALL FlextBus methods demonstrated!")
    print("🎯 Next: See 12_dispatcher_patterns.py for FlextDispatcher")
    print("=" * 60)


if __name__ == "__main__":
    main()
