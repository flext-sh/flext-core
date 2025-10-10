#!/usr/bin/env python3
"""11 - FlextCore.Bus: Message Bus and Command/Query Handling.

This example demonstrates the COMPLETE FlextCore.Bus API for implementing
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
from typing import cast
from uuid import uuid4

from flext_core import (
    FlextConstants,
    FlextCore,
    FlextLogger,
    FlextModels,
    FlextResult,
    FlextService,
    FlextTypes,
)


class BusMessagingService(FlextService[FlextTypes.Dict]):
    """Service demonstrating ALL FlextCore.Bus patterns with FlextMixins.Service infrastructure.

    This service inherits from FlextService to demonstrate:
    - Inherited container property (FlextCore.Container singleton)
    - Inherited logger property (FlextLogger with service context - BUS MESSAGING FOCUS!)
    - Inherited context property (FlextCore.Context for request/correlation tracking)
    - Inherited config property (FlextCore.Config with bus processing settings)
    - Inherited metrics property (FlextMetrics for bus observability)

    FlextCore.Bus provides:
    - Command Bus: Execute commands with single handlers
    - Query Bus: Execute queries and return results
    - Handler Registration: Register and discover handlers
    - Middleware: Pre/post processing of messages
    - Auto-discovery: Automatic handler registration
    - Error Handling: Bus-level error management
    - Performance Metrics: Execution tracking
    """

    def __init__(self) -> None:
        """Initialize with inherited FlextMixins.Service infrastructure.

        Inherited properties (no manual instantiation needed):
        - self.logger: FlextLogger with service context (bus messaging operations)
        - self.container: FlextCore.Container singleton (for service dependencies)
        - self.context: FlextCore.Context (for correlation tracking)
        - self.config: FlextCore.Config (for bus configuration)
        - self.metrics: FlextMetrics (for bus observability)
        """
        super().__init__()

        # Demonstrate inherited logger (no manual instantiation needed!)
        self.logger.info(
            "BusMessagingService initialized with inherited infrastructure",
            extra={
                "service_type": "FlextCore.Bus Messaging & CQRS demonstration",
                "bus_features": [
                    "command_bus",
                    "query_bus",
                    "handler_registry",
                    "middleware_pipeline",
                    "auto_discovery",
                    "error_handling",
                    "performance_tracking",
                ],
            },
        )

    def execute(self) -> FlextResult[FlextTypes.Dict]:
        """Execute all FlextCore.Bus pattern demonstrations.

        Runs comprehensive bus messaging demonstrations:
        1. Basic bus usage with commands and queries
        2. Command bus pattern with domain commands
        3. Middleware pattern with logging, validation, and performance
        4. Handler discovery and management
        5. Auto handlers with automatic registration
        6. Error handling with bus-level error management
        7. New FlextResult methods in bus context (v0.9.9+)
        8. Deprecated patterns (for educational comparison)

        Returns:
            FlextResult[FlextTypes.Dict]: Execution summary with demonstration results

        """
        self.logger.info("Starting comprehensive FlextCore.Bus demonstration")

        try:
            # Run all 8 demonstrations
            self.demonstrate_basic_bus()
            self.demonstrate_command_bus()
            self.demonstrate_middleware()
            self.demonstrate_handler_discovery()
            self.demonstrate_auto_handlers()
            self.demonstrate_error_handling()
            self.demonstrate_new_flextresult_methods()
            self.demonstrate_deprecated_patterns()

            summary: FlextTypes.Dict = {
                "status": "completed",
                "demonstrations": 8,
                "patterns": [
                    "basic_bus",
                    "command_bus",
                    "middleware",
                    "handler_discovery",
                    "auto_handlers",
                    "error_handling",
                    "new_flextresult_methods",
                    "deprecated_patterns",
                ],
                "bus_executed": True,
            }

            self.logger.info(
                "FlextCore.Bus demonstration completed successfully",
                extra={"summary": summary},
            )

            return FlextResult[FlextTypes.Dict].ok(summary)

        except Exception as e:
            error_msg = f"FlextCore.Bus demonstration failed: {e}"
            self.logger.exception(error_msg, extra={"error_type": type(e).__name__})
            return FlextResult[FlextTypes.Dict].fail(error_msg)

    # ========== BASIC BUS USAGE ==========

    def demonstrate_basic_bus(self) -> None:
        """Show basic bus creation and usage."""
        print("\n=== Basic Bus Usage ===")

        # Create bus with configuration
        config = FlextModels.Cqrs.Bus(
            enable_middleware=True,
            enable_metrics=True,
            execution_timeout=int(FlextConstants.Defaults.TIMEOUT),
            max_cache_size=FlextConstants.Performance.BatchProcessing.DEFAULT_SIZE,
        )

        bus = FlextCore.Bus(bus_config=config)
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

        def handle_get_user(
            query: GetUserQuery,
        ) -> FlextResult[FlextTypes.Dict]:
            """Handle user query."""
            print(f"  Getting user: {query.user_id}")
            return FlextResult[FlextTypes.Dict].ok({
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
        bus = FlextCore.Bus()
        print("✅ Command bus created")

        # Define domain commands
        @dataclass
        class PlaceOrderCommand:
            """Command to place an order."""

            customer_id: str
            items: list[FlextTypes.Dict]
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

            logger: FlextLogger

            def __init__(self) -> None:
                super().__init__()
                self._orders: FlextTypes.NestedDict = {}
                self.logger = FlextCore.create_logger(__name__)

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

                self.logger.info("Order placed: %s", order_id)
                return FlextResult[str].ok(order_id)

            def handle_cancel_order(self, cmd: CancelOrderCommand) -> FlextResult[None]:
                """Cancel an order."""
                if cmd.order_id not in self._orders:
                    return FlextResult[None].fail("Order not found")

                order = self._orders[cmd.order_id]
                if order["status"] != "placed":
                    return FlextResult[None].fail(
                        f"Cannot cancel order in {order['status']} status",
                    )

                order["status"] = "cancelled"
                order["cancel_reason"] = cmd.reason
                order["cancelled_at"] = time.time()

                self.logger.info(f"Order cancelled: {cmd.order_id}")
                return FlextResult[None].ok(None)

            def handle_ship_order(self, cmd: ShipOrderCommand) -> FlextResult[None]:
                """Ship an order."""
                if cmd.order_id not in self._orders:
                    return FlextResult[None].fail("Order not found")

                order = self._orders[cmd.order_id]
                if order["status"] != "placed":
                    return FlextResult[None].fail(
                        f"Cannot ship order in {order['status']} status",
                    )

                order["status"] = "shipped"
                order["carrier"] = cmd.carrier
                order["tracking_number"] = cmd.tracking_number
                order["shipped_at"] = time.time()

                self.logger.info(f"Order shipped: {cmd.order_id}")
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
        result = bus.execute(place_cmd)
        if result.is_success:
            order_id = result.unwrap()
            print(f"  ✅ Order placed: {order_id}")

            print("\n2. Ship Order:")
            ship_cmd = ShipOrderCommand(
                order_id=str(order_id),
                carrier="FedEx",
                tracking_number="1234567890",
            )
            ship_result = bus.execute(ship_cmd)
            print(f"  {'✅' if ship_result.is_success else '❌'} Order shipped")

            print("\n3. Try to Cancel Shipped Order:")
            cancel_cmd = CancelOrderCommand(
                order_id=str(order_id),
                reason="Customer request",
            )
            cancel_result = bus.execute(cancel_cmd)
            if cancel_result.is_failure:
                print(f"  ✅ Correctly rejected: {cancel_result.error}")

    # ========== MIDDLEWARE PATTERN ==========

    def demonstrate_middleware(self) -> None:
        """Show middleware pattern."""
        print("\n=== Middleware Pattern ===")

        bus = FlextCore.Bus()

        # Logging middleware
        class LoggingMiddleware:
            """Logs all bus operations."""

            logger: FlextLogger

            def __init__(self) -> None:
                super().__init__()
                self.logger = FlextCore.create_logger(__name__)

            def __call__(
                self,
                message: object,
                next_handler: Callable[[object], FlextResult[object]],
            ) -> FlextResult[object]:
                """Log before and after."""
                message_type = type(message).__name__
                self.logger.info("Processing: %s", message_type)

                start = time.time()
                result = next_handler(message)
                duration = time.time() - start

                self.logger.info(
                    "Completed: %s",
                    message_type,
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
                                f"Field {field_name} is required",
                            )

                return next_handler(message)

        # Performance middleware
        class PerformanceMiddleware:
            """Tracks performance metrics."""

            def __init__(self) -> None:
                super().__init__()
                self._metrics: dict[str, FlextTypes.FloatList] = {}

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

            def get_stats(self) -> dict[str, FlextTypes.FloatDict]:
                """Get performance statistics."""
                stats: dict[str, FlextTypes.FloatDict] = {}
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
            f"  Result: {'✅' if result.is_success else '❌'} {result.unwrap() if result.is_success else result.error}",
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

        bus = FlextCore.Bus()

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

        def handle_query1(query: Query1) -> FlextResult[FlextTypes.Dict]:
            return FlextResult[FlextTypes.Dict].ok({
                "id": query.id,
                "found": True,
            })

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

        bus = FlextCore.Bus()

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

        # Create handler from callable
        handler = FlextCore.Handlers.from_callable(callable_func=calculate_wrapper)

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

        def search(keyword: str, limit: int = 10) -> FlextTypes.StringList:
            """Perform search."""
            # Simulate search
            return [f"{keyword}-{i}" for i in range(limit)]

        # Create query handler automatically
        def search_wrapper(query: object) -> object:
            if isinstance(query, SearchQuery):
                return search(query.keyword, query.limit)
            return []

        query_handler = FlextCore.Handlers.from_callable(
            callable_func=search_wrapper, handler_type="query"
        )

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
        config = FlextModels.Cqrs.Bus(
            enable_middleware=True,
            enable_metrics=True,
            execution_timeout=1,
        )
        bus = FlextCore.Bus(bus_config=config)

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
            f"  Result: {'✅' if result.is_success else '❌'} {result.error if result.is_failure else ''}",
        )

        print("\n2. Success case:")
        cmd = FailingCommand(fail_type="ok")
        result = bus.execute(cmd)
        print(f"  Result: {'✅' if result.is_success else '❌'} Success")

    # ========== DEPRECATED PATTERNS ==========

    def demonstrate_new_flextresult_methods(self) -> None:
        """Demonstrate the 5 new FlextResult methods in bus messaging context.

        Shows how the new v0.9.9+ methods work with bus messaging patterns:
        - from_callable: Safe bus operations
        - flow_through: Message pipeline composition
        - lash: Bus fallback recovery
        - alt: Handler alternatives
        - value_or_call: Lazy bus initialization
        """
        print("\n" + "=" * 60)
        print("NEW FlextResult METHODS - BUS MESSAGING CONTEXT")
        print("Demonstrating v0.9.9+ methods with FlextCore.Bus patterns")
        print("=" * 60)

        # 1. from_callable - Safe Bus Operations
        print("\n=== 1. from_callable: Safe Bus Operations ===")

        def risky_bus_registration() -> FlextCore.Bus:
            """Bus registration that might raise exceptions."""
            bus_config = FlextModels.Cqrs.Bus(
                enable_middleware=True,
                enable_metrics=True,
                execution_timeout=int(FlextConstants.Defaults.TIMEOUT),
            )
            if not bus_config:
                msg = "Bus configuration failed"
                raise ValueError(msg)
            return FlextCore.Bus(bus_config=bus_config)

        # Safe bus creation without try/except
        bus_result = FlextResult[FlextCore.Bus].from_callable(risky_bus_registration)
        if bus_result.is_success:
            bus = bus_result.unwrap()
            print(f"✅ Bus created safely: {type(bus).__name__}")
            # Note: Middleware status can be checked via bus._config_model.enable_middleware
        else:
            print(f"❌ Bus creation failed: {bus_result.error}")

        # 2. flow_through - Message Pipeline Composition
        print("\n=== 2. flow_through: Message Pipeline Composition ===")

        def validate_message_format(
            data: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            """Validate message has required fields."""
            if not data.get("type"):
                return FlextResult[dict[str, object]].fail("Message type required")
            if not data.get("payload"):
                return FlextResult[dict[str, object]].fail("Message payload required")
            return FlextResult[dict[str, object]].ok(data)

        def enrich_with_metadata(
            data: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            """Add bus metadata to message."""
            enriched: dict[str, object] = {
                **data,
                "message_id": str(uuid4()),
                "timestamp": time.time(),
                "bus_version": "1.0",
            }
            return FlextResult[dict[str, object]].ok(enriched)

        def register_in_bus(
            data: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            """Register message in bus tracking."""
            message_id = str(data.get("message_id", "unknown"))
            enriched: dict[str, object] = {
                **data,
                "registered": True,
                "tracking_id": f"TRACK-{message_id[:8]}",
            }
            return FlextResult[dict[str, object]].ok(enriched)

        def validate_complete(
            data: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            """Validate message is ready for bus."""
            required_fields = [
                "type",
                "payload",
                "message_id",
                "timestamp",
                "registered",
            ]
            missing = [f for f in required_fields if f not in data]
            if missing:
                return FlextResult[dict[str, object]].fail(
                    f"Missing fields: {', '.join(missing)}"
                )
            return FlextResult[dict[str, object]].ok(data)

        # Flow through complete bus message pipeline
        message_data: dict[str, object] = cast(
            "dict[str, object]",
            {
                "type": "user.created",
                "payload": {"user_id": "USER-001", "name": "Alice"},
            },
        )
        pipeline_result = (
            FlextResult[dict[str, object]]
            .ok(message_data)
            .flow_through(
                validate_message_format,
                enrich_with_metadata,
                register_in_bus,
                validate_complete,
            )
        )

        if pipeline_result.is_success:
            final_message = pipeline_result.unwrap()
            message_id_value = str(final_message.get("message_id", "N/A"))
            print(f"✅ Bus message pipeline complete: {final_message['type']}")
            print(f"   Message ID: {message_id_value[:36]}")
            print(f"   Tracking ID: {final_message.get('tracking_id', 'N/A')}")
            print(f"   Registered: {final_message.get('registered', False)}")
        else:
            print(f"❌ Pipeline failed: {pipeline_result.error}")

        # 3. lash - Bus Fallback Recovery
        print("\n=== 3. lash: Bus Fallback Recovery ===")

        def primary_handler() -> FlextResult[str]:
            """Primary handler that might fail."""
            return FlextResult[str].fail("Primary handler unavailable")

        def fallback_handler(error: str) -> FlextResult[str]:
            """Fallback handler when primary fails."""
            print(f"   ⚠️  Primary failed: {error}, using fallback...")
            result_id = f"FALLBACK-{uuid4().hex[:8]}"
            return FlextResult[str].ok(result_id)

        # Try primary handler, fall back on failure
        handler_result = primary_handler().lash(fallback_handler)
        if handler_result.is_success:
            result_id = handler_result.unwrap()
            print(f"✅ Handler execution successful: {result_id}")
        else:
            print(f"❌ All handlers failed: {handler_result.error}")

        # 4. alt - Handler Alternatives
        print("\n=== 4. alt: Handler Alternatives ===")

        def get_custom_bus_config() -> FlextResult[FlextModels.Cqrs.Bus]:
            """Try to get custom bus configuration."""
            return FlextResult[FlextModels.Cqrs.Bus].fail("Custom config not found")

        def get_default_bus_config() -> FlextResult[FlextModels.Cqrs.Bus]:
            """Provide default bus configuration."""
            config = FlextModels.Cqrs.Bus(
                enable_middleware=True,
                enable_metrics=True,
                execution_timeout=int(FlextConstants.Defaults.TIMEOUT),
                max_cache_size=FlextConstants.Performance.BatchProcessing.DEFAULT_SIZE,
            )
            return FlextResult[FlextModels.Cqrs.Bus].ok(config)

        # Try custom config, fall back to default
        config_result = get_custom_bus_config().alt(get_default_bus_config())
        if config_result.is_success:
            config = config_result.unwrap()
            print(f"✅ Bus config acquired: {type(config).__name__}")
            print(f"   Middleware: {config.enable_middleware}")
            print(f"   Metrics: {config.enable_metrics}")
        else:
            print(f"❌ No config available: {config_result.error}")

        # 5. value_or_call - Lazy Bus Initialization
        print("\n=== 5. value_or_call: Lazy Bus Initialization ===")

        def create_expensive_bus() -> FlextCore.Bus:
            """Create and configure a new bus (expensive operation)."""
            print("   ⚙️  Creating new bus with full configuration...")
            config = FlextModels.Cqrs.Bus(
                enable_middleware=True,
                enable_metrics=True,
                execution_timeout=int(FlextConstants.Defaults.TIMEOUT),
                max_cache_size=1000,
            )
            # Register some handlers (expensive setup)
            # Return directly to avoid RET504
            return FlextCore.Bus(bus_config=config)

        # Try to get existing bus, create new one if not available
        bus_fail_result = FlextResult[FlextCore.Bus].fail("No existing bus")
        bus = bus_fail_result.value_or_call(create_expensive_bus)
        print(f"✅ Bus acquired: {type(bus).__name__}")
        print(f"   Config enabled: {hasattr(bus, 'config')}")

        # Try again with successful result (lazy function NOT called)
        existing_bus = FlextCore.Bus()
        bus_success_result = FlextResult[FlextCore.Bus].ok(existing_bus)
        bus_cached = bus_success_result.value_or_call(create_expensive_bus)
        print(f"✅ Existing bus used: {type(bus_cached).__name__}")
        print("   No expensive creation needed")

        print("\n" + "=" * 60)
        print("✅ NEW FlextResult METHODS BUS MESSAGING DEMO COMPLETE!")
        print("All 5 methods demonstrated with FlextCore.Bus messaging context")
        print("=" * 60)

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
            "Global handler registry is DEPRECATED! Use FlextCore.Bus instance.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (global):")
        print("HANDLERS = {}")
        print("HANDLERS[CommandType] = handler")

        print("\n✅ CORRECT WAY (bus instance):")
        print("bus = FlextCore.Bus()")
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
    """Main entry point demonstrating all FlextCore.Bus capabilities."""
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

    # New FlextResult methods (v0.9.9+)
    service.demonstrate_new_flextresult_methods()

    # Deprecation warnings
    service.demonstrate_deprecated_patterns()

    print("\n" + "=" * 60)
    print("✅ ALL FlextCore.Bus methods demonstrated!")
    print("🎯 Next: See 12_dispatcher_patterns.py for FlextCore.Dispatcher")
    print("=" * 60)


if __name__ == "__main__":
    main()
