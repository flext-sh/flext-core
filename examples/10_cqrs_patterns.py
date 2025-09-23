#!/usr/bin/env python3
"""10 - FlextCqrs: Command Query Responsibility Segregation Patterns.

This example demonstrates the COMPLETE FlextCqrs API for implementing
CQRS patterns with commands, queries, events, and separation of read/write models.

Key Concepts Demonstrated:
- Commands: Write operations that modify state
- Queries: Read operations that don't modify state
- Command Handlers: Process commands and emit events
- Query Handlers: Process queries and return data
- Results: Specialized CQRS result types
- Operations: Command and query execution
- Decorators: CQRS decorators for handlers
- Event Sourcing: Track changes through events

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from decimal import Decimal
from uuid import uuid4

from flext_core import (
    FlextConstants,
    FlextContainer,
    FlextCqrs,
    FlextLogger,
    FlextModels,
    FlextResult,
    FlextService,
)

# ========== CQRS SERVICE ==========


class CqrsPatternService(FlextService[dict[str, object]]):
    """Service demonstrating ALL FlextCqrs patterns."""

    def __init__(self) -> None:
        """Initialize with dependencies."""
        super().__init__()
        self._logger = FlextLogger(__name__)
        self._container = FlextContainer.get_global()
        # In-memory stores for demo
        self._write_store: dict[str, object] = {}
        self._read_store: dict[str, object] = {}
        self._event_store: list[FlextModels.DomainEvent] = []

    def execute(self) -> FlextResult[dict[str, object]]:
        """Execute method required by FlextService."""
        self._logger.info("Executing CQRS demo")
        return FlextResult[dict[str, object]].ok({
            "status": "processed",
            "cqrs_executed": True,
        })

    # ========== COMMANDS AND QUERIES ==========

    def demonstrate_commands_queries(self) -> None:
        """Show basic command and query patterns."""
        print("\n=== Commands and Queries ===")

        # Define Commands (write operations)
        @dataclass
        class CreateUserCommand:
            """Command to create a new user."""

            user_id: str
            name: str
            email: str

        @dataclass
        class UpdateUserCommand:
            """Command to update a user."""

            user_id: str
            name: str | None = None
            email: str | None = None

        # Define Queries (read operations)
        @dataclass
        class GetUserQuery:
            """Query to get a user by ID."""

            user_id: str

        @dataclass
        class ListUsersQuery:
            """Query to list all users."""

            limit: int = 10
            offset: int = 0

        # Command Handler
        class UserCommandHandler:
            """Handles user write operations."""

            def __init__(
                self,
                write_store: dict[str, object],
                event_store: list[FlextModels.DomainEvent],
            ) -> None:
                self._write_store = write_store
                self._event_store = event_store
                self._logger = FlextLogger(__name__)

            def handle_create(self, cmd: CreateUserCommand) -> FlextResult[str]:
                """Handle user creation."""
                self._logger.info(f"Creating user {cmd.user_id}")

                if cmd.user_id in self._write_store:
                    return FlextResult[str].fail("User already exists")

                # Store user
                self._write_store[cmd.user_id] = {
                    "id": cmd.user_id,
                    "name": cmd.name,
                    "email": cmd.email,
                    "created_at": time.time(),
                }

                # Emit event
                event = FlextModels.DomainEvent(
                    aggregate_id=cmd.user_id,
                    event_type="UserCreated",
                    data={
                        "user_id": cmd.user_id,
                        "name": cmd.name,
                        "email": cmd.email,
                    },
                )
                self._event_store.append(event)

                return FlextResult[str].ok(cmd.user_id)

            def handle_update(self, cmd: UpdateUserCommand) -> FlextResult[None]:
                """Handle user update."""
                if cmd.user_id not in self._write_store:
                    return FlextResult[None].fail("User not found")

                user = self._write_store[cmd.user_id]
                if isinstance(user, dict):
                    if cmd.name:
                        user["name"] = cmd.name
                    if cmd.email:
                        user["email"] = cmd.email
                    user["updated_at"] = time.time()

                # Emit event
                event = FlextModels.DomainEvent(
                    aggregate_id=cmd.user_id,
                    event_type="UserUpdated",
                    data={
                        "user_id": cmd.user_id,
                        "changes": {"name": cmd.name, "email": cmd.email},
                    },
                )
                self._event_store.append(event)

                return FlextResult[None].ok(None)

        # Query Handler
        class UserQueryHandler:
            """Handles user read operations."""

            def __init__(self, read_store: dict[str, object]) -> None:
                self._read_store = read_store
                self._logger = FlextLogger(__name__)

            def handle_get(self, query: GetUserQuery) -> FlextResult[dict[str, object]]:
                """Handle get user query."""
                user = self._read_store.get(query.user_id)
                if not user:
                    return FlextResult[dict[str, object]].fail("User not found")
                if isinstance(user, dict):
                    return FlextResult[dict[str, object]].ok(dict(user))
                return FlextResult[dict[str, object]].fail("Invalid user data")

            def handle_list(
                self, query: ListUsersQuery
            ) -> FlextResult[list[dict[str, object]]]:
                """Handle list users query."""
                users = list(self._read_store.values())
                # Filter to only dict objects
                dict_users: list[dict[str, object]] = [
                    u for u in users if isinstance(u, dict)
                ]
                paginated: list[dict[str, object]] = dict_users[
                    query.offset : query.offset + query.limit
                ]
                return FlextResult[list[dict[str, object]]].ok(paginated)

        # Use handlers
        cmd_handler = UserCommandHandler(self._write_store, self._event_store)
        query_handler = UserQueryHandler(self._read_store)

        # Execute commands
        print("\n1. Executing Commands:")
        create_cmd = CreateUserCommand("USER-001", "Alice", "alice@example.com")
        create_result = cmd_handler.handle_create(create_cmd)
        print(
            f"  Create user: {'✅' if create_result.is_success else '❌'} {create_result.unwrap() if create_result.is_success else create_result.error}"
        )

        # Sync read model (eventual consistency)
        if create_result.is_success:
            write_data = self._write_store[create_cmd.user_id]
            if isinstance(write_data, dict):
                self._read_store[create_cmd.user_id] = write_data.copy()

        update_cmd = UpdateUserCommand("USER-001", name="Alice Smith")
        update_result = cmd_handler.handle_update(update_cmd)
        print(f"  Update user: {'✅' if update_result.is_success else '❌'}")

        # Execute queries
        print("\n2. Executing Queries:")
        get_query = GetUserQuery("USER-001")
        get_result = query_handler.handle_get(get_query)
        if get_result.is_success:
            user = get_result.unwrap()
            print(f"  Get user: ✅ {user['name']} ({user['email']})")

        list_query = ListUsersQuery(limit=5)
        list_result = query_handler.handle_list(list_query)
        print(f"  List users: ✅ {len(list_result.unwrap())} users")

    # ========== CQRS RESULTS ==========

    def demonstrate_cqrs_results(self) -> None:
        """Show specialized CQRS result types."""
        print("\n=== CQRS Results ===")

        # Command result
        cmd_result = FlextCqrs.Results.success(
            data={
                "aggregate_id": "ORDER-123",
                "version": 1,
                "events": ["OrderCreated", "OrderValidated"],
            }
        )
        print("✅ Command Result:")
        if cmd_result.is_success:
            data = cmd_result.unwrap()
            if isinstance(data, dict):
                aggregate_id: str = str(data.get("aggregate_id", "Unknown"))
                version: str = str(data.get("version", "Unknown"))
                events: str = str(data.get("events", "Unknown"))
                print(f"  Aggregate: {aggregate_id}")
                print(f"  Version: {version}")
                print(f"  Events: {events}")

        # Query result
        query_result = FlextCqrs.Results.success(
            data={
                "data": {"id": "USER-456", "name": "Bob"},
                "metadata": {"cache_hit": True, "query_time": 0.005},
            }
        )
        print("\n✅ Query Result:")
        if query_result.is_success:
            result_data = query_result.unwrap()
            if isinstance(result_data, dict):
                query_data: str = str(result_data.get("data", "Unknown"))
                metadata: str = str(result_data.get("metadata", "Unknown"))
                print(f"  Data: {query_data}")
                print(f"  Metadata: {metadata}")

        # Batch result
        batch_result = FlextCqrs.Results.success(
            data={
                "succeeded": 8,
                "failed": 2,
                "results": [
                    {"id": "1", "status": "success"},
                    {"id": "2", "status": "failed", "error": "Validation error"},
                ],
            }
        )
        print("\n✅ Batch Result:")
        if batch_result.is_success:
            batch_data = batch_result.unwrap()
            if isinstance(batch_data, dict):
                succeeded: str = str(batch_data.get("succeeded", "Unknown"))
                failed: str = str(batch_data.get("failed", "Unknown"))
                print(f"  Succeeded: {succeeded}")
                print(f"  Failed: {failed}")

        # Error result
        error_result = FlextCqrs.Results.failure(
            message="Invalid email format",
            error_code=FlextConstants.Errors.VALIDATION_ERROR,
            error_data={"field": "email"},
        )
        print("\n❌ Error Result:")
        if not error_result.is_success:
            print(f"  Error: {error_result.error}")
        else:
            print("  Unexpected success result")

    # ========== CQRS OPERATIONS ==========

    def demonstrate_cqrs_operations(self) -> None:
        """Show CQRS operation execution patterns."""
        print("\n=== CQRS Operations ===")

        # Define operation handlers
        class OrderOperations:
            """Order CQRS operations."""

            def create_order(
                self, customer_id: str, items: list[dict[str, object]]
            ) -> FlextResult[str]:
                """Create order command."""
                _ = customer_id  # Used in real implementation
                order_id = f"ORDER-{uuid4().hex[:8]}"
                print(f"  Creating order {order_id}")
                # Validate and process
                if not items:
                    return FlextResult[str].fail("No items provided")
                return FlextResult[str].ok(order_id)

            def get_order(self, order_id: str) -> FlextResult[dict[str, object]]:
                """Get order query."""
                print(f"  Getting order {order_id}")
                # Simulate fetch
                return FlextResult[dict[str, object]].ok({
                    "id": order_id,
                    "status": "pending",
                    "total": Decimal("99.99"),
                })

        # Use OrderOperations for demonstration
        operations = OrderOperations()

        # Demonstrate OrderOperations
        print("\n0. OrderOperations Demo:")
        order_result = operations.create_order(
            "CUST-123", [{"product": "Widget", "qty": 1}]
        )
        if order_result.is_success:
            order_id = order_result.unwrap()
            get_result = operations.get_order(order_id)
            if get_result.is_success:
                order_data = get_result.unwrap()
                print(f"  ✅ Created and retrieved order: {order_data}")

        # Execute command operation
        print("\n1. Command Operation:")
        cmd_result = FlextCqrs.Operations.create_command(
            command_data={
                "command_type": "CreateOrder",
                "customer_id": "CUST-789",
                "items": [{"product": "Widget", "qty": 2}],
            }
        )
        if cmd_result.is_success:
            command = cmd_result.unwrap()
            print(f"  ✅ Command created: {command.command_type}")

        # Execute query operation
        print("\n2. Query Operation:")
        query_result = FlextCqrs.Operations.create_query(
            query_data={"filters": {"order_id": "ORDER-123"}}
        )
        if query_result.is_success:
            query = query_result.unwrap()
            print(f"  ✅ Query created: {query.query_id}")

        # Batch operations
        print("\n3. Batch Operations:")

        def process_item(item_id: str) -> FlextResult[str]:
            """Process single item."""
            if item_id.endswith("3"):  # Simulate failure for item 3
                return FlextResult[str].fail(f"Failed to process {item_id}")
            return FlextResult[str].ok(f"Processed {item_id}")

        items = ["ITEM-1", "ITEM-2", "ITEM-3", "ITEM-4"]
        # Process batch manually (no execute_batch wrapper exists)
        batch_results = [process_item(item) for item in items]

        succeeded = sum(1 for r in batch_results if r.is_success)
        failed = sum(1 for r in batch_results if r.is_failure)
        print(f"  Batch complete: {succeeded} succeeded, {failed} failed")

        # Pipeline operations
        print("\n4. Pipeline Operations:")

        def validate(data: dict[str, object]) -> FlextResult[dict[str, object]]:
            """Validation step."""
            if not data.get("amount"):
                return FlextResult[dict[str, object]].fail("Amount required")
            return FlextResult[dict[str, object]].ok(data)

        def enrich(data: dict[str, object]) -> FlextResult[dict[str, object]]:
            """Enrichment step."""
            data["timestamp"] = time.time()
            data["processed"] = True
            return FlextResult[dict[str, object]].ok(data)

        def persist(data: dict[str, object]) -> FlextResult[str]:
            """Persistence step."""
            _ = data  # Used in real implementation
            record_id = f"REC-{uuid4().hex[:8]}"
            print(f"    Persisted as {record_id}")
            return FlextResult[str].ok(record_id)

        pipeline = [validate, enrich]
        # Execute pipeline manually using railway pattern
        initial_data: dict[str, object] = {"amount": 100.0, "currency": "USD"}
        pipeline_result = FlextResult[dict[str, object]].ok(initial_data)
        for step in pipeline:
            if pipeline_result.is_failure:
                break
            pipeline_result = pipeline_result.flat_map(step)

        # Handle the final persist step separately since it returns a different type
        if pipeline_result.is_success:
            persist_result = persist(pipeline_result.unwrap())
            if persist_result.is_success:
                print(f"  ✅ Pipeline complete: {persist_result.unwrap()}")

    # ========== CQRS DECORATORS ==========

    def demonstrate_cqrs_decorators(self) -> None:
        """Show CQRS decorator patterns."""
        print("\n=== CQRS Decorators ===")

        # Command handler using proper decorator
        @dataclass
        class CreateProductCommand:
            name: str
            price: Decimal

        @FlextCqrs.Decorators.command_handler(CreateProductCommand)
        def create_product(cmd: CreateProductCommand) -> FlextResult[str]:
            """Create product command with decorator."""
            product_id = f"PROD-{uuid4().hex[:8]}"
            print(f"  Creating product: {cmd.name} (${cmd.price})")
            return FlextResult[str].ok(product_id)

        # Query handler (use command_handler for query too)
        @dataclass
        class GetProductQuery:
            product_id: str

        @FlextCqrs.Decorators.command_handler(GetProductQuery)
        def get_product(query: GetProductQuery) -> FlextResult[dict[str, object]]:
            """Get product query."""
            print(f"  Getting product: {query.product_id}")
            return FlextResult[dict[str, object]].ok({
                "id": query.product_id,
                "name": "Widget",
                "price": "29.99",
            })

        # Event handler (no decorator needed, just a function)
        def on_product_created(event: FlextModels.DomainEvent) -> None:
            """Handle product created event."""
            print(f"  Event handled: {event.event_type}")
            print(f"    Data: {event.data}")

        # Use decorated functions
        print("\n1. Decorated Command:")
        cmd = CreateProductCommand(name="Super Widget", price=Decimal("49.99"))
        result = create_product(cmd)
        if result.is_success:
            print(f"  ✅ Product ID: {result.unwrap()}")

        print("\n2. Decorated Query:")
        if result.is_success:
            query = GetProductQuery(product_id=result.unwrap())
            query_result = get_product(query)
            if query_result.is_success:
                print(f"  ✅ Product: {query_result.unwrap()}")

        print("\n3. Event Handler:")
        event = FlextModels.DomainEvent(
            aggregate_id=result.unwrap() if result.is_success else "PROD-000",
            event_type="ProductCreated",
            data={"name": "Super Widget", "price": "49.99"},
        )
        on_product_created(event)

    # ========== EVENT SOURCING ==========

    def demonstrate_event_sourcing(self) -> None:
        """Show event sourcing with CQRS."""
        print("\n=== Event Sourcing with CQRS ===")

        # Event-sourced aggregate
        class BankAccount:
            """Event-sourced bank account."""

            def __init__(self, account_id: str) -> None:
                self.id = account_id
                self.balance = Decimal(0)
                self.events: list[FlextModels.DomainEvent] = []
                self.version = 0

            def deposit(self, amount: Decimal) -> FlextResult[None]:
                """Deposit money."""
                if amount <= 0:
                    return FlextResult[None].fail("Amount must be positive")

                # Create event
                event = FlextModels.DomainEvent(
                    aggregate_id=self.id,
                    event_type="MoneyDeposited",
                    data={"amount": str(amount)},
                )

                # Apply event
                self._apply(event)
                return FlextResult[None].ok(None)

            def withdraw(self, amount: Decimal) -> FlextResult[None]:
                """Withdraw money."""
                if amount <= 0:
                    return FlextResult[None].fail("Amount must be positive")
                if amount > self.balance:
                    return FlextResult[None].fail("Insufficient funds")

                # Create event
                event = FlextModels.DomainEvent(
                    aggregate_id=self.id,
                    event_type="MoneyWithdrawn",
                    data={"amount": str(amount)},
                )

                # Apply event
                self._apply(event)
                return FlextResult[None].ok(None)

            def apply_event(self, event: FlextModels.DomainEvent) -> None:
                """Apply event to state (public method for replay)."""
                if event.event_type == "MoneyDeposited":
                    self.balance += Decimal(str(event.data["amount"]))
                elif event.event_type == "MoneyWithdrawn":
                    self.balance -= Decimal(str(event.data["amount"]))

                self.events.append(event)
                self.version += 1

            def _apply(self, event: FlextModels.DomainEvent) -> None:
                """Apply event to state (internal method)."""
                self.apply_event(event)

            def get_events(self) -> list[FlextModels.DomainEvent]:
                """Get uncommitted events."""
                return self.events.copy()

            def mark_events_committed(self) -> None:
                """Clear uncommitted events."""
                self.events.clear()

        # Event store
        class EventStore:
            """Simple event store."""

            def __init__(self) -> None:
                self._streams: dict[str, list[FlextModels.DomainEvent]] = {}

            def save_events(
                self, aggregate_id: str, events: list[FlextModels.DomainEvent]
            ) -> None:
                """Save events to store."""
                if aggregate_id not in self._streams:
                    self._streams[aggregate_id] = []
                self._streams[aggregate_id].extend(events)
                print(f"  Saved {len(events)} events for {aggregate_id}")

            def get_events(self, aggregate_id: str) -> list[FlextModels.DomainEvent]:
                """Get events for aggregate."""
                return self._streams.get(aggregate_id, [])

            def replay_events(self, account: BankAccount) -> None:
                """Replay events to rebuild state."""
                events = self.get_events(account.id)
                for event in events:
                    account.apply_event(event)
                account.mark_events_committed()

        # Use event sourcing
        print("\n1. Create and use account:")
        account = BankAccount("ACC-001")
        event_store = EventStore()

        # Perform operations
        account.deposit(Decimal(1000))
        print(f"  Deposited $1000, balance: ${account.balance}")

        account.withdraw(Decimal(250))
        print(f"  Withdrew $250, balance: ${account.balance}")

        # Save events
        events = account.get_events()
        event_store.save_events(account.id, events)
        account.mark_events_committed()

        print(f"\n2. Event stream ({len(event_store.get_events('ACC-001'))} events):")
        for event in event_store.get_events("ACC-001"):
            print(f"  - {event.event_type}: {event.data}")

        # Rebuild from events
        print("\n3. Rebuild from events:")
        new_account = BankAccount("ACC-001")
        print(f"  Initial balance: ${new_account.balance}")

        event_store.replay_events(new_account)
        print(f"  After replay: ${new_account.balance}")
        print(f"  Version: {new_account.version}")

    # ========== READ/WRITE MODEL SEPARATION ==========

    def demonstrate_model_separation(self) -> None:
        """Show separation of read and write models."""
        print("\n=== Read/Write Model Separation ===")

        # Write model (normalized)
        class ProductWriteModel:
            """Normalized write model."""

            def __init__(self) -> None:
                self.products: dict[str, dict[str, object]] = {}
                self.categories: dict[str, dict[str, object]] = {}
                self.product_categories: dict[str, str] = {}

            def add_product(
                self, product_id: str, name: str, price: Decimal, category_id: str
            ) -> None:
                """Add product to write model."""
                self.products[product_id] = {
                    "id": product_id,
                    "name": name,
                    "price": str(price),
                }
                self.product_categories[product_id] = category_id

        # Read model (denormalized)
        class ProductReadModel:
            """Denormalized read model for queries."""

            def __init__(self) -> None:
                self.product_views: dict[str, dict[str, object]] = {}

            def update_from_event(self, event: FlextModels.DomainEvent) -> None:
                """Update read model from event."""
                if event.event_type == "ProductAdded":
                    data = event.data
                    self.product_views[str(data["product_id"])] = {
                        "id": data["product_id"],
                        "name": data["name"],
                        "price": data["price"],
                        "category": data["category_name"],  # Denormalized
                        "last_updated": time.time(),
                    }

            def get_product_view(self, product_id: str) -> dict[str, object] | None:
                """Get denormalized product view."""
                return self.product_views.get(product_id)

        # Projector
        class ReadModelProjector:
            """Projects events to read model."""

            def __init__(self, read_model: ProductReadModel) -> None:
                self._read_model = read_model

            def project(self, event: FlextModels.DomainEvent) -> None:
                """Project event to read model."""
                self._read_model.update_from_event(event)
                print(f"  Projected {event.event_type} to read model")

        # Use separated models
        write_model = ProductWriteModel()
        read_model = ProductReadModel()
        projector = ReadModelProjector(read_model)

        print("\n1. Write to normalized model:")
        write_model.add_product(
            "PROD-A", "Laptop", Decimal("999.99"), "CAT-ELECTRONICS"
        )
        print(f"  Products: {len(write_model.products)}")
        print(f"  Write model: {write_model.products}")

        print("\n2. Project to denormalized read model:")
        event = FlextModels.DomainEvent(
            aggregate_id="PROD-A",
            event_type="ProductAdded",
            data={
                "product_id": "PROD-A",
                "name": "Laptop",
                "price": "999.99",
                "category_name": "Electronics",  # Denormalized
            },
        )
        projector.project(event)

        print("\n3. Query from read model:")
        view = read_model.get_product_view("PROD-A")
        if view:
            print(f"  Product view: {view}")
            print(f"  Category (denormalized): {view['category']}")

    # ========== DEPRECATED PATTERNS ==========

    def demonstrate_deprecated_patterns(self) -> None:
        """Show deprecated CQRS patterns."""
        print("\n=== ⚠️ DEPRECATED PATTERNS ===")

        # OLD: Mixed read/write operations (DEPRECATED)
        warnings.warn(
            "Mixed read/write operations are DEPRECATED! Separate commands and queries.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("❌ OLD WAY (mixed operations):")
        print("def save_and_get_user(data):")
        print("    user = save_user(data)  # Write")
        print("    return get_user(user.id)  # Read")

        print("\n✅ CORRECT WAY (separated):")
        print("# Command")
        print("def create_user_command(data) -> FlextResult[str]:")
        print("    return save_user(data)")
        print()
        print("# Query")
        print("def get_user_query(user_id) -> FlextResult[User]:")
        print("    return get_user(user_id)")

        # OLD: Shared models (DEPRECATED)
        warnings.warn(
            "Shared read/write models are DEPRECATED! Use separate models.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (shared model):")
        print("class User:")
        print("    # Used for both reads and writes")
        print("    def save(self): ...")
        print("    def load(self): ...")

        print("\n✅ CORRECT WAY (separated):")
        print("class UserWriteModel:  # Normalized for writes")
        print("class UserReadModel:   # Denormalized for reads")

        # OLD: Direct database access (DEPRECATED)
        warnings.warn(
            "Direct database access in handlers is DEPRECATED! Use repositories.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (direct DB):")
        print("def handle_command(cmd):")
        print("    db.execute('INSERT INTO ...')")

        print("\n✅ CORRECT WAY (repository):")
        print("def handle_command(cmd, repository):")
        print("    repository.save(aggregate)")
        print("    # Repository handles persistence")


def main() -> None:
    """Main entry point demonstrating all FlextCqrs capabilities."""
    service = CqrsPatternService()

    print("=" * 60)
    print("FLEXTCQRS COMPLETE API DEMONSTRATION")
    print("Command Query Responsibility Segregation Patterns")
    print("=" * 60)

    # Core patterns
    service.demonstrate_commands_queries()
    service.demonstrate_cqrs_results()

    # Operations
    service.demonstrate_cqrs_operations()
    service.demonstrate_cqrs_decorators()

    # Advanced patterns
    service.demonstrate_event_sourcing()
    service.demonstrate_model_separation()

    # Deprecation warnings
    service.demonstrate_deprecated_patterns()

    print("\n" + "=" * 60)
    print("✅ ALL FlextCqrs patterns demonstrated!")
    print("🎯 Next: See 11_bus_messaging.py for FlextBus patterns")
    print("=" * 60)


if __name__ == "__main__":
    main()
