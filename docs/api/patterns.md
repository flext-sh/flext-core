# Patterns API Reference

Advanced patterns and design implementations in FLEXT Core.

## Command Pattern (CQRS)

Commands represent intent to change system state.

### Basic Command

```python
from flext_core import FlextResult

class CreateOrderCommand:
    """Command to create a new order."""

    def __init__(self, customer_id: str, items: list[dict]):
        self.customer_id = customer_id
        self.items = items
        self.command_id = f"cmd_{generate_id()}"
        self.timestamp = datetime.now()

    def validate(self) -> FlextResult[None]:
        """Validate command data."""
        if not self.customer_id:
            return FlextResult.fail("Customer ID required")

        if not self.items:
            return FlextResult.fail("Order must have items")

        for item in self.items:
            if item.get("quantity", 0) <= 0:
                return FlextResult.fail("Invalid item quantity")

        return FlextResult.ok(None)
```

### Command Handler

```python
from flext_core import FlextResult

class CreateOrderHandler:
    """Handler for CreateOrderCommand."""

    def __init__(self, order_repository, inventory_service, event_bus):
        self.order_repository = order_repository
        self.inventory_service = inventory_service
        self.event_bus = event_bus

    def handle(self, command: CreateOrderCommand) -> FlextResult[Order]:
        """Process the command."""
        # Validate command
        validation = command.validate()
        if validation.is_failure:
            return FlextResult.fail(validation.error)

        # Check inventory
        for item in command.items:
            stock_check = self.inventory_service.check_stock(
                item["product_id"],
                item["quantity"]
            )
            if stock_check.is_failure:
                return FlextResult.fail(f"Insufficient stock: {stock_check.error}")

        # Create order
        order = Order(
            id=generate_id("order"),
            customer_id=command.customer_id,
            items=command.items,
            status="pending"
        )

        # Save order
        save_result = self.order_repository.save(order)
        if save_result.is_failure:
            return save_result

        # Publish event
        self.event_bus.publish(OrderCreatedEvent(
            order_id=order.id,
            customer_id=order.customer_id,
            total=order.calculate_total()
        ))

        return FlextResult.ok(order)
```

### Command Bus

```python
class CommandBus:
    """Route commands to handlers."""

    def __init__(self):
        self.handlers: dict[type, Any] = {}

    def register(self, command_type: type, handler: Any) -> None:
        """Register handler for command type."""
        self.handlers[command_type] = handler

    def dispatch(self, command: Any) -> FlextResult[Any]:
        """Dispatch command to handler."""
        handler = self.handlers.get(type(command))
        if not handler:
            return FlextResult.fail(f"No handler for {type(command).__name__}")

        try:
            return handler.handle(command)
        except Exception as e:
            return FlextResult.fail(f"Handler error: {str(e)}")

# Usage
bus = CommandBus()
bus.register(CreateOrderCommand, CreateOrderHandler(repo, inventory, events))

command = CreateOrderCommand("customer_123", items)
result = bus.dispatch(command)
```

## Query Pattern (CQRS)

Queries retrieve data without modifying state.

### Basic Query

```python
class GetOrderByIdQuery:
    """Query to get order by ID."""

    def __init__(self, order_id: str, include_items: bool = True):
        self.order_id = order_id
        self.include_items = include_items

class GetOrdersByCustomerQuery:
    """Query to get customer orders."""

    def __init__(self, customer_id: str, status: str = None,
                 limit: int = 10, offset: int = 0):
        self.customer_id = customer_id
        self.status = status
        self.limit = limit
        self.offset = offset
```

### Query Handler

```python
class OrderQueryHandler:
    """Handler for order queries."""

    def __init__(self, read_repository):
        self.repository = read_repository

    def get_by_id(self, query: GetOrderByIdQuery) -> FlextResult[Order]:
        """Get order by ID."""
        order = self.repository.find_by_id(query.order_id)
        if not order:
            return FlextResult.fail(f"Order {query.order_id} not found")

        if not query.include_items:
            order.items = []  # Clear items if not requested

        return FlextResult.ok(order)

    def get_by_customer(self, query: GetOrdersByCustomerQuery) -> FlextResult[list[Order]]:
        """Get orders by customer."""
        filters = {"customer_id": query.customer_id}
        if query.status:
            filters["status"] = query.status

        orders = self.repository.find_by_filters(
            filters=filters,
            limit=query.limit,
            offset=query.offset
        )

        return FlextResult.ok(orders)
```

## Handler Chain Pattern

Chain handlers for cross-cutting concerns.

### Handler Middleware

```python
from abc import ABC, abstractmethod

class HandlerMiddleware(ABC):
    """Base middleware for handlers."""

    def __init__(self, next_handler = None):
        self.next = next_handler

    @abstractmethod
    def handle(self, request: Any) -> FlextResult[Any]:
        """Process request."""
        pass

class LoggingMiddleware(HandlerMiddleware):
    """Log all requests."""

    def __init__(self, logger, next_handler = None):
        super().__init__(next_handler)
        self.logger = logger

    def handle(self, request: Any) -> FlextResult[Any]:
        self.logger.info(f"Processing: {type(request).__name__}")

        if self.next:
            result = self.next.handle(request)
            self.logger.info(f"Result: {'success' if result.success else 'failure'}")
            return result

        return FlextResult.fail("No handler configured")

class ValidationMiddleware(HandlerMiddleware):
    """Validate requests."""

    def handle(self, request: Any) -> FlextResult[Any]:
        if hasattr(request, 'validate'):
            validation = request.validate()
            if validation.is_failure:
                return validation

        if self.next:
            return self.next.handle(request)

        return FlextResult.fail("No handler configured")

class AuthorizationMiddleware(HandlerMiddleware):
    """Authorize requests."""

    def __init__(self, auth_service, next_handler = None):
        super().__init__(next_handler)
        self.auth_service = auth_service

    def handle(self, request: Any) -> FlextResult[Any]:
        if hasattr(request, 'user_id'):
            auth_result = self.auth_service.authorize(
                request.user_id,
                type(request).__name__
            )
            if auth_result.is_failure:
                return FlextResult.fail(f"Unauthorized: {auth_result.error}")

        if self.next:
            return self.next.handle(request)

        return FlextResult.fail("No handler configured")
```

### Building Handler Pipeline

```python
def build_handler_pipeline(core_handler, logger, auth_service):
    """Build handler with middleware chain."""
    # Build chain: Logging -> Validation -> Authorization -> Core
    return LoggingMiddleware(
        logger,
        ValidationMiddleware(
            AuthorizationMiddleware(
                auth_service,
                core_handler
            )
        )
    )

# Usage
core_handler = CreateOrderHandler(repo, inventory, events)
pipeline = build_handler_pipeline(core_handler, logger, auth_service)

command = CreateOrderCommand("customer_123", items)
result = pipeline.handle(command)
```

## Validation Patterns

Advanced validation with business rules.

### Validation Rules

```python
from typing import Protocol

class ValidationRule(Protocol):
    """Validation rule protocol."""

    def validate(self, value: Any) -> FlextResult[Any]:
        """Validate value."""
        ...

class RequiredRule:
    """Value is required."""

    def __init__(self, message: str = "Value is required"):
        self.message = message

    def validate(self, value: Any) -> FlextResult[Any]:
        if value is None or (isinstance(value, str) and not value.strip()):
            return FlextResult.fail(self.message)
        return FlextResult.ok(value)

class MinLengthRule:
    """Minimum length validation."""

    def __init__(self, min_length: int, message: str = None):
        self.min_length = min_length
        self.message = message or f"Minimum length is {min_length}"

    def validate(self, value: str) -> FlextResult[str]:
        if len(value) < self.min_length:
            return FlextResult.fail(self.message)
        return FlextResult.ok(value)

class EmailRule:
    """Email format validation."""

    def validate(self, value: str) -> FlextResult[str]:
        if "@" not in value or "." not in value.split("@")[1]:
            return FlextResult.fail("Invalid email format")
        return FlextResult.ok(value.lower())

class RangeRule:
    """Numeric range validation."""

    def __init__(self, min_val: float = None, max_val: float = None):
        self.min_val = min_val
        self.max_val = max_val

    def validate(self, value: float) -> FlextResult[float]:
        if self.min_val is not None and value < self.min_val:
            return FlextResult.fail(f"Value must be >= {self.min_val}")
        if self.max_val is not None and value > self.max_val:
            return FlextResult.fail(f"Value must be <= {self.max_val}")
        return FlextResult.ok(value)
```

### Composite Validator

```python
class Validator:
    """Composite validator with multiple rules."""

    def __init__(self):
        self.rules: dict[str, list[ValidationRule]] = {}

    def add_rule(self, field: str, rule: ValidationRule) -> 'Validator':
        """Add validation rule for field."""
        if field not in self.rules:
            self.rules[field] = []
        self.rules[field].append(rule)
        return self

    def validate(self, data: dict) -> FlextResult[dict]:
        """Validate all fields."""
        errors = []
        validated = {}

        for field, rules in self.rules.items():
            value = data.get(field)

            for rule in rules:
                result = rule.validate(value)
                if result.is_failure:
                    errors.append(f"{field}: {result.error}")
                    break
                value = result.unwrap()

            if not errors:
                validated[field] = value

        if errors:
            return FlextResult.fail("; ".join(errors))

        return FlextResult.ok(validated)

# Usage
user_validator = (
    Validator()
    .add_rule("name", RequiredRule("Name is required"))
    .add_rule("name", MinLengthRule(2, "Name too short"))
    .add_rule("email", RequiredRule("Email is required"))
    .add_rule("email", EmailRule())
    .add_rule("age", RangeRule(0, 150))
)

result = user_validator.validate({
    "name": "John",
    "email": "john@example.com",
    "age": 30
})
```

## Event Handling Patterns

Domain events and event sourcing foundations.

### Domain Events

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DomainEvent:
    """Base domain event."""
    event_id: str
    aggregate_id: str
    event_type: str
    timestamp: datetime
    data: dict

    @classmethod
    def create(cls, aggregate_id: str, event_type: str, data: dict):
        """Create new domain event."""
        return cls(
            event_id=generate_id("evt"),
            aggregate_id=aggregate_id,
            event_type=event_type,
            timestamp=datetime.now(),
            data=data
        )

class OrderCreatedEvent(DomainEvent):
    """Order created event."""

    def __init__(self, order_id: str, customer_id: str, total: Decimal):
        super().__init__(
            event_id=generate_id("evt"),
            aggregate_id=order_id,
            event_type="OrderCreated",
            timestamp=datetime.now(),
            data={
                "order_id": order_id,
                "customer_id": customer_id,
                "total": str(total)
            }
        )
```

### Event Bus

```python
class EventBus:
    """Publish and subscribe to events."""

    def __init__(self):
        self.handlers: dict[str, list[Callable]] = {}

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to event type."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    def publish(self, event: DomainEvent) -> None:
        """Publish event to subscribers."""
        handlers = self.handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

# Usage
event_bus = EventBus()

def on_order_created(event: OrderCreatedEvent):
    """Handle order created event."""
    print(f"New order: {event.data['order_id']}")
    # Send confirmation email
    # Update inventory
    # Generate invoice

event_bus.subscribe("OrderCreated", on_order_created)
event_bus.publish(OrderCreatedEvent(order_id, customer_id, total))
```

## Repository Pattern

Abstract data access with repositories.

### Repository Interface

```python
from abc import ABC, abstractmethod

class Repository(ABC):
    """Base repository interface."""

    @abstractmethod
    def find_by_id(self, entity_id: str) -> FlextResult[Any]:
        """Find entity by ID."""
        pass

    @abstractmethod
    def save(self, entity: Any) -> FlextResult[None]:
        """Save entity."""
        pass

    @abstractmethod
    def delete(self, entity_id: str) -> FlextResult[None]:
        """Delete entity."""
        pass

    @abstractmethod
    def find_all(self, limit: int = 100, offset: int = 0) -> FlextResult[list]:
        """Find all entities."""
        pass

class OrderRepository(Repository):
    """Order repository implementation."""

    def __init__(self, database):
        self.db = database

    def find_by_id(self, order_id: str) -> FlextResult[Order]:
        try:
            data = self.db.query_one("SELECT * FROM orders WHERE id = ?", order_id)
            if not data:
                return FlextResult.fail(f"Order {order_id} not found")
            return FlextResult.ok(Order.from_dict(data))
        except Exception as e:
            return FlextResult.fail(f"Database error: {e}")

    def save(self, order: Order) -> FlextResult[None]:
        try:
            self.db.execute(
                "INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?)",
                order.id, order.customer_id, order.status, order.to_json()
            )
            return FlextResult.ok(None)
        except Exception as e:
            return FlextResult.fail(f"Save failed: {e}")
```

## Unit of Work Pattern

Manage transactions across repositories.

```python
class UnitOfWork:
    """Coordinate transactions across repositories."""

    def __init__(self, database):
        self.database = database
        self.orders = OrderRepository(database)
        self.customers = CustomerRepository(database)
        self.inventory = InventoryRepository(database)

    def __enter__(self):
        """Start transaction."""
        self.database.begin_transaction()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Complete or rollback transaction."""
        if exc_type:
            self.rollback()
        else:
            self.commit()

    def commit(self) -> FlextResult[None]:
        """Commit transaction."""
        try:
            self.database.commit()
            return FlextResult.ok(None)
        except Exception as e:
            return FlextResult.fail(f"Commit failed: {e}")

    def rollback(self) -> None:
        """Rollback transaction."""
        self.database.rollback()

# Usage
with UnitOfWork(database) as uow:
    # Create order
    order = Order(...)
    uow.orders.save(order)

    # Update inventory
    for item in order.items:
        product = uow.inventory.find_by_id(item.product_id)
        product.reduce_stock(item.quantity)
        uow.inventory.save(product)

    # Update customer
    customer = uow.customers.find_by_id(order.customer_id)
    customer.add_order(order.id)
    uow.customers.save(customer)

    # All changes committed together
```

## Best Practices

1. **Use commands** for operations that change state
2. **Use queries** for read operations
3. **Validate early** in command handlers
4. **Publish events** for decoupled communication
5. **Use repositories** to abstract data access
6. **Apply middleware** for cross-cutting concerns
7. **Implement unit of work** for transaction management
8. **Keep handlers focused** on single responsibility

---

For implementation examples, see [Examples Guide](../examples/overview.md).
