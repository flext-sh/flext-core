# Layer 2: Domain API Reference

This section covers **Layer 2** (Domain) classes that implement Domain-Driven Design (DDD) patterns including Entities, Value Objects, Aggregates, and Domain Services.

> **Architecture**: Layer 2 depends only on Layer 1 (Foundation), Layer 0.5 (Runtime), and Layer 0 (Constants). See [Architecture Overview](../architecture/overview.md) for complete layer hierarchy.

## Domain Models

### FlextModels - DDD Base Classes

The `FlextModels` module provides base classes for implementing Domain-Driven Design patterns with Pydantic v2.

```python
from flext_core import FlextModels
from typing import List
from decimal import Decimal

# Entity - Identity-based domain objects
class User(FlextModels.Entity):
    """User entity with business logic."""
    name: str
    email: str
    age: int

    def model_post_init(self, __context) -> None:
        """Validate after initialization."""
        if self.age < 0:
            raise ValueError("Age cannot be negative")
        if "@" not in self.email:
            raise ValueError("Invalid email format")

# Value Object - Immutable value-based objects
class Address(FlextModels.Value):
    """Address value object."""
    street: str
    city: str
    postal_code: str
    country: str

# Aggregate Root - Entity that maintains consistency boundaries
class Order(FlextModels.AggregateRoot):
    """Order aggregate root."""
    customer_id: str
    items: List[OrderItem]
    total: Decimal
    status: OrderStatus

    def add_item(self, item: OrderItem) -> FlextResult[None]:
        """Add item to order with business rules."""
        if self.status != OrderStatus.PENDING:
            return FlextResult[None].fail("Can only add items to pending orders")

        self.items.append(item)
        self.total += item.price * item.quantity
        return FlextResult[None].ok(None)

    def confirm(self) -> FlextResult[None]:
        """Confirm order with validation."""
        if len(self.items) == 0:
            return FlextResult[None].fail("Cannot confirm empty order")

        self.status = OrderStatus.CONFIRMED
        return FlextResult[None].ok(None)
```

**Key Classes:**

- `FlextModels.Entity` - Base class for entities with identity
- `FlextModels.Value` - Base class for immutable value objects
- `FlextModels.AggregateRoot` - Base class for aggregate roots

### FlextService - Domain Service Base

Base class for domain services that encapsulate business logic.

```python
from flext_core import FlextService, FlextResult

class UserService(FlextService):
    """Domain service for user operations."""

    def create_user(self, name: str, email: str) -> FlextResult[User]:
        """Create user with domain rules."""
        # Business logic validation
        if not self._is_valid_email(email):
            return FlextResult[User].fail("Invalid email domain")

        # Create entity
        user = User(id=f"user_{name.lower()}", name=name, email=email)

        # Domain events
        self.add_domain_event(UserCreatedEvent(user.id))

        return FlextResult[User].ok(user)

    def _is_valid_email(self, email: str) -> bool:
        """Domain-specific email validation."""
        allowed_domains = ["company.com", "partner.com"]
        domain = email.split("@")[1] if "@" in email else ""
        return domain in allowed_domains

class OrderService(FlextService):
    """Domain service for order operations."""

    def process_order(self, order: Order) -> FlextResult[Order]:
        """Process order with complex business rules."""
        # Inventory check
        inventory_result = self._check_inventory(order)
        if inventory_result.is_failure:
            return inventory_result

        # Payment processing
        payment_result = self._process_payment(order)
        if payment_result.is_failure:
            return payment_result

        # Update order status
        order.status = OrderStatus.PROCESSING
        self.add_domain_event(OrderProcessedEvent(order.id))

        return FlextResult[Order].ok(order)
```

**Key Methods:**

- `add_domain_event(event)` - Add domain event for publishing
- Business logic encapsulation for complex operations

### Phase 1 Context Enrichment (v0.9.9)

**Status**: ✅ **COMPLETED** - Major enhancement providing zero-boilerplate context management for distributed tracing and audit trails.

#### Context Enrichment Methods

FlextService now provides automatic context management methods:

```python
from flext_core import FlextService, FlextTypes, FlextResult

class PaymentService(FlextService[FlextTypes.Dict]):
    """Service with automatic context enrichment."""

    def process_payment(self, payment_id: str, amount: float, user_id: str) -> FlextResult[dict]:
        # Generate correlation ID for distributed tracing
        correlation_id = self._with_correlation_id()

        # Set user context for audit trail
        self._with_user_context(user_id, payment_id=payment_id)

        # Set operation context for tracking
        self._with_operation_context("process_payment", amount=amount)

        # All logs now include full context automatically
        self.logger.info("Processing payment", payment_id=payment_id, amount=amount)

        return FlextResult[dict].ok({"status": "completed", "correlation_id": correlation_id})
```

**Available Context Methods:**

- `_with_correlation_id(correlation_id=None)` - Set/generate correlation ID for distributed tracing
- `_with_user_context(user_id, **user_data)` - Set user context for audit trails
- `_with_operation_context(operation_name, **data)` - Set operation context for tracking
- `_enrich_context(**context_data)` - Add custom metadata to logs
- `_clear_operation_context()` - Clean up operation context after completion

#### Complete Automation Helper

`execute_with_context_enrichment()` provides full automation:

```python
class OrderService(FlextService[Order]):
    def process_order(self, order_id: str, customer_id: str, correlation_id: str | None = None) -> FlextResult[Order]:
        """Process order with complete automation."""
        return self.execute_with_context_enrichment(
            operation_name="process_order",
            correlation_id=correlation_id,
            user_id=customer_id,
            order_id=order_id,
        )
        # Automatically handles:
        # - Correlation ID generation/setting
        # - User context enrichment
        # - Operation context tracking
        # - Performance tracking
        # - Operation logging (start/complete/error)
        # - Context cleanup
```

#### Benefits

- ✅ **Zero Boilerplate** - No manual context setup required
- ✅ **Distributed Tracing** - Automatic correlation ID generation
- ✅ **Audit Trail** - User context automatically captured
- ✅ **Ecosystem Ready** - Available to all 32+ dependent projects
- ✅ **Performance Tracking** - Operation lifecycle monitoring

See `examples/automation_showcase.py` for complete working examples.

## Domain Events

### Event System

Domain events for decoupled communication between domain objects.

```python
from datetime import datetime
from decimal import Decimal

class UserCreatedEvent(DomainEvent):
    """Event raised when user is created."""
    user_id: str
    email: str
    created_at: datetime

class OrderConfirmedEvent(DomainEvent):
    """Event raised when order is confirmed."""
    order_id: str
    customer_id: str
    total_amount: Decimal

# Publishing events
user = User(id="user_123", name="Alice", email="alice@company.com")
event = UserCreatedEvent(
    user_id=user.id,
    email=user.email,
    created_at=datetime.utcnow()
)

# Events are collected by aggregate roots and published by infrastructure
```

## Domain Mixins

### Reusable Behaviors

Mixins that provide common domain behaviors.

Each mixin ships with ready-to-use behavior:

- `FlextMixins.Auditable` keeps `created_at` and `updated_at` timestamps in sync.
- `FlextMixins.SoftDeletable` manages `is_deleted`/`deleted_at` flags and exposes `delete()`/`restore()` helpers.
- `FlextMixins.Versioned` performs optimistic locking by auto-incrementing the `version` field whenever state changes.

```python
from flext_core import FlextModels, FlextMixins

class AuditableEntity(FlextModels.Entity, FlextMixins.Auditable):
    """Entity with audit timestamps managed automatically."""
    name: str
    # created_at / updated_at are added by the mixin on save or mutation

class SoftDeletableEntity(FlextModels.Entity, FlextMixins.SoftDeletable):
    """Entity that gains soft-delete helpers from the mixin."""
    name: str
    # call instance.delete() / restore() provided by the mixin; it tracks deleted_at/is_deleted for you

class VersionedEntity(FlextModels.Entity, FlextMixins.Versioned):
    """Entity with optimistic locking handled by the mixin."""
    name: str
    version: int = 1
    # mixin auto-increments version on state changes; override hooks only if custom logic is required
```

**Available Mixins:**

- `FlextMixins.Auditable` - Created/updated timestamps
- `FlextMixins.SoftDeletable` - Soft deletion support
- `FlextMixins.Versioned` - Optimistic locking version
- `FlextMixins.Serializable` - JSON serialization helpers

## Domain Utilities

### Helper Functions

Utility functions for common domain operations.

```python
from decimal import Decimal
from flext_core import FlextUtilities

# Validation helpers
is_valid_email = FlextUtilities.Validation.is_email("user@domain.com")
is_valid_phone = FlextUtilities.Validation.is_phone("+1234567890")

# Type guards
def is_active_user(user: User) -> bool:
    return FlextUtilities.TypeGuards.is_entity(user) and not user.is_deleted

# Conversion helpers
price_str = FlextUtilities.Conversion.decimal_to_string(Decimal("19.99"))
price_decimal = FlextUtilities.Conversion.string_to_decimal("19.99")

# String utilities
slug = FlextUtilities.String.slugify("Hello World!")  # "hello-world"
camel = FlextUtilities.String.to_camel_case("hello_world")  # "helloWorld"
```

## Quality Metrics

| Layer | Module         | Coverage | Status       | Description                   |
| ----- | -------------- | -------- | ------------ | ----------------------------- |
| **2** | `models.py`    | 64%      | 🔄 Improving | DDD base classes and patterns |
| **2** | `service.py`   | 100%     | ✅ Stable    | Domain service infrastructure |
| **2** | `mixins.py`    | 77%      | ✅ Good      | Reusable domain behaviors     |
| **2** | `utilities.py` | 65%      | 🔄 Improving | Domain utility functions      |

## Usage Examples

### Complete Domain Model Example

```python
from typing import List
from decimal import Decimal
from datetime import datetime
from flext_core import FlextModels, FlextService, FlextResult

# Value Objects
class Money(FlextModels.Value):
    amount: Decimal
    currency: str

class Address(FlextModels.Value):
    street: str
    city: str
    postal_code: str
    country: str

# Entity
class Customer(FlextModels.Entity):
    name: str
    email: str
    addresses: List[Address]

# Aggregate Root
class Order(FlextModels.AggregateRoot):
    customer_id: str
    items: List[OrderItem]
    total: Money
    status: OrderStatus

    def add_item(self, product: Product, quantity: int) -> FlextResult[None]:
        if self.status != OrderStatus.PENDING:
            return FlextResult[None].fail("Can only modify pending orders")

        item = OrderItem(product=product, quantity=quantity)
        self.items.append(item)
        self.total = Money(
            amount=self.total.amount + (product.price.amount * quantity),
            currency=self.total.currency
        )
        return FlextResult[None].ok(None)

# Domain Service
class OrderService(FlextService):
    def place_order(self, customer: Customer, items: List[OrderItem]) -> FlextResult[Order]:
        # Business rule validation
        if not items:
            return FlextResult[Order].fail("Order must contain at least one item")

        # Calculate total
        total_amount = Decimal("0")
        for item in items:
            total_amount += item.product.price.amount * item.quantity

        total = Money(amount=total_amount, currency="USD")

        # Create order
        order = Order(
            id=f"order_{datetime.utcnow().timestamp()}",
            customer_id=customer.id,
            items=items,
            total=total,
            status=OrderStatus.PENDING
        )

        # Add domain event
        self.add_domain_event(OrderPlacedEvent(order.id, customer.id, total.amount))

        return FlextResult[Order].ok(order)

# Sample products and order items used below
product1 = Product(
    id="prod_storage",
    name="Cloud Storage Plan",
    price=Money(amount=Decimal("19.99"), currency="USD"),
)

product2 = Product(
    id="prod_analytics",
    name="Realtime Analytics Add-on",
    price=Money(amount=Decimal("39.99"), currency="USD"),
)

item1 = OrderItem(product=product1, quantity=1)
item2 = OrderItem(product=product2, quantity=2)

# Usage
customer = Customer(
    id="cust_123",
    name="Alice Johnson",
    email="alice@company.com",
    addresses=[]
)

service = OrderService()
result = service.place_order(customer, [item1, item2])

if result.is_success:
    order = result.unwrap()
    print(f"Order placed: {order.id} for ${order.total.amount}")
```

This domain layer provides a solid foundation for implementing business logic with proper separation of concerns and domain-driven design principles.
