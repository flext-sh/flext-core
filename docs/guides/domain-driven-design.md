# Domain-Driven Design with FlextModels

**Status**: Production Ready | **Version**: 0.10.0 | **Pattern**: Clean Architecture Foundation

FlextModels provides domain-driven design (DDD) patterns through semantic base classes for building rich domain models with proper entity and value object semantics.

## Core Concepts

### Domain-Driven Design (DDD)

DDD is an approach to software development that emphasizes:

1. **Ubiquitous Language**: Domain experts and developers use the same terminology
2. **Bounded Contexts**: Clear boundaries around domain concepts
3. **Entity Invariants**: Objects protect their own business rules
4. **Value Objects**: Immutable objects compared by value, not identity
5. **Aggregates**: Clusters of entities that maintain invariants together

### FlextModels Architecture

```
FlextModels.AggregateRoot
        ↓
FlextModels.Entity
        ↓
m.Value
        ↓
Pydantic BaseModel
        ↓
Python 3.13+ Type System
```

## Building Blocks

### Value Objects: Immutable by Semantics

Value objects have **no identity** - they're compared by their values:

```python
from flext_core import FlextModels
from decimal import Decimal

class Money(m.Value):
    """Money is a value object - represented by amount and currency."""
    amount: Decimal
    currency: str  # "USD", "EUR", "GBP", etc.

    def add(self, other: "Money") -> "Money":
        """Add two money amounts (same currency)."""
        if self.currency != other.currency:
            raise ValueError("Cannot add different currencies")
        return Money(amount=self.amount + other.amount, currency=self.currency)

    def multiply(self, factor: Decimal) -> "Money":
        """Multiply money by a factor."""
        return Money(amount=self.amount * factor, currency=self.currency)

# Value objects compared by value
money1 = Money(amount=Decimal("100"), currency="USD")
money2 = Money(amount=Decimal("100"), currency="USD")
money3 = Money(amount=Decimal("50"), currency="USD")

assert money1 == money2  # Same values = equal
assert money1 != money3  # Different values = not equal
assert money1 is not money2  # Different objects
```

#### When to Use Value Objects

- Represent domain concepts that are described by their values
- Are immutable - never change after creation
- Can be part of entities (embedded)
- Examples: Money, Email, PhoneNumber, Address, Color, Temperature

```python
from flext_core import FlextModels

class Email(m.Value):
    """Email address - value object."""
    address: str

class PhoneNumber(m.Value):
    """Phone number - value object."""
    country_code: str
    number: str

class Address(m.Value):
    """Physical address - value object."""
    street: str
    city: str
    postal_code: str
    country: str
```

### Entities: Identity and Mutability

Entities have **identity** - they're unique regardless of their values:

```python
from flext_core import FlextModels
from decimal import Decimal
from datetime import datetime

class Order(FlextModels.Entity):
    """Order is an entity - identified by order_id."""
    order_id: str  # Unique identifier
    customer_id: str
    items: list[dict]
    total: Decimal
    status: str  # "pending", "shipped", "delivered"
    created_at: datetime

    def add_item(self, item: dict) -> None:
        """Mutate order by adding an item."""
        self.items.append(item)
        self.recalculate_total()

    def ship(self) -> None:
        """Ship the order."""
        if self.status != "pending":
            raise ValueError("Can only ship pending orders")
        self.status = "shipped"

    def recalculate_total(self) -> None:
        """Recalculate order total from items."""
        self.total = sum(item["price"] * item["quantity"] for item in self.items)

# Entities compared by identity (order_id)
order1 = Order(order_id="ORD-001", customer_id="CUST-1", items=[], total=Decimal("0"), status="pending", created_at=datetime.now())
order2 = Order(order_id="ORD-001", customer_id="CUST-1", items=[], total=Decimal("0"), status="pending", created_at=datetime.now())
order3 = Order(order_id="ORD-002", customer_id="CUST-1", items=[], total=Decimal("0"), status="pending", created_at=datetime.now())

assert order1 == order2  # Same identity (order_id) = equal
assert order1 != order3  # Different identity = not equal
assert order1 is not order2  # Different objects (but same identity)
```

#### When to Use Entities

- Represent domain concepts with identity (unique identifier)
- Can be mutable - change over their lifetime
- Have business logic and methods
- Have lifecycle (create, update, delete)
- Examples: User, Order, Account, Product

```python
from flext_core import FlextModels
from datetime import datetime

class User(FlextModels.Entity):
    """User entity - identified by user_id."""
    user_id: str
    username: str
    email: str
    created_at: datetime
    is_active: bool = True

    def deactivate(self) -> None:
        """Business logic: deactivate user."""
        self.is_active = False

class Product(FlextModels.Entity):
    """Product entity - identified by sku."""
    sku: str
    name: str
    price: Decimal
    inventory: int

    def decrease_inventory(self, quantity: int) -> None:
        """Business logic: decrease inventory."""
        if quantity > self.inventory:
            raise ValueError("Insufficient inventory")
        self.inventory -= quantity
```

### Aggregate Roots: Consistency Boundaries

Aggregate roots are entities that enforce invariants across related entities:

```python
from flext_core import FlextModels
from decimal import Decimal
from datetime import datetime

class OrderItem(FlextModels.Entity):
    """Order line item - part of Order aggregate."""
    item_id: str
    product_id: str
    quantity: int
    unit_price: Decimal

class ShippingInfo(m.Value):
    """Shipping address - value object."""
    address: str
    city: str
    postal_code: str
    country: str

class Order(FlextModels.AggregateRoot):
    """Order aggregate root - maintains invariants."""
    order_id: str  # Aggregate identity
    customer_id: str
    items: list[OrderItem]
    total: Decimal
    status: str  # "pending", "confirmed", "shipped", "delivered"
    shipping_info: ShippingInfo
    created_at: datetime

    # Aggregate Invariants (business rules that must always be true)
    def __init__(self, **data):
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """Enforce aggregate invariants."""
        # Invariant 1: No empty orders
        if not self.items:
            raise ValueError("Order must have at least one item")

        # Invariant 2: Total must match items
        calculated_total = sum(
            item.quantity * item.unit_price
            for item in self.items
        )
        if self.total != calculated_total:
            raise ValueError("Order total does not match items")

        # Invariant 3: Can only change status to valid states
        valid_states = ["pending", "confirmed", "shipped", "delivered", "cancelled"]
        if self.status not in valid_states:
            raise ValueError(f"Invalid status: {self.status}")

    def add_item(self, item: OrderItem) -> None:
        """Add item maintaining invariants."""
        self.items.append(item)
        self._recalculate_total()
        self._validate_invariants()

    def remove_item(self, item_id: str) -> None:
        """Remove item maintaining invariants."""
        self.items = [item for item in self.items if item.item_id != item_id]
        self._recalculate_total()
        self._validate_invariants()

    def confirm(self) -> None:
        """Transition to confirmed state."""
        if self.status != "pending":
            raise ValueError("Can only confirm pending orders")
        self.status = "confirmed"

    def ship(self) -> None:
        """Transition to shipped state."""
        if self.status != "confirmed":
            raise ValueError("Can only ship confirmed orders")
        self.status = "shipped"

    def _recalculate_total(self) -> None:
        """Recalculate total from items."""
        self.total = sum(
            item.quantity * item.unit_price
            for item in self.items
        )

# Using the aggregate
order = Order(
    order_id="ORD-001",
    customer_id="CUST-1",
    items=[
        OrderItem(item_id="I1", product_id="P1", quantity=2, unit_price=Decimal("50")),
    ],
    total=Decimal("100"),
    status="pending",
    shipping_info=ShippingInfo(
        address="123 Main St",
        city="New York",
        postal_code="10001",
        country="USA",
    ),
    created_at=datetime.now(),
)

# Add item (aggregate maintains invariants)
order.add_item(OrderItem(item_id="I2", product_id="P2", quantity=1, unit_price=Decimal("75")))

# Total is automatically recalculated
assert order.total == Decimal("175")

# Transition through valid states
order.confirm()
order.ship()

# Invalid operations throw errors
try:
    order.confirm()  # Can't confirm shipped order
except ValueError as e:
    print(f"Business rule violation: {e}")
```

#### When to Use Aggregate Roots

- Represent a cluster of related entities
- Maintain consistency across contained entities
- Enforce business rule invariants
- Act as the entry point for modifications
- Examples: Order with OrderItems, ShoppingCart with CartItems, Invoice with InvoiceLines

## Domain Events and CQRS Integration

Domain events capture important state changes inside aggregates. FLEXT surfaces domain events through `FlextModels.DomainEvent` and dispatcher publishing so other bounded contexts can react without direct coupling:

```python
from flext_core import FlextResult
from flext_core.dispatcher import FlextDispatcher
from flext_core.handlers import h
from flext_core.models import FlextModels


class InventoryAdjusted(FlextModels.DomainEvent):
    sku: str
    quantity: int


class InventoryAdjustedHandler(h[InventoryAdjusted, bool]):
    def handle(self, message: InventoryAdjusted) -> FlextResult[bool]:
        # Side-effect: notify downstream system or persist projection
        return FlextResult[bool].ok(True)


class Product(FlextModels.AggregateRoot):
    sku: str
    inventory: int

    def decrease_inventory(self, quantity: int) -> None:
        if quantity > self.inventory:
            raise ValueError("Insufficient inventory")
        self.inventory -= quantity
        self.add_domain_event(InventoryAdjusted(sku=self.sku, quantity=quantity))


dispatcher = FlextDispatcher()
dispatcher.register_handler(InventoryAdjusted, InventoryAdjustedHandler())

product = Product(sku="ABC", inventory=10)
product.decrease_inventory(3)

# Domain events collected on the aggregate can be emitted through the dispatcher
dispatch_result = dispatcher.publish_events(product.commit_domain_events())
assert dispatch_result.is_success
```

Key points:

- Aggregates collect domain events via `add_domain_event` and return them with `get_domain_events` or `commit_domain_events`.
- `FlextDispatcher.register_handler` accepts command, query, or event handlers; event handlers can subclass `h` for validation and telemetry.
- `publish_event` / `publish_events` reuse the dispatcher pipeline, so middleware (logging, retries, timeouts) is applied consistently across commands, queries, and domain events.

## Real-World Examples

### Example 1: E-Commerce Order System

```python
from flext_core import FlextModels, FlextResult
from decimal import Decimal
from datetime import datetime
from enum import Enum

class OrderStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

class Money(m.Value):
    """Money value object."""
    amount: Decimal
    currency: str = "USD"

class Address(m.Value):
    """Address value object."""
    street: str
    city: str
    postal_code: str
    country: str

class OrderLine(FlextModels.Entity):
    """Order line item."""
    line_id: str
    product_id: str
    product_name: str
    quantity: int
    unit_price: Money

    def subtotal(self) -> Money:
        """Calculate line subtotal."""
        amount = self.unit_price.amount * self.quantity
        return Money(amount=amount, currency=self.unit_price.currency)

class Order(FlextModels.AggregateRoot):
    """Order aggregate root."""
    order_id: str
    customer_id: str
    lines: list[OrderLine]
    shipping_address: Address
    billing_address: Address
    status: OrderStatus
    subtotal: Money
    tax: Money
    total: Money
    created_at: datetime
    updated_at: datetime

    def __init__(self, **data):
        super().__init__(**data)
        self._validate_invariants()

    def _validate_invariants(self):
        """Enforce order invariants."""
        # Must have at least one line
        if not self.lines:
            raise ValueError("Order must have at least one line item")

        # Total must match calculation
        calculated_subtotal = sum(
            line.subtotal().amount
            for line in self.lines
        )
        if self.subtotal.amount != calculated_subtotal:
            raise ValueError("Order subtotal calculation mismatch")

        # Total must be subtotal + tax
        calculated_total = self.subtotal.amount + self.tax.amount
        if self.total.amount != calculated_total:
            raise ValueError("Order total calculation mismatch")

    def add_line(self, line: OrderLine) -> FlextResult[bool]:
        """Add line to order."""
        if self.status != OrderStatus.PENDING:
            return FlextResult[bool].fail(
                "Can only modify pending orders",
                error_code="ORDER_NOT_MODIFIABLE",
            )

        self.lines.append(line)
        self._recalculate_totals()

        try:
            self._validate_invariants()
            return FlextResult[bool].| ok(value=True)
        except ValueError as e:
            return FlextResult[bool].fail(
                str(e),
                error_code="ORDER_INVARIANT_VIOLATION",
            )

    def remove_line(self, line_id: str) -> FlextResult[bool]:
        """Remove line from order."""
        if self.status != OrderStatus.PENDING:
            return FlextResult[bool].fail(
                "Can only modify pending orders",
                error_code="ORDER_NOT_MODIFIABLE",
            )

        self.lines = [line for line in self.lines if line.line_id != line_id]
        self._recalculate_totals()

        try:
            self._validate_invariants()
            return FlextResult[bool].| ok(value=True)
        except ValueError as e:
            return FlextResult[bool].fail(
                str(e),
                error_code="ORDER_INVARIANT_VIOLATION",
            )

    def confirm(self) -> FlextResult[bool]:
        """Confirm order (transition to confirmed state)."""
        if self.status != OrderStatus.PENDING:
            return FlextResult[bool].fail(
                f"Cannot confirm order in {self.status} state",
                error_code="INVALID_STATE_TRANSITION",
            )

        self.status = OrderStatus.CONFIRMED
        self.updated_at = datetime.now()
        return FlextResult[bool].| ok(value=True)

    def ship(self) -> FlextResult[bool]:
        """Ship order (transition to shipped state)."""
        if self.status != OrderStatus.CONFIRMED:
            return FlextResult[bool].fail(
                f"Cannot ship order in {self.status} state",
                error_code="INVALID_STATE_TRANSITION",
            )

        self.status = OrderStatus.SHIPPED
        self.updated_at = datetime.now()
        return FlextResult[bool].| ok(value=True)

    def _recalculate_totals(self):
        """Recalculate order totals."""
        subtotal_amount = sum(
            line.subtotal().amount
            for line in self.lines
        )
        self.subtotal = Money(amount=subtotal_amount)

        # Tax = 10% of subtotal
        tax_amount = subtotal_amount * Decimal("0.10")
        self.tax = Money(amount=tax_amount)

        # Total = subtotal + tax
        total_amount = subtotal_amount + tax_amount
        self.total = Money(amount=total_amount)

# Usage
order = Order(
    order_id="ORD-001",
    customer_id="CUST-1",
    lines=[
        OrderLine(
            line_id="L1",
            product_id="P1",
            product_name="Widget",
            quantity=2,
            unit_price=Money(amount=Decimal("50")),
        ),
    ],
    shipping_address=Address(
        street="123 Main St",
        city="New York",
        postal_code="10001",
        country="USA",
    ),
    billing_address=Address(
        street="123 Main St",
        city="New York",
        postal_code="10001",
        country="USA",
    ),
    status=OrderStatus.PENDING,
    subtotal=Money(amount=Decimal("100")),
    tax=Money(amount=Decimal("10")),
    total=Money(amount=Decimal("110")),
    created_at=datetime.now(),
    updated_at=datetime.now(),
)

# Add another line
add_result = order.add_line(
    OrderLine(
        line_id="L2",
        product_id="P2",
        product_name="Gadget",
        quantity=1,
        unit_price=Money(amount=Decimal("75")),
    )
)

if add_result.is_success:
    print(f"✅ Line added. New total: {order.total.amount}")
else:
    print(f"❌ Failed to add line: {add_result.error}")

# Confirm order
confirm_result = order.confirm()
if confirm_result.is_success:
    print("✅ Order confirmed")

# Ship order
ship_result = order.ship()
if ship_result.is_success:
    print("✅ Order shipped")
```

### Example 2: User Authentication System

```python
from flext_core import FlextModels, FlextResult
from datetime import datetime, timedelta
import re

class Email(m.Value):
    """Email value object."""
    address: str

    def __init__(self, address: str):
        if not re.match(r'^[^@]+@[^@]+\.[^@]+$', address):
            raise ValueError(f"Invalid email: {address}")
        super().__init__(address=address)

class Password(m.Value):
    """Password value object (hashed representation)."""
    hash: str

    @classmethod
    def from_plain(cls, plain: str) -> "Password":
        """Create password from plain text."""
        if len(plain) < 8:
            raise ValueError("Password must be at least 8 characters")
        # In real implementation, use bcrypt or similar
        return cls(hash=f"hashed_{plain}")

class User(FlextModels.AggregateRoot):
    """User aggregate root."""
    user_id: str
    email: Email
    password_hash: Password
    username: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login_at: datetime | None = None

    def login(self, plain_password: str) -> FlextResult[bool]:
        """Attempt to login."""
        if not self.is_active:
            return FlextResult[bool].fail(
                "User account is inactive",
                error_code="USER_INACTIVE",
            )

        if not self.is_verified:
            return FlextResult[bool].fail(
                "User account is not verified",
                error_code="USER_NOT_VERIFIED",
            )

        # Check password (simplified)
        password_check = Password.from_plain(plain_password)
        if password_check.hash != self.password_hash.hash:
            return FlextResult[bool].fail(
                "Invalid password",
                error_code="INVALID_PASSWORD",
            )

        self.last_login_at = datetime.now()
        return FlextResult[bool].| ok(value=True)

    def deactivate(self) -> FlextResult[bool]:
        """Deactivate user account."""
        if not self.is_active:
            return FlextResult[bool].fail(
                "User is already deactivated",
                error_code="ALREADY_DEACTIVATED",
            )

        self.is_active = False
        return FlextResult[bool].| ok(value=True)

    def verify_email(self) -> FlextResult[bool]:
        """Mark email as verified."""
        if self.is_verified:
            return FlextResult[bool].fail(
                "Email is already verified",
                error_code="ALREADY_VERIFIED",
            )

        self.is_verified = True
        return FlextResult[bool].| ok(value=True)

# Usage
user = User(
    user_id="USER-1",
    email=Email(address="user@example.com"),
    password_hash=Password.from_plain("securepassword"),
    username="alice",
    is_active=True,
    is_verified=False,
    created_at=datetime.now(),
)

# Try to login (fails - email not verified)
login_result = user.login("securepassword")
print(f"Login attempt: {login_result.error}")  # "User account is not verified"

# Verify email
verify_result = user.verify_email()
print(f"Email verification: {verify_result.is_success}")  # True

# Now login works
login_result = user.login("securepassword")
print(f"Login: {login_result.is_success}")  # True
```

## Integration with FlextResult

Always use `FlextResult` for operations that can fail:

```python
from flext_core import FlextModels, FlextResult

class User(FlextModels.Entity):
    username: str
    email: str

    def update_email(self, new_email: str) -> FlextResult[bool]:
        """Update user email with validation."""
        if not new_email or "@" not in new_email:
            return FlextResult[bool].fail(
                "Invalid email format",
                error_code="INVALID_EMAIL",
            )

        self.email = new_email
        return FlextResult[bool].| ok(value=True)

# Usage
user = User(username="alice", email="alice@example.com")
result = user.update_email("bob@example.com")

if result.is_success:
    print("✅ Email updated")
else:
    print(f"❌ Failed: {result.error}")
```

## CQRS: Command Query Responsibility Segregation

**CQRS** separates read operations (Queries) from write operations (Commands). This DDD pattern enables better scalability, performance, and clear intent.

### Commands: Write Operations

Commands represent requests to **change state**. They always return `FlextResult`:

```python
from flext_core import FlextModels, FlextResult, FlextService
from dataclasses import dataclass

# Command definitions (no logic, just data transfer objects)
@dataclass
class CreateUserCommand:
    username: str
    email: str
    password: str

@dataclass
class UpdateUserEmailCommand:
    user_id: str
    new_email: str

@dataclass
class DeleteUserCommand:
    user_id: str

# Command handler in service
class UserCommandService(FlextService):
    """Handles all user write operations."""

    def handle_create_user(self, cmd: CreateUserCommand) -> FlextResult[dict]:
        """Execute create user command."""
        # Validate business rules
        if not "@" in cmd.email:
            return FlextResult[dict].fail("Invalid email", error_code="INVALID_EMAIL")

        # Create aggregate
        user = User(id=f"user_{cmd.username}", username=cmd.username, email=cmd.email)

        # Publish domain event
        self.add_domain_event(UserCreatedEvent(user.entity_id, user.username))

        # Return result
        return FlextResult[dict].ok({"user_id": user.entity_id, "username": user.username})

    def handle_update_email(self, cmd: UpdateUserEmailCommand) -> FlextResult[bool]:
        """Execute update email command."""
        # Load aggregate
        user = self._load_user(cmd.user_id)
        if not user:
            return FlextResult[bool].fail("User not found")

        # Execute business logic
        result = user.update_email(cmd.new_email)
        if result.is_failure:
            return result

        # Publish event
        self.add_domain_event(UserEmailUpdatedEvent(cmd.user_id, cmd.new_email))

        return FlextResult[bool].| ok(value=True)
```

### Queries: Read Operations

Queries represent requests to **retrieve data**. They return `FlextResult`:

```python
from flext_core import FlextService, FlextResult

# Query definitions
@dataclass
class GetUserByIdQuery:
    user_id: str

@dataclass
class ListUsersQuery:
    limit: int = 10
    offset: int = 0

@dataclass
class SearchUsersQuery:
    username: str

# Query handler in service
class UserQueryService(FlextService):
    """Handles all user read operations."""

    def __init__(self, user_repository):
        super().__init__()
        self.user_repository = user_repository

    def handle_get_user(self, query: GetUserByIdQuery) -> FlextResult[dict]:
        """Execute get user by ID query."""
        user = self.user_repository.find_by_id(query.user_id)
        if not user:
            return FlextResult[dict].fail(f"User {query.user_id} not found")

        return FlextResult[dict].ok({
            "id": user.entity_id,
            "username": user.username,
            "email": user.email,
            "created_at": user.created_at,
        })

    def handle_list_users(self, query: ListUsersQuery) -> FlextResult[list]:
        """Execute list users query with pagination."""
        users = self.user_repository.list(
            limit=query.limit,
            offset=query.offset
        )

        return FlextResult[list].ok([
            {
                "id": u.entity_id,
                "username": u.username,
                "email": u.email,
            }
            for u in users
        ])

    def handle_search_users(self, query: SearchUsersQuery) -> FlextResult[list]:
        """Execute search users query."""
        users = self.user_repository.search_by_username(query.username)
        if not users:
            return FlextResult[list].ok([])  # Empty result is still success

        return FlextResult[list].ok([
            {"id": u.entity_id, "username": u.username}
            for u in users
        ])
```

### Dispatcher: Unified Command/Query Bus

Separate services by responsibility and dispatch through a unified bus:

```python
from flext_core import FlextDispatcher

# Setup dispatcher
dispatcher = FlextDispatcher.get_global()

# Register command handlers
command_service = UserCommandService()
dispatcher.register_command("CreateUserCommand", command_service.handle_create_user)
dispatcher.register_command("UpdateUserEmailCommand", command_service.handle_update_email)

# Register query handlers
query_service = UserQueryService(user_repository)
dispatcher.register_query("GetUserByIdQuery", query_service.handle_get_user)
dispatcher.register_query("ListUsersQuery", query_service.handle_list_users)

# Usage: Execute command
create_cmd = CreateUserCommand(username="alice", email="alice@example.com", password="secret")
result = dispatcher.dispatch_command(create_cmd)

if result.is_success:
    user_id = result.value["user_id"]
    print(f"✅ Created user: {user_id}")

# Usage: Execute query
get_query = GetUserByIdQuery(user_id=user_id)
result = dispatcher.dispatch_query(get_query)

if result.is_success:
    user_data = result.value
    print(f"User: {user_data['username']}")
```

### Benefits of CQRS in FLEXT

1. **Clear Intent**: Commands for writes, Queries for reads
2. **Scalability**: Scale read/write sides independently
3. **Performance**: Optimize queries separately from commands
4. **Testing**: Easier to test command/query logic in isolation
5. **Type Safety**: `FlextResult[T]` ensures predictable contracts
6. **Error Handling**: Consistent `FlextResult` return types

### When to Use CQRS

- ✅ Complex domain logic with distinct read/write paths
- ✅ Different scaling needs for reads vs writes
- ✅ Large aggregates with many operations
- ❌ Simple CRUD applications (overhead not justified)

## Best Practices

### 1. Protect Invariants in **init**

```python
# ✅ CORRECT - Validate in constructor
class Order(FlextModels.AggregateRoot):
    items: list[OrderItem]

    def __init__(self, **data):
        super().__init__(**data)
        if not self.items:
            raise ValueError("Order must have items")

# ❌ WRONG - No invariant protection
class Order(FlextModels.AggregateRoot):
    items: list[OrderItem]
    # No validation - could have empty items
```

### 2. Use Semantic Types for Values

```python
# ✅ CORRECT - Semantic value objects
class Money(m.Value):
    amount: Decimal
    currency: str

class Email(m.Value):
    address: str

# ❌ WRONG - Using primitives
class Order:
    total: Decimal  # Should be Money
    customer_email: str  # Should be Email
```

### 3. Enforce Business Rules

```python
# ✅ CORRECT - Business logic in entity methods
class ShoppingCart(FlextModels.Entity):
    items: list[CartItem]

    def add_item(self, item: CartItem) -> FlextResult[bool]:
        if len(self.items) >= 100:
            return FlextResult[bool].fail("Cart is full")
        self.items.append(item)
        return FlextResult[bool].| ok(value=True)

# ❌ WRONG - Business logic in caller
def add_to_cart(cart, item):
    if len(cart.items) >= 100:  # This belongs in entity!
        return False
    cart.items.append(item)
    return True
```

## Key Takeaways

1. **Value Objects**: Immutable, compared by value, no identity
2. **Entities**: Have identity, mutable, encapsulate business logic
3. **Aggregates**: Clusters of entities maintaining invariants
4. **Ubiquitous Language**: Use domain terms in code
5. **Invariants**: Protect business rules in entity methods
6. **FlextResult**: Use for operations that can fail

## Next Steps

1. **Service Patterns**: Explore [Service Patterns](./service-patterns.md) for domain service implementation
2. **Dependency Injection**: See [Advanced DI](./dependency-injection-advanced.md) for service composition
3. **Error Handling**: Check [Error Handling Guide](./error-handling.md) for domain error patterns
4. **Railway Patterns**: Review [Railway-Oriented Programming](./railway-oriented-programming.md) for result composition
5. **Architecture**: Study [Clean Architecture](../architecture/clean-architecture.md) for layer organization

## See Also

- [Service Patterns](./service-patterns.md) - Domain services with FlextService
- [Dependency Injection Advanced](./dependency-injection-advanced.md) - Service composition patterns
- [Railway-Oriented Programming](./railway-oriented-programming.md) - Result composition with FlextResult
- [Error Handling Guide](./error-handling.md) - Domain error handling patterns
- [Clean Architecture](../architecture/clean-architecture.md) - Architecture patterns and layers
- [API Reference: FlextModels](../api-reference/domain.md#flextmodels) - Complete models API
- **FLEXT CLAUDE.md**: Development patterns and standards

---

**Example from FLEXT Ecosystem**: See `src/flext_tests/test_models.py` for 200+ test cases demonstrating DDD patterns with FlextModels.
