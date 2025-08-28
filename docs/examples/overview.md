# Examples Guide

Comprehensive examples demonstrating FLEXT Core patterns and best practices.

## Overview

This guide provides practical, working examples of FLEXT Core's key patterns and features. Each example is self-contained and demonstrates real-world usage patterns.

## Quick Start Examples

### Available Imports

```python
# Core patterns - Railway-oriented programming
from flext_core import FlextResult

# Dependency injection
from flext_core import FlextContainer, get_flext_container

# Domain patterns
from flext_core import FlextEntity, FlextValue, FlextAggregates

# Configuration management
from flext_core import FlextConfig

# Utilities
from flext_core.utilities import generate_id, generate_uuid
from flext_core import get_logger
```

## Example 1: Railway-Oriented Programming with FlextResult

### Basic Error Handling

```python
from flext_core import FlextResult
from typing import Optional

def parse_integer(value: str) -> FlextResult[int]:
    """Parse string to integer with error handling."""
    try:
        return FlextResult[None].ok(int(value))
    except ValueError:
        return FlextResult[None].fail(f"'{value}' is not a valid integer")

def divide_numbers(a: int, b: int) -> FlextResult[float]:
    """Safe division with error handling."""
    if b == 0:
        return FlextResult[None].fail("Cannot divide by zero")
    return FlextResult[None].ok(a / b)

# Chain operations
result = (
    parse_integer("10")
    .flat_map(lambda a: parse_integer("2")
              .flat_map(lambda b: divide_numbers(a, b)))
)

if result.success:
    print(f"Result: {result.unwrap()}")  # Result: 5.0
else:
    print(f"Error: {result.error}")
```

### Complex Business Logic

```python
from flext_core import FlextResult
from decimal import Decimal
from datetime import datetime

class Order:
    def __init__(self, id: str, total: Decimal):
        self.id = id
        self.total = total
        self.status = "pending"

def validate_order(order: Order) -> FlextResult[Order]:
    """Validate order before processing."""
    if order.total <= 0:
        return FlextResult[None].fail("Order total must be positive")
    if order.status != "pending":
        return FlextResult[None].fail(f"Cannot process {order.status} order")
    return FlextResult[None].ok(order)

def check_inventory(order: Order) -> FlextResult[Order]:
    """Check if items are in stock."""
    # Simulated inventory check
    in_stock = True
    if not in_stock:
        return FlextResult[None].fail("Items out of stock")
    return FlextResult[None].ok(order)

def charge_payment(order: Order) -> FlextResult[str]:
    """Process payment for order."""
    # Simulated payment processing
    if order.total > 10000:
        return FlextResult[None].fail("Payment amount exceeds limit")

    transaction_id = f"txn_{datetime.now().timestamp():.0f}"
    return FlextResult[None].ok(transaction_id)

def ship_order(order: Order, transaction_id: str) -> FlextResult[dict]:
    """Ship the order after successful payment."""
    tracking_number = f"TRACK-{order.id}-{transaction_id[:8]}"

    return FlextResult[None].ok({
        "order_id": order.id,
        "transaction_id": transaction_id,
        "tracking_number": tracking_number,
        "status": "shipped"
    })

# Complete order processing pipeline
def process_order(order: Order) -> FlextResult[dict]:
    """Process order through complete pipeline."""
    return (
        validate_order(order)
        .flat_map(check_inventory)
        .flat_map(lambda o: charge_payment(o)
                  .flat_map(lambda txn_id: ship_order(o, txn_id)))
        .map_error(lambda e: f"Order processing failed: {e}")
    )

# Usage
order = Order("ORD-123", Decimal("99.99"))
result = process_order(order)

if result.success:
    print(f"Order processed: {result.unwrap()}")
else:
    print(f"Processing failed: {result.error}")
```

## Example 2: Dependency Injection with FlextContainer

### Service Registration and Resolution

```python
"""
Real example using FlextContainer — FLEXT Core's DI system.
This example works with the current implementation.
"""

from flext_core import FlextContainer, FlextResult

# Simple services for DI example
class DatabaseService:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    def save(self, data: dict) -> FlextResult[str]:
        """Simulate database save."""
        return FlextResult[None].ok(f"Saved {data} to {self.connection_string}")

class UserService:
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service

    def create_user(self, name: str, email: str) -> FlextResult[dict]:
        """Create user using injected database service."""
        user_data = {"name": name, "email": email}

        save_result = self.db_service.save(user_data)
        if save_result.is_failure:
            return FlextResult[None].fail(f"Save failed: {save_result.error}")

        return FlextResult[None].ok(user_data)

# Container setup and usage
def setup_container() -> FlextContainer:
    """Setup dependency injection container."""
    container = FlextContainer()

    # Register database service
    db_service = DatabaseService("sqlite:///app.db")
    reg_result = container.register("database", db_service)
    if reg_result.is_failure:
        raise RuntimeError(f"Failed to register database: {reg_result.error}")

    # Register user service with dependency
    user_service = UserService(db_service)
    reg_result = container.register("user_service", user_service)
    if reg_result.is_failure:
        raise RuntimeError(f"Failed to register user service: {reg_result.error}")

    return container

# Usage example
if __name__ == "__main__":
    # Setup container
    container = setup_container()

    # Get service from container
    service_result = container.get("user_service")
    if service_result.success:
        user_service = service_result.data

        # Use service
        create_result = user_service.create_user("John", "john@test.com")
        if create_result.success:
            print(f"✅ User created: {create_result.data}")
        else:
            print(f"❌ Create failed: {create_result.error}")
    else:
        print(f"❌ Service not found: {service_result.error}")
```

## Example 3: Domain-Driven Design Patterns

### Entities with Business Logic

```python
from flext_core import FlextEntity, FlextResult
from datetime import datetime
from typing import Optional

class Account(FlextEntity):
    """Bank account entity with business rules."""

    account_number: str
    owner_name: str
    balance: float
    is_active: bool = True
    created_at: datetime
    daily_withdrawal_limit: float = 1000.0
    daily_withdrawn: float = 0.0

    def deposit(self, amount: float) -> FlextResult[float]:
        """Deposit money into account."""
        if amount <= 0:
            return FlextResult[None].fail("Deposit amount must be positive")

        if not self.is_active:
            return FlextResult[None].fail("Account is not active")

        self.balance += amount
        self.add_domain_event("MoneyDeposited", {
            "account": self.account_number,
            "amount": amount,
            "new_balance": self.balance
        })

        return FlextResult[None].ok(self.balance)

    def withdraw(self, amount: float) -> FlextResult[float]:
        """Withdraw money from account."""
        if amount <= 0:
            return FlextResult[None].fail("Withdrawal amount must be positive")

        if not self.is_active:
            return FlextResult[None].fail("Account is not active")

        if amount > self.balance:
            return FlextResult[None].fail("Insufficient funds")

        if self.daily_withdrawn + amount > self.daily_withdrawal_limit:
            return FlextResult[None].fail(f"Exceeds daily limit of {self.daily_withdrawal_limit}")

        self.balance -= amount
        self.daily_withdrawn += amount

        self.add_domain_event("MoneyWithdrawn", {
            "account": self.account_number,
            "amount": amount,
            "new_balance": self.balance
        })

        return FlextResult[None].ok(self.balance)

    def transfer_to(self, target: 'Account', amount: float) -> FlextResult[None]:
        """Transfer money to another account."""
        # Withdraw from this account
        withdraw_result = self.withdraw(amount)
        if withdraw_result.is_failure:
            return FlextResult[None].fail(f"Transfer failed: {withdraw_result.error}")

        # Deposit to target account
        deposit_result = target.deposit(amount)
        if deposit_result.is_failure:
            # Rollback withdrawal
            self.balance += amount
            self.daily_withdrawn -= amount
            return FlextResult[None].fail(f"Transfer failed: {deposit_result.error}")

        self.add_domain_event("MoneyTransferred", {
            "from": self.account_number,
            "to": target.account_number,
            "amount": amount
        })

        return FlextResult[None].ok(None)

# Usage
account1 = Account(
    id="acc_001",
    account_number="1234567890",
    owner_name="Alice Smith",
    balance=1000.0,
    created_at=datetime.now()
)

account2 = Account(
    id="acc_002",
    account_number="0987654321",
    owner_name="Bob Jones",
    balance=500.0,
    created_at=datetime.now()
)

# Perform transfer
transfer_result = account1.transfer_to(account2, 250.0)
if transfer_result.success:
    print(f"Transfer successful")
    print(f"Account 1 balance: {account1.balance}")  # 750.0
    print(f"Account 2 balance: {account2.balance}")  # 750.0

    # Check domain events
    for event in account1.get_events():
        print(f"Event: {event}")
```

### Value Objects for Domain Concepts

```python
from flext_core import FlextValue, FlextResult
from decimal import Decimal

class Money(FlextValue):
    """Immutable money value object."""

    amount: Decimal
    currency: str = "USD"

    def add(self, other: 'Money') -> FlextResult['Money']:
        """Add two money values."""
        if self.currency != other.currency:
            return FlextResult[None].fail(f"Cannot add {self.currency} and {other.currency}")

        return FlextResult[None].ok(Money(
            amount=self.amount + other.amount,
            currency=self.currency
        ))

    def multiply(self, factor: Decimal) -> 'Money':
        """Multiply money by a factor."""
        return Money(
            amount=self.amount * factor,
            currency=self.currency
        )

    def __str__(self) -> str:
        return f"{self.currency} {self.amount:.2f}"

class Address(FlextValue):
    """Immutable address value object."""

    street: str
    city: str
    state: str
    postal_code: str
    country: str = "USA"

    @property
    def full_address(self) -> str:
        """Get formatted full address."""
        return f"{self.street}, {self.city}, {self.state} {self.postal_code}, {self.country}"

    def is_same_city(self, other: 'Address') -> bool:
        """Check if addresses are in the same city."""
        return self.city == other.city and self.state == other.state

# Usage
price = Money(amount=Decimal("19.99"), currency="USD")
tax = Money(amount=Decimal("1.80"), currency="USD")

total_result = price.add(tax)
if total_result.success:
    print(f"Total: {total_result.unwrap()}")  # USD 21.79

address = Address(
    street="123 Main St",
    city="San Francisco",
    state="CA",
    postal_code="94102"
)
print(f"Address: {address.full_address}")
```

### Aggregate Roots for Consistency

```python
from flext_core import FlextAggregates, FlextResult
from typing import List
from datetime import datetime

class ShoppingCart(FlextAggregates):
    """Shopping cart aggregate maintaining consistency."""

    customer_id: str
    items: List[dict] = []
    created_at: datetime
    updated_at: datetime
    status: str = "active"

    def add_item(self, product_id: str, name: str,
                 price: float, quantity: int) -> FlextResult[None]:
        """Add item to cart with validation."""
        if self.status != "active":
            return FlextResult[None].fail("Cannot modify inactive cart")

        if quantity <= 0:
            return FlextResult[None].fail("Quantity must be positive")

        if price < 0:
            return FlextResult[None].fail("Price cannot be negative")

        # Check if item already exists
        for item in self.items:
            if item["product_id"] == product_id:
                item["quantity"] += quantity
                self.updated_at = datetime.now()
                self.add_domain_event("ItemQuantityUpdated", {
                    "cart_id": self.id,
                    "product_id": product_id,
                    "new_quantity": item["quantity"]
                })
                return FlextResult[None].ok(None)

        # Add new item
        self.items.append({
            "product_id": product_id,
            "name": name,
            "price": price,
            "quantity": quantity
        })

        self.updated_at = datetime.now()
        self.add_domain_event("ItemAddedToCart", {
            "cart_id": self.id,
            "product_id": product_id,
            "quantity": quantity
        })

        return FlextResult[None].ok(None)

    def remove_item(self, product_id: str) -> FlextResult[None]:
        """Remove item from cart."""
        if self.status != "active":
            return FlextResult[None].fail("Cannot modify inactive cart")

        for i, item in enumerate(self.items):
            if item["product_id"] == product_id:
                self.items.pop(i)
                self.updated_at = datetime.now()
                self.add_domain_event("ItemRemovedFromCart", {
                    "cart_id": self.id,
                    "product_id": product_id
                })
                return FlextResult[None].ok(None)

        return FlextResult[None].fail(f"Item {product_id} not found in cart")

    def calculate_total(self) -> float:
        """Calculate cart total."""
        return sum(item["price"] * item["quantity"] for item in self.items)

    def checkout(self) -> FlextResult[float]:
        """Checkout cart and return total."""
        if self.status != "active":
            return FlextResult[None].fail("Cart already checked out")

        if not self.items:
            return FlextResult[None].fail("Cart is empty")

        total = self.calculate_total()
        self.status = "checked_out"
        self.updated_at = datetime.now()

        self.add_domain_event("CartCheckedOut", {
            "cart_id": self.id,
            "total": total,
            "items_count": len(self.items)
        })

        return FlextResult[None].ok(total)

# Usage
cart = ShoppingCart(
    id="cart_123",
    customer_id="customer_456",
    created_at=datetime.now(),
    updated_at=datetime.now()
)

# Add items
cart.add_item("prod_1", "Laptop", 999.99, 1)
cart.add_item("prod_2", "Mouse", 29.99, 2)
cart.add_item("prod_1", "Laptop", 999.99, 1)  # Increases quantity

print(f"Cart total: ${cart.calculate_total():.2f}")

# Checkout
checkout_result = cart.checkout()
if checkout_result.success:
    print(f"Checked out. Total: ${checkout_result.unwrap():.2f}")

    # View events
    for event in cart.get_events():
        print(f"Event: {event}")
```

## Example 4: Configuration Management with FlextConfig

### Environment-Based Configuration

```python
"""
Example using FlextConfig — FLEXT Core configuration system.
Based on the current implementation.
"""

from flext_core import FlextConfig
from typing import Optional

class AppSettings(FlextConfig):
    """Application configuration using FLEXT Core settings."""

    # Basic settings with defaults
    app_name: str = "FLEXT Demo App"
    debug: bool = False
    port: int = 8000

    # Database settings
    database_url: str = "sqlite:///app.db"
    max_connections: int = 10

    # Optional settings
    redis_url: Optional[str] = None

    class Config:
        env_prefix = "APP_"

# Usage example
if __name__ == "__main__":
    # Load configuration (from env vars or defaults)
    settings = AppSettings()

    print(f"✅ App: {settings.app_name}")
    print(f"✅ Debug: {settings.debug}")
    print(f"✅ Port: {settings.port}")
    print(f"✅ Database: {settings.database_url}")
    print(f"✅ Redis: {settings.redis_url or 'Not configured'}")

    # Environment-aware settings
    if settings.debug:
        print("🔧 Running in debug mode")
    else:
        print("🚀 Running in production mode")
```

## Example 5: Command and Query Patterns (CQRS)

### Commands for State Changes

```python
from flext_core import FlextResult
from dataclasses import dataclass
from typing import Protocol

@dataclass
class CreateProductCommand:
    """Command to create a new product."""
    name: str
    description: str
    price: float
    stock: int

    def validate(self) -> FlextResult[None]:
        """Validate command data."""
        if not self.name:
            return FlextResult[None].fail("Product name is required")
        if self.price < 0:
            return FlextResult[None].fail("Price cannot be negative")
        if self.stock < 0:
            return FlextResult[None].fail("Stock cannot be negative")
        return FlextResult[None].ok(None)

@dataclass
class UpdateStockCommand:
    """Command to update product stock."""
    product_id: str
    quantity_change: int  # Can be positive or negative

    def validate(self) -> FlextResult[None]:
        """Validate command data."""
        if not self.product_id:
            return FlextResult[None].fail("Product ID is required")
        if self.quantity_change == 0:
            return FlextResult[None].fail("Quantity change cannot be zero")
        return FlextResult[None].ok(None)

class CommandHandler(Protocol):
    """Protocol for command handlers."""

    def handle(self, command) -> FlextResult:
        """Handle the command."""
        ...

class CreateProductHandler:
    """Handler for CreateProductCommand."""

    def __init__(self, repository):
        self.repository = repository

    def handle(self, command: CreateProductCommand) -> FlextResult[str]:
        """Create a new product."""
        # Validate command
        validation = command.validate()
        if validation.is_failure:
            return FlextResult[None].fail(validation.error)

        # Create product
        product_id = f"prod_{hash(command.name) % 10000:04d}"
        product = {
            "id": product_id,
            "name": command.name,
            "description": command.description,
            "price": command.price,
            "stock": command.stock
        }

        # Save to repository
        save_result = self.repository.save(product)
        if save_result.is_failure:
            return FlextResult[None].fail(f"Failed to save: {save_result.error}")

        return FlextResult[None].ok(product_id)

# Usage
command = CreateProductCommand(
    name="Wireless Keyboard",
    description="Bluetooth mechanical keyboard",
    price=79.99,
    stock=50
)

handler = CreateProductHandler(repository)
result = handler.handle(command)
if result.success:
    print(f"Product created with ID: {result.unwrap()}")
```

### Queries for Data Retrieval

```python
@dataclass
class GetProductByIdQuery:
    """Query to get product by ID."""
    product_id: str

@dataclass
class SearchProductsQuery:
    """Query to search products."""
    search_term: str
    min_price: float = 0
    max_price: float = float('inf')
    in_stock_only: bool = False
    limit: int = 10
    offset: int = 0

class GetProductByIdHandler:
    """Handler for GetProductByIdQuery."""

    def __init__(self, repository):
        self.repository = repository

    def handle(self, query: GetProductByIdQuery) -> FlextResult[dict]:
        """Get product by ID."""
        if not query.product_id:
            return FlextResult[None].fail("Product ID is required")

        product = self.repository.find_by_id(query.product_id)
        if not product:
            return FlextResult[None].fail(f"Product {query.product_id} not found")

        return FlextResult[None].ok(product)

class SearchProductsHandler:
    """Handler for SearchProductsQuery."""

    def __init__(self, repository):
        self.repository = repository

    def handle(self, query: SearchProductsQuery) -> FlextResult[list]:
        """Search for products."""
        # Build search criteria
        criteria = {
            "search_term": query.search_term,
            "price_range": (query.min_price, query.max_price),
            "in_stock": query.in_stock_only
        }

        # Execute search
        products = self.repository.search(
            criteria,
            limit=query.limit,
            offset=query.offset
        )

        return FlextResult[None].ok(products)

# Usage
query = SearchProductsQuery(
    search_term="keyboard",
    min_price=50,
    max_price=200,
    in_stock_only=True
)

handler = SearchProductsHandler(repository)
result = handler.handle(query)
if result.success:
    products = result.unwrap()
    for product in products:
        print(f"Found: {product['name']} - ${product['price']}")
```

## Example 6: Event-Driven Patterns

### Domain Events

```python
from flext_core import FlextResult
from dataclasses import dataclass
from datetime import datetime
from typing import List, Callable

@dataclass
class DomainEvent:
    """Base class for domain events."""
    event_id: str
    aggregate_id: str
    occurred_at: datetime
    event_type: str

@dataclass
class OrderPlacedEvent(DomainEvent):
    """Event when order is placed."""
    order_id: str
    customer_id: str
    total_amount: float
    items_count: int

@dataclass
class PaymentProcessedEvent(DomainEvent):
    """Event when payment is processed."""
    order_id: str
    payment_id: str
    amount: float
    payment_method: str

class EventBus:
    """Simple event bus for publishing and subscribing."""

    def __init__(self):
        self.handlers: dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to an event type."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    def publish(self, event: DomainEvent) -> None:
        """Publish an event to all subscribers."""
        event_handlers = self.handlers.get(event.event_type, [])
        for handler in event_handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Handler error: {e}")

# Event handlers
def send_order_confirmation(event: OrderPlacedEvent):
    """Send email confirmation when order is placed."""
    print(f"Sending confirmation for order {event.order_id} to customer {event.customer_id}")

def update_inventory(event: OrderPlacedEvent):
    """Update inventory when order is placed."""
    print(f"Updating inventory for {event.items_count} items in order {event.order_id}")

def generate_invoice(event: PaymentProcessedEvent):
    """Generate invoice when payment is processed."""
    print(f"Generating invoice for payment {event.payment_id} (${event.amount})")

# Setup event bus
event_bus = EventBus()
event_bus.subscribe("OrderPlaced", send_order_confirmation)
event_bus.subscribe("OrderPlaced", update_inventory)
event_bus.subscribe("PaymentProcessed", generate_invoice)

# Publish events
order_event = OrderPlacedEvent(
    event_id="evt_001",
    aggregate_id="order_123",
    occurred_at=datetime.now(),
    event_type="OrderPlaced",
    order_id="order_123",
    customer_id="customer_456",
    total_amount=299.99,
    items_count=3
)

payment_event = PaymentProcessedEvent(
    event_id="evt_002",
    aggregate_id="order_123",
    occurred_at=datetime.now(),
    event_type="PaymentProcessed",
    order_id="order_123",
    payment_id="pay_789",
    amount=299.99,
    payment_method="credit_card"
)

event_bus.publish(order_event)
event_bus.publish(payment_event)
```

## Running the Examples

### Prerequisites

```bash
# Install FLEXT Core
pip install flext-core

# Or install from source
cd flext-core
poetry install

# Verify installation
python -c "from flext_core import FlextResult; print('Installation successful')"
```

### Running Individual Examples

Save any example to a Python file and run:

```bash
python example_railway.py
python example_domain.py
python example_cqrs.py
python example_events.py
```

### Creating Your Own Examples

```python
#!/usr/bin/env python3
"""Template for FLEXT Core examples."""

from flext_core import FlextResult, FlextContainer, FlextConfig
from flext_core import FlextEntity, FlextValue, FlextAggregates

def main():
    """Main example function."""
    # Your code here
    result = FlextResult[None].ok("Success!")
    print(result.unwrap())

if __name__ == "__main__":
    main()
```

## Best Practices

### 1. Always Use FlextResult for Error Handling

```python
# Good: Explicit error handling
def process_data(data: dict) -> FlextResult[dict]:
    if not data:
        return FlextResult[None].fail("Empty data")
    return FlextResult[None].ok(processed_data)

# Bad: Using exceptions
def process_data_bad(data: dict) -> dict:
    if not data:
        raise ValueError("Empty data")
    return processed_data
```

### 2. Chain Operations with flat_map

```python
# Good: Chained operations
result = (
    validate_input(data)
    .flat_map(transform_data)
    .flat_map(save_to_database)
    .map(format_response)
)

# Bad: Nested if statements
result = validate_input(data)
if result.success:
    result = transform_data(result.unwrap())
    if result.success:
        result = save_to_database(result.unwrap())
        if result.success:
            result = format_response(result.unwrap())
```

### 3. Use Dependency Injection for Testability

```python
# Good: Dependencies injected
class Service:
    def __init__(self, repository, cache, logger):
        self.repository = repository
        self.cache = cache
        self.logger = logger

# Bad: Hard-coded dependencies
class ServiceBad:
    def __init__(self):
        self.repository = DatabaseRepository()
        self.cache = RedisCache()
        self.logger = Logger()
```

### 4. Keep Domain Logic in Entities

```python
# Good: Business logic in entity
class Order(FlextEntity):
    def apply_discount(self, percentage: float) -> FlextResult[None]:
        if percentage < 0 or percentage > 100:
            return FlextResult[None].fail("Invalid discount percentage")
        self.total *= (1 - percentage / 100)
        return FlextResult[None].ok(None)

# Bad: Business logic scattered
def apply_discount_to_order(order: dict, percentage: float) -> dict:
    order["total"] *= (1 - percentage / 100)
    return order
```

## Additional Resources

- **[API Reference](../api/core.md)**: Complete API documentation
- **[Architecture Guide](../architecture/overview.md)**: Architectural patterns and principles
- **[Configuration Guide](../configuration/overview.md)**: Configuration management details
- **[Best Practices](../development/best-practices.md)**: Development guidelines
- **[Getting Started](../getting-started/quickstart.md)**: Quick start guide

## Repository Examples

For more comprehensive examples, check the `examples/` directory in the repository:

- `01_flext_result_railway_pattern.py`: Advanced railway pattern usage
- `02_flext_container_dependency_injection.py`: Complex DI scenarios
- `03_flext_commands_cqrs_pattern.py`: CQRS implementation
- `04_flext_utilities_modular.py`: Utility functions
- `05_flext_validation_advanced_system.py`: Validation patterns
- `06_flext_entity_valueobject_ddd_patterns.py`: DDD examples
- And many more...

---

These examples demonstrate the core patterns and best practices of FLEXT Core. Each example is self-contained and can be run independently.
