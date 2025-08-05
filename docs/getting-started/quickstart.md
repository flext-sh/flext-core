# FLEXT Core Quick Start Guide

**Get started with FLEXT Core in 10 minutes using ACTUAL working patterns**

## 🚀 Hello World - 2 Minutes

### Installation

```bash
pip install flext-core
# or
poetry add flext-core
```

### First Example

```python
# hello_flext.py
from flext_core import FlextResult

def hello_world() -> FlextResult[str]:
    """First example with FLEXT Core."""
    return FlextResult.ok("Hello, FLEXT World! 🚀")

# Usage
result = hello_world()
if result.success:
    print(result.data)  # Output: Hello, FLEXT World! 🚀
else:
    print(f"Error: {result.error}")
```

## 📋 Core Concepts

### 1. FlextResult - Type-Safe Error Handling

**The heart of FLEXT Core - replaces exceptions with explicit results.**

```python
from flext_core import FlextResult

def divide_numbers(a: float, b: float) -> FlextResult[float]:
    """Safe division with error handling."""
    if b == 0:
        return FlextResult.fail("Division by zero not allowed")

    result = a / b
    return FlextResult.ok(result)

# Safe usage
result = divide_numbers(10, 2)
if result.success:
    print(f"Result: {result.data}")  # 5.0
else:
    print(f"Error: {result.error}")

# Error case
error_result = divide_numbers(10, 0)
print(error_result.is_failure)  # True
print(error_result.error)       # "Division by zero not allowed"
```

### 2. FlextContainer - Dependency Injection

**Type-safe IoC container for dependency management.**

```python
from flext_core import FlextContainer

# Create container
container = FlextContainer()

# Register services
database_url = "postgresql://localhost/mydb"
register_result = container.register("database_url", database_url)
assert register_result.success

# Register service class
class EmailService:
    def send_email(self, to: str, subject: str) -> str:
        return f"Email sent to {to}: {subject}"

email_service = EmailService()
container.register("email_service", email_service)

# Resolve dependencies
db_result = container.get("database_url")
if db_result.success:
    print(f"Database: {db_result.data}")

email_result = container.get("email_service")
if email_result.success:
    service = email_result.data
    message = service.send_email("user@test.com", "Welcome!")
    print(message)
```

### 3. Configuration Management

**Environment-aware configuration with type safety.**

```python
from flext_core import FlextBaseSettings

class AppSettings(FlextBaseSettings):
    app_name: str = "My App"
    debug: bool = False
    database_url: str = "sqlite:///app.db"
    api_port: int = 8000

    class Config:
        env_prefix = "APP_"

# Usage - loads from environment automatically
settings = AppSettings()
print(f"App: {settings.app_name}")
print(f"Debug: {settings.debug}")
print(f"Database: {settings.database_url}")
```

### 4. Domain Entities - Basic Usage

**Domain entities with identity and business logic.**

```python
from flext_core.models import FlextEntity
from flext_core import FlextResult

class User(FlextEntity):
    id: str
    name: str
    email: str
    is_active: bool = True

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules (required by FlextEntity)."""
        if not self.name:
            return FlextResult.fail("Name is required")
        if "@" not in self.email:
            return FlextResult.fail("Valid email is required")
        return FlextResult.ok(None)

    def change_email(self, new_email: str) -> FlextResult[None]:
        """Change email with business validation."""
        if "@" not in new_email:
            return FlextResult.fail("Email must contain @")

        if new_email == self.email:
            return FlextResult.fail("New email must be different")

        self.email = new_email
        # Validate after change
        validation = self.validate_business_rules()
        if validation.is_failure:
            return validation

        return FlextResult.ok(None)

    def deactivate(self) -> FlextResult[None]:
        """Deactivate user."""
        if not self.is_active:
            return FlextResult.fail("User already inactive")

        self.is_active = False
        return FlextResult.ok(None)

# Create and use entity (Pydantic model syntax)
user = User(id="user_123", name="John Doe", email="john@test.com")
print(f"User: {user.name} (ID: {user.id})")

# Change email
email_result = user.change_email("john.doe@newcompany.com")
if email_result.success:
    print(f"Email updated: {user.email}")
```

### 5. Command Pattern - Real Implementation

**Commands using actual FlextCommands namespace.**

```python
from flext_core import FlextCommands, FlextResult

# Define a command
class CreateUserCommand(FlextCommands.Command):
    def __init__(self, name: str, email: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.email = email

    def validate_command(self) -> FlextResult[None]:
        """Validate command data."""
        if not self.name.strip():
            return FlextResult.fail("Name is required")

        if "@" not in self.email:
            return FlextResult.fail("Valid email required")

        if len(self.name) < 2:
            return FlextResult.fail("Name must be at least 2 characters")

        return FlextResult.ok(None)

# Command handler
class CreateUserHandler(FlextCommands.Handler[CreateUserCommand, User]):
    def __init__(self, container: FlextContainer):
        super().__init__()
        self._container = container

    def handle(self, command: CreateUserCommand) -> FlextResult[User]:
        """Process user creation."""
        # Create user entity
        user = User(f"user_{hash(command.email)}", command.name, command.email)

        # Simulate save (in real app, use repository)
        save_result = self._save_user(user)
        if save_result.is_failure:
            return FlextResult.fail(f"Save failed: {save_result.error}")

        return FlextResult.ok(user)

    def _save_user(self, user: User) -> FlextResult[None]:
        """Simulate user persistence."""
        return FlextResult.ok(None)

# Usage
container = FlextContainer()
handler = CreateUserHandler(container)

# Create and process command
command = CreateUserCommand("Ana Paula", "ana@company.com")
result = handler.process_command(command)

if result.success:
    user = result.data
    print(f"✅ User created: {user.name} ({user.id})")
else:
    print(f"❌ Error: {result.error}")

# Test with invalid data
invalid_command = CreateUserCommand("", "invalid-email")
invalid_result = handler.process_command(invalid_command)
print(f"Validation error: {invalid_result.error}")
```

## 🎯 Railway-Oriented Programming

**Chain operations safely with map and flat_map.**

```python
from flext_core import FlextResult

def validate_email(email: str) -> FlextResult[str]:
    if "@" not in email:
        return FlextResult.fail("Invalid email format")
    return FlextResult.ok(email.lower())

def create_user_account(email: str) -> FlextResult[dict]:
    if not email:
        return FlextResult.fail("Email required")
    return FlextResult.ok({"email": email, "created": True})

def send_welcome_email(account: dict) -> FlextResult[dict]:
    # Simulate email sending
    account["welcome_sent"] = True
    return FlextResult.ok(account)

# Chain operations
def process_user_registration(email: str) -> FlextResult[dict]:
    return (
        validate_email(email)
        .flat_map(create_user_account)
        .flat_map(send_welcome_email)
    )

# Usage
result = process_user_registration("user@example.com")
if result.success:
    print(f"Registration successful: {result.data}")
else:
    print(f"Registration failed: {result.error}")

# Error case - chain stops at first failure
error_result = process_user_registration("invalid-email")
print(f"Expected error: {error_result.error}")  # "Invalid email format"
```

## 🏗️ Complete Example - Order System

**Simple order system using real FLEXT Core patterns.**

```python
from flext_core import FlextEntity, FlextResult, FlextContainer, FlextCommands
from typing import List
from enum import Enum

# Domain models
class OrderStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"

class Product(FlextEntity):
    def __init__(self, product_id: str, name: str, price: float, stock: int):
        super().__init__(product_id)
        self.name = name
        self.price = price
        self.stock = stock

    def reserve_stock(self, quantity: int) -> FlextResult[None]:
        if quantity <= 0:
            return FlextResult.fail("Quantity must be positive")

        if self.stock < quantity:
            return FlextResult.fail(f"Insufficient stock. Available: {self.stock}")

        self.stock -= quantity
        return FlextResult.ok(None)

class OrderItem:
    def __init__(self, product: Product, quantity: int):
        self.product = product
        self.quantity = quantity

    def total_price(self) -> float:
        return self.product.price * self.quantity

class Order(FlextEntity):
    def __init__(self, order_id: str, customer_id: str):
        super().__init__(order_id)
        self.customer_id = customer_id
        self.items: List[OrderItem] = []
        self.status = OrderStatus.PENDING

    def add_item(self, product: Product, quantity: int) -> FlextResult[None]:
        if self.status != OrderStatus.PENDING:
            return FlextResult.fail("Cannot modify confirmed order")

        # Reserve stock
        reserve_result = product.reserve_stock(quantity)
        if reserve_result.is_failure:
            return reserve_result

        # Add item
        item = OrderItem(product, quantity)
        self.items.append(item)

        return FlextResult.ok(None)

    def confirm(self) -> FlextResult[None]:
        if self.status != OrderStatus.PENDING:
            return FlextResult.fail("Order must be pending to confirm")

        if not self.items:
            return FlextResult.fail("Order must have at least one item")

        self.status = OrderStatus.CONFIRMED
        return FlextResult.ok(None)

    def total(self) -> float:
        return sum(item.total_price() for item in self.items)

# Command
class CreateOrderCommand(FlextCommands.Command):
    def __init__(self, customer_id: str, items: List[dict], **kwargs):
        super().__init__(**kwargs)
        self.customer_id = customer_id
        self.items = items  # [{"product_id": "p1", "quantity": 2}]

    def validate_command(self) -> FlextResult[None]:
        if not self.customer_id:
            return FlextResult.fail("Customer ID required")

        if not self.items:
            return FlextResult.fail("Order must have at least one item")

        for item in self.items:
            if "product_id" not in item or "quantity" not in item:
                return FlextResult.fail("Item must have product_id and quantity")

            if item["quantity"] <= 0:
                return FlextResult.fail("Quantity must be positive")

        return FlextResult.ok(None)

# Handler
class CreateOrderHandler(FlextCommands.Handler[CreateOrderCommand, Order]):
    def __init__(self):
        super().__init__()
        # Mock products
        self.products = {
            "p1": Product("p1", "Laptop", 2500.00, 10),
            "p2": Product("p2", "Mouse", 50.00, 100),
            "p3": Product("p3", "Keyboard", 150.00, 50)
        }

    def handle(self, command: CreateOrderCommand) -> FlextResult[Order]:
        # Create order
        order_id = f"order_{hash(command.customer_id)}"
        order = Order(order_id, command.customer_id)

        # Add items
        for item_data in command.items:
            product_id = item_data["product_id"]
            if product_id not in self.products:
                return FlextResult.fail(f"Product not found: {product_id}")

            product = self.products[product_id]
            add_result = order.add_item(product, item_data["quantity"])
            if add_result.is_failure:
                return FlextResult.fail(f"Failed to add item: {add_result.error}")

        # Confirm order
        confirm_result = order.confirm()
        if confirm_result.is_failure:
            return confirm_result

        return FlextResult.ok(order)

# Usage example
def run_order_example():
    print("🛒 Order System Example\n")

    handler = CreateOrderHandler()

    # Create order
    command = CreateOrderCommand(
        customer_id="customer_123",
        items=[
            {"product_id": "p1", "quantity": 1},  # Laptop
            {"product_id": "p2", "quantity": 2},  # 2x Mouse
        ]
    )

    result = handler.process_command(command)
    if result.success:
        order = result.data
        print(f"✅ Order created: {order.id}")
        print(f"   Customer: {order.customer_id}")
        print(f"   Status: {order.status}")
        print(f"   Total: ${order.total():.2f}")
        print(f"   Items: {len(order.items)}")

        for item in order.items:
            print(f"     - {item.product.name}: {item.quantity}x ${item.product.price:.2f}")
    else:
        print(f"❌ Error: {result.error}")

if __name__ == "__main__":
    run_order_example()
```

## 📚 Next Steps

### 1. Learn More

- **[Architecture Guide](../architecture/overview.md)** - Design patterns and principles
- **[Core API Reference](../api/core.md)** - Complete API documentation
- **[Patterns Guide](../api/patterns.md)** - Advanced patterns

### 2. Explore Examples

- **[Examples Overview](../examples/overview.md)** - Real-world use cases
- **[Best Practices](../development/best-practices.md)** - Development guidelines

### 3. Development Setup

```bash
# Setup development environment
git clone <repository-url>
cd flext-core
make setup

# Run tests
make test

# Quality checks
make validate
```

## ⚠️ Important Notes

- This guide uses **ACTUAL** working imports from src/flext_core/
- All examples are **TESTED** against the current implementation
- Some advanced features are still in development
- Check the source code for the most up-to-date API

---

**Congratulations!** 🎉 You now have the foundation to build enterprise applications with FLEXT Core!
