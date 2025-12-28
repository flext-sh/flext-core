# QUICK START

Get started with FLEXT-Core in 5 minutes. This guide covers the essentials; see the full guides for deeper coverage and rationale.

## Installation

```bash
# Add to your project
poetry add flext-core

# Or with pip
pip install flext-core
```

**Requirements**: Python 3.13+

## Core Concepts (2 minutes)

FLEXT-Core provides three foundational patterns:

### 1. Railway-Oriented Programming (FlextResult[T])

Instead of exceptions, operations return either success or failure:

```python
from flext_core import FlextResult

def validate_email(email: str) -> FlextResult[str]:
    """Returns success or failure, never raises exceptions."""
    if "@" not in email:
        return FlextResult[str].fail("Invalid email")
    return FlextResult[str].ok(email)

# Safe value extraction
result = validate_email("user@example.com")
if result.is_success:
    email = result.unwrap()
    print(f"Valid: {email}")
else:
    print(f"Error: {result.error}")
```

### 2. Dependency Injection (FlextContainer)

Centralized service management with explicit lifecycles:

```python
from flext_core import FlextContainer, FlextLogger

# Register services
container = FlextContainer.get_global()
logger = FlextLogger(__name__)
container.register("logger", logger, singleton=True)

# Use services
logger_result = container.get("logger")
if logger_result.is_success:
    logger = logger_result.unwrap()
    logger.info("Application started")
```

### 3. Domain-Driven Design (FlextModels)

Model your business domain with explicit boundaries:

```python
from flext_core import FlextModels

class Email(m.Value):
    """Immutable value object compared by value."""
    address: str

class User(FlextModels.Entity):
    """Entity with identity."""
    id: str
    name: str
    email: Email

# Use your models
user = User(id="123", name="Alice", email=Email(address="alice@example.com"))
print(f"Created: {user.name}")
```

## Common Use Cases (3 minutes)

### Use Case 1: Validation Pipeline

Chain validations together:

```python
from flext_core import FlextResult

def validate_password(password: str) -> FlextResult[str]:
    if len(password) < 8:
        return FlextResult[str].fail("Password too short")
    return FlextResult[str].ok(password)

def validate_username(username: str) -> FlextResult[str]:
    if len(username) < 3:
        return FlextResult[str].fail("Username too short")
    return FlextResult[str].ok(username)

# Chain validations (railway pattern)
def register_user(username: str, password: str) -> FlextResult[dict]:
    return (
        validate_username(username)
        .flat_map(lambda u: validate_password(password).map(lambda p: {"username": u, "password": p}))
    )

# Test it
result = register_user("alice", "SecurePass123")
if result.is_success:
    data = result.unwrap()
    print(f"Registered: {data['username']}")
else:
    print(f"Registration failed: {result.error}")
```

### Use Case 2: Service with Dependency Injection

```python
from flext_core import FlextService, FlextResult, FlextContainer

class EmailService(FlextService):
    """Example service."""

    def send_welcome_email(self, email: str) -> FlextResult[str]:
        """Send welcome email."""
        if not email:
            return FlextResult[str].fail("Email required")
        # Send email logic here
        return FlextResult[str].ok(f"Email sent to {email}")

# Register and use
container = FlextContainer.get_global()
email_service = EmailService()
container.register("email_service", email_service, singleton=True)

# Retrieve and use
service_result = container.get("email_service")
if service_result.is_success:
    service = service_result.unwrap()
    send_result = service.send_welcome_email("user@example.com")
    if send_result.is_success:
        print(send_result.unwrap())
```

### Use Case 3: Domain Models with Validation

```python
from pydantic import Field
from flext_core import FlextModels, FlextService, FlextResult

class OrderItem(m.Value):
    """Immutable order item."""
    product_id: str
    quantity: int = Field(ge=1)  # >= 1
    price: float = Field(gt=0)   # > 0

class Order(FlextModels.Entity):
    """Order with identity."""
    id: str
    items: list[OrderItem]
    customer_id: str

class OrderService(FlextService):
    """Service with business logic."""

    def create_order(self, customer_id: str, items: list[dict]) -> FlextResult[Order]:
        """Create order with validation."""
        if not customer_id:
            return FlextResult[Order].fail("Customer ID required")

        if not items:
            return FlextResult[Order].fail("At least one item required")

        # Create order
        order_items = [OrderItem(**item) for item in items]
        order = Order(id="ORD-001", items=order_items, customer_id=customer_id)

        return FlextResult[Order].ok(order)

# Use it
service = OrderService()
result = service.create_order(
    customer_id="CUST-123",
    items=[
        {"product_id": "PROD-1", "quantity": 2, "price": 29.99},
        {"product_id": "PROD-2", "quantity": 1, "price": 49.99},
    ]
)

if result.is_success:
    order = result.unwrap()
    print(f"Order created: {order.entity_id} with {len(order.items)} items")
else:
    print(f"Order failed: {result.error}")
```

### Use Case 4: Dispatcher-Driven CQRS

Route commands through the dispatcher to keep orchestration and side effects consistent:

```python
from flext_core import FlextDispatcher, FlextRegistry, FlextResult, FlextService

class CreateUser(FlextService.Command):
    """Command payload for creating users."""


class UserService(FlextService):
    """Domain service implementing the command handler."""

    def handle_create_user(self, command: CreateUser) -> FlextResult[str]:
        if not command.email:
            return FlextResult[str].fail("Email required")
        # persist user and raise domain event here
        return FlextResult[str].ok(command.email)


registry = FlextRegistry()
registry.register_command(CreateUser, UserService().handle_create_user)

dispatcher = FlextDispatcher(registry=registry)
result = dispatcher.dispatch(CreateUser(email="user@example.com"))

if result.is_success:
    print(f"Created: {result.unwrap()}")
```

## Key Patterns Cheat Sheet

### Pattern 1: Success/Failure Handling

```python
# Return FlextResult instead of raising exceptions
def operation() -> FlextResult[str]:
    if something_wrong:
        return FlextResult[str].fail("Error message")
    return FlextResult[str].ok("Success value")

# Check result
if result.is_success:
    value = result.unwrap()
else:
    error = result.error
```

### Pattern 2: Transform Values

```python
# Use map() to transform success values
result = (
    FlextResult[int].ok(10)
    .map(lambda x: x * 2)  # Transform to 20
    .map(lambda x: f"Result: {x}")  # Transform to "Result: 20"
)
```

### Pattern 3: Chain Operations

```python
# Use flat_map() to chain operations that return FlextResult
result = (
    get_user(user_id)  # Returns FlextResult[User]
    .flat_map(lambda user: update_profile(user))  # Returns FlextResult[User]
    .flat_map(lambda user: send_confirmation(user))  # Returns FlextResult[str]
)
```

### Pattern 4: Error Recovery

```python
# Use map_error() to handle errors
result = (
    risky_operation()
    .map_error(lambda err: f"Failed: {err}")  # Transform error
)

# Or provide fallback
result = operation().unwrap_or("default value")
```

### Pattern 5: Access Both APIs

```python
# Both .data and .value work (backward compatibility)
result = FlextResult[str].ok("test")
assert result.value == result.data == "test"
```

## Testing Example

```python
import pytest
from flext_core import FlextResult

def test_validation_success():
    """Test successful validation."""
    result = validate_email("user@example.com")
    assert result.is_success
    assert result.unwrap() == "user@example.com"

def test_validation_failure():
    """Test failed validation."""
    result = validate_email("invalid")
    assert not result.is_success
    assert "Invalid" in result.error

def test_chained_operations():
    """Test railway pattern chaining."""
    result = (
        FlextResult[int].ok(10)
        .map(lambda x: x * 2)
        .map(lambda x: x + 5)
    )
    assert result.is_success
    assert result.unwrap() == 25
```

## Next Steps

- **Configuration**: [Configuration Guide](guides/configuration.md) - Learn FlextSettings patterns
- **Error Handling**: [Error Handling Guide](guides/error-handling.md) - Deep dive into railway patterns
- **Testing**: [Testing Guide](guides/testing.md) - Testing with pytest and fixtures
- **Architecture**: [Clean Architecture](architecture/clean-architecture.md) - 5-layer patterns
- **Patterns**: [Architecture Patterns](architecture/patterns.md) - 9 design patterns
- **API Reference**: [API Documentation](api-reference/) - Complete API reference
- **Troubleshooting**: [Troubleshooting Guide](guides/troubleshooting.md) - Common issues and fixes

## Common Questions

**Q: Do I have to use FlextResult everywhere?**
A: For business logic, yes. Railway pattern prevents error handling bugs. For external APIs, conversion at boundaries is fine.

**Q: Can I use exceptions?**
A: In infrastructure code (I/O, database), yes. But convert to FlextResult at domain layer boundaries.

**Q: Is FlextContainer a service locator anti-pattern?**
A: No - it's the foundation for dependency injection. Dependency injection is preferred, but container can bootstrap services and provide global access.

**Q: What's the difference between .data and .value?**
A: They're identical - both access the success value. `.data` is legacy, `.value` is current. Both work.

**Q: How do I handle async operations?**
A: FlextResult works with async/await normally:

```python
async def get_user_async(user_id: str) -> FlextResult[User]:
    if not user_id:
        return FlextResult[User].fail("User ID required")
    user = await fetch_from_database(user_id)
    return FlextResult[User].ok(user)

# Use it
result = await get_user_async("123")
if result.is_success:
    user = result.unwrap()
```

**Q: Can I create custom exception types?**
A: Yes, but prefer FlextResult for business errors. Custom exceptions for framework/infrastructure errors are fine.

**Q: Where do I put my code?**
A: Follow clean architecture layers:

- **Domain**: Business logic (models, services, validation)
- **Application**: Use cases (commands, queries, handlers)
- **Infrastructure**: External dependencies (databases, APIs, config)

## Command Reference

```bash
# Run tests
PYTHONPATH=src poetry run pytest tests/unit/ -v

# Check types
PYTHONPATH=src poetry run pyrefly check src/

# Lint code
ruff check .

# Format code
ruff format .

# Full validation (recommended before commit)
make validate
```

## Quick Reference

| Task                    | Code                                   |
| ----------------------- | -------------------------------------- |
| **Return Success**      | `FlextResult[T].ok(value)`             |
| **Return Error**        | `FlextResult[T].fail("error message")` |
| **Check Success**       | `result.is_success`                    |
| **Extract Value**       | `result.unwrap()`                      |
| **Get Error**           | `result.error`                         |
| **Transform Value**     | `result.map(lambda x: x * 2)`          |
| **Chain Operations**    | `result.flat_map(next_operation)`      |
| **Register Service**    | `container.register("name", service)`  |
| **Get Service**         | `container.get("name")`                |
| **Create Entity**       | `User(id="1", name="Alice")`           |
| **Create Value Object** | `Email(address="alice@example.com")`   |

## Getting Help

- **Documentation**: See guides/ and api-reference/
- **Troubleshooting**: See [guides/troubleshooting.md](guides/troubleshooting.md)
- **Examples**: Check examples/ directory
- **Issues**: Report bugs on GitHub

---

**Ready to dive deeper?** Start with the [Configuration Guide](guides/configuration.md) to learn how to configure FLEXT-Core for your application.
