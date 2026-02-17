# Layer 2: Domain API Reference

This section covers **Layer 2 (Domain)** classes that implement Domain-Driven Design patterns and participate in dispatcher-driven CQRS flows.

> **Architecture**: Layer 2 depends only on Layers 0–1 and the runtime bridge. See the [Architecture Overview](../architecture/overview.md) for the complete hierarchy.

## Domain Models

### FlextModels — DDD Base Classes {#flextmodels}

Base classes for entities, value objects, and aggregate roots implemented with Pydantic v2.

```python
from decimal import Decimal
from typing import List
from flext_core import FlextModels, FlextResult

class Email(m.Value):
    """Immutable value object compared by value."""

    address: str


class User(FlextModels.Entity):
    """Entity with identity and invariants."""

    id: str
    name: str
    email: Email

    def model_post_init(self, __context) -> None:
        """Validate invariants after initialization."""
        if "@" not in self.email.address:
            raise ValueError("Invalid email format")


class OrderItem(m.Value):
    """Item belonging to an order."""

    product_id: str
    quantity: int
    price: Decimal


class Order(FlextModels.AggregateRoot):
    """Aggregate root enforcing order consistency."""

    customer_id: str
    items: List[OrderItem]
    total: Decimal

    def add_item(self, item: OrderItem) -> FlextResult[bool]:
        if item.quantity <= 0:
            return FlextResult[bool].fail("Quantity must be positive")
        self.items.append(item)
        self.total += item.price * item.quantity
        return FlextResult[bool].| ok(value=True)
```

### FlextService — Service Base

Base class for domain services that encapsulate business logic, domain events, and CQRS handlers.

```python
from flext_core import FlextDispatcher, FlextRegistry, FlextResult, FlextService

class CreateUser(FlextService.Command):
    """Command payload for creating a user."""

    email: str


class UserCreated(FlextService.Event):
    """Domain event emitted after a user is created."""

    user_id: str


class UserService(FlextService):
    """Domain service with command and event handlers."""

    def handle_create_user(self, command: CreateUser) -> FlextResult[str]:
        if not command.email:
            return FlextResult[str].fail("Email required")

        user_id = f"user-{command.email}"  # Persist in a repository in real scenarios
        self.add_domain_event(UserCreated(user_id=user_id))
        return FlextResult[str].ok(user_id)

    def handle_user_created(self, event: UserCreated) -> FlextResult[bool]:
        # Perform read-side updates or notifications
        return FlextResult[bool].| ok(value=True)


registry = FlextRegistry()
service = UserService()
registry.register_command(CreateUser, service.handle_create_user)
registry.register_event(UserCreated, service.handle_user_created)

result = FlextDispatcher(registry=registry).dispatch(CreateUser(email="user@example.com"))
if result.is_success:
    print(f"Created user: {result.value}")
```

Domain models and services remain independent of frameworks while integrating cleanly with the dispatcher and application orchestration.
