# Layer 2: Domain API Reference

<!-- TOC START -->

- [Domain Models](#domain-models)
  - [FlextModels — DDD Base Classes {#flextmodels}](#flextmodels-ddd-base-classes-flextmodels)
  - [FlextService — Service Base](#flextservice-service-base)
- [Verification Commands](#verification-commands)

<!-- TOC END -->

Layer 2 covers Domain-Driven Design building blocks used by command/query handlers and service orchestration.

Canonical references:

- `../architecture/overview.md`
- `../architecture/cqrs.md`
- `../../README.md`

Layer rule: domain code depends only on lower layers and shared runtime contracts.

## Domain Models

### FlextModels — DDD Base Classes {#flextmodels}

Base classes for entities, value objects, and aggregate roots implemented with Pydantic v2.

```python
from decimal import Decimal
from flext_core import m, r

class Email(m.Value):
    """Immutable value object compared by value."""

    address: str


class User(m.Entity):
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


class Order(m.AggregateRoot):
    """Aggregate root enforcing order consistency."""

    customer_id: str
    items: list[OrderItem]
    total: Decimal

    def add_item(self, item: OrderItem) -> r[bool]:
        if item.quantity <= 0:
            return r[bool].fail("Quantity must be positive")
        self.items.append(item)
        self.total += item.price * item.quantity
        return r[bool].ok(True)
```

### FlextService — Service Base

Base class for domain services that encapsulate business logic, domain events, and CQRS handlers.

```python
from flext_core import FlextDispatcher, FlextRegistry, FlextService, r

class CreateUser(FlextService.Command):
    """Command payload for creating a user."""

    email: str


class UserCreated(FlextService.Event):
    """Domain event emitted after a user is created."""

    user_id: str


class UserService(FlextService[str]):
    """Domain service with command and event handlers."""

    def handle_create_user(self, command: CreateUser) -> r[str]:
        if not command.email:
            return r[str].fail("Email required")

        user_id = f"user-{command.email}"  # Persist in a repository in real scenarios
        self.add_domain_event(UserCreated(user_id=user_id))
        return r[str].ok(user_id)

    def handle_user_created(self, event: UserCreated) -> r[bool]:
        # Perform read-side updates or notifications
        return r[bool].ok(True)


registry = FlextRegistry()
service = UserService()
registry.register_command(CreateUser, service.handle_create_user)
registry.register_event(UserCreated, service.handle_user_created)

result = FlextDispatcher(registry=registry).dispatch(CreateUser(email="user@example.com"))
if result.is_success:
    print(f"Created user: {result.value}")
```

Domain models and services remain independent of frameworks while integrating cleanly with the dispatcher and application orchestration.

## Verification Commands

Run from `flext-core/`:

```bash
make lint
make type-check
make test-fast
```
