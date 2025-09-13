# Architecture Overview

## Architecture Overview

> **Note**: For verified implementation details, see [../ACTUAL_CAPABILITIES.md](../ACTUAL_CAPABILITIES.md)

FLEXT Core implements basic layered architecture with separation of concerns. The system provides foundation patterns for railway-oriented programming and dependency injection.

### Layer Organization

The codebase is organized into logical layers with clear boundaries:

```
Foundation → Domain → Application → Infrastructure
```

## Layer Organization

### 1. Core/Foundation Layer (Innermost)

The fundamental patterns that everything else builds upon:

```python
# src/flext_core/
├── result.py           # FlextResult[T] - Railway-oriented programming
├── container.py        # FlextContainer - Dependency injection
├── constants.py        # Core enums and constants
├── typings.py
└── exceptions.py       # Exception hierarchy
```

**Key Patterns:**

```python
from flext_core import FlextResult

# All operations return FlextResult for composability
def divide(a: float, b: float) -> FlextResult[float]:
    if b == 0:
        return FlextResult[None].fail("Division by zero")
    return FlextResult[None].ok(a / b)

# Chain operations without exception handling
result = (
    divide(10, 2)
    .map(lambda x: x * 2)
    .flat_map(lambda x: divide(x, 4))
)
```

### 2. Domain Layer

Business logic and domain models, independent of frameworks:

```python
# src/flext_core/
├── entities.py         # FlextModels.Entity - Entities with identity
├── value_objects.py    # FlextModels.Value - Immutable values
└── domain_services.py  # FlextDomainService - Domain operations
```

**Domain Modeling Example:**

```python
from flext_core import FlextModels
from decimal import Decimal

class Money(FlextModels.Value):
    """Value object - compared by value, immutable."""
    amount: Decimal
    currency: str

    def add(self, other: Money) -> FlextResult[Money]:
        if self.currency != other.currency:
            return FlextResult[None].fail("Currency mismatch")
        return FlextResult[None].ok(Money(
            amount=self.amount + other.amount,
            currency=self.currency
        ))

class Account(FlextModels.Entity):
    """Entity - has identity, mutable state."""
    account_number: str
    balance: Money
    owner_id: str

    def withdraw(self, amount: Money) -> FlextResult[None]:
        result = self.balance.subtract(amount)
        if result.success:
            self.balance = result.unwrap()
            self.add_domain_event("MoneyWithdrawn", {
                "account_id": self.id,
                "amount": str(amount.amount)
            })
            return FlextResult[None].ok(None)
        return result.map(lambda _: None)

class BankingContext(FlextModels.AggregateRoot):
    """Aggregate root - consistency boundary."""
    accounts: list[Account]

    def transfer(self, from_id: str, to_id: str,
                 amount: Money) -> FlextResult[None]:
        # Ensures transactional consistency
        from_account = self.find_account(from_id)
        to_account = self.find_account(to_id)

        withdraw_result = from_account.withdraw(amount)
        if withdraw_result.is_failure:
            return withdraw_result

        deposit_result = to_account.deposit(amount)
        if deposit_result.is_failure:
            # Rollback logic here
            return deposit_result

        self.add_domain_event("TransferCompleted", {
            "from": from_id,
            "to": to_id,
            "amount": str(amount.amount)
        })
        return FlextResult[None].ok(None)
```

### 3. Application Layer

Use case orchestration and application services:

```python
# src/flext_core/
├── commands.py         # Command patterns (CQRS)
├── handlers.py         # Command/query handlers
├── validation.py       # Business validation rules
└── interfaces.py       # Port interfaces
```

**CQRS Pattern Example:**

```python
from flext_core import FlextCommand, FlextMessageHandler, FlextResult

class TransferMoneyCommand(FlextCommand):
    """Command - represents intent to change state."""
    from_account: str
    to_account: str
    amount: Decimal
    currency: str

class TransferMoneyHandler(FlextMessageHandler[TransferMoneyCommand, None]):
    """Handler - executes business logic."""

    def __init__(self, repository, event_bus):
        self.repository = repository
        self.event_bus = event_bus

    def handle(self, command: TransferMoneyCommand) -> FlextResult[None]:
        # Load aggregate
        banking = self.repository.get_banking_context()

        # Execute domain logic
        money = Money(amount=command.amount, currency=command.currency)
        result = banking.transfer(
            command.from_account,
            command.to_account,
            money
        )

        # Persist changes
        if result.success:
            self.repository.save(banking)
            # Publish domain events
            for event in banking.get_uncommitted_events():
                self.event_bus.publish(event)

        return result
```

### 4. Infrastructure Layer

External concerns and framework integrations:

```python
# src/flext_core/
├── config.py           # Configuration management
├── loggings.py         # Structured logging
├── payload.py          # Event/message infrastructure
├── observability.py    # Monitoring and metrics
└── protocols.py        # External system protocols
```

**Infrastructure Example:**

```python
from flext_core import FlextConfig, FlextLogger

class DatabaseSettings(FlextConfig):
    """Configuration with environment support."""
    database_url: str
    pool_size: int = 5
    timeout: int = 30

    class Config:
        env_prefix = "DB_"

# Structured logging with correlation
logger = FlextLogger(__name__)

def connect_database(settings: DatabaseSettings) -> FlextResult[Connection]:
    logger.info("Connecting to database",
                url=settings.database_url,
                pool_size=settings.pool_size)
    try:
        conn = create_connection(settings.database_url)
        return FlextResult[None].ok(conn)
    except Exception as e:
        logger.error("Database connection failed", error=str(e))
        return FlextResult[None].fail(f"Connection failed: {e}")
```

## Architectural Patterns

### Railway-Oriented Programming

All operations return `FlextResult[T]`, enabling functional composition without exceptions:

```python
def process_order(order_data: dict) -> FlextResult[Order]:
    return (
        validate_order_data(order_data)
        .flat_map(create_order)
        .flat_map(calculate_pricing)
        .flat_map(apply_discounts)
        .flat_map(save_order)
        .map(send_confirmation_email)
    )
```

### Dependency Injection

Global container pattern for service management:

```python
from flext_core import get_flext_container

# Register services at startup
container = FlextContainer.get_global()
container.register("database", DatabaseService())
container.register("cache", CacheService())
container.register("email", EmailService())

# Inject dependencies
class OrderService:
    def __init__(self):
        self.db = container.get("database").unwrap()
        self.cache = container.get("cache").unwrap()
        self.email = container.get("email").unwrap()
```

### Domain Event Pattern

Domain events for decoupled communication:

```python
class OrderPlaced(FlextDomainEvent):
    order_id: str
    customer_id: str
    total: Decimal

class Order(FlextModels.AggregateRoot):
    def place(self) -> FlextResult[None]:
        # Business logic
        self.status = "placed"

        # Raise domain event
        self.add_domain_event(OrderPlaced(
            order_id=self.id,
            customer_id=self.customer_id,
            total=self.total
        ))

        return FlextResult[None].ok(None)
```

## Design Principles

### SOLID Principles

1. **Single Responsibility**: Each module has one reason to change
2. **Open/Closed**: Open for extension via inheritance and composition
3. **Liskov Substitution**: All FlextResult operations are substitutable
4. **Interface Segregation**: Small, focused protocols
5. **Dependency Inversion**: Depend on abstractions (protocols)

### DDD Tactical Patterns

- **Entities**: Objects with identity (`FlextModels.Entity`)
- **Value Objects**: Immutable values (`FlextModels.Value`)
- **Aggregates**: Consistency boundaries (`FlextModels.AggregateRoot`)
- **Domain Services**: Stateless operations (`FlextDomainService`)
- **Domain Events**: State change notifications

### Functional Patterns

- **Railway-Oriented**: Two-track error handling
- **Monadic Composition**: map, flat_map, map_error
- **Immutability**: Value objects and frozen configs
- **Pure Functions**: No side effects in domain logic

## Testing Architecture

### Unit Testing

Test individual components in isolation:

```python
def test_money_addition():
    money1 = Money(amount=Decimal("10.00"), currency="USD")
    money2 = Money(amount=Decimal("20.00"), currency="USD")

    result = money1.add(money2)

    assert result.success
    assert result.unwrap().amount == Decimal("30.00")

def test_money_currency_mismatch():
    money1 = Money(amount=Decimal("10.00"), currency="USD")
    money2 = Money(amount=Decimal("20.00"), currency="EUR")

    result = money1.add(money2)

    assert result.is_failure
    assert "Currency mismatch" in result.error
```

### Integration Testing

Test component interactions:

```python
def test_transfer_between_accounts(container):
    # Setup
    container.register("repository", InMemoryRepository())
    container.register("event_bus", InMemoryEventBus())

    handler = TransferMoneyHandler(
        container.get("repository").unwrap(),
        container.get("event_bus").unwrap()
    )

    # Execute
    command = TransferMoneyCommand(
        from_account="123",
        to_account="456",
        amount=Decimal("100.00"),
        currency="USD"
    )
    result = handler.handle(command)

    # Verify
    assert result.success
    events = container.get("event_bus").unwrap().get_events()
    assert any(e.type == "TransferCompleted" for e in events)
```

## Performance Considerations

### Optimization Strategies

1. **Lazy Loading**: Aggregates load entities on demand
2. **Caching**: Container caches service instances
3. **Batch Operations**: Process multiple items in single result
4. **Async Support**: Future async/await integration planned

### Memory Management

- Value objects are immutable and shareable
- Entities track changes for efficient persistence
- Events are lightweight data carriers

## Migration Path

### From Exception-Based Code

```python
# Before: Exception-based
def old_divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Division by zero")
    return a / b

# After: Railway-oriented
def new_divide(a: float, b: float) -> FlextResult[float]:
    if b == 0:
        return FlextResult[None].fail("Division by zero")
    return FlextResult[None].ok(a / b)
```

### From Procedural to Domain-Driven

```python
# Before: Procedural
def transfer_money(from_id, to_id, amount):
    from_account = db.get_account(from_id)
    to_account = db.get_account(to_id)

    if from_account.balance < amount:
        return False

    from_account.balance -= amount
    to_account.balance += amount

    db.save(from_account)
    db.save(to_account)
    return True

# After: Domain-driven
def transfer_money(command: TransferMoneyCommand) -> FlextResult[None]:
    return (
        repository.get_banking_context()
        .flat_map(lambda banking: banking.transfer(
            command.from_account,
            command.to_account,
            Money(command.amount, command.currency)
        ))
        .flat_map(lambda _: repository.save(banking))
    )
```

## Future Architecture

### Planned Enhancements

1. **Event Sourcing**: Complete event store implementation
2. **CQRS Bus**: Auto-discovery and routing
3. **Plugin System**: Dynamic module loading
4. **Async/Await**: Full async support
5. **Distributed Patterns**: Saga orchestration

### Ecosystem Growth

FLEXT Core serves as the foundation for 32+ projects, ensuring consistent patterns across:

- Data extraction (Singer taps)
- Data loading (Singer targets)
- Data transformation (DBT)
- API services (FastAPI)
- Background workers (Celery)

---

**Architecture Philosophy**: Make the right thing easy and the wrong thing hard.
