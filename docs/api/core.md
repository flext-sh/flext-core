# Core API - FLEXT Core

Complete reference for the fundamental APIs of FLEXT Core

## 🎯 Overview

The Core API provides the foundational components to build robust applications. All components are type-safe, testable, and follow clean architecture patterns.

## 📦 Primary Imports

```python
# Recommended modern imports
from flext_core import (
    FlextResult,           # Error handling type-safe
    FlextContainer,        # Dependency injection
    FlextSettings,         # Configuration management
    FlextEntity,           # Domain entities
    FlextValueObject,      # Immutable value objects
    FlextAggregateRoot,    # Domain aggregates
)

# Legacy imports (compat re-exports still available via __init__)
```

## 🎭 FlextResult[T] - Type-Safe Error Handling

Replaces exceptions with explicit, type-safe results.

### Creating Results

```python
from flext_core import FlextResult

# Success result
def fetch_user(user_id: str) -> FlextResult[dict]:
    user_data = {"id": user_id, "name": "John"}
    return FlextResult.ok(user_data)

# Error result
def validate_email(email: str) -> FlextResult[str]:
    if "@" not in email:
        return FlextResult.fail("Email must contain @")
    return FlextResult.ok(email)

# Factory methods
success_result = FlextResult.ok("success data")
error_result = FlextResult.fail("error message")
```

### Checking Status

```python
result = fetch_user("123")

# Success/failure check
if result.success:
    print(f"Data: {result.data}")
else:
    print(f"Error: {result.error}")

# Boolean properties
assert result.success is True
assert result.is_failure is False
```

### Data Handling

```python
# Safe data access
result = fetch_user("123")

if result.success:
    # result.data is guaranteed non-None when success=True
    user_data = result.data
    print(f"User: {user_data['name']}")

if result.is_failure:
    # result.error is guaranteed non-None when is_failure=True
    error_msg = result.error
    print(f"Error: {error_msg}")
```

### Composition and Chaining

```python
def create_and_save_user(name: str, email: str) -> FlextResult[str]:
    # Validation
    email_result = validate_email(email)
    if email_result.is_failure:
        return email_result  # Propagate error

    # Creation
    user_result = create_user(name, email_result.data)
    if user_result.is_failure:
        return FlextResult.fail(f"Creation failed: {user_result.error}")

    # Save
    save_result = save_user(user_result.data)
    if save_result.is_failure:
        return FlextResult.fail(f"Save failed: {save_result.error}")

    return FlextResult.ok("User created successfully")
```

### Complete API

```python
class FlextResult[T]:
    """Type-safe result container."""

    # Factory methods
    @classmethod
    def ok(cls, data: T) -> FlextResult[T]:
        """Create success result."""

    @classmethod
    def fail(cls, error: str) -> FlextResult[T]:
        """Create failure result."""

    # Properties
    @property
    def success(self) -> bool:
        """True if result represents success."""

    @property
    def is_failure(self) -> bool:
        """True if result represents failure."""

    @property
    def data(self) -> T | None:
        """Success data (None if failure)."""

    @property
    def error(self) -> str | None:
        """Error message (None if success)."""

    # Methods
    def get_data_or(self, default: T) -> T:
        """Get data or return default if failure."""

    def get_error_or(self, default: str) -> str:
        """Get error or return default if success."""
```

## 🏗️ FlextContainer - Dependency Injection

**Type-safe, enterprise-grade dependency injection container.**

### Service Registration

```python
from flext_core import FlextContainer

# Create container
container = FlextContainer()

# Basic registration
result = container.register("database", DatabaseService())
if result.success:
    print("Service registered successfully")

# Factory registration
def create_email_service() -> EmailService:
    return EmailService(smtp_host="localhost", port=587)

container.register_factory("email", create_email_service)

# Singleton registration (default)
container.register("cache", RedisCache(), singleton=True)
```

### Dependency Resolution

```python
# Get registered service
db_result = container.get("database")
if db_result.success:
    database = db_result.data  # type: DatabaseService
    users = database.fetch_users()

# Resolution with type hint
email_service_result = container.get_typed("email", EmailService)
if email_service_result.success:
    service = email_service_result.data
    service.send_email("test@example.com", "Hello")
```

### Automatic Injection

```python
# Class with dependencies
class UserService:
    def __init__(self, database: DatabaseService, email: EmailService):
        self.database = database
        self.email = email

    def create_user(self, name: str, email_addr: str) -> FlextResult[str]:
        # Logic using dependencies
        user = {"name": name, "email": email_addr}
        save_result = self.database.save_user(user)

        if save_result.success:
            self.email.send_welcome_email(email_addr)
            return FlextResult.ok("User created")
        else:
            return FlextResult.fail("Failed to save user")

# Explicit instance registration (advanced auto-wiring not available)
user_service = UserService(database, email_service)
container.register("user_service", user_service)
```

### Configuration and Lifecycle

```python
# Container configuration
container.configure(
    auto_wire=True,           # Auto-resolve dependencies
    strict_mode=True,         # Fail on missing dependencies
    lazy_loading=False,       # Eager instantiation
    cache_instances=True      # Cache resolved instances
)

# Lifecycle management
result = container.start()   # Initialize all services
if result.success:
    print("Container started")

# Cleanup
container.stop()            # Cleanup resources
container.clear()           # Remove all registrations
```

### Complete API

```python
class FlextContainer:
    """Dependency injection container."""

    def register(
        self,
        key: str,
        instance: object,
        singleton: bool = True
    ) -> FlextResult[None]:
        """Register service instance."""

    def register_factory(
        self,
        key: str,
        factory: Callable[[], Any]
    ) -> FlextResult[None]:
        """Register service factory."""

    def get(self, key: str) -> FlextResult[Any]:
        """Resolve service by key."""

    def get_typed(self, key: str, expected_type: type[T]) -> FlextResult[T]:
        """Resolve service with type checking."""

    def has(self, key: str) -> bool:
        """Check if service is registered."""

    def remove(self, key: str) -> FlextResult[None]:
        """Remove service registration."""

    def get_all_keys(self) -> list[str]:
        """Get all registered service keys."""

    def configure(self, **options) -> None:
        """Configure container behavior."""

    def start(self) -> FlextResult[None]:
        """Initialize container and services."""

    def stop(self) -> FlextResult[None]:
        """Stop container and cleanup."""

    def clear(self) -> None:
        """Clear all registrations."""
```

## ⚙️ FlextSettings - Configuration Management

**Centralized configuration management with Pydantic v2.**

### Basic Configuration

```python
from flext_core import FlextSettings

class AppSettings(FlextSettings):
    debug: bool = False
    environment: str = "production"
    log_level: str = "INFO"
    database_url: str | None = None

    class Config:
        env_prefix = "APP_"
        case_sensitive = False

# Instance
settings = AppSettings()
```

### Environment Variables

```bash
export APP_DEBUG=true
export APP_LOG_LEVEL=DEBUG
export APP_DATABASE_URL="postgresql://localhost/flext"
```

```python
settings = AppSettings()
print(settings.debug)        # True
print(settings.log_level)    # "DEBUG"
print(settings.database_url) # Value from env
```

### Configuration Validation

```python
from pydantic import Field, ValidationError

class StrictSettings(FlextSettings):
    max_connections: int = Field(10, ge=1, le=1000)

try:
    StrictSettings(max_connections=-1)
except ValidationError as e:
    print(f"Configuration error: {e}")
```

## 🏛️ Domain Layer - Entities and Value Objects

### FlextEntity[TId] - Domain Entities (models API)

```python
from flext_core.models import FlextEntity

# ID types
UserId = NewType('UserId', str)

class User(FlextEntity[UserId]):
    """Domain entity with identity."""

    def __init__(self, user_id: UserId, name: str, email: str):
        super().__init__(user_id)
        self._name = name
        self._email = email
        self._created_at = datetime.now()

    @property
    def name(self) -> str:
        return self._name

    @property
    def email(self) -> str:
        return self._email

    def change_name(self, new_name: str) -> FlextResult[None]:
        """Business logic for name change."""
        if not new_name.strip():
            return FlextResult.fail("Name cannot be empty")

        self._name = new_name
        return FlextResult.ok(None)

    def __str__(self) -> str:
        return f"User(id={self.id}, name={self.name})"

# Usage
user_id = UserId("user_123")
user = User(user_id, "John Smith", "john@example.com")

# Identity comparison
other_user = User(user_id, "John Jones", "john2@example.com")
assert user == other_user  # Same ID = same entity
```

### FlextValueObject - Immutable Values

```python
from flext_core import FlextValueObject

class Email(FlextValueObject):
    """Immutable email value object."""

    def __init__(self, value: str):
        if not self._is_valid_email(value):
            raise ValueError(f"Invalid email: {value}")
        self._value = value.lower()

    @property
    def value(self) -> str:
        return self._value

    @property
    def domain(self) -> str:
        return self._value.split("@")[1]

    def _is_valid_email(self, email: str) -> bool:
        return "@" in email and "." in email.split("@")[1]

    def __str__(self) -> str:
        return self._value

class Money(FlextValueObject):
    """Money value object with currency."""

    def __init__(self, amount: float, currency: str = "BRL"):
        if amount < 0:
            raise ValueError("Amount cannot be negative")
        self._amount = amount
        self._currency = currency

    @property
    def amount(self) -> float:
        return self._amount

    @property
    def currency(self) -> str:
        return self._currency

    def add(self, other: 'Money') -> 'Money':
        if self._currency != other._currency:
            raise ValueError("Cannot add different currencies")
        return Money(self._amount + other._amount, self._currency)

    def __str__(self) -> str:
        return f"{self._amount:.2f} {self._currency}"

# Usage
email = Email("joao@EXAMPLE.com")  # Normalized to lowercase
print(email.value)     # "joao@example.com"
print(email.domain)    # "example.com"

money1 = Money(100.0, "BRL")
money2 = Money(50.0, "BRL")
total = money1.add(money2)  # Money(150.0, "BRL")
```

### FlextAggregateRoot - Domain Aggregates (models API)

```python
from flext_core import FlextAggregateRoot

class Order(FlextAggregateRoot[OrderId]):
    """Order aggregate root."""

    def __init__(self, order_id: OrderId, customer_id: CustomerId):
        super().__init__(order_id)
        self._customer_id = customer_id
        self._items: list[OrderItem] = []
        self._status = OrderStatus.PENDING
        self._total = Money(0.0)

    def add_item(self, product_id: ProductId, quantity: int, price: Money) -> FlextResult[None]:
        """Add item to order with business rules."""
        if self._status != OrderStatus.PENDING:
            return FlextResult.fail("Cannot modify confirmed order")

        if quantity <= 0:
            return FlextResult.fail("Quantity must be positive")

        item = OrderItem(product_id, quantity, price)
        self._items.append(item)
        self._recalculate_total()

        # Domain event
        self._add_domain_event(ItemAddedEvent(self.id, product_id, quantity))

        return FlextResult.ok(None)

    def confirm(self) -> FlextResult[None]:
        """Confirm order with business invariants."""
        if not self._items:
            return FlextResult.fail("Cannot confirm empty order")

        if self._total.amount == 0:
            return FlextResult.fail("Order total cannot be zero")

        self._status = OrderStatus.CONFIRMED

        # Domain event
        self._add_domain_event(OrderConfirmedEvent(self.id, self._total))

        return FlextResult.ok(None)

    def _recalculate_total(self) -> None:
        """Private method to maintain invariants."""
        total_amount = sum(item.total.amount for item in self._items)
        self._total = Money(total_amount)

# Usage
order = Order(OrderId("order_123"), CustomerId("customer_456"))

# Business operations
result = order.add_item(ProductId("prod_1"), 2, Money(50.0))
if result.success:
    print("Item added successfully")

confirm_result = order.confirm()
if confirm_result.success:
    print("Order confirmed")
```

## 📝 Best Practices

### 1. Error Handling

```python
# ✅ Always use FlextResult for operations that can fail
def save_user(user: User) -> FlextResult[None]:
    try:
        # Save operation
        return FlextResult.ok(None)
    except DatabaseError as e:
        return FlextResult.fail(f"Database error: {e}")

# ✅ Check results before using data
result = fetch_user("123")
if result.success:
    process_user(result.data)  # Safe to use
else:
    log_error(result.error)
```

### 2. Dependency Injection

```python
# ✅ Register dependencies at startup
def setup_container() -> FlextContainer:
    container = FlextContainer()

    # Infrastructure
    container.register("database", DatabaseService())
    container.register("cache", RedisCache())

    # Application services
    container.register("user_service", UserService)

    return container

# ✅ Resolve dependencies when needed
container = setup_container()
user_service = container.get("user_service").data
```

### 3. Configuration

```python
# ✅ Use environment-specific settings
settings = FlextCoreSettings()

if settings.debug:
    logging.getLogger().setLevel(logging.DEBUG)

# ✅ Validate configuration early
try:
    settings = FlextCoreSettings()
except ValidationError as e:
    print(f"Invalid configuration: {e}")
    exit(1)
```

### 4. Domain Modeling

```python
# ✅ Use value objects for complex values
class UserEmail(FlextValueObject):
    def __init__(self, value: str):
        # Validation in constructor
        pass

# ✅ Keep business logic in entities
class User(FlextEntity[UserId]):
    def change_email(self, new_email: UserEmail) -> FlextResult[None]:
        # Business rules here
        pass

# ✅ Use aggregates for consistency boundaries
class Order(FlextAggregateRoot[OrderId]):
    def add_item(self, item: OrderItem) -> FlextResult[None]:
        # Maintain order invariants
        pass
```

## 🔗 Compatibility and Migration

### FlextResult migration

```python
# Old (deprecated)
from flext_core import FlextResult

def old_function() -> FlextResult[str]:
    return FlextResult.success("data")

# New (recommended)
from flext_core import FlextResult

def new_function() -> FlextResult[str]:
    return FlextResult.ok("data")
```

### Migration from DIContainer to FlextContainer

```python
# Old (deprecated)
from flext_core import DIContainer

container = DIContainer()
container.set("service", service)
service = container.get("service")

# New (recommended)
from flext_core import FlextContainer

container = FlextContainer()
container.register("service", service)
service_result = container.get("service")
if service_result.success:
    service = service_result.data
```

---

This Core API provides all fundamental components needed to build robust, type-safe enterprise applications with FLEXT Core.
