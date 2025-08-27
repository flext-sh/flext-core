# Core API Reference

Complete reference for FLEXT Core's fundamental APIs.

## FlextResult[T] - Railway-Oriented Programming

Type-safe error handling that replaces exceptions with explicit results.

### Basic Usage

```python
from flext_core import FlextResult

def divide(a: float, b: float) -> FlextResult[float]:
    """Safe division without exceptions."""
    if b == 0:
        return FlextResult[None].fail("Cannot divide by zero")
    return FlextResult[None].ok(a / b)

# Use the result
result = divide(10, 2)
if result.success:
    print(f"Result: {result.unwrap()}")  # Result: 5.0
else:
    print(f"Error: {result.error}")
```

### Chaining Operations

```python
# Railway-oriented composition
result = (
    divide(10, 2)                    # FlextResult[None].ok(5.0)
    .map(lambda x: x * 2)            # Transform: 10.0
    .flat_map(lambda x: divide(x, 4)) # Chain: 2.5
    .map_error(lambda e: f"Math error: {e}")  # Transform errors
)

# Alternative chaining
def process_number(n: float) -> FlextResult[str]:
    return (
        validate_positive(n)
        .map(lambda x: x ** 2)
        .map(lambda x: f"Result: {x}")
        .or_else(lambda _: FlextResult[None].ok("Using default"))
    )
```

### API Methods

```python
class FlextResult[T]:
    # Factory methods
    @classmethod
    def ok(cls, data: T) -> FlextResult[T]:
        """Create success result with data."""

    @classmethod
    def fail(cls, error: str) -> FlextResult[T]:
        """Create failure result with error message."""

    # Properties
    @property
    def success(self) -> bool:
        """True if result is successful."""

    @property
    def is_failure(self) -> bool:
        """True if result is failure."""

    @property
    def data(self) -> T | None:
        """Success data (None if failure)."""

    @property
    def error(self) -> str | None:
        """Error message (None if success)."""

    # Transformation methods
    def map(self, func: Callable[[T], U]) -> FlextResult[U]:
        """Transform success value, pass through errors."""

    def flat_map(self, func: Callable[[T], FlextResult[U]]) -> FlextResult[U]:
        """Chain operations that return FlextResult."""

    def map_error(self, func: Callable[[str], str]) -> FlextResult[T]:
        """Transform error message, pass through success."""

    def or_else(self, func: Callable[[str], FlextResult[T]]) -> FlextResult[T]:
        """Provide alternative on failure."""

    # Extraction methods
    def unwrap(self) -> T:
        """Extract value (raises if failure)."""

    def unwrap_or(self, default: T) -> T:
        """Extract value or return default."""

    def unwrap_or_else(self, func: Callable[[str], T]) -> T:
        """Extract value or compute default from error."""
```

## FlextContainer - Dependency Injection

Global singleton container for service management.

### Service Registration

```python
from flext_core import get_flext_container

# Get global container
container = FlextContainer.get_global()

# Register service instance
class DatabaseService:
    def connect(self) -> str:
        return "Connected"

container.register("database", DatabaseService())

# Register factory function
def create_cache():
    return CacheService(ttl=300)

container.register_factory("cache", create_cache)

# Register with parameters
container.register("config", {"api_key": "secret", "timeout": 30})
```

### Service Resolution

```python
# Get service with error handling
db_result = container.get("database")
if db_result.success:
    db = db_result.unwrap()
    print(db.connect())

# Check if service exists
if container.has("cache"):
    cache = container.get("cache").unwrap()

# Get all registered services
services = container.list_services()
print(f"Registered: {services}")  # ["database", "cache", "config"]

# Remove service
container.unregister("cache")
```

### API Methods

```python
class FlextContainer:
    # Registration
    def register(self, key: str, instance: Any) -> FlextResult[None]:
        """Register service instance."""

    def register_factory(self, key: str, factory: Callable[[], Any]) -> FlextResult[None]:
        """Register service factory."""

    # Resolution
    def get(self, key: str) -> FlextResult[Any]:
        """Get service by key."""

    def has(self, key: str) -> bool:
        """Check if service is registered."""

    # Management
    def unregister(self, key: str) -> FlextResult[None]:
        """Remove service registration."""

    def list_services(self) -> list[str]:
        """Get all registered service keys."""

    def clear(self) -> None:
        """Remove all registrations."""

# Global container access
def FlextContainer.get_global() -> FlextContainer:
    """Get global singleton container."""
```

## FlextConfig - Configuration Management

Environment-aware configuration with Pydantic validation.

### Basic Configuration

```python
from flext_core import FlextConfig

class AppSettings(FlextConfig):
    """Application configuration."""
    app_name: str = "MyApp"
    debug: bool = False
    database_url: str = "sqlite:///app.db"
    api_port: int = 8000
    api_key: str | None = None

    class Config:
        env_prefix = "APP_"  # Read from APP_DEBUG, APP_DATABASE_URL, etc.
        env_file = ".env"    # Load from .env file
        case_sensitive = False

# Automatic environment loading
settings = AppSettings()
print(f"Debug: {settings.debug}")
print(f"Database: {settings.database_url}")
```

### Advanced Configuration

```python
from pydantic import Field, validator

class DatabaseSettings(FlextConfig):
    """Database configuration with validation."""
    host: str = "localhost"
    port: int = Field(5432, ge=1, le=65535)
    username: str
    password: str = Field(..., min_length=8)
    database: str = "myapp"
    pool_size: int = Field(5, ge=1, le=100)

    @validator("username")
    def validate_username(cls, v):
        if not v.isalnum():
            raise ValueError("Username must be alphanumeric")
        return v

    @property
    def connection_url(self) -> str:
        """Build connection URL."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    class Config:
        env_prefix = "DB_"
        env_file = ".env"
        env_file_encoding = "utf-8"
```

## Domain Patterns

### FlextEntity - Domain Entities

Entities with identity and business logic.

```python
from flext_core import FlextEntity

class User(FlextEntity):
    """User entity with business logic."""
    username: str
    email: str
    is_active: bool = True
    created_at: datetime

    def activate(self) -> FlextResult[None]:
        """Activate user account."""
        if self.is_active:
            return FlextResult[None].fail("User already active")

        self.is_active = True
        self.add_domain_event("UserActivated", {
            "user_id": self.id,
            "activated_at": datetime.now().isoformat()
        })
        return FlextResult[None].ok(None)

    def change_email(self, new_email: str) -> FlextResult[None]:
        """Change user email with validation."""
        if "@" not in new_email:
            return FlextResult[None].fail("Invalid email format")

        old_email = self.email
        self.email = new_email

        self.add_domain_event("EmailChanged", {
            "user_id": self.id,
            "old_email": old_email,
            "new_email": new_email
        })
        return FlextResult[None].ok(None)
```

### FlextValue - Immutable Values

Value objects for domain concepts.

```python
from flext_core import FlextValue
from decimal import Decimal

class Money(FlextValue):
    """Immutable money value object."""
    amount: Decimal
    currency: str

    def add(self, other: 'Money') -> FlextResult['Money']:
        """Add money amounts with same currency."""
        if self.currency != other.currency:
            return FlextResult[None].fail(f"Currency mismatch: {self.currency} != {other.currency}")

        return FlextResult[None].ok(Money(
            amount=self.amount + other.amount,
            currency=self.currency
        ))

    def multiply(self, factor: Decimal) -> 'Money':
        """Multiply money amount."""
        return Money(
            amount=self.amount * factor,
            currency=self.currency
        )

    def __str__(self) -> str:
        return f"{self.currency} {self.amount:.2f}"

class Email(FlextValue):
    """Email value object with validation."""
    address: str

    def __init__(self, **data):
        address = data.get('address', '')
        if "@" not in address or "." not in address.split("@")[1]:
            raise ValueError(f"Invalid email: {address}")
        data['address'] = address.lower()  # Normalize
        super().__init__(**data)

    @property
    def domain(self) -> str:
        """Get email domain."""
        return self.address.split("@")[1]

    @property
    def username(self) -> str:
        """Get email username."""
        return self.address.split("@")[0]
```

### FlextAggregateRoot - Consistency Boundaries

Aggregates that maintain consistency.

```python
from flext_core import FlextAggregateRoot

class ShoppingCart(FlextAggregateRoot):
    """Shopping cart aggregate."""
    customer_id: str
    items: list[CartItem] = []
    status: str = "active"

    def add_item(self, product_id: str, quantity: int,
                 price: Money) -> FlextResult[None]:
        """Add item to cart with validation."""
        if self.status != "active":
            return FlextResult[None].fail("Cannot modify inactive cart")

        if quantity <= 0:
            return FlextResult[None].fail("Quantity must be positive")

        # Check for existing item
        for item in self.items:
            if item.product_id == product_id:
                item.quantity += quantity
                self.add_domain_event("ItemQuantityUpdated", {
                    "cart_id": self.id,
                    "product_id": product_id,
                    "new_quantity": item.quantity
                })
                return FlextResult[None].ok(None)

        # Add new item
        self.items.append(CartItem(
            product_id=product_id,
            quantity=quantity,
            price=price
        ))

        self.add_domain_event("ItemAddedToCart", {
            "cart_id": self.id,
            "product_id": product_id,
            "quantity": quantity,
            "price": str(price.amount)
        })

        return FlextResult[None].ok(None)

    def checkout(self) -> FlextResult[Money]:
        """Checkout cart and calculate total."""
        if not self.items:
            return FlextResult[None].fail("Cart is empty")

        if self.status != "active":
            return FlextResult[None].fail("Cart already checked out")

        total = Money(amount=Decimal("0"), currency="USD")
        for item in self.items:
            item_total = item.price.multiply(Decimal(item.quantity))
            add_result = total.add(item_total)
            if add_result.is_failure:
                return add_result
            total = add_result.unwrap()

        self.status = "checked_out"
        self.add_domain_event("CartCheckedOut", {
            "cart_id": self.id,
            "total": str(total.amount),
            "items_count": len(self.items)
        })

        return FlextResult[None].ok(total)
```

## Utility Functions

### Logging

```python
from flext_core import get_logger

# Get configured logger
logger = get_logger(__name__)

# Structured logging
logger.info("Processing request",
            request_id="123",
            user_id="456",
            action="create_order")

logger.error("Operation failed",
             error="Database connection lost",
             retry_count=3)
```

### ID Generation

```python
from flext_core.utilities import generate_id, generate_uuid

# Generate unique IDs
entity_id = generate_id("user")  # "user_1234567890abcdef"
request_id = generate_uuid()     # "550e8400-e29b-41d4-a716-446655440000"
```

## Error Handling Patterns

### Graceful Degradation

```python
def get_user_with_fallback(user_id: str) -> FlextResult[User]:
    """Get user with cache fallback."""
    # Try primary source
    result = database.get_user(user_id)
    if result.success:
        return result

    # Try cache
    cache_result = cache.get_user(user_id)
    if cache_result.success:
        logger.warning("Using cached user data", user_id=user_id)
        return cache_result

    # Return error
    return FlextResult[None].fail(f"User {user_id} not found")
```

### Error Aggregation

```python
def validate_order(order: Order) -> FlextResult[Order]:
    """Validate order with multiple checks."""
    errors = []

    if not order.items:
        errors.append("Order has no items")

    if order.total.amount <= 0:
        errors.append("Order total must be positive")

    if not order.customer_id:
        errors.append("Customer ID is required")

    if errors:
        return FlextResult[None].fail("; ".join(errors))

    return FlextResult[None].ok(order)
```

## Best Practices

1. **Always use FlextResult** for operations that can fail
2. **Check success before unwrap** to avoid exceptions
3. **Use map/flat_map** for composing operations
4. **Register services early** in application lifecycle
5. **Validate configuration** at startup
6. **Keep domain logic** in entities and value objects
7. **Use aggregates** for consistency boundaries
8. **Raise domain events** for important state changes

---

For more examples, see the [Examples Guide](../examples/overview.md).
