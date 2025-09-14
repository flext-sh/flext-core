# API Reference

Complete API documentation for FLEXT-Core foundation library.

## Core Imports

All FLEXT-Core functionality is available through root-level imports:

```python
from flext_core import (
    # Core patterns
    FlextResult,           # Railway-oriented programming
    FlextContainer,        # Dependency injection
    FlextModels,           # Domain modeling

    # Configuration and services
    FlextConfig,           # Settings management
    FlextLogger,           # Structured logging
    FlextDomainService,    # Service base class

    # Utilities and types
    FlextConstants,        # System constants
    FlextTypes,            # Type definitions
    FlextUtilities,        # Helper functions
    FlextValidations,      # Validation patterns
)
```

## FlextResult[T]

Railway-oriented programming with type-safe error handling.

### Class Definition

```python
class FlextResult[T]:
    """Monadic result type for explicit error handling."""
```

### Properties

#### `.success: bool`
True if operation succeeded, False if failed.

```python
result = FlextResult[str].ok("success")
assert result.success == True

failure = FlextResult[str].fail("error")
assert failure.success == False
```

#### `.is_failure: bool`
True if operation failed, False if succeeded.

```python
result = FlextResult[str].ok("success")
assert result.is_failure == False
```

#### `.value: T`
Access the success value (new API).

```python
result = FlextResult[str].ok("hello")
if result.success:
    value = result.value  # "hello"
```

#### `.data: T`
Access the success value (legacy compatibility).

```python
result = FlextResult[str].ok("hello")
if result.success:
    value = result.data  # "hello" - maintains compatibility
```

#### `.error: str | None`
Access the error message if operation failed.

```python
result = FlextResult[str].fail("Something went wrong")
if result.is_failure:
    error_msg = result.error  # "Something went wrong"
```

### Static Methods

#### `.ok(value: T) -> FlextResult[T]`
Create a successful result.

```python
success = FlextResult[int].ok(42)
assert success.success
assert success.unwrap() == 42
```

#### `.fail(error: str, error_code: str | None = None) -> FlextResult[T]`
Create a failed result.

```python
failure = FlextResult[int].fail("Invalid input", error_code="VALIDATION_ERROR")
assert failure.is_failure
assert failure.error == "Invalid input"
```

### Instance Methods

#### `.unwrap() -> T`
Extract value after success check. Raises ValueError if called on failure.

```python
result = FlextResult[str].ok("success")
if result.success:
    value = result.unwrap()  # Safe extraction
    print(value)  # "success"

# Unsafe - will raise ValueError
failure = FlextResult[str].fail("error")
# failure.unwrap()  # ValueError: Cannot unwrap failure: error
```

#### `.unwrap_or(default: T) -> T`
Extract value or return default if failed.

```python
success = FlextResult[str].ok("hello")
value1 = success.unwrap_or("default")  # "hello"

failure = FlextResult[str].fail("error")
value2 = failure.unwrap_or("default")  # "default"
```

#### `.expect(message: str) -> T`
Extract value or raise with custom message.

```python
result = FlextResult[str].ok("success")
value = result.expect("Must have value")  # "success"

failure = FlextResult[str].fail("error")
# failure.expect("Must work")  # ValueError: Must work
```

#### `.map(func: Callable[[T], U]) -> FlextResult[U]`
Transform success value, preserve failure.

```python
def to_upper(s: str) -> str:
    return s.upper()

success = FlextResult[str].ok("hello")
result = success.map(to_upper)
assert result.unwrap() == "HELLO"

failure = FlextResult[str].fail("error")
result = failure.map(to_upper)
assert result.is_failure  # Error preserved
```

#### `.flat_map(func: Callable[[T], FlextResult[U]]) -> FlextResult[U]`
Chain operations that return FlextResult (monadic bind).

```python
def divide(a: float, b: float) -> FlextResult[float]:
    if b == 0:
        return FlextResult.fail("Division by zero")
    return FlextResult.ok(a / b)

result = (
    FlextResult[float].ok(10.0)
    .flat_map(lambda x: divide(x, 2))  # 5.0
    .flat_map(lambda x: divide(x, 0))  # Fails here
)
assert result.is_failure
assert "Division by zero" in result.error
```

#### `.map_error(func: Callable[[str], str]) -> FlextResult[T]`
Transform error message, preserve success.

```python
def format_error(error: str) -> str:
    return f"ERROR: {error.upper()}"

failure = FlextResult[str].fail("invalid input")
result = failure.map_error(format_error)
assert result.error == "ERROR: INVALID INPUT"

success = FlextResult[str].ok("hello")
result = success.map_error(format_error)
assert result.success  # Success preserved
```

#### `.filter(predicate: Callable[[T], bool], error: str) -> FlextResult[T]`
Filter success value with predicate.

```python
def is_positive(n: int) -> bool:
    return n > 0

success = FlextResult[int].ok(5)
result = success.filter(is_positive, "Must be positive")
assert result.success

failure_case = FlextResult[int].ok(-1)
result = failure_case.filter(is_positive, "Must be positive")
assert result.is_failure
assert result.error == "Must be positive"
```

### Chaining Operations

Railway pattern allows elegant operation chaining:

```python
def process_user_data(data: dict) -> FlextResult[User]:
    """Complete data processing pipeline."""
    return (
        validate_required_fields(data)
        .flat_map(lambda d: validate_email_format(d))
        .flat_map(lambda d: validate_username_unique(d))
        .flat_map(lambda d: hash_password(d))
        .map(lambda d: create_user_object(d))
        .flat_map(lambda u: save_to_database(u))
        .map(lambda u: add_audit_trail(u))
    )

# Success path: all operations succeed
# Failure path: first failure stops the chain
```

## FlextContainer

Global dependency injection container with type safety.

### Class Definition

```python
class FlextContainer:
    """Type-safe dependency injection container."""
```

### Class Methods

#### `.get_global() -> FlextContainer`
Get the global singleton container instance.

```python
container = FlextContainer.get_global()
# Same instance across the application
container2 = FlextContainer.get_global()
assert container is container2
```

### Instance Methods

#### `.register(key: str, service: Any) -> FlextResult[None]`
Register a service with the container.

```python
container = FlextContainer.get_global()

# Register service instance
service = DatabaseService(url="postgresql://localhost/db")
result = container.register("database", service)
assert result.success

# Register factory function
def create_logger() -> Logger:
    return Logger(__name__)

result = container.register_factory("logger", create_logger)
assert result.success
```

#### `.get(key: str) -> FlextResult[Any]`
Retrieve a service from the container.

```python
# Retrieve registered service
db_result = container.get("database")
if db_result.success:
    database = db_result.unwrap()
    # Use database service

# Handle missing service
missing_result = container.get("nonexistent")
assert missing_result.is_failure
assert "not found" in missing_result.error.lower()
```

#### `.has(key: str) -> bool`
Check if a service is registered.

```python
container.register("test_service", "test_value")
assert container.has("test_service") == True
assert container.has("missing_service") == False
```

#### `.unregister(key: str) -> FlextResult[None]`
Remove a service from the container.

```python
container.register("temp_service", "temp_value")
result = container.unregister("temp_service")
assert result.success
assert not container.has("temp_service")
```

### Usage Patterns

#### Service Registration at Startup

```python
def setup_services():
    """Register all application services."""
    container = FlextContainer.get_global()

    # Database connection
    db_config = DatabaseConfig()
    database = DatabaseService(db_config.url)
    container.register("database", database)

    # Email service
    email_config = EmailConfig()
    email_service = EmailService(email_config)
    container.register("email", email_service)

    # Logger
    logger = FlextLogger(__name__)
    container.register("logger", logger)

    return container
```

#### Service Injection in Classes

```python
class UserService:
    def __init__(self):
        container = FlextContainer.get_global()

        # Inject dependencies
        self._db = container.get("database").unwrap()
        self._email = container.get("email").unwrap()
        self._logger = container.get("logger").unwrap()

    def create_user(self, data: dict) -> FlextResult[User]:
        """Create user using injected services."""
        # Use self._db, self._email, self._logger
        pass
```

## FlextModels

Domain modeling with entities, value objects, and aggregates.

### FlextModels.Entity

Base class for entities with identity and lifecycle.

```python
class User(FlextModels.Entity):
    """User entity with business logic."""
    username: str
    email: str
    created_at: datetime
    is_active: bool = True

    def activate(self) -> FlextResult[None]:
        """Activate user account."""
        if self.is_active:
            return FlextResult.fail("User already active")

        self.is_active = True
        self.add_domain_event("UserActivated", {
            "user_id": self.id,
            "activated_at": datetime.now().isoformat()
        })
        return FlextResult.ok(None)

# Usage
user = User(
    id="user_123",  # Required for entities
    username="johndoe",
    email="john@example.com",
    created_at=datetime.now()
)

result = user.activate()
if result.success:
    print("User activated")
```

#### Entity Methods

##### `.add_domain_event(event_type: str, data: dict) -> None`
Add a domain event to the entity.

```python
user = User(id="user_123", username="john", email="john@example.com")
user.add_domain_event("UserCreated", {
    "user_id": user.id,
    "username": user.username,
    "created_at": datetime.now().isoformat()
})
```

##### `.get_domain_events() -> list[dict]`
Retrieve all domain events.

```python
events = user.get_domain_events()
for event in events:
    print(f"Event: {event['event_type']}, Data: {event['data']}")
```

##### `.clear_domain_events() -> None`
Clear all domain events (typically after processing).

```python
# After publishing events to event bus
user.clear_domain_events()
```

### FlextModels.Value

Base class for immutable value objects.

```python
class Money(FlextModels.Value):
    """Immutable money value object."""
    amount: Decimal
    currency: str

    def add(self, other: 'Money') -> FlextResult['Money']:
        """Add money with currency validation."""
        if self.currency != other.currency:
            return FlextResult.fail(f"Currency mismatch: {self.currency} vs {other.currency}")

        return FlextResult.ok(Money(
            amount=self.amount + other.amount,
            currency=self.currency
        ))

    def __str__(self) -> str:
        return f"{self.amount} {self.currency}"

# Usage
price1 = Money(amount=Decimal("10.50"), currency="USD")
price2 = Money(amount=Decimal("5.25"), currency="USD")

total_result = price1.add(price2)
if total_result.success:
    total = total_result.unwrap()
    print(total)  # 15.75 USD

# Value objects are immutable
# price1.amount = Decimal("20.00")  # This would raise an error
```

### FlextModels.AggregateRoot

Base class for aggregate roots (consistency boundaries).

```python
class Order(FlextModels.AggregateRoot):
    """Order aggregate root."""
    customer_id: str
    items: list[OrderItem] = Field(default_factory=list)
    status: OrderStatus = OrderStatus.PENDING
    total: Money = Field(default_factory=lambda: Money(amount=Decimal("0"), currency="USD"))

    def add_item(self, product: Product, quantity: int) -> FlextResult[None]:
        """Add item with business rule validation."""
        if self.status != OrderStatus.PENDING:
            return FlextResult.fail("Cannot modify confirmed order")

        if quantity <= 0:
            return FlextResult.fail("Quantity must be positive")

        if product.stock < quantity:
            return FlextResult.fail(f"Insufficient stock: {product.stock}")

        item = OrderItem(product=product, quantity=quantity)
        self.items.append(item)

        # Update total
        item_total = Money(
            amount=product.price.amount * quantity,
            currency=product.price.currency
        )
        total_result = self.total.add(item_total)
        if total_result.is_failure:
            return total_result

        self.total = total_result.unwrap()

        # Raise domain event
        self.add_domain_event("ItemAdded", {
            "order_id": self.id,
            "product_id": product.id,
            "quantity": quantity,
            "item_total": str(item_total)
        })

        return FlextResult.ok(None)

    def confirm(self) -> FlextResult[None]:
        """Confirm the order."""
        if not self.items:
            return FlextResult.fail("Cannot confirm empty order")

        if self.status != OrderStatus.PENDING:
            return FlextResult.fail("Order already processed")

        self.status = OrderStatus.CONFIRMED
        self.add_domain_event("OrderConfirmed", {
            "order_id": self.id,
            "total": str(self.total),
            "item_count": len(self.items)
        })

        return FlextResult.ok(None)

# Usage
order = Order(id="order_123", customer_id="customer_456")
product = Product(id="prod_1", name="Widget", price=Money(Decimal("10.00"), "USD"), stock=100)

add_result = order.add_item(product, 2)
if add_result.success:
    print(f"Order total: {order.total}")  # Order total: 20.00 USD

confirm_result = order.confirm()
if confirm_result.success:
    print(f"Order {order.id} confirmed")
```

## FlextConfig

Pydantic-based configuration with environment support.

### Class Definition

```python
class FlextConfig(BaseSettings):
    """Base configuration class with environment support."""
```

### Usage

```python
class DatabaseConfig(FlextConfig):
    """Database configuration."""
    host: str = Field("localhost", description="Database host")
    port: int = Field(5432, ge=1, le=65535)
    database: str = Field("myapp")
    username: str = Field("postgres")
    password: str = Field("", description="Database password")
    ssl_mode: bool = Field(False)

    @property
    def connection_url(self) -> str:
        """Build connection URL."""
        auth = f"{self.username}:{self.password}@" if self.password else f"{self.username}@"
        ssl = "?sslmode=require" if self.ssl_mode else ""
        return f"postgresql://{auth}{self.host}:{self.port}/{self.database}{ssl}"

    class Config:
        env_prefix = "DB_"  # Reads from DB_HOST, DB_PORT, etc.
        env_file = ".env"

# Usage
config = DatabaseConfig()  # Automatically loads from environment
print(config.connection_url)

# Override with environment variables:
# DB_HOST=production-db.example.com
# DB_PORT=5433
# DB_PASSWORD=secret123
```

### Application Configuration

```python
class AppConfig(FlextConfig):
    """Main application configuration."""
    app_name: str = Field("My FLEXT App")
    version: str = Field("0.9.0")
    debug: bool = Field(False)
    environment: str = Field("development", regex="^(development|staging|production)$")

    # API settings
    api_host: str = Field("0.0.0.0")
    api_port: int = Field(8000, ge=1, le=65535)
    api_workers: int = Field(1, ge=1, le=16)

    # Security
    secret_key: str = Field(..., min_length=32)  # Required
    cors_origins: list[str] = Field(default_factory=list)

    # Optional integrations
    redis_url: str | None = Field(None)
    sentry_dsn: str | None = Field(None)

    class Config:
        env_prefix = "APP_"
        env_file = ".env"
        case_sensitive = False

# Usage with validation
try:
    app_config = AppConfig()
    print(f"Starting {app_config.app_name} on {app_config.api_host}:{app_config.api_port}")
except ValidationError as e:
    print(f"Configuration error: {e}")
```

## FlextLogger

Structured logging with contextual information.

### Class Definition

```python
class FlextLogger:
    """Structured logger with context support."""
```

### Basic Usage

```python
# Create logger
logger = FlextLogger(__name__)

# Basic logging
logger.info("Application started")
logger.error("Database connection failed")
logger.warning("High memory usage detected", memory_percent=85)
logger.debug("Processing user request", user_id="user_123")

# With structured context
logger.info("Order processed", extra={
    "order_id": "order_456",
    "customer_id": "customer_789",
    "amount": "125.50",
    "currency": "USD"
})
```

### Contextual Logging

```python
class UserService:
    def __init__(self):
        self._logger = FlextLogger(__name__)
        self._container = FlextContainer.get_global()

    def create_user(self, data: dict) -> FlextResult[User]:
        """Create user with comprehensive logging."""
        self._logger.info("Starting user creation", extra={
            "username": data.get("username"),
            "email": data.get("email")
        })

        try:
            # Validate input
            if not data.get("username"):
                self._logger.error("User creation failed: missing username")
                return FlextResult.fail("Username is required")

            # Create user
            user = User(**data)

            # Log success
            self._logger.info("User created successfully", extra={
                "user_id": user.id,
                "username": user.username,
                "created_at": user.created_at.isoformat()
            })

            return FlextResult.ok(user)

        except Exception as e:
            self._logger.error("User creation failed with exception", extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "username": data.get("username")
            })
            return FlextResult.fail(f"User creation failed: {str(e)}")
```

## FlextDomainService

Base class for domain services with dependency injection.

### Class Definition

```python
class FlextDomainService(BaseModel):
    """Base class for domain services."""
```

### Usage

```python
class UserRegistrationService(FlextDomainService):
    """Domain service for user registration business logic."""

    def __init__(self, **data):
        super().__init__(**data)
        container = FlextContainer.get_global()
        self._logger = FlextLogger(__name__)
        self._email_service = container.get("email").unwrap()
        self._user_repository = container.get("user_repository").unwrap()

    def register_user(self, registration_data: dict) -> FlextResult[User]:
        """Complete user registration process."""
        self._logger.info("Starting user registration", extra={
            "username": registration_data.get("username")
        })

        # Validation pipeline
        validation_result = self._validate_registration_data(registration_data)
        if validation_result.is_failure:
            return validation_result

        # Business logic pipeline
        return (
            self._check_username_availability(registration_data["username"])
            .flat_map(lambda _: self._create_user_account(registration_data))
            .flat_map(lambda user: self._send_welcome_email(user))
            .flat_map(lambda user: self._save_user(user))
            .map(lambda user: self._log_registration_success(user))
        )

    def _validate_registration_data(self, data: dict) -> FlextResult[dict]:
        """Validate registration input."""
        if not data.get("username"):
            return FlextResult.fail("Username is required")

        if not data.get("email"):
            return FlextResult.fail("Email is required")

        if len(data.get("password", "")) < 8:
            return FlextResult.fail("Password must be at least 8 characters")

        return FlextResult.ok(data)

    def _check_username_availability(self, username: str) -> FlextResult[None]:
        """Check if username is available."""
        existing_user = self._user_repository.find_by_username(username)
        if existing_user.success:
            return FlextResult.fail(f"Username '{username}' is already taken")

        return FlextResult.ok(None)

    def _create_user_account(self, data: dict) -> FlextResult[User]:
        """Create user account with hashed password."""
        import hashlib

        password_hash = hashlib.sha256(data["password"].encode()).hexdigest()
        user = User(
            id=f"user_{hashlib.md5(data['email'].encode()).hexdigest()[:8]}",
            username=data["username"],
            email=data["email"],
            password_hash=password_hash,
            created_at=datetime.now(),
            is_active=False  # Requires email verification
        )

        return FlextResult.ok(user)

    def _send_welcome_email(self, user: User) -> FlextResult[User]:
        """Send welcome email to new user."""
        email_result = self._email_service.send_welcome_email(user.email, user.username)
        if email_result.is_failure:
            self._logger.warning("Welcome email failed", extra={
                "user_id": user.id,
                "error": email_result.error
            })
            # Don't fail registration for email issues

        return FlextResult.ok(user)

    def _save_user(self, user: User) -> FlextResult[User]:
        """Save user to repository."""
        save_result = self._user_repository.save(user)
        if save_result.is_failure:
            return FlextResult.fail(f"Failed to save user: {save_result.error}")

        return FlextResult.ok(user)

    def _log_registration_success(self, user: User) -> User:
        """Log successful registration."""
        self._logger.info("User registration completed", extra={
            "user_id": user.id,
            "username": user.username,
            "email": user.email
        })
        return user

# Usage
registration_service = UserRegistrationService()
result = registration_service.register_user({
    "username": "johndoe",
    "email": "john@example.com",
    "password": "securepassword123"
})

if result.success:
    user = result.unwrap()
    print(f"User {user.username} registered successfully!")
else:
    print(f"Registration failed: {result.error}")
```

## Utility Functions

### FlextUtilities

Common utility functions used across the ecosystem.

```python
from flext_core import FlextUtilities

# ID generation
entity_id = FlextUtilities.generate_id("user")  # "user_abc123def"
uuid_str = FlextUtilities.generate_uuid()       # "550e8400-e29b-41d4-a716-446655440000"

# Date utilities
iso_timestamp = FlextUtilities.now_iso()        # "2025-09-14T15:30:45.123456Z"
unix_timestamp = FlextUtilities.now_unix()      # 1631629845

# Validation helpers
is_valid = FlextUtilities.is_valid_email("test@example.com")  # True
is_strong = FlextUtilities.is_strong_password("MySecure123!") # True
```

## Error Handling Best Practices

### Consistent Error Patterns

```python
def business_operation(data: dict) -> FlextResult[ProcessedData]:
    """Example of consistent error handling."""

    # Input validation
    if not data:
        return FlextResult.fail("Input data cannot be empty", error_code="VALIDATION_ERROR")

    # Business logic with early returns
    if not data.get("required_field"):
        return FlextResult.fail("Required field missing", error_code="MISSING_FIELD")

    # Chain operations with railway pattern
    return (
        validate_data(data)
        .flat_map(lambda d: enrich_data(d))
        .flat_map(lambda d: apply_business_rules(d))
        .flat_map(lambda d: persist_data(d))
        .map(lambda d: create_response(d))
    )

# Usage with proper error handling
result = business_operation(input_data)
if result.success:
    processed = result.unwrap()
    logger.info("Operation succeeded", extra={"result_id": processed.id})
else:
    logger.error("Operation failed", extra={"error": result.error})
    # Handle error appropriately
```

### Testing API Usage

```python
import pytest
from flext_core import FlextResult, FlextContainer, FlextModels

def test_flext_result_success():
    """Test successful FlextResult operations."""
    result = FlextResult[str].ok("success")
    assert result.success
    assert result.unwrap() == "success"
    assert result.value == "success"  # New API
    assert result.data == "success"   # Legacy API compatibility

def test_flext_result_failure():
    """Test failed FlextResult operations."""
    result = FlextResult[str].fail("error message")
    assert result.is_failure
    assert result.error == "error message"

    with pytest.raises(ValueError):
        result.unwrap()  # Should raise on failure

def test_flext_result_chaining():
    """Test railway pattern chaining."""
    def double(x: int) -> int:
        return x * 2

    def divide_by_two(x: int) -> FlextResult[int]:
        return FlextResult.ok(x // 2)

    result = (
        FlextResult[int].ok(10)
        .map(double)                    # 20
        .flat_map(divide_by_two)       # 10
        .map(double)                   # 20
    )

    assert result.success
    assert result.unwrap() == 20

def test_container_dependency_injection():
    """Test dependency injection container."""
    container = FlextContainer()

    # Register service
    container.register("test_service", "test_value")

    # Retrieve service
    result = container.get("test_service")
    assert result.success
    assert result.unwrap() == "test_value"

    # Test missing service
    missing = container.get("nonexistent")
    assert missing.is_failure

def test_domain_models():
    """Test domain modeling patterns."""
    class TestUser(FlextModels.Entity):
        name: str
        email: str

        def activate(self) -> FlextResult[None]:
            self.is_active = True
            self.add_domain_event("UserActivated", {"user_id": self.id})
            return FlextResult.ok(None)

    user = TestUser(id="test_123", name="John", email="john@example.com")
    result = user.activate()

    assert result.success
    events = user.get_domain_events()
    assert len(events) == 1
    assert events[0]["event_type"] == "UserActivated"
```

---

**API Reference Complete**: This documentation covers all public APIs in FLEXT-Core foundation library. For usage examples, see [Examples](examples/overview.md). For architectural details, see [Architecture](architecture.md).