# API Reference

Complete API documentation for FLEXT-Core foundation library components.

---

## Core Imports

All FLEXT-Core functionality is available through root-level imports:

```python
from flext_core import (
    # Railway pattern
    FlextResult,           # Error handling with monadic operations

    # Dependency injection
    FlextContainer,        # Service container with type safety

    # Domain modeling
    FlextModels,           # Entity/Value/Aggregate patterns
    FlextDomainService,    # Service base class

    # Configuration and core
    FlextConfig,           # Environment-aware configuration
    FlextCore,             # Main orchestrator class
    FlextContext,          # Request/operation context

    # CQRS and processing
    FlextCommands,         # CQRS command patterns
    FlextProcessing,       # Handler patterns and execution
    FlextHandlers,         # Alias for FlextProcessing (backward compatibility)

    # Validation and guards
    FlextValidations,      # Validation patterns and predicates
    FlextGuards,           # Type guards and validation decorators
    Predicates,            # Predicate functions

    # Type system
    FlextTypes,            # Type definitions and aliases
    FlextTypeAdapters,     # Type adapters for conversions
    T, U, V, E, F, P, R,   # Type variables
    T_co,                  # Covariant type variable

    # Infrastructure
    FlextLogger,           # Structured logging
    FlextExceptions,       # Exception hierarchy
    FlextConstants,        # System constants
    FlextProtocols,        # Interface definitions

    # Utilities and extensions
    FlextUtilities,        # Helper functions
    FlextDecorators,       # Decorator utilities
    FlextMixins,           # Reusable behaviors
    FlextFields,           # Field definitions

    # Version management
    FlextVersionManager,   # Version information
    __version__,           # Current version string
)
```

---

## FlextResult[T]

Railway-oriented programming pattern for type-safe error handling.

### Creation

```python
# Success case
result = FlextResult[str].ok("success value")

# Failure case
result = FlextResult[str].fail("error message")
result = FlextResult[str].fail("error", error_code="CUSTOM_ERROR")
```

### Core Operations

```python
# Check status
result.is_success        # bool
result.is_failure        # bool

# Safe value access
if result.is_success:
    value = result.unwrap()  # Extract value after success check

# Alternative access
value = result.unwrap_or("default")  # With fallback
```

### Monadic Operations

```python
# Transform success values
result.map(lambda x: x.upper())

# Chain operations that return FlextResult
result.flat_map(lambda x: process_value(x))

# Filter with predicate
result.filter(lambda x: len(x) > 5, "Too short")
```

---

## FlextContainer

Dependency injection container with type-safe service management.

### Basic Usage

```python
# Get global singleton
container = FlextContainer.get_global()

# Register services
container.register("database", DatabaseService())
container.register("logger", LoggerService())

# Retrieve services
db_result = container.get("database")
if db_result.is_success:
    database = db_result.unwrap()
```

### Factory Registration

```python
# Register factory function
container.register_factory("connection", lambda: create_connection())

# Register singleton (cached after first creation)
container.register_singleton("config", ConfigService())
```

---

## FlextModels

Domain modeling patterns following Domain-Driven Design principles.

### Entity Pattern

```python
class User(FlextModels.Entity):
    name: str
    email: str
    is_active: bool = False

    def activate(self) -> FlextResult[None]:
        if self.is_active:
            return FlextResult[None].fail("Already active")
        self.is_active = True
        return FlextResult[None].ok(None)
```

### Value Object Pattern

```python
class Money(FlextModels.Value):
    amount: int
    currency: str

    def add(self, other: "Money") -> FlextResult["Money"]:
        if self.currency != other.currency:
            return FlextResult[Money].fail("Currency mismatch")
        return FlextResult[Money].ok(
            Money(amount=self.amount + other.amount, currency=self.currency)
        )
```

---

## FlextConfig

Environment-aware configuration management with Pydantic integration.

### Basic Configuration

```python
class AppConfig(FlextConfig):
    database_url: str
    api_key: str
    debug: bool = False
    timeout: int = 30

# Automatically loads from environment variables
config = AppConfig()
```

### Environment Integration

```python
# Reads from:
# - DATABASE_URL environment variable
# - API_KEY environment variable
# - DEBUG environment variable (with bool conversion)
# - TIMEOUT environment variable (with int conversion)
```

---

## FlextValidations

Validation patterns and predicate-based validation system.

### Type Validators

```python
# Built-in type validation
FlextValidations.TypeValidators.validate_string("test")
FlextValidations.TypeValidators.validate_integer(42)
FlextValidations.TypeValidators.validate_email("user@example.com")
```

### Custom Validators

```python
def validate_positive(value: int) -> FlextResult[int]:
    if value <= 0:
        return FlextResult[int].fail("Must be positive")
    return FlextResult[int].ok(value)

# Compose validators
composite = FlextValidations.create_composite_validator([
    validate_positive,
    lambda x: FlextResult[int].ok(x) if x < 100 else FlextResult[int].fail("Too large")
])
```

---

## FlextLogger

Structured logging with correlation tracking and environment awareness.

### Basic Usage

```python
logger = FlextLogger(__name__)

logger.info("Operation completed", user_id="123", duration_ms=45.2)
logger.error("Operation failed", error="Database timeout", user_id="123")
```

### Context Management

```python
# Add persistent context
logger = logger.bind(service="user-service", version="1.0")

# Performance tracking
with logger.performance_timer():
    # Operation being timed
    process_data()
```

---

## FlextDomainService

Base class for domain services with dependency injection integration.

### Service Implementation

```python
class UserService(FlextDomainService):
    def __init__(self) -> None:
        super().__init__()
        self._logger = FlextLogger(__name__)

    def create_user(self, data: dict) -> FlextResult[User]:
        # Validate input
        if not data.get("email"):
            return FlextResult[User].fail("Email required")

        # Create user
        user = User(**data)
        self._logger.info("User created", user_id=user.id)
        return FlextResult[User].ok(user)
```

---

## Error Handling

### FlextResult Error Propagation

```python
def process_user_data(data: dict) -> FlextResult[ProcessedUser]:
    return (
        validate_user_data(data)           # FlextResult[dict]
        .flat_map(enrich_user_data)        # FlextResult[dict]
        .map(create_user_object)           # FlextResult[User]
        .flat_map(save_user)               # FlextResult[ProcessedUser]
    )

# Automatic error propagation - first failure stops the chain
result = process_user_data({"email": "test@example.com"})
```

### Exception Handling

```python
# Convert exceptions to FlextResult
def safe_operation() -> FlextResult[str]:
    try:
        risky_operation()
        return FlextResult[str].ok("success")
    except Exception as e:
        return FlextResult[str].fail(f"Operation failed: {e}")
```

---

## Type System

### Core Types

```python
# Type variables for generic programming
T = TypeVar('T')  # Generic type
U = TypeVar('U')  # Second generic type

# Result type aliases
FlextTypes.Core.StringResult = FlextResult[str]
FlextTypes.Core.IntResult = FlextResult[int]
```

### Configuration Types

```python
FlextTypes.Config.LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
FlextTypes.Config.Environment = Literal["development", "testing", "production"]
```

---

## Integration Patterns

### Service Composition

```python
class ComposedService(FlextDomainService):
    def __init__(self) -> None:
        super().__init__()
        container = FlextContainer.get_global()

        # Get dependencies
        self._user_service = container.get("user_service").unwrap()
        self._email_service = container.get("email_service").unwrap()

    def register_user(self, data: dict) -> FlextResult[User]:
        return (
            self._user_service.create_user(data)
            .flat_map(lambda user: self._email_service.send_welcome(user))
        )
```

### Error Aggregation

```python
# Collect multiple results
results = [
    validate_name(data.get("name")),
    validate_email(data.get("email")),
    validate_age(data.get("age"))
]

# Combine results
combined = FlextResult.combine(*results)
if combined.is_success:
    # All validations passed
    validated_data = combined.unwrap()
```

---

## Performance Considerations

### FlextResult Optimization

- Use `map()` for simple transformations
- Use `flat_map()` for operations returning FlextResult
- Avoid deep nesting with early returns
- Consider `unwrap_or()` for simple fallbacks

### Container Performance

- Register singletons for expensive-to-create services
- Use factories for lightweight, stateful services
- Minimize container lookups in hot paths

### Logging Performance

- Use structured logging with consistent field names
- Avoid expensive operations in log messages
- Use appropriate log levels (DEBUG only in development)

---

**API Status**: This reference covers FLEXT-Core v0.9.0 foundation patterns with 84% test coverage. All documented APIs are functional and tested. For development workflow and contribution guidelines, see [development.md](development.md).