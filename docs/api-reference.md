# API Reference

**FLEXT-Core Foundation Library API Documentation**
**Date**: September 17, 2025 | **Version**: 0.9.0

---

## Overview

FLEXT-Core API reference is maintained at the workspace level to ensure consistency across the ecosystem. This document provides foundation-specific usage patterns and examples.

**Complete API Documentation**: [Workspace API Reference](../../docs/api/flext-core.md)

---

## Foundation Exports

### Core Imports

```python
from flext_core import (
    # Error handling and composition
    FlextResult,

    # Dependency injection
    FlextContainer,

    # Domain modeling
    FlextModels,

    # Configuration management
    FlextConfig,

    # Validation patterns
    FlextValidations,

    # CQRS patterns
    FlextCommands,

    # Structured logging
    FlextLogger,

    # Type system
    FlextTypes,

    # Utilities
    FlextUtilities,

    # Constants
    FlextConstants,

    # Type adaptation (minimal)
    FlextTypeAdapters
)
```

---

## Usage Patterns

### FlextResult[T] - Railway Pattern

**Primary API for error handling across FLEXT ecosystem**

```python
from flext_core import FlextResult

# Creation
success = FlextResult[str].ok("value")
failure = FlextResult[str].fail("error message")

# Access patterns (both supported for compatibility)
if result.is_success:
    value = result.value    # New API
    value = result.data     # Legacy API (maintained)
    value = result.unwrap() # Explicit API

# Composition
result = (
    initial_operation()
    .map(transform_value)           # Transform success
    .flat_map(chain_operation)      # Chain operations
    .map_error(handle_error)        # Handle errors
    .filter(predicate, "Error")     # Conditional filtering
)

# Pattern matching
match result:
    case result if result.is_success:
        process(result.unwrap())
    case result if result.is_failure:
        log_error(result.error)
```

### FlextContainer - Dependency Injection

**Singleton container for service lifecycle management**

```python
from flext_core import FlextContainer

# Get global singleton
container = FlextContainer.get_global()

# Service registration
register_result = container.register("service_name", service_instance)
if register_result.is_success:
    print("Service registered")

# Service retrieval
service_result = container.get("service_name")
if service_result.is_success:
    service = service_result.unwrap()
    # Use service

# Type-safe registration pattern
class DatabaseService:
    def connect(self) -> str:
        return "Connected"

db_service = DatabaseService()
container.register("database", db_service)
```

### FlextModels - Domain Modeling

**Domain-driven design patterns**

```python
from flext_core import FlextModels, FlextResult

# Entity (has identity)
class User(FlextModels.Entity):
    name: str
    email: str

    def change_email(self, new_email: str) -> FlextResult[None]:
        # Business logic with validation
        if "@" not in new_email:
            return FlextResult[None].fail("Invalid email")

        self.email = new_email
        self.add_domain_event("EmailChanged", {"new_email": new_email})
        return FlextResult[None].ok(None)

# Value Object (immutable)
class Money(FlextModels.Value):
    amount: float
    currency: str

    def add(self, other: "Money") -> FlextResult["Money"]:
        if self.currency != other.currency:
            return FlextResult["Money"].fail("Currency mismatch")
        return FlextResult["Money"].ok(
            Money(amount=self.amount + other.amount, currency=self.currency)
        )

# Aggregate Root (consistency boundary)
class Account(FlextModels.AggregateRoot):
    balance: Money
    owner: User

    def transfer_to(self, target: "Account", amount: Money) -> FlextResult[None]:
        # Ensure business invariants
        if self.balance.amount < amount.amount:
            return FlextResult[None].fail("Insufficient funds")

        # Business logic
        return FlextResult[None].ok(None)
```

### FlextConfig - Configuration Management

**Environment-aware configuration with Pydantic**

```python
from flext_core import FlextConfig

class ApplicationConfig(FlextConfig):
    """Application configuration with environment variables."""

    # Required settings
    database_url: str
    api_key: str

    # Optional with defaults
    debug: bool = False
    port: int = 8000
    timeout: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = False
        env_prefix = "APP_"

# Automatic environment variable loading
# APP_DATABASE_URL, APP_API_KEY, APP_DEBUG, etc.
config = ApplicationConfig()

# Validation with FlextResult integration
def validate_config(config: ApplicationConfig) -> FlextResult[None]:
    if not config.database_url.startswith(("postgresql://", "mysql://")):
        return FlextResult[None].fail("Invalid database URL")
    return FlextResult[None].ok(None)
```

---

## Advanced Patterns

### Error Handling Chain

```python
def complex_operation(data: dict) -> FlextResult[str]:
    """Complex business operation with multiple steps."""
    return (
        validate_input(data)
        .flat_map(parse_data)
        .flat_map(process_business_logic)
        .flat_map(serialize_result)
        .map_error(lambda e: f"Operation failed: {e}")
    )

def validate_input(data: dict) -> FlextResult[dict]:
    if not data:
        return FlextResult[dict].fail("Empty input")
    return FlextResult[dict].ok(data)

def parse_data(data: dict) -> FlextResult[dict]:
    try:
        # Parsing logic
        return FlextResult[dict].ok(parsed_data)
    except Exception as e:
        return FlextResult[dict].fail(f"Parse error: {e}")
```

### Service Layer Pattern

```python
from flext_core import FlextContainer, FlextResult, FlextLogger

class UserService:
    """Business service using FLEXT patterns."""

    def __init__(self):
        container = FlextContainer.get_global()
        self._db = container.get("database").unwrap()
        self._logger = FlextLogger(__name__)

    def create_user(self, data: dict) -> FlextResult[User]:
        """Create user with business validation."""
        self._logger.info("Creating user", extra={"email": data.get("email")})

        return (
            self._validate_user_data(data)
            .flat_map(self._check_email_uniqueness)
            .flat_map(self._create_user_entity)
            .flat_map(self._save_user)
            .map(self._log_user_created)
        )

    def _validate_user_data(self, data: dict) -> FlextResult[dict]:
        # Validation logic
        pass

    def _check_email_uniqueness(self, data: dict) -> FlextResult[dict]:
        # Business rule validation
        pass
```

---

## Foundation-Specific APIs

### FlextTypeAdapters (Minimal Design)

**Simplified type adaptation interface**

```python
from flext_core import FlextTypeAdapters

# Simple object-to-dict adaptation
adapters = FlextTypeAdapters()
result_dict = adapters.adapt_to_dict(some_object)

# The design is intentionally minimal (22 lines total)
# For complex type adaptation, use Pydantic TypeAdapter directly:
from pydantic import TypeAdapter

adapter = TypeAdapter(YourType)
validated = adapter.validate_python(data)
```

### FlextUtilities

**Foundation utility functions**

```python
from flext_core import FlextUtilities

# ID generation
unique_id = FlextUtilities.Generators.generate_id()
timestamp = FlextUtilities.Generators.generate_iso_timestamp()

# Type guards
is_valid = FlextUtilities.Guards.is_not_none(value)
is_string = FlextUtilities.Guards.is_string(value)
```

---

## Type Safety Guidelines

### Strict Type Annotations

```python
# ✅ Complete type annotations
def process_user(user_data: dict[str, str]) -> FlextResult[User]:
    return FlextResult[User].ok(User(**user_data))

# ✅ Generic type parameters
def map_results(
    results: list[FlextResult[T]],
    mapper: Callable[[T], U]
) -> list[FlextResult[U]]:
    return [result.map(mapper) for result in results]

# ❌ Missing type information
def process_user(user_data):  # Missing types
    return FlextResult.ok(User(**user_data))  # Missing generic
```

### Error Handling Standards

```python
# ✅ Explicit error handling
result = risky_operation()
if result.is_success:
    value = result.unwrap()
    # Continue processing
else:
    logger.error(f"Operation failed: {result.error}")
    return FlextResult[None].fail("Processing failed")

# ❌ Direct unwrap without checking
value = risky_operation().unwrap()  # Can raise exception
```

---

## Integration with FLEXT Ecosystem

All FLEXT projects should use these foundation patterns for consistency:

1. **Error Handling**: FlextResult[T] for all business operations
2. **Dependency Injection**: FlextContainer for service management
3. **Configuration**: FlextConfig for environment-aware settings
4. **Domain Modeling**: FlextModels for business entities
5. **Logging**: FlextLogger for structured logging

**Complete API Documentation**: [Workspace API Reference](../../docs/api/flext-core.md)

---

**Foundation API Reference** - Core patterns for the FLEXT ecosystem