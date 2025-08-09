# FLEXT Core Architecture Overview

Reality-based architecture overview aligned with current implementation

## 🎯 Overview

FLEXT Core is the foundation library for clean architecture patterns and railway-oriented programming. This documentation reflects the ACTUAL implementation in `src/flext_core/`.

## 🏗️ Actual Project Structure

Based on `src/flext_core/` — validated:

```text
src/flext_core/
├── __init__.py              # Public API gateway
├── result.py                # FlextResult[T] - Railway pattern
├── container.py             # FlextContainer - DI system
├── config.py                # FlextSettings
├── entities.py              # FlextEntity - Domain entities
├── value_objects.py         # FlextValueObject - Value objects
├── aggregate_root.py        # FlextAggregateRoot - DDD aggregates
├── commands.py              # FlextCommands namespace
├── handlers.py              # FlextHandlers namespace
├── validation.py            # FlextValidation namespace
├── loggings.py              # Structured logging
├── exceptions.py            # Exception hierarchy
├── utilities.py             # Utility functions
├── constants.py             # Core constants
├── typings.py               # Centralized type definitions
├── __version__.py           # Version management & compatibility
├── protocols.py             # Protocol definitions
├── mixins.py                # Behavior mixins
├── decorators.py            # Decorator patterns
├── fields.py                # Field metadata
├── guards.py                # Validation guards
├── payload.py               # Message patterns
├── core.py                  # FlextCore main class
└── domain_services.py       # Domain services
```

## 🔧 Implemented Core Patterns

### 1. FlextResult[T] - Railway Pattern

Functional — the central pattern of FLEXT Core:

```python
from flext_core import FlextResult

# Success case
result = FlextResult.ok("Success data")
assert result.success
assert result.data == "Success data"

# Failure case
result = FlextResult.fail("Error message")
assert result.is_failure
assert result.error == "Error message"

# Chaining operations
def validate_email(email: str) -> FlextResult[str]:
    if "@" not in email:
        return FlextResult.fail("Invalid email")
    return FlextResult.ok(email.lower())

def create_user(email: str) -> FlextResult[dict]:
    return (
        validate_email(email)
        .map(lambda valid_email: {"email": valid_email, "created": True})
    )
```

### 2. FlextContainer - Dependency Injection

Functional — type-safe DI system:

```python
from flext_core import FlextContainer

# Setup container
container = FlextContainer()

# Register services
database_service = DatabaseService("sqlite:///app.db")
result = container.register("database", database_service)
assert result.success

# Retrieve services
service_result = container.get("database")
if service_result.success:
    db_service = service_result.data
```

### 3. Domain Patterns

Available — API present, implementation evolving:

```python
from flext_core import FlextValueObject, FlextAggregateRoot
from flext_core.models import FlextEntity

# Domain entity
class User(FlextEntity):
    def __init__(self, user_id: str, name: str, email: str):
        super().__init__(user_id)
        self.name = name
        self.email = email

# Value object
class Email(FlextValueObject):
    def __init__(self, address: str):
        if "@" not in address:
            raise ValueError("Invalid email")
        self.address = address.lower()
```

### 4. Configuration Management

Functional — based on Pydantic:

```python
from flext_core import FlextSettings

class AppSettings(FlextSettings):
    app_name: str = "My App"
    debug: bool = False
    database_url: str = "sqlite:///app.db"

    class Config:
        env_prefix = "APP_"

settings = AppSettings()
```

## 🏛️ Architecture Layers

### Foundation Layer

- **result.py**: FlextResult[T] for error handling
- **container.py**: FlextContainer for DI
- **typings.py**: Centralized type system
- **constants.py**: Core constants

### Domain Layer

- **entities.py**: Rich domain entities
- **value_objects.py**: Immutable value objects
- **aggregate_root.py**: DDD aggregates
- **domain_services.py**: Domain services

### Application Layer

- **commands.py**: Command patterns (CQRS)
- **handlers.py**: Handler patterns
- **validation.py**: Input validation

### Infrastructure Layer

- **config.py**: Configuration management
- **loggings.py**: Structured logging
- **protocols.py**: External system contracts

## 🧪 Testability

### Core Pattern Testing

```python
import pytest
from flext_core import FlextResult, FlextContainer

def test_result_pattern():
    """Test FlextResult railway pattern."""
    # Success path
    result = FlextResult.ok("test")
    assert result.success
    assert result.data == "test"

    # Failure path
    result = FlextResult.fail("error")
    assert result.is_failure
    assert result.error == "error"

def test_container_pattern():
    """Test dependency injection."""
    container = FlextContainer()
    service = "test_service"

    # Register
    reg_result = container.register("test", service)
    assert reg_result.success

    # Retrieve
    get_result = container.get("test")
    assert get_result.success
    assert get_result.data == service
```

## 📊 Implementation Status

### ✅ Production Ready

- **FlextResult[T]**: Complete railway-oriented programming
- **FlextContainer**: Dependency injection system
- **Configuration**: FlextSettings with Pydantic
- **Basic logging**: Structured logging support

### 🔧 In Development

- **Domain patterns**: Entity/ValueObject/Aggregate APIs available
- **CQRS**: Command/Handler namespace structure exists
- **Validation**: Basic validation patterns

### 📋 Planned

- **Event Sourcing**: Complete event sourcing implementation
- **Advanced CQRS**: Query bus and auto-discovery
- **Plugin Architecture**: Hot-pluggable components

## 🔗 Integration Points

### Framework Compatibility

- **Pydantic V2**: Configuration and validation
- **Standard Library**: Minimal external dependencies
- **Type System**: Python 3.13+ type hints

### Ecosystem Integration

FLEXT Core serves as the foundation for related projects in the workspace.

## ⚠️ Reality Check

This documentation reflects the CURRENT code in `src/flext_core/`.

### What EXISTS

- FlextResult pattern fully implemented
- FlextContainer dependency injection working
- Configuration system functional
- Domain pattern APIs available

### What's PLANNED

- Complete CQRS implementation
- Event sourcing system
- Advanced domain patterns

### What DOESN'T exist (yet)

- "33 projects ecosystem" (not validated)
- Complete framework integrations
- Production-ready event sourcing

---

For details, check the code in `src/flext_core/` and tests in `tests/`.
