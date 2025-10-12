# Foundation Layers API Reference

This section covers **Layers 0, 0.5, and 1** - the foundational architecture of FLEXT-Core v0.9.9.

## Architecture Overview

FLEXT-Core's foundation is built on three layers:

- **Layer 0**: Pure Constants (FlextCore.Constants, FlextCore.Types, FlextCore.Protocols) - zero dependencies
- **Layer 0.5**: Runtime Bridge (FlextCore.Runtime) - external library integration
- **Layer 1**: Foundation (FlextCore.Result, FlextCore.Container, FlextCore.Exceptions) - core primitives

See [Architecture Overview](../architecture/overview.md) for complete layer details.

---

## Layer 0: Pure Constants

### FlextCore.Constants - Centralized Constants

Immutable constants and configurations with **zero dependencies** (pure Python).

```python
from flext_core import FlextCore

# Error codes (50+ codes)
error_code = FlextCore.Constants.Errors.VALIDATION_FAILED

# Configuration defaults
timeout = FlextCore.Constants.Config.DEFAULT_TIMEOUT

# Validation patterns
email_pattern = FlextCore.Constants.Validation.EMAIL_PATTERN
```

### FlextCore.Types - Type System

Comprehensive type system with 50+ TypeVars, protocols, and type aliases.

```python
from flext_core import FlextCore

# Common TypeVars
T = FlextCore.Types.T  # Covariant type
U = FlextCore.Types.U  # Invariant type

# Domain types
TCommand = FlextCore.Types.TCommand
TQuery = FlextCore.Types.TQuery
TEvent = FlextCore.Types.TEvent
```

### FlextCore.Protocols - Runtime Interfaces

Runtime-checkable protocol definitions for type safety.

```python
from flext_core import FlextCore

# Check protocol compliance at runtime
if isinstance(obj, FlextCore.Protocols.Foundation.Configurable):
    result = obj.configure(config)
```

---

## Layer 0.5: Runtime Bridge

### FlextCore.Runtime - External Library Integration

Bridge to external libraries (structlog, dependency_injector) with **no Layer 1+ imports**.

```python
from flext_core import FlextCore

# Type guards using Layer 0 patterns
if FlextCore.Runtime.is_valid_email(email):
    process_email(email)

# Serialization utilities
json_data = FlextCore.Runtime.serialize_to_json(data)
```

**Key Features**:
- Email, URL, UUID validation
- JSON serialization with FLEXT defaults
- Direct access to structlog, dependency_injector
- No circular dependencies

---

## Layer 1: Foundation (Core Primitives)

### FlextCore.Result[T] - Railway-Oriented Programming

The `FlextCore.Result[T]` class provides monadic error handling without exceptions, implementing the railway-oriented programming pattern.

```python
from flext_core import FlextCore

# Creating results
success_result = FlextCore.Result[str].ok("Success message")
failure_result = FlextCore.Result[str].fail("Error message")

# Type-safe operations
result = FlextCore.Result[int].ok(42)

# Railway operations
def divide(a: float, b: float) -> FlextCore.Result[float]:
    if b == 0:
        return FlextCore.Result[float].fail("Division by zero")
    return FlextCore.Result[float].ok(a / b)

# Monadic composition
result = divide(10, 2).map(lambda x: x * 2)  # FlextCore.Result[float].ok(10.0)
```

**Key Methods:**

- `ok(value)` - Create a success result
- `fail(error)` - Create a failure result
- `is_success` - Check if result is success
- `is_failure` - Check if result is failure
- `unwrap()` - Extract value (throws on failure)
- `unwrap_or(default)` - Extract value with default
- `error` - Access the error message (for failure results)
- `map(transform)` - Transform success value
- `flat_map(transform)` - Transform and flatten
- `filter(predicate)` - Filter success values

### FlextCore.Container - Dependency Injection

Global dependency injection container with type-safe service registration and resolution.

```python
from flext_core import FlextCore

# Get global container
container = FlextCore.Container.get_global()

# Register services
container.register("logger", LoggerService())
container.register("database", DatabaseService(), singleton=True)

# Resolve services
logger_result = container.get("logger")
if logger_result.is_success:
    logger = logger_result.unwrap()
    logger.info("Application started")
```

**Key Methods:**

- `get_global()` - Get the global container instance
- `register(key, service, singleton=True)` - Register a service
- `get(key)` - Resolve a service by key
- `clear()` - Clear all registered services

### FlextCore.Exceptions - Exception Hierarchy

Comprehensive exception hierarchy with error codes and context.

````python
from flext_core import FlextException, ErrorCode
class ValidationException(FlextException):
    def __init__(self, field: str, value: object):
        super().__init__(
            error_code=ErrorCode.VALIDATION_ERROR,
            message=f"Invalid value for {field}: {value}",
            context={"field": field, "value": value}
        )
- `ValidationException` - Field validation errors
- `ConfigurationException` - Configuration errors


## Quality Metrics

| Layer | Module          | Coverage | Status       | Description                    |
| ----- | --------------- | -------- | ------------ | ------------------------------ |
| **0** | `constants.py`  | 100%     | ✅ Complete  | 50+ error codes, validation patterns |
| **0** | `typings.py`    | 100%     | ✅ Complete  | 50+ TypeVars, type aliases     |
| **0** | `protocols.py`  | 99%      | ✅ Stable    | Runtime-checkable interfaces   |
| **0.5** | `runtime.py`  | N/A      | ✅ Stable    | External library bridge        |
| **1** | `result.py`     | 95%      | ✅ Stable    | Railway pattern implementation |
| **1** | `container.py`  | 99%      | ✅ Stable    | Dependency injection container |
| **1** | `exceptions.py` | 62%      | 🔄 Improving | Exception hierarchy            |

## Usage Examples

### Complete Railway-Oriented Example

```python
from flext_core import FlextCore

class UserService:
    def __init__(self):
        self.logger = FlextCore.Logger(__name__)

from flext_core import FlextCore
# Note: FlextCore.Logger is documented in the Logging section

class UserService:
    def __init__(self):
        self.logger = FlextCore.Logger(__name__)

        # Business logic
        user = User(id=f"user_{name.lower()}", name=name, email=email)

        # Logging
        self.logger.info(f"Created user: {user.name}")

        return FlextCore.Result[User].ok(user)

    def _validate_input(self, name: str, email: str) -> FlextCore.Result[None]:
        if not name:
            return FlextCore.Result[None].fail("Name is required")
        if "@" not in email:
            return FlextCore.Result[None].fail("Invalid email format")
        return FlextCore.Result[None].ok(None)

# Dependency injection setup
container = FlextCore.Container.get_global()
container.register("user_service", UserService())

# Usage
service_result = container.get("user_service")
if service_result.is_success:
    service = service_result.unwrap()
    result = service.create_user("Alice", "alice@example.com")

    if result.is_success:
        user = result.unwrap()
        print(f"✅ Created user: {user.name}")
    else:
        print(f"❌ Failed: {result.error}")
```

This foundation layer provides the essential building blocks for robust, maintainable Python applications with type safety and clean error handling.
