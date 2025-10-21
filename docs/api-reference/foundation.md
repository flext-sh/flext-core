# Foundation Layers API Reference

This section covers **Layers 0, 0.5, and 1** - the foundational architecture of FLEXT-Core v0.9.9.

## Architecture Overview

FLEXT-Core's foundation is built on three layers:

- **Layer 0**: Pure Constants (FlextConstants, FlextTypes, FlextProtocols) - zero dependencies
- **Layer 0.5**: Runtime Bridge (FlextRuntime) - external library integration
- **Layer 1**: Foundation (FlextResult, FlextContainer, FlextExceptions) - core primitives

See [Architecture Overview](../architecture/overview.md) for complete layer details.

---

## Layer 0: Pure Constants

### FlextConstants - Centralized Constants

Immutable constants and configurations with **zero dependencies** (pure Python).

```python
from flext_core import FlextConstants

# Error codes (50+ codes)
error_code = FlextConstants.Errors.VALIDATION_FAILED

# Configuration defaults
timeout = FlextConstants.Configuration.DEFAULT_TIMEOUT

# Validation patterns
email_pattern = FlextConstants.Validation.EMAIL_PATTERN
```

### FlextTypes - Type System

Comprehensive type system with 50+ TypeVars, protocols, and type aliases.

```python
from flext_core import FlextTypes

# Common TypeVars
T = FlextTypes.T  # Covariant type
U = FlextTypes.U  # Invariant type

# Domain types
TCommand = FlextTypes.TCommand
TQuery = FlextTypes.TQuery
TEvent = FlextTypes.TEvent
```

### FlextProtocols - Runtime Interfaces

Runtime-checkable protocol definitions for type safety.

```python
from flext_core import FlextProtocols

# Check protocol compliance at runtime
if isinstance(obj, FlextProtocols.Configurable):
    result = obj.configure(config)
```

---

## Layer 0.5: Runtime Bridge

### FlextRuntime - External Library Integration

Bridge to external libraries (structlog, dependency_injector) with **no Layer 1+ imports**.

```python
from flext_core import FlextRuntime

# Type guards using Layer 0 patterns
if FlextRuntime.is_valid_email(email):
    process_email(email)

# Serialization utilities
json_data = FlextRuntime.serialize_to_json(data)
```

**Key Features**:

- Email, URL, UUID validation
- JSON serialization with FLEXT defaults
- Direct access to structlog, dependency_injector
- No circular dependencies

---

## Layer 1: Foundation (Core Primitives)

### FlextResult[T] - Railway-Oriented Programming

The `FlextResult[T]` class provides monadic error handling without exceptions, implementing the railway-oriented programming pattern.

```python
from flext_core import FlextResult

# Creating results
success_result = FlextResult[str].ok("Success message")
failure_result = FlextResult[str].fail("Error message")

# Type-safe operations
result = FlextResult[int].ok(42)

# Railway operations
def divide(a: float, b: float) -> FlextResult[float]:
    if b == 0:
        return FlextResult[float].fail("Division by zero")
    return FlextResult[float].ok(a / b)

# Monadic composition
result = divide(10, 2).map(lambda x: x * 2)  # FlextResult[float].ok(10.0)
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

### FlextContainer - Dependency Injection

Global dependency injection container with type-safe service registration and resolution.

```python
from flext_core import FlextContainer

# Get global container
container = FlextContainer.get_global()

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

### FlextExceptions - Exception Hierarchy

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
from flext_core import FlextContainer, FlextLogger, FlextResult

class UserService:
    def __init__(self):
        self.logger = FlextLogger(__name__)

    def create_user(self, name: str, email: str) -> FlextResult[dict]:
        """Create user with validation."""
        self.logger.info("Creating user", extra={"name": name})

        # Validation
        if not name:
            return FlextResult[dict].fail("Name is required")
        if "@" not in email:
            return FlextResult[dict].fail("Invalid email format")

        # Business logic
        user = {"id": f"user_{name.lower()}", "name": name, "email": email}

        # Logging
        self.logger.info(f"Created user: {user['name']}")

        return FlextResult[dict].ok(user)

# Dependency injection setup
container = FlextContainer.get_global()
container.register("user_service", UserService())

# Usage
service_result = container.get("user_service")
if service_result.is_success:
    service = service_result.unwrap()
    result = service.create_user("Alice", "alice@example.com")

    if result.is_success:
        user = result.unwrap()
        print(f"✅ Created user: {user['name']}")
    else:
        print(f"❌ Failed: {result.error}")
```

This foundation layer provides the essential building blocks for robust, maintainable Python applications with type safety and clean error handling.
````
