# Foundation Layer API Reference

This section covers the core foundation classes that provide the fundamental building blocks of FLEXT-Core.

## Core Foundation Classes

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
    def __init__(self, field: str, value: Any):
        super().__init__(
            error_code=ErrorCode.VALIDATION_ERROR,
            message=f"Invalid value for {field}: {value}",
            context={"field": field, "value": value}
        )
- `ValidationException` - Field validation errors
- `ConfigurationException` - Configuration errors

### FlextConstants - Centralized Constants

Centralized constants and enumerations used throughout the framework.

```python
from flext_core import FlextConstants

# HTTP status codes
status = FlextConstants.HttpStatus.OK

# Common patterns
pattern = FlextConstants.Regex.EMAIL

# Framework defaults
timeout = FlextConstants.Defaults.REQUEST_TIMEOUT
````

### FlextTypes - Type System

Comprehensive type system with TypeVars, Protocols, and type aliases.

```python
from flext_core import FlextTypes

# Generic types
UserId = FlextTypes.UserId  # NewType based on str
EntityId = FlextTypes.EntityId  # Generic entity ID

# Protocols
class Repository(Protocol):
    def get_by_id(self, id: EntityId) -> FlextResult[Entity]:
        ...

# Type aliases
JsonDict = FlextTypes.JsonDict  # Dict[str, Any]
```

## Quality Metrics

| Module          | Coverage | Status       | Description                    |
| --------------- | -------- | ------------ | ------------------------------ |
| `result.py`     | 95%      | ✅ Stable    | Railway pattern implementation |
| `container.py`  | 99%      | ✅ Stable    | Dependency injection container |
| `exceptions.py` | 62%      | 🔄 Improving | Exception hierarchy            |
| `constants.py`  | 100%     | ✅ Complete  | Constants and enumerations     |
| `typings.py`    | 100%     | ✅ Complete  | Type system definitions        |

## Usage Examples

### Complete Railway-Oriented Example

```python
from flext_core import FlextResult, FlextContainer, FlextLogger

class UserService:
    def __init__(self):
        self.logger = FlextLogger(__name__)

from flext_core import FlextResult, FlextContainer, FlextLogger
# Note: FlextLogger is documented in the Logging section

class UserService:
    def __init__(self):
        self.logger = FlextLogger(__name__)

        # Business logic
        user = User(id=f"user_{name.lower()}", name=name, email=email)

        # Logging
        self.logger.info(f"Created user: {user.name}")

        return FlextResult[User].ok(user)

    def _validate_input(self, name: str, email: str) -> FlextResult[None]:
        if not name:
            return FlextResult[None].fail("Name is required")
        if "@" not in email:
            return FlextResult[None].fail("Invalid email format")
        return FlextResult[None].ok(None)

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
        print(f"✅ Created user: {user.name}")
    else:
        print(f"❌ Failed: {result.error}")
```

This foundation layer provides the essential building blocks for robust, maintainable Python applications with type safety and clean error handling.
