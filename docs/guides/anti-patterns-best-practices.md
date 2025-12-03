# Anti-Patterns and Best Practices

**Status**: Production Ready | **Version**: 0.9.9 | **Focus**: Common FLEXT-Core mistakes and solutions

This guide documents common anti-patterns found in FLEXT ecosystem projects and their correct solutions. Every pattern includes code examples from the actual codebase.

## Table of Contents

1. [Error Handling Anti-Patterns](#error-handling-anti-patterns)
2. [Type Safety Anti-Patterns](#type-safety-anti-patterns)
3. [Architecture Anti-Patterns](#architecture-anti-patterns)
4. [Dependency Injection Anti-Patterns](#dependency-injection-anti-patterns)
5. [Model Anti-Patterns](#model-anti-patterns)
6. [Configuration Anti-Patterns](#configuration-anti-patterns)

---

## Error Handling Anti-Patterns

### Anti-Pattern 1: Using Exceptions for Business Logic

**Problem**: Raising exceptions for normal business errors creates unpredictable control flow.

```python
# ❌ ANTI-PATTERN - Exception-based
def validate_user(data: dict) -> User:
    """Returns User or raises exception."""
    if "email" not in data:
        raise ValueError("Email is required")  # Business error
    if len(data.get("password", "")) < 8:
        raise ValueError("Password too short")  # Business error
    return User(**data)

# Caller must handle exceptions
try:
    user = validate_user(data)
except ValueError as e:
    print(f"Validation failed: {e}")
```

**Why it's wrong**:

- Exceptions are for exceptional (unexpected) conditions, not business logic
- Multiple exception types make error handling verbose
- Stack traces pollute logs
- Performance cost of exception handling
- Difficult to compose with other operations

**Solution**: Use FlextResult railway pattern

```python
# ✅ CORRECT - Railway pattern
from flext_core import FlextResult

def validate_user(data: dict) -> FlextResult[User]:
    """Returns FlextResult wrapping success or failure."""
    if "email" not in data:
        return FlextResult[User].fail("Email is required")
    if len(data.get("password", "")) < 8:
        return FlextResult[User].fail("Password too short")
    return FlextResult[User].ok(User(**data))

# Caller handles results
result = validate_user(data)
if result.is_success:
    user = result.unwrap()
else:
    print(f"Validation failed: {result.error}")
```

**Benefits**:

- Clear success/failure semantics
- Composable with `flat_map`
- No exception overhead
- Explicit error handling
- Full context available

### Anti-Pattern 2: Swallowing Errors

**Problem**: Catching exceptions and ignoring them hides problems.

```python
# ❌ ANTI-PATTERN - Silent failure
def load_config() -> dict:
    """Loads config silently failing on error."""
    try:
        with open("config.json") as f:
            return json.load(f)
    except Exception:
        pass  # SILENT FAILURE!
    return {}
```

**Why it's wrong**:

- Problems go unnoticed in production
- Debugging becomes impossible
- Cascading failures downstream
- No audit trail

**Solution**: Propagate errors with context

```python
# ✅ CORRECT - Explicit error handling
from flext_core import FlextResult

def load_config() -> FlextResult[dict]:
    """Loads config with explicit error handling."""
    try:
        with open("config.json") as f:
            return FlextResult[dict].ok(json.load(f))
    except FileNotFoundError:
        return FlextResult[dict].fail(
            "Config file not found",
            error_code="CONFIG_NOT_FOUND",
            error_data={"filename": "config.json"},
        )
    except json.JSONDecodeError as e:
        return FlextResult[dict].fail(
            f"Invalid JSON in config: {e}",
            error_code="CONFIG_PARSE_ERROR",
        )
```

**Benefits**:

- Errors are visible
- Error context is preserved
- Downstream code can react
- Debugging is straightforward

### Anti-Pattern 3: Ignoring Error Information

**Problem**: Not capturing error codes and metadata.

```python
# ❌ ANTI-PATTERN - Missing context
result = FlextResult[dict].fail("An error occurred")
# No error code, no metadata - hard to debug
```

**Solution**: Include structured error information

```python
# ✅ CORRECT - Rich error context
from flext_core import FlextConstants

result = FlextResult[dict].fail(
    "Database connection failed after 3 retries",
    error_code=FlextConstants.Errors.DATABASE_ERROR,
    error_data={
        "host": "db.example.com",
        "port": 5432,
        "retry_count": 3,
        "last_error": "Connection timeout",
    },
)
```

---

## Type Safety Anti-Patterns

### Anti-Pattern 4: Using `Any` Type

**Problem**: `Any` disables type checking.

```python
# ❌ ANTI-PATTERN - Disables type checking
from typing import Any

def process_data(data: Any) -> Any:
    """Returns Any - type checker can't help."""
    return data.something()  # IDE doesn't know what methods are available
```

**Why it's wrong**:

- Type checker can't validate
- IDE autocomplete doesn't work
- Errors only discovered at runtime
- Type safety defeated

**Solution**: Use specific types or generics

```python
# ✅ CORRECT - Type-safe
from typing import TypeVar, Generic

T = TypeVar("T")

def process_data(data: dict[str, object]) -> dict[str, object]:
    """Specific types - type checker validates."""
    return data  # IDE knows dict methods

# Or with generics
class Container(Generic[T]):
    def process(self, data: T) -> T:
        """Generic preserves type."""
        return data

# Type checker knows exact type when used
container = Container[str]()
result = container.process("hello")  # Type is str
```

### Anti-Pattern 5: Untyped Container Retrieval

**Problem**: Getting services without type information.

```python
# ❌ ANTI-PATTERN - Lost type information
from flext_core import FlextContainer

container = FlextContainer.get_global()
logger = container.get("logger").unwrap()  # Type is object
logger.debug("Message")  # IDE doesn't know if debug() exists
```

**Solution**: Use type-safe retrieval

```python
# ✅ CORRECT - Type preserved
from flext_core import FlextContainer, FlextLogger

container = FlextContainer.get_global()

# Type-safe retrieval
result = container.get_typed("logger", FlextLogger)
if result.is_success:
    logger: FlextLogger = result.unwrap()
    logger.debug("Message")  # IDE knows FlextLogger methods
```

### Anti-Pattern 6: Type Ignores Without Justification

**Problem**: Suppressing type errors hides real problems.

```python
# ❌ ANTI-PATTERN - Suppresses type safety
def calculate_total(items: list[Item]) -> Decimal:
    total = 0
    for item in items:
        total += item.price
    return total  # Returns int, not Decimal
```

**Solution**: Fix the type error

```python
# ✅ CORRECT - Proper typing
from decimal import Decimal

def calculate_total(items: list[Item]) -> Decimal:
    total = Decimal("0")  # Correct type from start
    for item in items:
        total += item.price
    return total
```

---

## Architecture Anti-Patterns

### Anti-Pattern 7: Circular Dependencies

**Problem**: Modules importing each other violates layer hierarchy.

```
config.py imports → result.py
    ↓
result.py imports ← config.py
        CIRCULAR!
```

```python
# config.py
# ❌ ANTI-PATTERN - Imports from higher layer
from flext_core.result import FlextResult  # config is higher than result

# result.py
# ❌ ANTI-PATTERN - Imports from lower layer
from flext_core.config import FlextConfig  # result is lower than config
```

**Why it's wrong**:

- Creates import order dependencies
- Breaks code organization
- Hard to understand dependencies
- Causes import failures

**Solution**: Respect layer hierarchy (only import downward)

```
Layer 0: FlextConstants, t, p (no imports from other layers)
Layer 0.5: FlextRuntime (imports Layer 0 only)
Layer 1: FlextResult, FlextContainer (imports Layer 0, 0.5 only)
Layer 2: FlextModels, FlextService (imports Layer 0-1 only)
Layer 3: h, FlextDispatcher (imports Layer 0-2 only)
Layer 4: FlextConfig, FlextLogger (imports all lower layers)
```

```python
# ✅ CORRECT - Respect hierarchy
# config.py (Layer 4) - can import from all lower layers
from flext_core.result import FlextResult
from flext_core.constants import FlextConstants

# result.py (Layer 1) - imports only from Layer 0
from flext_core.constants import FlextConstants
from flext_core.typings import t
```

### Anti-Pattern 8: Multiple Exports per Module

**Problem**: Single module exports multiple public classes.

```python
# ❌ ANTI-PATTERN - Multiple exports violates architecture
# flext_core/models.py
class FlextModels:
    pass

class DomainModel:  # Second export - WRONG!
    pass

class ValueObject:  # Third export - WRONG!
    pass

# In __init__.py
from flext_core.models import FlextModels, DomainModel, ValueObject
# Violates single class per module rule
```

**Why it's wrong**:

- Breaks modularity principle
- Makes dependencies unclear
- Hard to find where things are defined
- Violates FLEXT architecture standard

**Solution**: One public class per module

```python
# ✅ CORRECT - Single export per module
# flext_core/models.py
class FlextModels:
    """Single main class per module."""
    class Value:
        """Nested helper - OK."""
        pass

    class Entity:
        """Nested helper - OK."""
        pass

# In __init__.py
from flext_core.models import FlextModels
# Clear, single responsibility
```

### Anti-Pattern 9: God Objects

**Problem**: Single class doing too much.

```python
# ❌ ANTI-PATTERN - God object (3,000+ lines)
class FlextMeltano:
    """Everything in one class - config, validation, services, streams..."""

    def __init__(self, config_path: str):
        pass

    def validate_config(self):
        pass

    def load_streams(self):
        pass

    def run_tap(self):
        pass

    def run_target(self):
        pass

    def run_dbt(self):
        pass

    # ... 100+ more methods
```

**Why it's wrong**:

- Hard to understand
- Hard to test
- Hard to reuse
- Violates Single Responsibility Principle

**Solution**: Decompose into focused classes

```python
# ✅ CORRECT - Separated concerns
class MeltanoConfig:
    """Handles configuration only."""
    def load(self, path: str) -> FlextResult[dict]:
        pass

class MeltanoValidator:
    """Handles validation only."""
    def validate_config(self, config: dict) -> FlextResult[None]:
        pass

class MeltanoStreamManager:
    """Handles stream operations."""
    def load_streams(self, config: dict) -> FlextResult[list]:
        pass

class MeltanoExecutor:
    """Handles execution (tap, target, dbt)."""
    def run_tap(self, config: dict) -> FlextResult[None]:
        pass
```

---

## Dependency Injection Anti-Patterns

### Anti-Pattern 10: Creating New Containers

**Problem**: Creating multiple container instances.

```python
# ❌ ANTI-PATTERN - Multiple containers
def service_a():
    container = FlextContainer()  # New container
    return container.get("logger")

def service_b():
    container = FlextContainer()  # DIFFERENT container!
    return container.get("logger")

# service_a and service_b get different logger instances!
```

**Why it's wrong**:

- Defeats singleton pattern
- Each container has its own services
- Different code paths get different instances
- State is not shared

**Solution**: Use global singleton

```python
# ✅ CORRECT - Global singleton
def service_a():
    container = FlextContainer.get_global()  # Same instance
    return container.get("logger")

def service_b():
    container = FlextContainer.get_global()  # Same instance!
    return container.get("logger")

# Both get same logger instance
assert service_a() is service_b()
```

### Anti-Pattern 11: Not Checking Container Results

**Problem**: Assuming service retrieval succeeds.

```python
# ❌ ANTI-PATTERN - No error handling
from flext_core import FlextContainer

container = FlextContainer.get_global()
logger = container.get("logger").unwrap()  # May crash
service = container.get("non_existent").unwrap()  # CRASH!
```

**Why it's wrong**:

- Code assumes service exists
- Runtime crashes on missing service
- No error context
- Hard to debug

**Solution**: Check results

```python
# ✅ CORRECT - Explicit error handling
from flext_core import FlextContainer

container = FlextContainer.get_global()

# Check result
logger_result = container.get("logger")
if logger_result.is_failure:
    print(f"Logger not available: {logger_result.error}")
    return FlextResult[None].fail("Logger unavailable")

logger = logger_result.unwrap()
logger.info("Service started")
```

---

## Model Anti-Patterns

### Anti-Pattern 12: Validation in Models Without FlextResult

**Problem**: Models validate but don't report errors properly.

```python
# ❌ ANTI-PATTERN - Validation via exceptions
from pydantic import BaseModel

class User(BaseModel):
    email: str
    age: int

    @field_validator("email")
    def validate_email(cls, v):
        if "@" not in v:
            raise ValueError("Invalid email")  # Pydantic exception
        return v

    @field_validator("age")
    def validate_age(cls, v):
        if v < 0:
            raise ValueError("Age must be positive")  # Pydantic exception
        return v

# Usage - Pydantic raises ValidationError
try:
    user = User(email="invalid", age=-5)
except ValidationError as e:
    print(f"Validation failed: {e}")
```

**Solution**: Wrap Pydantic validation in FlextResult

```python
# ✅ CORRECT - FlextResult for validation
from flext_core import FlextResult
from pydantic import BaseModel, ValidationError

class User(BaseModel):
    email: str
    age: int

def create_user(data: dict) -> FlextResult[User]:
    """Create user with FlextResult validation."""
    try:
        user = User(**data)
        return FlextResult[User].ok(user)
    except ValidationError as e:
        return FlextResult[User].fail(
            f"User validation failed: {e}",
            error_code="USER_VALIDATION_ERROR",
            error_data={"validation_errors": str(e)},
        )

# Usage
result = create_user({"email": "invalid", "age": -5})
if result.is_success:
    user = result.unwrap()
else:
    print(f"User creation failed: {result.error}")
```

### Anti-Pattern 13: Mutable Value Objects

**Problem**: Value objects that can be modified.

```python
# ❌ ANTI-PATTERN - Mutable value object
from flext_core import FlextModels

class Money(FlextModels.Value):
    amount: float  # Mutable!
    currency: str

money = Money(amount=100.0, currency="USD")
money.amount = 50.0  # Can be changed - violates value semantics!

# Now logic breaks
if money1 == money2:
    # Are they really equal? amount might have been modified elsewhere
    pass
```

**Why it's wrong**:

- Value semantics require immutability
- Modification breaks equality
- Shared references cause problems
- Unexpected side effects

**Solution**: Mark value objects as frozen

```python
# ✅ CORRECT - Immutable value object
from flext_core import FlextModels
from pydantic import ConfigDict
from decimal import Decimal

class Money(FlextModels.Value):
    model_config = ConfigDict(frozen=True)  # Immutable
    amount: Decimal
    currency: str

money = Money(amount=Decimal("100"), currency="USD")
money.amount = Decimal("50")  # TypeError: frozen object cannot be modified

# Now safe - value objects can't be modified
```

---

## Configuration Anti-Patterns

### Anti-Pattern 14: Hardcoded Configuration

**Problem**: Configuration values hardcoded in source.

```python
# ❌ ANTI-PATTERN - Hardcoded config
def connect_database():
    # Hardcoded values - can't change per environment!
    connection = psycopg2.connect(
        host="localhost",  # Hardcoded!
        port=5432,  # Hardcoded!
        database="flext_dev",  # Hardcoded!
        user="admin",  # SECURITY ISSUE!
        password="secret123",  # SECURITY ISSUE!
    )
    return connection
```

**Why it's wrong**:

- Can't change per environment
- Secrets in source code
- Hard to deploy to different environments
- Security risk

**Solution**: Use configuration management

```python
# ✅ CORRECT - Configuration from environment
from flext_core import FlextConfig
from pydantic_settings import BaseSettings, SettingsConfigDict

class DatabaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = "localhost"
    port: int = 5432
    database: str
    user: str
    password: str  # SecretStr recommended

db_config = DatabaseConfig()  # Loads from environment variables

def connect_database():
    connection = psycopg2.connect(
        host=db_config.host,
        port=db_config.port,
        database=db_config.database,
        user=db_config.user,
        password=str(db_config.password),
    )
    return connection
```

Usage:

```bash
# Development
DB_HOSTING=localhost DB_PORT=5432 DB_DATABASE=flext_dev \
DB_USER=admin DB_PASSWORD=dev_pass python app.py

# Production
DB_HOST=prod.db.example.com DB_PORT=5432 DB_DATABASE=flext_prod \
DB_USER=prod_user DB_PASSWORD=$DB_PASSWORD python app.py
```

### Anti-Pattern 15: No Configuration Validation

**Problem**: Configuration not validated.

```python
# ❌ ANTI-PATTERN - No validation
config = {
    "timeout": "not_a_number",  # Should be int
    "log_level": "INVALID",  # Should be DEBUG/INFO/WARNING/ERROR
    "api_key": "",  # Should be non-empty
}

# Later in code - crashes with cryptic error
time.sleep(config["timeout"])  # TypeError: float argument required
```

**Solution**: Validate configuration

```python
# ✅ CORRECT - Validated configuration
from flext_core import FlextConfig
from pydantic import BaseSettings, Field, field_validator

class AppConfig(BaseSettings):
    timeout: int = Field(gt=0, description="Timeout in seconds")
    log_level: str = Field(pattern="^(DEBUG|INFO|WARNING|ERROR)$")
    api_key: str = Field(min_length=1, description="API key required")

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v):
        if v > 3600:
            raise ValueError("Timeout cannot exceed 1 hour")
        return v

# Pydantic validates automatically on construction
try:
    config = AppConfig(
        timeout="not_a_number",  # Invalid
        log_level="INVALID",  # Invalid
        api_key="",  # Invalid
    )
except ValidationError as e:
    print(f"Config validation failed:\n{e}")
    # Clear errors, easy to fix
```

---

## Summary: Anti-Pattern Checklist

### Error Handling

- ✅ Use `FlextResult` for business logic errors
- ✅ Include error codes and metadata
- ✅ Never swallow errors silently
- ❌ Don't use exceptions for normal errors
- ❌ Don't ignore error information

### Type Safety

- ✅ Use specific types, not `Any`
- ✅ Use `get_typed()` for container retrieval
- ✅ Add type hints to all parameters
- ❌ Don't use `type: ignore` without justification
- ❌ Don't lose type information

### Architecture

- ✅ Respect layer hierarchy (downward only)
- ✅ One class per module with `Flext` prefix
- ✅ Import from `__init__.py`, not internal modules
- ❌ Don't create circular dependencies
- ❌ Don't create god objects

### Dependency Injection

- ✅ Use `FlextContainer.get_global()`
- ✅ Check `FlextResult` before using services
- ✅ Register services during initialization
- ❌ Don't create multiple containers
- ❌ Don't assume service exists

### Models

- ✅ Use `FlextModels` for DDD patterns
- ✅ Wrap validation in `FlextResult`
- ✅ Make value objects immutable
- ❌ Don't mix mutable and value semantics
- ❌ Don't validate without error handling

### Configuration

- ✅ Use `pydantic_settings.BaseSettings`
- ✅ Validate configuration on load
- ✅ Use environment variables
- ❌ Don't hardcode configuration
- ❌ Don't skip configuration validation

---

## See Also

- [Railway-Oriented Programming](./railway-oriented-programming.md)
- [Clean Architecture](../architecture/clean-architecture.md)
- [Development Standards](../standards/development.md)
- **FLEXT CLAUDE.md**: Architecture principles and zero-tolerance standards

---

**Updated**: 2025-10-21 | **Version**: 0.9.9 | **Based on**: Actual FLEXT ecosystem patterns and lessons learned
