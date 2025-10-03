# FlextExceptions - Exception Handling Guide

**Version**: 1.0.0
**Authority**: flext-core Foundation Library
**Last Updated**: 2025-10-02
**Status**: Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Exception Hierarchy](#exception-hierarchy)
4. [Usage Patterns](#usage-patterns)
5. [Migration Guide](#migration-guide)
6. [Best Practices](#best-practices)
7. [Ecosystem Integration](#ecosystem-integration)

---

## Overview

FlextExceptions provides a structured exception hierarchy for the FLEXT ecosystem, replacing generic Python exceptions with context-rich, traceable errors. All exceptions include error codes, correlation IDs, and structured context for improved debugging and monitoring.

### Why FlextExceptions

**Before** (Generic Exceptions):

```python
raise ValueError("Invalid email format")
# ❌ No error code
# ❌ No correlation ID
# ❌ No structured context
# ❌ Hard to monitor/trace
```

**After** (FlextExceptions):

```python
raise FlextExceptions.ValidationError(
    "Invalid email format",
    field="email",
    value="invalid@",
)
# ✅ Error code: VALIDATION_ERROR
# ✅ Correlation ID: UUID for tracing
# ✅ Structured context: field + value
# ✅ Easy to monitor/trace
```

### Key Benefits

- ✅ **Error Codes**: Programmatic error classification and handling
- ✅ **Correlation IDs**: Distributed tracing across services
- ✅ **Structured Context**: Rich debugging information
- ✅ **Backward Compatible**: Inherits from Python exceptions
- ✅ **Ecosystem Consistency**: Unified patterns across 32+ projects

---

## Quick Start

### Basic Import

```python
from flext_core import FlextExceptions
```

### Simple Usage

```python
# Validation error
raise FlextExceptions.ValidationError(
    "Email is required",
    field="email",
)

# Configuration error
raise FlextExceptions.ConfigurationError(
    "Database URL not configured",
    config_key="database_url",
    config_file="settings.yaml",
)

# Not found error
raise FlextExceptions.NotFoundError(
    "User not found",
    resource_type="User",
    resource_id="user-123",
)

# Type error
raise FlextExceptions.TypeError(
    "Expected string, got integer",
    expected_type="str",
    actual_type="int",
)
```

### Catching Exceptions

```python
# Catch specific FlextException
try:
    validate_email(email)
except FlextExceptions.ValidationError as e:
    logger.error(
        f"Validation failed: {e.message}",
        error_code=e.error_code,
        correlation_id=e.correlation_id,
        field=e.field,
        value=e.value,
    )

# Backward compatibility - still works with generic exceptions
try:
    validate_email(email)
except ValueError as e:  # FlextExceptions.ValidationError inherits from ValueError
    logger.error(f"Validation failed: {e}")
```

---

## Exception Hierarchy

### FlextExceptions Structure

```
FlextExceptions (Base Exception)
│
├── ValidationError (inherits from ValueError)
│   ├── Use for: Input validation, data validation, business rules
│   └── Context: field, value, validation_details
│
├── ConfigurationError (inherits from KeyError)
│   ├── Use for: Configuration issues, missing settings
│   └── Context: config_key, config_file, config_section
│
├── NotFoundError (inherits from FileNotFoundError)
│   ├── Use for: Missing resources, entities not found
│   └── Context: resource_type, resource_id, resource_path
│
├── TypeError (inherits from TypeError)
│   ├── Use for: Type mismatches, invalid conversions
│   └── Context: expected_type, actual_type, field
│
├── OperationError (inherits from RuntimeError)
│   ├── Use for: Operation failures, runtime errors
│   └── Context: operation, state, details
│
└── TimeoutError (inherits from TimeoutError)
    ├── Use for: Operation timeouts, deadline exceeded
    └── Context: timeout_seconds, operation, elapsed_time
```

### Common Context Fields

All FlextExceptions support these fields:

| Field            | Type     | Description                         | Example                                |
| ---------------- | -------- | ----------------------------------- | -------------------------------------- |
| `error_code`     | str      | Auto-generated error classification | "VALIDATION_ERROR"                     |
| `correlation_id` | UUID     | Unique ID for distributed tracing   | "550e8400-e29b-41d4-a716-446655440000" |
| `message`        | str      | Human-readable error message        | "Email is required"                    |
| `field`          | str      | Field/attribute name that failed    | "email"                                |
| `value`          | Any      | Actual value that caused the error  | "invalid@"                             |
| `timestamp`      | datetime | When the error occurred             | "2025-10-02T10:30:00Z"                 |

### Exception-Specific Context

**ValidationError**:

- `field`: Field name that failed validation
- `value`: Actual value that failed
- `validation_details`: Additional validation context

**ConfigurationError**:

- `config_key`: Configuration key name
- `config_file`: Configuration file path
- `config_section`: Configuration section name

**NotFoundError**:

- `resource_type`: Type of resource not found
- `resource_id`: Resource identifier
- `resource_path`: File/path not found

**TypeError**:

- `expected_type`: Expected type (str, int, etc.)
- `actual_type`: Actual type received
- `field`: Field name with type mismatch

---

## Usage Patterns

### 1. Input Validation

```python
from flext_core import FlextExceptions, FlextResult

def validate_user_input(data: dict) -> FlextResult[dict]:
    """Validate user input with structured error handling."""

    # Required field validation
    if not data.get("email"):
        raise FlextExceptions.ValidationError(
            "Email is required",
            field="email",
            value=data.get("email"),
        )

    # Format validation
    if "@" not in data["email"]:
        raise FlextExceptions.ValidationError(
            "Invalid email format",
            field="email",
            value=data["email"],
            validation_details="Email must contain @ symbol",
        )

    # Type validation
    if not isinstance(data.get("age"), int):
        raise FlextExceptions.TypeError(
            "Age must be an integer",
            field="age",
            expected_type="int",
            actual_type=type(data.get("age")).__name__,
        )

    return FlextResult[dict].ok(data)
```

### 2. Configuration Management

```python
from flext_core import FlextExceptions
import yaml

def load_configuration(config_file: str) -> dict:
    """Load configuration with structured error handling."""

    # File existence check
    if not os.path.exists(config_file):
        raise FlextExceptions.NotFoundError(
            f"Configuration file not found: {config_file}",
            resource_type="configuration_file",
            resource_path=config_file,
        )

    # Parse configuration
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Required configuration validation
    required_keys = ["database_url", "api_key", "environment"]
    for key in required_keys:
        if key not in config:
            raise FlextExceptions.ConfigurationError(
                f"Missing required configuration: {key}",
                config_key=key,
                config_file=config_file,
            )

    return config
```

### 3. Resource Management

```python
from flext_core import FlextExceptions, FlextResult

class UserRepository:
    """User repository with FlextExceptions error handling."""

    def get_user(self, user_id: str) -> FlextResult[User]:
        """Get user by ID with structured error handling."""

        # Query database
        user = self.db.query(User).filter_by(id=user_id).first()

        if not user:
            raise FlextExceptions.NotFoundError(
                f"User not found: {user_id}",
                resource_type="User",
                resource_id=user_id,
            )

        return FlextResult[User].ok(user)

    def create_user(self, data: dict) -> FlextResult[User]:
        """Create user with validation."""

        # Validate required fields
        if not data.get("username"):
            raise FlextExceptions.ValidationError(
                "Username is required",
                field="username",
                value=data.get("username"),
            )

        # Check for duplicates
        existing_user = self.db.query(User).filter_by(
            username=data["username"]
        ).first()

        if existing_user:
            raise FlextExceptions.ValidationError(
                f"Username already exists: {data['username']}",
                field="username",
                value=data["username"],
                validation_details="Username must be unique",
            )

        # Create user
        user = User(**data)
        self.db.add(user)
        self.db.commit()

        return FlextResult[User].ok(user)
```

### 4. Type Validation

```python
from flext_core import FlextExceptions
from typing import TypeVar, Generic

T = TypeVar('T')

class TypeValidator(Generic[T]):
    """Type validator with FlextExceptions."""

    def validate_type(self, value: object, expected_type: type[T]) -> T:
        """Validate value type with structured error handling."""

        if not isinstance(value, expected_type):
            raise FlextExceptions.TypeError(
                f"Expected {expected_type.__name__}, got {type(value).__name__}",
                expected_type=expected_type.__name__,
                actual_type=type(value).__name__,
                value=value,
            )

        return value
```

### 5. Operation Timeouts

```python
from flext_core import FlextExceptions, FlextResult
import time

def wait_for_service(timeout_seconds: int = 30) -> FlextResult[None]:
    """Wait for service with timeout handling."""

    start_time = time.time()

    while True:
        if service_is_ready():
            return FlextResult[None].ok(None)

        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise FlextExceptions.TimeoutError(
                f"Service not ready after {timeout_seconds}s",
                timeout_seconds=timeout_seconds,
                operation="wait_for_service",
                elapsed_time=elapsed,
            )

        time.sleep(1)
```

---

## Migration Guide

### Step 1: Identify Generic Exceptions

```bash
# Find all generic exceptions in your codebase
grep -r "raise ValueError\|raise KeyError\|raise TypeError" src/ --include="*.py"
```

### Step 2: Map to FlextExceptions

| Generic Exception   | FlextException       | Use Case               |
| ------------------- | -------------------- | ---------------------- |
| `ValueError`        | `ValidationError`    | Input/data validation  |
| `KeyError`          | `ConfigurationError` | Configuration/settings |
| `FileNotFoundError` | `NotFoundError`      | Missing resources      |
| `TypeError`         | `TypeError`          | Type mismatches        |
| `RuntimeError`      | `OperationError`     | Operation failures     |
| `TimeoutError`      | `TimeoutError`       | Timeout scenarios      |

### Step 3: Add Import

```python
# Add to your module imports
from flext_core import FlextExceptions
```

### Step 4: Replace Exceptions

**Before**:

```python
def validate_email(email: str) -> None:
    if not email:
        raise ValueError("Email is required")
    if "@" not in email:
        raise ValueError("Invalid email format")
```

**After**:

```python
def validate_email(email: str) -> None:
    if not email:
        raise FlextExceptions.ValidationError(
            "Email is required",
            field="email",
            value=email,
        )
    if "@" not in email:
        raise FlextExceptions.ValidationError(
            "Invalid email format",
            field="email",
            value=email,
            validation_details="Email must contain @ symbol",
        )
```

### Step 5: Update Exception Handlers

**Before**:

```python
try:
    validate_email(email)
except ValueError as e:
    logger.error(f"Validation failed: {e}")
```

**After** (Enhanced):

```python
try:
    validate_email(email)
except FlextExceptions.ValidationError as e:
    logger.error(
        "Validation failed",
        error_code=e.error_code,
        correlation_id=e.correlation_id,
        field=e.field,
        value=e.value,
    )
```

**After** (Backward Compatible):

```python
try:
    validate_email(email)
except ValueError as e:  # Still works!
    logger.error(f"Validation failed: {e}")
```

### Step 6: Validate Migration

```bash
# Run tests
poetry run pytest tests/ -v

# Check for undefined names
ruff check src/ --select F821,F401

# Verify coverage maintained
poetry run pytest --cov=src --cov-fail-under=75
```

---

## Best Practices

### 1. Always Provide Context

❌ **Bad** - No context:

```python
raise FlextExceptions.ValidationError("Invalid input")
```

✅ **Good** - Rich context:

```python
raise FlextExceptions.ValidationError(
    "Invalid input",
    field="email",
    value=user_input,
    validation_details="Email format required",
)
```

### 2. Use Specific Exception Types

❌ **Bad** - Generic exception:

```python
raise FlextExceptions.OperationError("Something failed")
```

✅ **Good** - Specific exception:

```python
raise FlextExceptions.ValidationError(
    "Email validation failed",
    field="email",
    value=email,
)
```

### 3. Include Error Details

❌ **Bad** - Vague message:

```python
raise FlextExceptions.ConfigurationError("Config error")
```

✅ **Good** - Detailed message:

```python
raise FlextExceptions.ConfigurationError(
    "Database URL not configured in settings.yaml",
    config_key="database_url",
    config_file="settings.yaml",
    config_section="database",
)
```

### 4. Catch Specific Exceptions

❌ **Bad** - Catch all:

```python
try:
    operation()
except Exception as e:
    pass
```

✅ **Good** - Catch specific:

```python
try:
    operation()
except FlextExceptions.ValidationError as e:
    logger.error("Validation failed", extra={"error": e})
except FlextExceptions.ConfigurationError as e:
    logger.error("Configuration error", extra={"error": e})
```

### 5. Log Structured Context

❌ **Bad** - String formatting:

```python
except FlextExceptions.ValidationError as e:
    logger.error(f"Error: {e}")
```

✅ **Good** - Structured logging:

```python
except FlextExceptions.ValidationError as e:
    logger.error(
        "Validation failed",
        error_code=e.error_code,
        correlation_id=e.correlation_id,
        field=e.field,
        value=e.value,
    )
```

### 6. Maintain Backward Compatibility

✅ **Good** - Both patterns work:

```python
# New code uses FlextExceptions
try:
    new_operation()
except FlextExceptions.ValidationError as e:
    handle_error(e)

# Legacy code still works
try:
    legacy_operation()
except ValueError as e:  # FlextExceptions.ValidationError inherits from ValueError
    handle_error(e)
```

---

## Ecosystem Integration

### Domain Libraries

All FLEXT domain libraries should use FlextExceptions:

**flext-api** (HTTP operations):

```python
from flext_core import FlextExceptions

class ApiClient:
    def make_request(self, url: str) -> FlextResult[dict]:
        if not url:
            raise FlextExceptions.ValidationError(
                "URL is required",
                field="url",
                value=url,
            )
        # ... rest of implementation
```

**flext-ldap** (LDAP operations):

```python
from flext_core import FlextExceptions

class LdapClient:
    def connect(self, server: str) -> FlextResult[None]:
        if not server:
            raise FlextExceptions.ConfigurationError(
                "LDAP server not configured",
                config_key="ldap_server",
            )
        # ... rest of implementation
```

**flext-cli** (CLI operations):

```python
from flext_core import FlextExceptions

class CliHandler:
    def validate_args(self, args: dict) -> FlextResult[dict]:
        if not args.get("command"):
            raise FlextExceptions.ValidationError(
                "Command is required",
                field="command",
                value=args.get("command"),
            )
        # ... rest of implementation
```

### Error Monitoring Integration

**Prometheus Metrics**:

```python
from prometheus_client import Counter

error_counter = Counter(
    'flext_errors_total',
    'Total errors by type',
    ['error_code', 'error_type']
)

try:
    operation()
except FlextExceptions.ValidationError as e:
    error_counter.labels(
        error_code=e.error_code,
        error_type='ValidationError'
    ).inc()
    raise
```

**Distributed Tracing**:

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

try:
    with tracer.start_as_current_span("operation") as span:
        operation()
except FlextExceptions.ValidationError as e:
    span.set_attribute("error.code", e.error_code)
    span.set_attribute("error.correlation_id", str(e.correlation_id))
    span.set_attribute("error.field", e.field)
    raise
```

---

## Summary

FlextExceptions provides:

✅ **Structured error handling** with error codes and correlation IDs
✅ **Rich context** for debugging and monitoring
✅ **Backward compatibility** with Python exceptions
✅ **Ecosystem consistency** across 32+ projects
✅ **Production ready** - battle-tested in flext-core v0.9.9

**Get Started**: Import from flext-core and start using today!

```python
from flext_core import FlextExceptions

# Your journey to better error handling starts here
```

---

**Document Version**: 1.0.0
**Foundation**: flext-core v0.9.9
**Authority**: FLEXT Foundation Library
**Status**: Production Ready
