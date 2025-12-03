# Foundation Layers API Reference

This reference covers **Layers 0, 0.5, and 1**, the primitives that support dispatcher-driven CQRS without leaking higher-layer dependencies.

## Architecture Overview

- **Layer 0** — Pure constants, typing helpers, and runtime protocols with zero dependencies.
- **Layer 0.5** — Runtime bridge that adapts external libraries without importing application or domain code.
- **Layer 1** — Core primitives (`FlextResult`, `FlextContainer`, `FlextExceptions`) that Domain and Application layers rely on.

See the [Architecture Overview](../architecture/overview.md) for the full layering model.

---

## Layer 0: Pure Constants

### FlextConstants — Centralized Defaults

Immutable defaults and identifiers with no runtime dependencies.

```python
from flext_core import FlextConstants

# Error code and configuration defaults
error_code = FlextConstants.Errors.VALIDATION_FAILED
request_timeout = FlextConstants.Configuration.DEFAULT_TIMEOUT
email_pattern = FlextConstants.Validation.EMAIL_PATTERN
```

### t — Type System

Common type variables, aliases, and CQRS markers used throughout the codebase.

```python
from flext_core import t

T = t.T
U = t.U
TCommand = t.TCommand
TQuery = t.TQuery
TEvent = t.TEvent
```

### p — Runtime Interfaces

Runtime-checkable protocols that keep boundary contracts explicit.

```python
from flext_core import p

if isinstance(service, p.Configurable):
    service.configure(config)
```

---

## Layer 0.5: Runtime Bridge

### FlextRuntime — External Library Integration

Adapters for external libraries (for example, `structlog`) that stay isolated from the dispatcher and domain layers.

```python
from flext_core import FlextRuntime

if FlextRuntime.is_valid_email(email):
    process_email(email)

data = FlextRuntime.to_json_serializable(payload)
```

---

## Layer 1: Foundation (Core Primitives)

### FlextResult[T] — Railway-Oriented Programming

Monadic success/failure handling used across services, handlers, and decorators.

```python
from flext_core import FlextResult

result = (
    FlextResult[int].ok(10)
    .map(lambda value: value * 2)
    .map(lambda value: f"Result: {value}")
)
```

### FlextContainer — Dependency Injection

Lightweight DI container with explicit lifecycles that works cleanly with dispatcher-driven handlers.

```python
from flext_core import FlextContainer, FlextLogger

container = FlextContainer.get_global()
container.register("logger", FlextLogger(__name__), singleton=True)

logger_result = container.get("logger")
if logger_result.is_success:
    logger = logger_result.unwrap()
    logger.info("Application started")
```

### FlextExceptions — Exception Hierarchy

Structured exception types with contextual metadata for infrastructure concerns.

```python
from flext_core import FlextException, ErrorCode

class ValidationException(FlextException):
    """Raised when domain validation fails."""

    def __init__(self, field: str, value: object):
        super().__init__(
            error_code=ErrorCode.VALIDATION_ERROR,
            message=f"Invalid value for {field}: {value}",
            context={"field": field, "value": value},
        )
```

The foundation layers provide stable, dependency-light building blocks for dispatcher orchestration, domain modeling, and infrastructure integration.
