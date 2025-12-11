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

### FlextResult[T] — Railway-Oriented Programming {#flextresult}

Monadic success/failure handling used across services, handlers, and decorators.

```python
from flext_core import FlextResult

result = (
    FlextResult[int].ok(10)
    .map(lambda value: value * 2)
    .map(lambda value: f"Result: {value}")
)
```

### FlextContainer — Dependency Injection {#flextcontainer}

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

## Short Alias Reference

FLEXT-Core provides short aliases for frequently used types to keep code concise and readable:

```python
# ✅ CORRECT - Import short aliases from their modules
from flext_core.result import r       # FlextResult alias
from flext_core.typings import t      # FlextTypes alias
from flext_core.constants import c    # FlextConstants alias
from flext_core.models import m       # FlextModels alias
from flext_core.protocols import p    # FlextProtocols alias
from flext_core.utilities import u    # FlextUtilities alias
from flext_core.exceptions import e   # FlextExceptions alias
from flext_core.context import x      # FlextContext alias (via mixins)
from flext_core.service import s      # FlextService alias
from flext_core.decorators import d   # FlextDecorators alias
from flext_core.handlers import h     # FlextHandlers alias
```

**Usage Examples**:

```python
# Result operations
def process(value: str) -> r[str]:
    if not value:
        return r[str].fail("Empty value")
    return r[str].ok(value.upper())

# Type annotations
config_dict: t.ConfigurationDict = {"key": "value"}

# Constants
error_code = c.Errors.VALIDATION_ERROR

# Models
user = m.Entity(id="123", name="Alice")

# Protocols
if isinstance(service, p.Config):
    service.configure(config)

# Utilities
if u.chk().eq(value, expected):
    process(value)
```

**Complete Alias Reference**:

| Alias | Full Name         | Module       | Purpose                                      |
| ----- | ----------------- | ------------ | -------------------------------------------- |
| `r`   | `FlextResult`     | `result`     | Railway-oriented result type                 |
| `t`   | `FlextTypes`      | `typings`    | Type aliases and TypeVars                    |
| `c`   | `FlextConstants`  | `constants`  | Immutable constants and defaults             |
| `m`   | `FlextModels`     | `models`     | Domain models (Entity, Value, AggregateRoot) |
| `p`   | `FlextProtocols`  | `protocols`  | Runtime-checkable protocols                  |
| `u`   | `FlextUtilities`  | `utilities`  | General-purpose utility functions            |
| `e`   | `FlextExceptions` | `exceptions` | Exception hierarchy                          |
| `x`   | `FlextMixins`     | `mixins`     | Reusable mixin behaviors (context access)    |
| `s`   | `FlextService`    | `service`    | Domain service base class                    |
| `d`   | `FlextDecorators` | `decorators` | Cross-cutting decorators                     |
| `h`   | `FlextHandlers`   | `handlers`   | CQRS handler base class                      |

See the [Type System Guidelines](../../README.md#type-system-guidelines) in the main README for detailed usage patterns.

---

The foundation layers provide stable, dependency-light building blocks for dispatcher orchestration, domain modeling, and infrastructure integration.

## Related Documentation

**Within Project**:

- [Getting Started](../guides/getting-started.md) - Installation and basic usage
- [Railway-Oriented Programming](../guides/railway-oriented-programming.md) - FlextResult pattern
- [Dependency Injection Advanced](../guides/dependency-injection-advanced.md) - FlextContainer usage
- [Architecture Overview](../architecture/README.md) - System architecture and layering
- [Domain API Reference](./domain.md) - Domain layer APIs
- [Application API Reference](./application.md) - Application layer APIs

**External Resources**:

- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
