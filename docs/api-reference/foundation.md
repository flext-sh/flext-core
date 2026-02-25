# Foundation Layers API Reference

<!-- TOC START -->

- [Architecture Overview](#architecture-overview)
- [Layer 0: Pure Constants](#layer-0-pure-constants)
  - [FlextConstants - Centralized Defaults](#flextconstants-centralized-defaults)
  - [t — Type System](#t-type-system)
  - [p — Runtime Interfaces](#p-runtime-interfaces)
- [Layer 0.5: Runtime Bridge](#layer-05-runtime-bridge)
  - [FlextRuntime — External Library Integration](#flextruntime-external-library-integration)
- [Layer 1: Foundation (Core Primitives)](#layer-1-foundation-core-primitives)
  - [FlextResult[T] — Railway-Oriented Programming {#flextresult}](#flextresultt-railway-oriented-programming-flextresult)
  - [FlextContainer — Dependency Injection {#flextcontainer}](#flextcontainer-dependency-injection-flextcontainer)
  - [FlextExceptions — Exception Hierarchy](#flextexceptions-exception-hierarchy)
- [Short Alias Reference](#short-alias-reference)
- [Related Documentation](#related-documentation)
- [Verification Commands](#verification-commands)

<!-- TOC END -->

This reference covers Layers 0, 0.5, and 1: the primitives that support dispatcher-driven CQRS without leaking higher-layer dependencies.

Canonical references:

- `../architecture/overview.md`
- `../architecture/clean-architecture.md`
- `../../README.md`

## Architecture Overview

- **Layer 0** — Pure constants, typing helpers, and runtime protocols with zero dependencies.
- **Layer 0.5** — Runtime bridge that adapts external libraries without importing application or domain code.
- **Layer 1** — Core primitives (`FlextResult`, `FlextContainer`, `FlextExceptions`) that Domain and Application layers rely on.

See the Architecture Overview for the full layering model.

______________________________________________________________________

## Layer 0: Pure Constants

### FlextConstants - Centralized Defaults

Immutable defaults and identifiers with no runtime dependencies.

````python
from flext_core import c

# Error code and configuration defaults
error_code = c.Errors.VALIDATION_FAILED
request_timeout = c.Configuration.DEFAULT_TIMEOUT
email_pattern = c.Validation.EMAIL_PATTERN
```text

### t — Type System

Common type variables, aliases, and CQRS markers used throughout the codebase.

```python
from flext_core import t

T = t.T
U = t.U
TCommand = t.TCommand
TQuery = t.TQuery
TEvent = t.TEvent
```text

### p — Runtime Interfaces

Runtime-checkable protocols that keep boundary contracts explicit.

```python
from flext_core import p

if isinstance(service, p.Configurable):
    service.configure(config)
```text

______________________________________________________________________

## Layer 0.5: Runtime Bridge

### FlextRuntime — External Library Integration

Adapters for external libraries (for example, `structlog`) that stay isolated from the dispatcher and domain layers.

```python
from flext_core import FlextRuntime

if FlextRuntime.is_valid_email(email):
    process_email(email)

data = FlextRuntime.to_json_serializable(payload)
```text

______________________________________________________________________

## Layer 1: Foundation (Core Primitives)

### FlextResult[T] — Railway-Oriented Programming {#flextresult}

Monadic success/failure handling used across services, handlers, and decorators.

```python
from flext_core import r

result = (
    r[int].ok(10)
    .map(lambda value: value * 2)
    .map(lambda value: f"Result: {value}")
)
```text

### FlextContainer — Dependency Injection {#flextcontainer}

Lightweight DI container with explicit lifecycles that works cleanly with dispatcher-driven handlers.

```python
from flext_core import FlextContainer, FlextLogger

container = FlextContainer.get_global()
container.register("logger", FlextLogger(__name__), singleton=True)

logger_result = container.get("logger")
if logger_result.is_success:
    logger = logger_result.value
    logger.info("Application started")
```text

### FlextExceptions — Exception Hierarchy

Structured exception types with contextual metadata for infrastructure concerns.

```python
from flext_core import c, e

class ValidationException(e.BaseError):
    """Raised when domain validation fails."""

    def __init__(self, field: str, value: object):
        super().__init__(
            error_code=c.Errors.VALIDATION_ERROR,
            message=f"Invalid value for {field}: {value}",
            context={"field": field, "value": value},
        )
```text

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
```text

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
```text

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

See the Type System Guidelines in the main README for detailed usage patterns.

______________________________________________________________________

The foundation layers provide stable, dependency-light building blocks for dispatcher orchestration, domain modeling, and infrastructure integration.

## Related Documentation

**Within Project**:

- Getting Started - Installation and basic usage
- Railway-Oriented Programming - FlextResult pattern
- Dependency Injection Advanced - FlextContainer usage
- Architecture Overview - System architecture and layering
- Domain API Reference - Domain layer APIs
- Application API Reference - Application layer APIs

**External Resources**:

- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

## Verification Commands

Run from `flext-core/`:

```bash
make lint
make type-check
make test-fast
```text
````
