# FLEXT-Core Guides

Comprehensive guides for each major component of FLEXT-Core.

**Status**: Updated 2026-04-14

## Core Components

### Railway Results (`r[T]`)
- **Purpose**: Error handling as values instead of exceptions
- **Key Methods**: `ok()`, `fail()`, `map()`, `flat_map()`, `unwrap()`
- **Example**: See [Quick Start — Railway Results](../quick-start.md#1-railway-results-rt)

### Dependency Injection (`FlextContainer`)
- **Purpose**: Type-safe service registration and resolution
- **Key Methods**: `factory()`, `service()`, `resource()`, `resolve()`, `scope()`
- **Example**: See [Quick Start — Dependency Injection](../quick-start.md#2-dependency-injection)

### CQRS Dispatcher
- **Purpose**: Route typed messages (commands/queries) to handlers
- **Key Methods**: `register_handler()`, `dispatch()`, `publish()`
- **Example**: See [Quick Start — CQRS Dispatcher](../quick-start.md#3-cqrs-dispatcher)

### Domain Services (`s`)
- **Purpose**: Base class for business logic with DI and validation
- **Key Features**: Pydantic v2, container access, bootstrap configuration
- **Example**: See [Quick Start — Services & Models](../quick-start.md#4-services--models)

### Domain Models (`m.*`)
- **Purpose**: DDD building blocks (Entity, Aggregate, Command, Query, Event)
- **All Models**: Inherit from Pydantic BaseModel with validation
- **Example**: See [Architecture — Domain Models](../architecture/README.md#domain-models)

### Settings & Configuration
- **Purpose**: Typed configuration with env override
- **Key Class**: `FlextSettings` (singleton, Pydantic BaseSettings)
- **Example**: See [Quick Start — Settings](../quick-start.md#example-4-settings--configuration)

### Utilities (`u.*`)
- **Purpose**: Reusable helper functions (parsers, guards, converters, loggers)
- **Access**: Flat namespace — `u.method()` (e.g., `u.fetch_logger()`)
- **Count**: 20+ utility modules

### Protocols (`p.*`)
- **Purpose**: Runtime-checkable abstract contracts
- **Types**: Result, Container, Service, Handler, Settings, Context, Logger
- **Usage**: `isinstance(obj, p.Service)` returns True if structurally compatible

## Quick Navigation

| Need | File |
|------|------|
| 5-min overview | [Quick Start](../quick-start.md) |
| System architecture | [Architecture](../architecture/README.md) |
| API reference | [API Reference](../api-reference/) |
| Development | [Development](../development/) |

## Practical Walkthroughs

- **Validation Pipeline** — See [Quick Start Example 1](../quick-start.md#example-1-data-validation-pipeline)
- **Error Recovery** — See [Quick Start Example 2](../quick-start.md#example-2-error-recovery)
- **Structured Logging** — See [Quick Start Example 3](../quick-start.md#example-3-structured-logging)
- **Settings & Config** — See [Quick Start Example 4](../quick-start.md#example-4-settings--configuration)

---

**Last Updated**: 2026-04-14 | **Status**: Current

