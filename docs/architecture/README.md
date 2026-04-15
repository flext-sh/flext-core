# FLEXT-Core Architecture


<!-- TOC START -->
- [Overview](#overview)
- [Dependency Flow (Inward-Only)](#dependency-flow-inward-only)
- [Core Modules](#core-modules)
  - [Result Handling (`result.py`)](#result-handling-resultpy)
  - [Dependency Injection (`container.py`)](#dependency-injection-containerpy)
  - [CQRS Dispatcher (`dispatcher.py`)](#cqrs-dispatcher-dispatcherpy)
  - [Domain Services (`service.py`)](#domain-services-servicepy)
  - [Domain Models (`models.py`)](#domain-models-modelspy)
  - [Configuration (`settings.py`)](#configuration-settingspy)
  - [Protocols (`protocols.py`)](#protocols-protocolspy)
  - [Utilities (`utilities.py`)](#utilities-utilitiespy)
  - [Constants (`constants.py`)](#constants-constantspy)
  - [Exceptions (`exceptions.py`)](#exceptions-exceptionspy)
- [Lazy Loading System](#lazy-loading-system)
- [See Also](#see-also)
<!-- TOC END -->

**Status**: Current (2026-04-14) | **Python**: 3.13+ | **Version**: 0.12.0-dev

## Overview

FLEXT-Core is organized around four foundational architectural principles:

1. **Railway-Oriented Programming** — Error handling as values via `r[T]`
2. **Dependency Injection** — Type-safe service container with scoped lifetimes
3. **CQRS & Event Sourcing** — Message dispatching with typed handlers
4. **Domain-Driven Design** — Pydantic v2 entities, aggregates, commands, queries

## Dependency Flow (Inward-Only)

```
┌──────────────────────────────────────────────────────┐
│ L3: Application                                      │
│     - Dispatcher, Services, Controllers              │
├──────────────────────────────────────────────────────┤
│ L2: Domain                                           │
│     - Models (Entity, Value, Aggregate)              │
│     - Commands, Queries, Events                      │
├──────────────────────────────────────────────────────┤
│ L1: Foundation                                       │
│     - Result (r[T]), Container, Logger               │
│     - Context, Settings, Utilities                   │
├──────────────────────────────────────────────────────┤
│ L0: Contracts                                        │
│     - Protocols (p.*), Pydantic BaseModel            │
│     - Type Guards, Constants                         │
└──────────────────────────────────────────────────────┘
```

**Key Rule**: Dependencies flow **inward only** (L3 → L2 → L1 → L0). No reverse imports.

## Core Modules

### Result Handling (`result.py`)

**Class**: `FlextResult[T]` (alias: `r`)

Provides Railway-Oriented Programming by wrapping success/failure states:

```python
from flext_core import r

# Success
result = r[int].ok(42)

# Failure
result = r[int].fail("Error message")

# Chain
result = r[int].ok(10).map(lambda x: x * 2).unwrap()
```

**Key Methods**:
- `ok(value)`, `fail(error)` — Constructor
- `success`, `error`, `error_code`, `error_data` — Inspect
- `map()`, `flat_map()`, `lash()` — Transform/chain
- `unwrap()`, `unwrap_or()`, `unwrap_type()` — Extract value

### Dependency Injection (`container.py`)

**Class**: `FlextContainer` (singleton)

Manages service registration and resolution with type safety:

```python
from flext_core import FlextContainer

container = FlextContainer.get_global()

# Register factory
container.factory("database", lambda: MyDatabase())

# Resolve
db = container.resolve("database").unwrap()

# Scoped container
with container.scope() as scoped:
    service = scoped.resolve("service").unwrap()
```

**Features**:
- Singleton pattern with double-checked locking
- Factory, service, and resource registration
- Scoped containers for short-lived instances
- Thread-safe initialization
- Result-bearing API (no exceptions)

### CQRS Dispatcher (`dispatcher.py`)

**Class**: `FlextDispatcher`

Routes typed messages (commands/queries) to handlers:

```python
from flext_core import FlextDispatcher, m

dispatcher = FlextDispatcher()

# Register handler
dispatcher.register_handler(CreateUserCommand, handle_create_user)

# Dispatch
result = dispatcher.dispatch(CreateUserCommand(username="alice"))
```

**Key Methods**:
- `register_handler(message_type, handler)` — Register handler
- `dispatch(message)` → `r[T]` — Execute handler
- `publish(event)` → `r[bool]` — Publish events

### Domain Services (`service.py`)

**Class**: `FlextService[TDomainResult]` (alias: `s`)

Base class for domain services with DI and result bearing:

```python
from flext_core import s, r, m


class UserService(s):
    def create_user(self, name: str) -> r[m.Entity]:
        # Access DI
        db = self.container.resolve("database").unwrap()

        # Business logic
        user = m.Entity(name=name)
        db.insert(user)

        return r[m.Entity].ok(user)
```

**Features**:
- DI-aware via `self.container`
- Pydantic v2 validation
- Service bootstrap with factories/resources
- Generic domain result type

### Domain Models (`models.py`)

**Facade**: `FlextModels` (alias: `m`)

DDD building blocks with Pydantic v2:

Types:
- `m.Entity` — Domain object with identity
- `m.Value` — Immutable value object
- `m.AggregateRoot` — Consistency boundary
- `m.Command` — Message for state change
- `m.Query` — Message for data retrieval
- `m.DomainEvent` — Recorded domain occurrence

All models inherit from Pydantic `BaseModel` with automatic validation.

### Configuration (`settings.py`)

**Class**: `FlextSettings`

Pydantic BaseSettings with env override and MRO inheritance:

```python
from flext_core import FlextSettings


class MySettings(FlextSettings):
    database_url: str = "sqlite://app.db"
    debug: bool = False

    class Config:
        env_prefix = "MY_APP_"


settings = FlextSettings.get_global()
```

**Features**:
- Singleton pattern with thread-safe initialization
- Environment variable override via `env_prefix`
- MRO-based fallback for inherited settings
- Pydantic v2 validation

### Protocols (`protocols.py`)

**Facade**: `FlextProtocols` (alias: `p`)

10+ structural typing contracts:

- `p.Result[T]` — Railway result carrier
- `p.Container` — DI container lifecycle
- `p.Service` — Service contract
- `p.Handler` — Message handler
- `p.Settings` — Configuration base
- `p.Context` — Request context
- `p.Logger` — Logging interface
- `p.Routable` — CQRS message
- `p.Entity` — Domain entity
- Plus 5+ more...

All protocols are runtime-checkable via `isinstance()`.

### Utilities (`utilities.py`)

**Facade**: `FlextUtilities` (alias: `u`)

20+ utility modules:

| Module | Purpose |
|--------|---------|
| `u.guards` | Type guards (TypeIs) |
| `u.parse` | String parsing |
| `u.map` | Dictionary mapping |
| `u.batch` | Batch operations |
| `u.fetch_logger()` | Get logger |
| `u.convert` | Type conversion |
| `u.discovery` | Module discovery |
| `u.enforce` | Runtime validation |

Access via flat namespace: `u.method()` (no subdivisions).

### Constants (`constants.py`)

**Facade**: `FlextConstants` (alias: `c`)

Global constants and defaults:

```python
from flext_core import c

# Error codes
c.ErrorCode.COMMAND_HANDLER_NOT_FOUND.value

# Defaults
c.ENV_PREFIX  # "FLEXT_"
c.DEFAULT_ENCODING  # "utf-8"
```

### Exceptions (`exceptions.py`)

**Facade**: `FlextExceptions` (alias: `e`)

Domain exception hierarchy:

- `e.FlextError` — Base exception
- `e.ValidationError` — Validation failures
- `e.HandlerNotFound` — Dispatch failures
- Plus domain-specific...

## Lazy Loading System

FLEXT-Core exports 90+ symbols via automatic lazy loading to reduce import time:

```python
# Lazy! Only loaded when accessed
from flext_core import m, r, c, p, u, e, s, h, d, t, x
```

Six submódule trees auto-generate exports:
- `_constants/` → 11 modules
- `_exceptions/` → 6 modules  
- `_models/` → 17 modules
- `_protocols/` → 9 modules
- `_typings/` → 7 modules
- `_utilities/` → 20 modules

## See Also

- [Clean Architecture Principles](clean-architecture.md)
- [CQRS & Event Sourcing](cqrs.md)
- [Design Patterns](patterns.md)
- [Architectural Decisions](decisions.md)
- [Modernization Roadmap](modernization-roadmap.md)

---

**Questions?** See [../quick-start.md](../quick-start.md) for practical examples.
