# FLEXT-Core

<!-- TOC START -->

- [🚀 Key Features](#-key-features)
- [📦 Installation](#-installation)
- [🛠️ Usage](#-usage)
  - [Railway-Oriented Results](#railway-oriented-results)
  - [Dependency Injection](#dependency-injection)
  - [CQRS Dispatching](#cqrs-dispatching)
  - [Services & Domain Logic](#services--domain-logic)
  - [Settings & Configuration](#settings--configuration)
- [🏗️ Architecture](#-architecture)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

<!-- TOC END -->

**FLEXT-Core** is the foundational library for the FLEXT ecosystem, providing architectural primitives for enterprise Python 3.13+ applications. It enforces:
- **Railway-Oriented Programming (ROP)**: Error handling as values via `r[T]`
- **Dependency Injection (DI)**: Type-safe container with scoped lifetimes
- **CQRS Dispatching**: Typed command/query routing with event support
- **Domain-Driven Design**: Pydantic v2-based entities, aggregates, commands, and events
- **Protocol-Based Architecture**: Structural typing for loose coupling

**Reviewed**: 2026-04-14 | **Version**: 0.12.0-dev | **Python**: 3.13+

Part of the [FLEXT](https://github.com/flext-sh/flext) ecosystem.

## 🚀 Key Features

- **Railway-Oriented Programming (`r[T]`)**: Error handling as values instead of exceptions; chainable operations via `map()`, `flat_map()`, `lash()`.
- **Dependency Injection (`FlextContainer`)**: Singleton container with factory, scoped, and resource registration; thread-safe initialization; `r` result-bearing API.
- **CQRS Dispatcher**: Typed message routing for commands, queries, and events; handler auto-discovery; reliability policies.
- **Type-Safe Services (`s[T]`)**: Base class for domain services with Pydantic v2 validation, DI integration, and runtime bootstrap.
- **Protocol-Based Contracts**: 10+ runtime-checkable protocols (`p.*`) for loose coupling: Result, Container, Service, Handler, Settings, Context, Logging.
- **Comprehensive Utilities (`u.*`)**: 20+ utility modules covering parsing, validation, collection handling, context management, guards, and more.
- **Structured Settings**: `FlextSettings` with Pydantic BaseSettings, env override, MRO-based inheritance, singleton pattern.
- **Lazy Export System**: Auto-generated `__init__.py` with lazy loading of 90+ symbols across 6 submódule trees.

## 📦 Installation

Install `flext-core` using pip:

```bash
pip install flext-core
```

Or with Poetry:

```bash
poetry add flext-core
```

## 🛠️ Usage

### Railway-Oriented Results

Use `r[T]` to encapsulate success/failure without exceptions.

```python
from flext_core import r


def divide(a: int, b: int) -> r[float]:
    if b == 0:
        return r[float].fail("Division by zero")
    return r[float].ok(a / b)


result = divide(10, 2)
if result.success:
    print(f"Result: {result.unwrap()}")  # 5.0
else:
    print(f"Error: {result.error}")

# Chain operations
result = (
    r[int].ok(10).map(lambda x: x * 2).flat_map(lambda x: r[int].ok(x + 5)).unwrap_or(0)
)
```

### Dependency Injection

Register and resolve services using `FlextContainer`.

```python
from flext_core import FlextContainer, r

# Get global container
container = FlextContainer.get_global()

# Register a factory
container.factory("database", lambda: MyDatabase())

# Resolve with error handling
db_result = container.resolve("database")
if db_result.success:
    db = db_result.unwrap()
    # Use db...
else:
    print(f"Failed to resolve: {db_result.error}")

# Scoped container
with container.scope() as scoped:
    # Scoped services live only within this context
    user_service = scoped.resolve("user_service").unwrap()
```

### CQRS Dispatching

Route commands and queries with the dispatcher.

```python
from pydantic import BaseModel
from flext_core import FlextDispatcher, r, m


# Define command
class CreateUser(m.Entity):
    username: str
    email: str


# Define handler
def handle_create_user(cmd: CreateUser) -> r[str]:
    return r[str].ok(f"User {cmd.username} created")


# Dispatch
dispatcher = FlextDispatcher()
dispatcher.register_handler(CreateUser, handle_create_user)

result = dispatcher.dispatch(CreateUser(username="alice", email="alice@example.com"))
```

### Services & Domain Logic

Create services with DI and result-bearing methods.

```python
from flext_core import s, r, m


class UserService(s):
    """Domain service for user operations."""

    def create_user(self, username: str) -> r[str]:
        # Access container
        db = self.container.resolve("database").unwrap()

        # Validate
        if not username:
            return r[str].fail("Username required")

        # Execute
        db.insert({"username": username})
        return r[str].ok(username)


# Use
svc = UserService()
result = svc.create_user("alice")
```

### Settings & Configuration

Manage typed configuration with `FlextSettings`.

```python
from pydantic_settings import BaseSettings
from flext_core import FlextSettings, c


class AppSettings(FlextSettings):
    """Application settings with env override."""

    database_url: str = "sqlite://app.db"
    debug: bool = False
    api_key: str = ""

    class Config:
        env_prefix = "FLEXT_APP_"  # Read from FLEXT_APP_DATABASE_URL, etc.


# Get settings singleton
settings = FlextSettings.get_global()
db_url = settings.database_url
```

## 🏗️ Architecture

FLEXT-Core is organized in **Inward-Only Dependency Flow** (`L3 -> L2 -> L1 -> L0`):

```
┌──────────────────────────────────────────────────┐
│ L3: Application (Dispatcher, Services)           │
├──────────────────────────────────────────────────┤
│ L2: Domain (Models, Commands, Queries, Events)   │
├──────────────────────────────────────────────────┤
│ L1: Foundation (Result, Container, Logger)       │
├──────────────────────────────────────────────────┤
│ L0: Contracts (Protocols, BaseModel, TypeGuards) │
└──────────────────────────────────────────────────┘
```

### Core Modules

| Module | Purpose | Key Classes |
|--------|---------|------------|
| `result.py` | Railway-like result type | `FlextResult[T]`, `r` |
| `container.py` | DI singleton container | `FlextContainer` |
| `dispatcher.py` | CQRS message router | `FlextDispatcher` |
| `service.py` | Service base class | `FlextService[T]`, `s` |
| `settings.py` | Configuration management | `FlextSettings` |
| `protocols.py` | Structural typing contracts | `p.*` (10+ protocols) |
| `models.py` | DDD building blocks | `m.*` (entities, aggregates, etc.) |
| `utilities.py` | Helper functions | `u.*` (20+ utility modules) |
| `constants.py` | Global constants | `c.*` (defaults, error codes) |
| `exceptions.py` | Domain exceptions | `e.*` (exception hierarchy) |
| `context.py` | Request context propagation | `FlextContext` |
| `loggings.py` | Structured logging | `FlextLogger`, `u.fetch_logger()` |

### Alias System

| Alias | Facade | Category |
|-------|--------|----------|
| `r` | `FlextResult` | Error handling |
| `c` | `FlextConstants` | Global defaults |
| `m` | `FlextModels` | Domain entities |
| `t` | `FlextTypes` | Type definitions |
| `p` | `FlextProtocols` | Abstract contracts |
| `u` | `FlextUtilities` | Helper functions |
| `e` | `FlextExceptions` | Exception types |
| `s` | `FlextService` | Service base |
| `d` | `FlextDecorators` | Decorator helpers |
| `h` | `FlextHandlers` | Handler protocols |
| `x` | `FlextMixins` | Mixin base classes |

### Lazy Loading & Exports

FLEXT-Core exports 90+ symbols via automatic lazy loading:
- `_constants/` → 11 submódules
- `_exceptions/` → 6 submódules  
- `_models/` → 17 submódules
- `_protocols/` → 9 submódules
- `_typings/` → 7 submódules
- `_utilities/` → 20 submódules

See [`__init__.py`](src/flext_core/__init__.py) for the complete export list.

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) to get started.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Documentation

For more detailed information:
- [Quick Start Guide](docs/quick-start.md)
- [Architecture Overview](docs/architecture/README.md)
- [API Reference](docs/api-reference/README.md)
- [Development Guide](docs/development/README.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
