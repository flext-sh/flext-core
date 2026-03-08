# FLEXT-Core

<!-- TOC START -->

- [🚀 Key Features](#-key-features)
- [📦 Installation](#-installation)
- [🛠️ Usage](#-usage)
  - [Railway-Oriented Results](#railway-oriented-results)
  - [Dependency Injection](#dependency-injection)
  - [CQRS Dispatching](#cqrs-dispatching)
- [🏗️ Architecture](#-architecture)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

<!-- TOC END -->

**FLEXT-Core** is the foundational library for the FLEXT ecosystem, providing a robust set of architectural primitives, patterns, and utilities for building enterprise-grade Python applications. It enforces Railway-Oriented Programming (ROP), Dependency Injection (DI), and Command Query Responsibility Segregation (CQRS) to ensure type safety, scalability, and maintainability.

**Reviewed**: 2026-02-17 | **Version**: 0.10.0-dev

Part of the [FLEXT](https://github.com/flext-sh/flext) ecosystem.

## 🚀 Key Features

- **Railway-Oriented Programming**: handling errors as values using `FlextResult[T, E]`, eliminating unexpected exceptions in business logic.
- **Dependency Injection**: A lightweight, type-safe DI container (`FlextContainer`) with scoped services and bridge integration.
- **CQRS Dispatcher**: A strictly typed `FlextDispatcher` for routing commands, queries, and events to their respective handlers.
- **Domain-Driven Design**: Base classes (`FlextModels`, `FlextService`) and mixins for rich domain modeling.
- **Protocol-Based Architecture**: Extensive use of Python `Protocol` for loose coupling and improved testability.
- **Infrastructure Helpers**: Built-in support for structured logging, configuration management, and context propagation.

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

Replace exception handling with `FlextResult` for predictable control flow.

```python
from flext_core import r

def divide(a: int, b: int) -> r[float]:
    if b == 0:
        return r[float].fail("Division by zero")
    return r[float].ok(a / b)

result = divide(10, 2)
if result.is_success:
    print(f"Result: {result.unwrap()}")
else:
    print(f"Error: {result.error}")
```

### Dependency Injection

Manage your application's dependencies cleanly using `FlextContainer` and `FlextRuntime`.

```python
from flext_core import FlextContainer, Provide, inject, FlextService

# 1. Register a service
container = FlextContainer.get_global()
container.register_factory("db_client", lambda: DatabaseClient())

# 2. Inject into functions
@inject
def get_user(user_id: str, db=Provide["db_client"]):
    return db.query(user_id)

# 3. Inject into Services
class UserService(FlextService):
    def get_user(self, user_id: str):
        # Access container directly via self.container
        db = self.container.get("db_client").unwrap()
        return db.query(user_id)
```

### CQRS Dispatching

Decouple your business logic using the `FlextDispatcher`.

```python
from dataclasses import dataclass
from flext_core import FlextDispatcher, r

# 1. Define a Command
@dataclass
class CreateUser:
    username: str
    email: str

# 2. Define a Handler
def handle_create_user(cmd: CreateUser) -> r[str]:
    # Business logic here...
    return r[str].ok(f"User {cmd.username} created")

# 3. Register and Dispatch
dispatcher = FlextDispatcher()
dispatcher.register_handler(CreateUser, handle_create_user)

result = dispatcher.dispatch(CreateUser("alice", "alice@example.com"))
```

## 🏗️ Architecture

FLEXT-Core is designed around Clean Architecture and SOLID principles.

- **Protocols First**: Interfaces are defined using `Protocol` to adhere to the Dependency Inversion Principle.
- **Layered Structure**:
  - **Runtime**: Bridges external libraries and provides the DI surface.
  - **Container**: Manages service lifecycles (Singleton, Factory, Scoped).
  - **Handlers/Dispatcher**: Orchestrates application flow.

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](docs/development/contributing.md) to get started.

1. Fork the repository
1. Create your feature branch (`git checkout -b feature/amazing-feature`)
1. Commit your changes (`git commit -m 'Add amazing feature'`)
1. Push to the branch (`git push origin feature/amazing-feature`)
1. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
