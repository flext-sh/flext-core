# FLEXT Core

FLEXT Core is a foundational framework providing core abstractions for building robust, maintainable applications following domain-driven design (DDD) and clean architecture principles.

## Overview

This module provides the core infrastructure components used across the FLEXT ecosystem:

- **Railway-oriented error handling** with `FlextResult[T]`
- **Dependency injection container** with type-safe service registration
- **CQRS implementation** with command bus and message dispatching
- **Domain modeling** with entities, value objects, and aggregates
- **Structured logging** with context propagation
- **Configuration management** with validation and environment support
- **Context management** for distributed tracing and correlation
- **Protocol definitions** for interface contracts
- **Utility functions** for validation, serialization, and type checking

## Architecture

The framework is organized into logical layers:

### Foundation Layer

Core types and constants that have no dependencies on other framework modules.

### Domain Layer

Business logic abstractions including entities, value objects, services, and domain events.

### Application Layer

Use case coordination with CQRS patterns, command/query handlers, and application services.

### Infrastructure Layer

External concerns including logging, configuration, dependency injection, and context management.

## Key Components

| Component         | Description                                                |
| ----------------- | ---------------------------------------------------------- |
| `FlextResult[T]`  | Railway-oriented error handling with monadic operations    |
| `FlextContainer`  | Dependency injection container with type-safe registration |
| `FlextBus`        | Command/query bus for CQRS message routing                 |
| `FlextDispatcher` | High-level message dispatch orchestration                  |
| `FlextContext`    | Hierarchical context management for tracing                |
| `FlextLogger`     | Structured logging with automatic context propagation      |
| `FlextConfig`     | Configuration management with Pydantic validation          |
| `FlextModels`     | DDD patterns (Entity, Value, AggregateRoot)                |

## Usage Examples

### Basic Setup

```python
from flext_core.bus import FlextBus
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.decorators import FlextDecorators
from flext_core.dispatcher import FlextDispatcher
from flext_core.exceptions import FlextExceptions
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.processors import FlextProcessors
from flext_core.protocols import FlextProtocols
from flext_core.registry import FlextRegistry
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.service import FlextService
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities, FlextResult

# Get unified facade
core = Flext)

# Railway-oriented error handling
result = FlextResult.success("operation completed")
if result.is_success:
    data = result.unwrap()
```

### Dependency Injection

```python
from flext_core import FlextContainer

container = FlextContainer()
container.register("logger", Flextcreate_logger("my-service"))
logger_result = container.get("logger")
```

### Domain Modeling

```python
from flext_core.models import FlextModels

class User(FlextModels.Entity):
    name: str
    email: str

    def validate(self) -> FlextResult[None]:
        if "@" not in self.email:
            return FlextResult.fail("Invalid email")
        return FlextResult.ok(None)
```

### CQRS Pattern

```python
from flext_core import FlextDispatcher

dispatcher = FlextDispatcher()
dispatcher.register_handler(CreateUserCommand, create_user_handler)
result = dispatcher.dispatch(CreateUserCommand(email="user@example.com"))
```

## Dependencies

- **Runtime**: structlog, dependency-injector, pydantic, returns
- **Type System**: typing, collections.abc
- **Standards**: Follows PEP8, provides type hints for Python 3.8+

## Extension Points

The framework is designed for extension through:

- **Custom handlers** inheriting from `FlextHandlers`
- **Protocol implementations** for interface contracts
- **Custom processors** for specialized message handling
- **Mixin composition** for reusable behaviors

All components integrate with the core infrastructure while maintaining clear separation of concerns.
