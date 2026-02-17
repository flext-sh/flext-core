# FLEXT Core

**Reviewed**: 2026-02-17 | **Scope**: Canonical rules alignment and link consistency

FLEXT Core is a foundational framework providing core abstractions for building robust, maintainable applications following domain-driven design (DDD) and clean architecture principles.

## Overview

This module provides the core infrastructure components used across the FLEXT ecosystem:

- **Railway-oriented error handling** with `FlextResult[T]`
- **Dependency injection container** with type-safe service registration
- **CQRS implementation** with command/query dispatcher and registry
- **Domain modeling** with entities, value objects, aggregates, and domain events
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

Use case coordination with CQRS patterns, command/query handlers, domain-event publishing, and application services.

### Infrastructure Layer

External concerns including logging, configuration, dependency injection, and context management.

## Key Components

| Component         | Description                                                |
| ----------------- | ---------------------------------------------------------- |
| `FlextResult[T]`  | Railway-oriented error handling with monadic operations    |
| `FlextContainer`  | Dependency injection container with type-safe registration |
| `FlextDispatcher` | Command/query dispatcher with reliability controls         |
| `FlextContext`    | Hierarchical context management for tracing                |
| `FlextLogger`     | Structured logging with automatic context propagation      |
| `FlextSettings`   | Configuration management with Pydantic validation          |
| `FlextModels`     | DDD patterns (Entity, Value, AggregateRoot)                |

## Usage Examples

### Basic Setup

```python
from flext_core.settings import FlextSettings
from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.decorators import FlextDecorators
from flext_core.dispatcher import FlextDispatcher
from flext_core.exceptions import FlextExceptions
from flext_core.handlers import h
from flext_core.loggings import FlextLogger
from flext_core.mixins import x
from flext_core.models import FlextModels
from flext_core.protocols import p
from flext_core.registry import FlextRegistry
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.service import FlextService
from flext_core.typings import t
from flext_core.utilities import u

# Create dispatcher/registry (registry optional for simple flows)
dispatcher = FlextDispatcher()
registry = FlextRegistry()

# Railway-oriented error handling
result = FlextResult.success("operation completed")
if result.is_success:
    data = result.value
```

### Dependency Injection

```python
from flext_core import FlextContainer
from flext_core.loggings import FlextLogger

container = FlextContainer()
container.register("logger", FlextLogger.create_module_logger(__name__))
logger_result = container.get("logger")
assert logger_result.value is container.get("logger").value
```

### Domain Modeling

```python
from flext_core.models import FlextModels

class User(FlextModels.Entity):
    name: str
    email: str

    def validate(self) -> FlextResult[bool]:
        if "@" not in self.email:
            return FlextResult[bool].fail("Invalid email")
        return FlextResult[bool].ok(True)
```

### Domain Events and Dispatcher Integration

Aggregate roots collect domain events that can be published through the dispatcher after a successful operation:

```python
from flext_core.dispatcher import FlextDispatcher
from flext_core.models import FlextModels


class InventoryAdjusted(FlextModels.DomainEvent):
    sku: str
    quantity: int


class Product(FlextModels.AggregateRoot):
    sku: str
    inventory: int

    def decrease_inventory(self, quantity: int) -> None:
        if quantity > self.inventory:
            raise ValueError("Insufficient inventory")
        self.inventory -= quantity
        self.add_domain_event(InventoryAdjusted(sku=self.sku, quantity=quantity))


dispatcher = FlextDispatcher()
product = Product(sku="ABC", inventory=10)
product.decrease_inventory(3)

# Publish events via dispatcher (with middleware/telemetry applied)
dispatcher.publish_events(product.commit_domain_events())
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

- **Custom handlers** inheriting from `h`
- **Protocol implementations** for interface contracts
- **Dispatcher middleware** for reliability, logging, or validation
- **Mixin composition** for reusable behaviors

All components integrate with the core infrastructure while maintaining clear separation of concerns.
