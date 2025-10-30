# Clean Architecture

FLEXT-Core implements **Clean Architecture** principles with a strict 5-layer hierarchy.

## Layer Hierarchy

```
┌─────────────────────────────────────┐
│  Layer 4: Infrastructure            │  External dependencies
│  (config.py, loggings.py, ...)      │
├─────────────────────────────────────┤
│  Layer 3: Application               │  Use cases & handlers
│  (handlers.py, bus.py, ...)         │
├─────────────────────────────────────┤
│  Layer 2: Domain                    │  Business rules
│  (models.py, service.py, ...)       │
├─────────────────────────────────────┤
│  Layer 1: Foundation                │  Core primitives
│  (result.py, container.py, ...)     │
├─────────────────────────────────────┤
│  Layer 0.5: Integration Bridge      │  External library bridge
│  (runtime.py)                       │
├─────────────────────────────────────┤
│  Layer 0: Pure Constants            │  Zero dependencies
│  (constants.py, typings.py, ...)    │
└─────────────────────────────────────┘
```

## CRITICAL RULE: Unidirectional Dependencies

**Dependencies flow INWARD ONLY:**

- Layer 4 can import from Layers 3, 2, 1, 0.5, 0
- Layer 3 can import from Layers 2, 1, 0.5, 0
- Layer 2 can import from Layers 1, 0.5, 0
- Layer 1 can import from Layers 0.5, 0
- Layer 0.5 can import from Layer 0 ONLY
- Layer 0 has ZERO imports from flext_core

```python
# ✅ CORRECT - Layer 2 imports from Layer 1
from flext_core import FlextResult, FlextContainer

# ❌ FORBIDDEN - Layer 1 importing from Layer 2
from flext_core.models import FlextModels  # NO!

# ❌ FORBIDDEN - Layer 0 importing from anywhere
import requests  # NO! (Layer 0 has zero dependencies)
```

## Layer Details

### Layer 0: Pure Constants (Zero Dependencies)

**Purpose:** Immutable constants, type definitions, protocols - no external dependencies.

**Files:**

- `constants.py` - 50+ error codes, validation patterns
- `typings.py` - 50+ TypeVars, type aliases
- `protocols.py` - Runtime-checkable interfaces

**Rule:** ZERO external imports. Pure Python only.

```python
# src/flext_core/constants.py
class FlextConstants:
    """Pure constants - zero dependencies."""

    class Errors:
        VALIDATION_FAILED = "VALIDATION_FAILED"
        RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
        # ... 50+ more
```

### Layer 0.5: Integration Bridge (External Libraries)

**Purpose:** Bridge to external libraries WITHOUT importing Layer 1+.

**Files:**

- `runtime.py` - structlog, dependency_injector integration

**Rule:** Can import external libraries, but NOT other flext_core layers.

```python
# src/flext_core/runtime.py
import structlog  # ✅ External library
from flext_core.constants import FlextConstants  # ✅ Layer 0 only

# ❌ NO - Never import Layer 1+
# from flext_core.result import FlextResult
```

### Layer 1: Foundation (Core Primitives)

**Purpose:** Core reusable primitives that depend only on Layers 0.5 and 0.

**Files:**

- `result.py` - FlextResult[T] monad
- `container.py` - Dependency injection singleton
- `exceptions.py` - Exception hierarchy

**Rule:** Can only import from Layers 0.5 and 0. NO business logic.

```python
from flext_core.constants import FlextConstants  # ✅ Layer 0
from flext_core.runtime import FlextRuntime  # ✅ Layer 0.5

class FlextResult:
    """Core result monad."""
    # No business logic, pure utility
```

### Layer 2: Domain (Business Rules)

**Purpose:** Business logic, domain entities, validation rules.

**Files:**

- `models.py` - Entity, Value, AggregateRoot base classes
- `service.py` - Domain service base class
- `mixins.py` - Reusable domain behaviors
- `utilities.py` - Domain utility functions

**Rule:** Can import from Layers 1, 0.5, 0. NO infrastructure dependencies.

```python
from flext_core import FlextResult, FlextContainer  # ✅ Layers 1

class UserService(FlextService):
    """Domain service - contains business logic."""

    def create_user(self, name: str, email: str) -> FlextResult[User]:
        """Business rule: validate and create user."""
        # Domain logic here
        pass

# ❌ NO - Don't import infrastructure
# from flext_core import FlextLogger  # NO! That's Layer 4
```

### Layer 3: Application (Use Cases & Handlers)

**Purpose:** Application orchestration - command handlers, query handlers, event processors.

**Files:**

- `handlers.py` - Message handler registry
- `bus.py` - Central message bus
- `dispatcher.py` - Unified dispatcher
- `processors.py` - Event processors
- `registry.py` - Component registry

**Rule:** Can import from Layers 2, 1, 0.5, 0. Coordinates domain services.

```python
from flext_core import FlextBus, FlextResult  # ✅ Layers 1, 3

class CreateUserHandler:
    """Application handler - orchestrates domain service."""

    def __init__(self, user_service):
        self.user_service = user_service

    def handle(self, command) -> FlextResult:
        # Call domain service
        return self.user_service.create_user(command.name, command.email)

# ❌ NO - Don't import infrastructure
# from flext_core import FlextLogger  # NO!
```

### Layer 4: Infrastructure (External Dependencies)

**Purpose:** Configuration, logging, context management, external integrations.

**Files:**

- `config.py` - FlextConfig (Pydantic Settings)
- `loggings.py` - FlextLogger (structlog)
- `context.py` - FlextContext (correlation IDs, tracing)

**Rule:** Can import ANYTHING. Final layer with external dependencies.

```python
from flext_core import FlextBus  # ✅ All layers available
import structlog  # ✅ External libraries
from database import connect  # ✅ External services

class FlextLogger:
    """Infrastructure layer - uses structlog."""
    # Can use any external library
```

## Dependency Injection Pattern

### Layered Dependency Resolution

```python
# Bootstrap application (from outside all layers)
from flext_core import FlextContainer, FlextConfig, FlextLogger, FlextBus

# 1. Configure (Layer 4)
config = FlextConfig(config_files=['config.toml'])
logger = FlextLogger(__name__)

# 2. Setup container (Layer 1)
container = FlextContainer.get_global()
container.register("config", config, singleton=True)
container.register("logger", logger, singleton=True)

# 3. Register domain services (Layer 2)
user_service = UserService()
container.register("user_service", user_service)

# 4. Register handlers (Layer 3)
bus = FlextBus()
bus.register_handler(CreateUserCommand, CreateUserHandler(user_service))
container.register("bus", bus)

# 5. Use application (request time)
bus_result = container.get("bus")
if bus_result.is_success:
    bus = bus_result.unwrap()
    command = CreateUserCommand(name="Alice", email="alice@example.com")
    result = bus.send_command(command)
```

## Single Responsibility Per Module

**FLEXT-Core Rule:** ONE public class per module with `Flext` prefix.

```python
# ✅ CORRECT - One class per file
# src/flext_core/result.py
class FlextResult:
    """Single public class."""

    class _Details:  # Nested helper - OK
        pass

# src/flext_core/container.py
class FlextContainer:
    """Single public class."""

# ❌ WRONG - Multiple top-level classes
# src/flext_core/mixed.py
class FlextResult:
    pass

class FlextContainer:  # WRONG! Second top-level class
    pass
```

## Testability Through Layering

### Independent Layer Testing

```python
# Test Layer 2 (Domain) without Layer 4 (Infrastructure)
from flext_core import FlextResult

def test_user_service():
    """Test domain logic independently."""
    service = UserService()

    result = service.create_user("Alice", "alice@example.com")

    assert result.is_success
    # No need for database, logging, config!

# Test Layer 3 (Application) with mocked Layer 2
from unittest.mock import Mock

def test_create_user_handler():
    """Test handler with mock service."""
    mock_service = Mock()
    mock_service.create_user.return_value = FlextResult[User].ok(...)

    handler = CreateUserHandler(mock_service)
    result = handler.handle(CreateUserCommand(...))

    assert result.is_success
    mock_service.create_user.assert_called_once()
```

## API Stability Contracts

### Layer 1 (Foundation) - NEVER BREAK

```python
# These APIs are GUARANTEED in 1.x
from flext_core import FlextResult, FlextContainer, FlextModels, FlextLogger, FlextConfig

# Both .data and .value work (backward compatibility)
result = FlextResult[str].ok("value")
assert result.value == "value"
assert result.data == "value"

container = FlextContainer.get_global()
container.register("service", MyService())

# Base classes always available
class User(FlextModels.Entity):
    pass
```

### Higher Layers - May Evolve

```python
# Layer 3 and 4 can add features/change implementation
# Example: FlextBus might add new middleware system in v1.1
from flext_core import FlextBus

bus = FlextBus()
# New in v1.1: advanced pipeline features
bus.add_middleware(MyMiddleware())
```

## Circular Dependency Prevention

### The Problem

```
Layer 2 (UserService) → Layer 1 (FlextResult) → Layer 0 (Constants)  ✅
Layer 2 (UserService) → Layer 3 (Handler) → Layer 2 (UserService)   ❌ CIRCULAR!
```

### The Solution

Use dependency injection to break the cycle:

```python
# ❌ WRONG - Direct dependency
class Handler:
    def __init__(self):
        self.service = UserService()  # Creates circular dependency

# ✅ CORRECT - Inject dependency
class Handler:
    def __init__(self, service: UserService):
        self.service = service  # Dependency provided by container
```

## Import Organization

### Correct Import Pattern

```python
# ✅ CORRECT - Root imports only
from flext_core import (
    FlextResult,
    FlextContainer,
    FlextModels,
    FlextLogger,
    FlextConfig,
)

# ❌ FORBIDDEN - Internal imports
from flext_core.result import FlextResult
from flext_core.models import FlextModels

# Why? 32+ projects depend on the root import interface
# Internal imports break the ecosystem
```

## Clean Architecture Benefits in FLEXT-Core

1. **Independent Testability**: Test any layer in isolation
2. **Easy to Understand**: Clear layer responsibilities
3. **Framework Independent**: Swap implementations without breaking logic
4. **Scalable**: Add new features in correct layer
5. **Maintainable**: Changes localized to appropriate layer
6. **Ecosystem Safe**: Breaking changes prevented by contracts

## Anti-Patterns to Avoid

### ❌ Mixing Layers

```python
# WRONG - Domain logic in handler
class CreateUserHandler:
    def handle(self, command):
        # Business logic mixed with handling
        if len(command.name) < 3:
            raise ValueError("Name too short")  # Domain rule in handler!
        # ...

# CORRECT - Domain logic in service
class UserService:
    def create_user(self, name):
        if len(name) < 3:
            return FlextResult.fail("Name too short")  # Domain rule here
        # ...

class CreateUserHandler:
    def handle(self, command):
        return self.service.create_user(command.name)  # Just delegate
```

### ❌ Upward Dependencies

```python
# WRONG - Layer 1 depending on Layer 2
from flext_core.models import FlextModels  # NO!

class FlextResult:
    def process_model(self, model: FlextModels.Entity):  # NO!
        pass

# CORRECT - No upward deps
class FlextResult:
    def process(self, value):
        # Generic, no knowledge of domain
        pass
```

## Summary

FLEXT-Core Clean Architecture:

- ✅ 5-layer strict hierarchy with unidirectional dependencies
- ✅ Each layer has clear responsibility
- ✅ Independent testability at each layer
- ✅ API stability contracts prevent breaking changes
- ✅ Scalable design for 32+ dependent projects
- ✅ Maintainable through clear separation of concerns

Use these principles when extending FLEXT-Core or building applications on top of it.
