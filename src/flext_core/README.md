# FLEXT Core Source Code

Core implementation of FLEXT Core's foundational patterns and architectural components.

## Overview

This directory contains the complete source implementation of FLEXT Core, providing railway-oriented programming, dependency injection, domain-driven design patterns, and enterprise features used across the FLEXT ecosystem.

## Module Organization

The source code follows Clean Architecture principles with clear separation of concerns:

### Foundation Layer

#### Core contracts and type system

| Module           | Purpose            | Key Components                    |
| ---------------- | ------------------ | --------------------------------- |
| `__init__.py`    | Public API exports | All public classes and functions  |
| `typings.py`     | Type definitions   | Type aliases, protocols, generics |
| `constants.py`   | System constants   | Enums, configuration defaults     |
| `__version__.py` | Version info       | Version string and metadata       |

### Core Patterns

#### Railway-oriented programming and DI

| Module          | Purpose              | Key Components                     |
| --------------- | -------------------- | ---------------------------------- |
| `result.py`     | Error handling       | `FlextResult[T]`, chaining methods |
| `container.py`  | Dependency injection | `FlextContainer`, service registry |
| `exceptions.py` | Exception hierarchy  | Domain exceptions, error types     |
| `utilities.py`  | Helper functions     | ID generation, common operations   |

### Configuration & Infrastructure

#### Settings, logging, and integration

| Module             | Purpose                  | Key Components                   |
| ------------------ | ------------------------ | -------------------------------- |
| `config.py`        | Configuration management | `FlextConfig`, env loading     |
| `config_base.py`   | Base configuration       | Configuration patterns           |
| `config_models.py` | Config models            | Database, cache, service configs |
| `loggings.py`      | Structured logging       | Logger factory, correlation IDs  |
| `payload.py`       | Message structures       | Events, messages, payloads       |
| `interfaces.py`    | Contracts                | Protocols, interfaces            |

### Domain Patterns

#### DDD building blocks

| Module               | Purpose         | Key Components                  |
| -------------------- | --------------- | ------------------------------- |
| `entities.py`        | Domain entities | Business entities with identity |
| `value_objects.py`   | Value objects   | Immutable domain values         |
| `models.py`          | Models          | Entities, Values, Aggregates    |
| `domain_services.py` | Domain services | Stateless domain operations     |
| `models.py`          | Domain models   | Shared model definitions        |

### Application Patterns

#### CQRS and command handling

| Module               | Purpose           | Key Components           |
| -------------------- | ----------------- | ------------------------ |
| `commands.py`        | Command patterns  | Commands, command bus    |
| `handlers.py`        | Request handlers  | Command/query handlers   |
| `handlers_base.py`   | Base handlers     | Handler abstractions     |
| `validation.py`      | Validation system | Validators, rules        |
| `validation_base.py` | Validation base   | Base validation patterns |

### Cross-Cutting Concerns

#### Mixins, decorators, and utilities

| Module          | Purpose             | Key Components                |
| --------------- | ------------------- | ----------------------------- |
| `mixins.py`     | Behavior mixins     | Timestamp, audit, soft delete |
| `decorators.py` | Function decorators | Retry, cache, logging         |
| `guards.py`     | Guard clauses       | Validation, assertions        |
| `context.py`    | Context management  | Request context, correlation  |
| `core.py`       | Core integration    | Main FlextCore class          |

### Specialized Modules

#### Domain-specific and compatibility

| Module                 | Purpose              | Key Components                  |
| ---------------------- | -------------------- | ------------------------------- |
| `observability.py`     | Monitoring patterns  | Metrics, tracing, health checks |
| `semantic.py`          | Semantic processing  | Domain language processing      |
| `schema_processing.py` | Schema handling      | Data schema operations          |
| `singer_base.py`       | Singer compatibility | Singer SDK base classes         |
| `legacy.py`            | Legacy support       | Backward compatibility          |
| `testing_utilities.py` | Test helpers         | Test fixtures, utilities        |

## Usage Examples

### Basic Imports

```python
# Essential imports for all FLEXT projects
from flext_core import FlextResult, FlextContainer, FlextModels.Entity
from flext_core.typings import FlextTypes, TAnyDict, TLogMessage
from flext_core.constants import FlextLogLevel, Platform
```

### Error Handling

```python
# Type-safe error handling throughout ecosystem
def process_data(data: dict) -> FlextResult[ProcessedData]:
    return (
        validate_input(data)
        .map(transform_data)
        .flat_map(save_to_database)
        .map(format_response)
    )
```

### Domain Models

```python
# Rich domain entities with business logic
class User(FlextModels.Entity):
    name: str
    email: str

    def activate(self) -> FlextResult[None]:
        if self.is_active:
            return FlextResult[object].fail("User already active")

        self.is_active = True
        self.add_domain_event({"type": "UserActivated"})
        return FlextResult[object].ok(None)
```

### Configuration

```python
# Environment-aware configuration management
class AppSettings(FlextConfig):
    database_url: str = "postgresql://localhost/app"
    log_level: str = "INFO"

    class Config:
        env_prefix = "APP_"
```

### Logging

```python
# Correlation ID support and enterprise observability
logger = FlextLogger(__name__)
with create_log_context(logger, request_id="123", user_id="456"):
    logger.info("Processing request", operation="create_user")
```

## Module Dependencies

### Internal Dependencies

Modules follow Clean Architecture dependency rules:

``` ascii
Foundation ← Core Patterns ← Configuration ← Domain ← Application ← Cross-Cutting
```

- Foundation modules have no internal dependencies
- Core patterns depend only on foundation
- Higher layers can depend on lower layers
- No circular dependencies allowed

### External Dependencies

**Runtime dependencies (minimal):**

- `pydantic >= 2.11.7` - Data validation
- `pydantic-settings >= 2.10.1` - Configuration
- `structlog >= 25.4.0` - Structured logging

**Development dependencies:**

- See `pyproject.toml` for complete list

## Development Guidelines

### Code Standards

- **Type Safety**: All public APIs fully typed
- **Error Handling**: Use FlextResult, avoid exceptions
- **Line Length**: 79 characters (PEP 8)
- **Documentation**: Docstrings for all public APIs
- **Testing**: 75% minimum coverage requirement

### Adding New Modules

1. **Determine Layer**: Place in appropriate architectural layer
2. **Define Exports**: Add to `__init__.py` if public
3. **Add Tests**: Create corresponding test file
4. **Document**: Include module docstring and examples
5. **Type Check**: Ensure MyPy passes

### Module Patterns

```python
"""Module description.

This module provides [functionality] as part of the [layer] layer.

Key Components:
    - Component1: Description
    - Component2: Description

Usage:
    >>> from flext_core.module import Component
    >>> component = Component()
"""

from typing import Optional
from flext_core import FlextResult

__all__ = ["Component1", "Component2"]

# Implementation follows...
```

## File Structure Summary

``` ascii
src/flext_core/
├── __init__.py              # Public API exports
├── __version__.py           # Version information
├── py.typed                 # Type hint marker
│
├── # Core Patterns
├── result.py                # FlextResult pattern
├── container.py             # Dependency injection
├── exceptions.py            # Exception hierarchy
├── utilities.py             # Utility functions
│
├── # Configuration
├── config.py                # Main configuration
├── config_base.py           # Base patterns
├── config_models.py         # Config models
├── loggings.py              # Logging system
│
├── # Domain Patterns
├── models.py                # Domain models (entities, values, aggregates)  
├── domain_services.py       # Domain services
│
├── # Application Patterns
├── commands.py              # Commands
├── handlers.py              # Handlers
├── validation.py            # Validation
│
├── # Cross-Cutting
├── mixins.py                # Mixins
├── decorators.py            # Decorators
├── interfaces.py            # Interfaces
│
└── # Supporting
    ├── typings.py           # Type definitions
    ├── constants.py         # Constants
    ├── models.py            # Models
    └── testing_utilities.py # Test helpers
```

## Key Design Decisions

### Why Railway-Oriented Programming

FlextResult provides explicit error handling without exceptions, making error paths visible and composable. This pattern is used consistently across all 32+ FLEXT projects.

### Why Minimal Dependencies

As a foundation library, FLEXT Core minimizes dependencies to avoid version conflicts and reduce the attack surface. Only essential, well-maintained packages are used.

### Why Clean Architecture

Clean Architecture ensures the business logic remains independent of frameworks and external concerns, making the codebase testable, maintainable, and adaptable to change.

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

**FLEXT Core v0.9.0** - Foundation library for the FLEXT ecosystem.
