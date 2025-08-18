# Architecture Guidelines

**Version**: 2.0.0  
**Status**: Active  
**Last Updated**: 2025-01-10

## Overview

This document defines the mandatory architectural guidelines for FLEXT Core, the foundation library serving 32+ projects in the FLEXT ecosystem.

## Core Architectural Principles

### Clean Architecture + DDD + CQRS

```
┌─────────────────────────────────────┐
│         PRESENTATION LAYER          │
├─────────────────────────────────────┤
│         APPLICATION LAYER           │
│    ┌─────────────┬─────────────┐   │
│    │  COMMANDS   │   QUERIES   │   │
│    └─────────────┴─────────────┘   │
├─────────────────────────────────────┤
│           DOMAIN LAYER              │
│  ┌─────────┬──────────┬──────────┐ │
│  │ENTITIES │ V.OBJECTS│AGGREGATES│ │
│  └─────────┴──────────┴──────────┘ │
├─────────────────────────────────────┤
│       INFRASTRUCTURE LAYER          │
└─────────────────────────────────────┘
```

### Railway-Oriented Programming

**MANDATORY**: All public methods MUST return `FlextResult[T]`

```python
def process_data(data: str) -> FlextResult[ProcessedData]:
    if not data:
        return FlextResult.fail("Empty data provided")

    try:
        processed = ProcessedData(data)
        return FlextResult.ok(processed)
    except Exception as e:
        return FlextResult.fail(str(e))
```

### Naming Convention

**MANDATORY**: All public exports MUST use `Flext` prefix

```python
# ✅ CORRECT
class FlextConfig: pass
class FlextContainer: pass
class FlextEntity: pass

# ❌ INCORRECT
class Config: pass
class Container: pass
class Entity: pass
```

## Implementation Patterns

### Module Structure Pattern

Every module MUST follow this structure:

```python
"""Module description explaining purpose and key components."""

from __future__ import annotations

from typing import  Optional, Any

from flext_core.result import FlextResult
from flext_core.exceptions import FlextError

from flext_core.typings import TAnyDict

# Public exports
__all__ = ["FlextComponent", "FlextService"]

# Implementation
class FlextComponent:
    """Component with proper Flext prefix."""

    def process(self, data: Any) -> FlextResult[str]:
        """All public methods return FlextResult."""
        return FlextResult.ok("processed")
```

### Abstraction → Implementation Pattern

```python
# base_module.py - Abstract definitions
from abc import ABC, abstractmethod

class FlextBaseService(ABC):
    """Abstract base class for services."""

    @abstractmethod
    def execute(self, request: Any) -> FlextResult[Any]:
        """Abstract method to be implemented."""
        ...

# module.py - Concrete implementations
class FlextService(FlextBaseService):
    """Concrete service implementation."""

    def execute(self, request: Any) -> FlextResult[Any]:
        """Concrete implementation."""
        return FlextResult.ok({"status": "success"})
```

### Exception Handling Pattern

**PROHIBITED**: Raw exceptions without FlextResult wrapping

```python
# ❌ INCORRECT
def bad_function(value: str) -> str:
    if not value:
        raise ValueError("Empty value")
    return value.upper()

# ✅ CORRECT
def good_function(value: str) -> FlextResult[str]:
    if not value:
        return FlextResult.fail("Empty value provided")

    try:
        result = value.upper()
        return FlextResult.ok(result)
    except Exception as e:
        return FlextResult.fail(f"Processing error: {e}")
```

### Type Safety Pattern

**MANDATORY**: Use centralized types from `flext_core.typings`

```python
from flext_core.typings import TAnyDict, TEntityId
from flext_core.constants import FlextEntityStatus

def process_entity(
    entity_id: TEntityId,
    data: TAnyDict,
    status: FlextEntityStatus
) -> FlextResult[bool]:
    """Process entity with proper typing."""
    # Implementation
    return FlextResult.ok(True)
```

## Anti-Patterns (Prohibited)

### Circular Imports

```python
# ❌ PROHIBITED
# file: base_handlers.py
from flext_core.handlers import FlextMessageHandler

# file: handlers.py
from flext_core.base_handlers import FlextBaseHandler
```

### Large Modules

Modules exceeding 1000 lines MUST be split into smaller, focused modules.

### Missing Exports

```python
# ❌ PROHIBITED
class FlextComponent: pass
# Missing __all__ declaration

# ✅ CORRECT
class FlextComponent: pass

__all__ = ["FlextComponent"]
```

### Overuse of Compatibility Layers

Minimize `*_compat.py` files. Prefer migration to modern APIs.

## Layered Architecture

### Layer 0: Foundation (No internal dependencies)

- `result.py` - FlextResult pattern
- `exceptions.py` - Exception hierarchy
- `typings.py` - Type definitions
- `constants.py` - System constants

### Layer 1: Infrastructure

- `config.py` - Configuration management
- `container.py` - Dependency injection
- `utilities.py` - Helper functions
- `loggings.py` - Structured logging

### Layer 2: Domain Models

- `entities.py` - Business entities
- `value_objects.py` - Immutable values
- `aggregate_root.py` - DDD aggregates
- `domain_services.py` - Domain logic

### Layer 3: Application Services

- `handlers.py` - Request handlers
- `commands.py` - Command patterns
- `validation.py` - Validation logic
- `decorators.py` - Cross-cutting concerns

### Layer 4: Integration

- `interfaces.py` - Contracts and protocols
- `legacy.py` - Backward compatibility

## Quality Gates

### Code Quality Requirements

```bash
make lint        # Ruff linting - ZERO errors allowed
make type-check  # MyPy strict - Target: ZERO errors
make test        # 75%+ coverage required
make validate    # All quality checks must pass
```

### Architectural Compliance Checklist

- [ ] All classes use `Flext` prefix
- [ ] All public methods return `FlextResult[T]`
- [ ] No circular imports detected
- [ ] All modules have `__all__` exports
- [ ] All modules include `from __future__ import annotations`
- [ ] Type hints on all public APIs
- [ ] Docstrings for all public components
- [ ] Line length ≤ 79 characters (PEP 8)

### Performance Requirements

- Module size: <1000 lines
- Import time: <10ms
- Memory usage: <50MB base
- Test execution: <1s per test

## Dependency Management

### Allowed Dependencies

```
Layer 0 (Foundation) ← Layer 1 (Infrastructure)
                    ← Layer 2 (Domain)
                    ← Layer 3 (Application)
                    ← Layer 4 (Integration)
```

### Prohibited Dependencies

- Layer 0 → Any other layer
- Layer 1 → Layers 2, 3, 4 (except utilities)
- Circular dependencies between any layers
- External dependencies in Layer 0

## Implementation Checklist

For each new module:

- [ ] Module docstring explaining purpose
- [ ] `from __future__ import annotations`
- [ ] Proper imports from `flext_core.*`
- [ ] All classes use `Flext` prefix
- [ ] All public methods return `FlextResult[T]`
- [ ] `__all__` export list defined
- [ ] Type hints on all functions
- [ ] Unit tests with adequate coverage
- [ ] No circular imports
- [ ] No raw exceptions
- [ ] Constants from `flext_core.constants`
- [ ] Types from `flext_core.typings`

## Metrics and Monitoring

### Module Health Metrics

| Metric                | Target | Current |
| --------------------- | ------ | ------- |
| Line count            | <1000  | ✓       |
| Cyclomatic complexity | <10    | ✓       |
| Import depth          | <5     | ✓       |
| Test coverage         | >75%   | 75%     |
| MyPy errors (src)     | 0      | 4       |
| MyPy errors (tests)   | <100   | 1,245   |

### Code Quality Trends

Track improvements over time:

- Type safety: Moving toward zero MyPy errors
- Test coverage: Maintaining 75%+ coverage
- Module size: Keeping modules focused
- API stability: Minimizing breaking changes

## Ecosystem Impact

FLEXT Core serves as foundation for **32+ FLEXT projects**:

### Breaking Change Policy

1. **Semantic Versioning**: Major.Minor.Patch
2. **Deprecation Period**: 2 release cycles minimum
3. **Migration Guides**: Required for all breaking changes
4. **Compatibility Testing**: Automated against dependent projects

### API Stability Requirements

- Public APIs marked `@final` when stable
- Abstract base classes versioned separately
- Backward compatibility for 2+ major versions
- Clear deprecation warnings with alternatives

### Change Impact Assessment

Before making changes:

1. Identify affected dependent projects
2. Assess breaking change severity
3. Provide migration path
4. Update compatibility matrix
5. Communicate changes clearly

## Best Practices

### Domain Modeling

```python
class FlextUser(FlextEntity):
    """Rich domain entity with business logic."""

    name: str
    email: str

    def activate(self) -> FlextResult[None]:
        """Business operation returning FlextResult."""
        if self.is_active:
            return FlextResult.fail("Already active")

        self.is_active = True
        self.add_event(UserActivatedEvent(self.id))
        return FlextResult.ok(None)
```

### Service Layer

```python
class FlextUserService:
    """Application service orchestrating domain logic."""

    def __init__(self, repository: FlextUserRepository):
        self.repository = repository

    def create_user(self, data: TAnyDict) -> FlextResult[FlextUser]:
        """Service method with FlextResult pattern."""
        validation_result = self.validate_user_data(data)
        if validation_result.is_failure:
            return validation_result

        user = FlextUser(**data)
        return self.repository.save(user)
```

### Testing Patterns

```python
def test_user_creation():
    """Test with FlextResult assertions."""
    service = FlextUserService(mock_repository)
    result = service.create_user({"name": "Test", "email": "test@example.com"})

    assert result.success
    assert result.unwrap().name == "Test"
```

## Compliance

- **MANDATORY**: This document defines required patterns
- **ENFORCEMENT**: Automated via CI/CD pipelines
- **REVIEW**: Quarterly architecture review meetings
- **UPDATES**: Document evolves with project needs

---

**Next Review**: 2025-04-10  
**Owner**: FLEXT Core Architecture Team  
**Status**: Actively Enforced
