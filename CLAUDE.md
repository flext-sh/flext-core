# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FLEXT Core is the **architectural foundation library** for the FLEXT data integration ecosystem. This pure Python library provides enterprise patterns (Clean Architecture, DDD, railway-oriented programming) used by 32+ dependent projects across the ecosystem.

**Status**: Core patterns stable (FlextResult, FlextContainer), 75% test coverage requirement, MyPy strict mode

**Key Characteristics**:

- Python 3.13+ only (no backward compatibility)
- Minimal dependencies: pydantic>=2.11.7, pydantic-settings>=2.10.1, structlog>=25.4.0
- Railway-oriented programming via FlextResult[T] pattern
- Foundation for all FLEXT ecosystem projects

## Essential Commands

### Development Workflow

```bash
# Initial setup
make setup                 # Complete dev environment setup with pre-commit hooks

# Before any commit (MANDATORY)
make validate              # Run ALL quality gates (lint + type + security + test)
make check                 # Quick validation (lint + type-check only)

# Individual quality checks
make lint                  # Ruff linting with comprehensive rules
make type-check            # MyPy strict mode checking
make test                  # Full test suite (75% coverage required)
make security              # Bandit + pip-audit scanning
make format                # Auto-format code (79 char line limit)
make fix                   # Auto-fix linting issues

# Testing commands
make test-unit             # Unit tests only
make test-integration      # Integration tests only
make test-fast             # Tests without coverage (quick feedback)
make coverage-html         # Generate HTML coverage report

# Run specific test file
poetry run pytest tests/unit/core/test_result.py -v
poetry run pytest tests/unit/core/test_container.py::TestFlextContainer::test_basic_registration -v

# Test with markers
poetry run pytest -m unit              # Unit tests only
poetry run pytest -m integration       # Integration tests only
poetry run pytest -m "not slow"        # Exclude slow tests
poetry run pytest -m core              # Core framework tests
poetry run pytest -m ddd               # Domain-driven design tests

# Development utilities
make shell                 # Python REPL with project loaded
make deps-show             # Show dependency tree
make doctor                # Complete health check
make clean                 # Clean build artifacts
make reset                 # Full reset (clean + setup)
```

## High-Level Architecture

### Core Pattern: FlextResult Railway

The foundation of error handling across the entire ecosystem - eliminates exceptions in business logic:

```python
from flext_core import FlextResult

def validate_user(data: dict) -> FlextResult[User]:
    """All operations return FlextResult for composability."""
    if not data.get("email"):
        return FlextResult.fail("Email required")
    return FlextResult.ok(User(**data))

# Railway-oriented composition
result = (
    validate_user(data)
    .flat_map(lambda u: save_user(u))      # Chain operations
    .map(lambda u: format_response(u))      # Transform success
    .map_error(lambda e: log_error(e))      # Handle errors
)

if result.success:
    return result.unwrap()  # Extract value
```

### Core Pattern: Dependency Injection

Global container pattern used across all FLEXT services:

```python
from flext_core import get_flext_container

# Global singleton container
container = get_flext_container()

# Register services (returns FlextResult)
container.register("db", DatabaseService())
container.register_factory("logger", lambda: create_logger())

# Retrieve with type safety
db_result = container.get("db")
if db_result.success:
    db = db_result.unwrap()
```

### Core Pattern: Domain Modeling

DDD patterns that all ecosystem projects inherit:

```python
from flext_core import FlextEntity, FlextValueObject, FlextAggregateRoot

class User(FlextEntity):
    """Entities have identity and lifecycle."""
    name: str
    email: Email  # Value object

    def activate(self) -> FlextResult[None]:
        """Business logic returns FlextResult."""
        if self.is_active:
            return FlextResult.fail("Already active")
        self.is_active = True
        # Domain events automatically tracked
        return FlextResult.ok(None)

class Email(FlextValueObject):
    """Value objects are immutable and compared by value."""
    address: str

    def validate(self) -> FlextResult[None]:
        if "@" not in self.address:
            return FlextResult.fail("Invalid email")
        return FlextResult.ok(None)
```

### Module Organization

The library is organized into logical layers following Clean Architecture:

```
src/flext_core/
├── Foundation Layer (Core Patterns)
│   ├── result.py           # FlextResult[T] railway pattern
│   ├── container.py        # Dependency injection
│   ├── exceptions.py       # Exception hierarchy
│   └── constants.py        # Core enums/constants
│
├── Domain Layer (DDD Patterns)
│   ├── entities.py         # FlextEntity base
│   ├── value_objects.py    # FlextValueObject base
│   ├── aggregate_root.py   # FlextAggregateRoot
│   └── domain_services.py  # Domain service patterns
│
├── Application Layer (CQRS/Handlers)
│   ├── commands.py         # Command patterns
│   ├── handlers.py         # Handler implementations
│   ├── handlers_base.py    # Base handler classes
│   └── validation.py       # Validation system
│
├── Infrastructure Layer (Cross-cutting)
│   ├── config.py           # Configuration management
│   ├── loggings.py        # Structured logging
│   ├── payload.py         # Event/message patterns
│   └── observability.py   # Monitoring patterns
│
└── Support Modules
    ├── mixins.py          # Reusable behaviors
    ├── decorators.py      # Enterprise decorators
    ├── utilities.py       # Helper functions
    └── legacy.py          # Backward compatibility
```

### Shared Domain Pattern

Examples and tests use a shared domain to avoid duplication:

```python
# ALWAYS import from shared_domain in examples/tests
from shared_domain import (
    SharedDomainFactory,    # Factory for test entities
    User,                   # Shared User entity
    Order,                  # Shared Order aggregate
    Email,                  # Shared Email value object
)

# NEVER create local domain models in examples
# This ensures consistency across all demonstrations
```

## Critical Development Guidelines

### Type Safety Requirements

- **MyPy Strict Mode**: Zero tolerance for type errors
- **Runtime Type Checking**: All public APIs validate types
- **Generic Types**: Use TypeVar for proper generic constraints
- **No Any Types**: Explicitly type all parameters and returns

### Error Handling Pattern

- **Always use FlextResult**: Never raise exceptions in business logic
- **Chain operations**: Use map/flat_map for composition
- **Handle all paths**: Both success and failure must be handled
- **Unwrap carefully**: Only unwrap when you've checked success

### Testing Requirements

- **Coverage minimum**: 75% (enforced in Makefile)
- **Test markers**: Use appropriate markers (unit, integration, slow)
- **Fixtures**: Use provided fixtures from conftest.py
- **Shared domain**: Import test entities from shared_test_domain.py

### Code Style

- **Line length**: 79 characters maximum (PEP8 strict)
- **Naming**: FlextXxx prefix for all public exports
- **Imports**: Absolute imports from flext_core
- **Docstrings**: Required for all public APIs

## Ecosystem Integration

### Dependency Chain

```
flext-core (this library)
    ↓
├── Infrastructure Libraries (6 projects)
│   ├── flext-db-oracle      # Uses FlextResult for DB operations
│   ├── flext-ldap           # Uses FlextContainer for services
│   ├── flext-grpc           # Uses FlextPayload for messages
│   └── flext-meltano        # Uses all core patterns
│
├── Application Services (5 projects)
│   ├── flext-api            # FastAPI with FlextResult responses
│   ├── flext-auth           # Auth with FlextEntity users
│   └── flext-web            # Web UI with FlextContainer DI
│
├── Singer Ecosystem (15 projects)
│   ├── Taps (5)             # Extract with FlextResult
│   ├── Targets (5)          # Load with FlextResult
│   └── DBT (4)              # Transform with validation
│
└── Go Services (via Python bridge)
    ├── FlexCore             # Runtime engine
    └── FLEXT Service        # Control panel
```

### Breaking Changes Impact

Changes to flext-core affect ALL 32 dependent projects. Before making breaking changes:

1. Check ecosystem compatibility
2. Document migration path
3. Update dependent projects
4. Follow semantic versioning

## Common Issues & Solutions

### MyPy Errors

```bash
# See all type errors with context
poetry run mypy src --show-error-codes --show-error-context

# Check specific file
poetry run mypy src/flext_core/result.py --strict
```

### Import Errors

```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Verify installation
python -c "from flext_core import FlextResult; print('✅ Working')"
```

### Test Failures

```bash
# Run with verbose output
poetry run pytest tests/unit/core/test_result.py -vvs

# Debug specific test
poetry run pytest tests/unit/core/test_result.py::TestFlextResult::test_map -vvs --pdb
```

## Architecture Gaps (TODOs)

### Critical Missing Components

1. **Event Sourcing**: FlextAggregateRoot exists but event store missing
2. **CQRS Bus**: Command/Query bus not implemented (handlers exist)
3. **Plugin System**: No foundation for FlexCore plugin architecture
4. **Cross-Language Bridge**: Python-Go type mapping undefined
5. **Observability**: Correlation IDs exist but tracing incomplete

These gaps are documented but not blocking current development.
