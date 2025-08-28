# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**References**: See [../CLAUDE.md](../CLAUDE.md) for FLEXT ecosystem standards and [README.md](README.md) for project overview.

## Development Commands

### Essential Development Workflow

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

# Development utilities
make shell                 # Python REPL with project loaded
make deps-show             # Show dependency tree
make deps-update           # Update all dependencies
make deps-audit            # Security audit of dependencies
make doctor                # Complete health check with diagnostics
make diagnose              # Project diagnostics (versions, environment)
make clean                 # Clean build artifacts
make clean-all             # Deep clean including venv
make reset                 # Full reset (clean + setup)
make pre-commit            # Run pre-commit hooks manually

# Build and documentation
make build                 # Build the package
make build-clean           # Clean and build
make docs                  # Build documentation with mkdocs
make docs-serve            # Serve documentation locally

# Single letter aliases for speed
make t                     # Alias for test
make l                     # Alias for lint
make f                     # Alias for format
make tc                    # Alias for type-check
make c                     # Alias for clean
make i                     # Alias for install
make v                     # Alias for validate
```

### Running Specific Tests

```bash
# Run specific test file
poetry run pytest tests/unit/core/test_result.py -v
poetry run pytest tests/unit/core/test_container.py::TestFlextContainer::test_basic_registration -v

# Test with markers
poetry run pytest -m unit              # Unit tests only
poetry run pytest -m integration       # Integration tests only
poetry run pytest -m "not slow"        # Exclude slow tests
poetry run pytest -m core              # Core framework tests
poetry run pytest -m ddd               # Domain-driven design tests

# Advanced test execution
poetry run pytest tests/unit/core/test_result.py::TestFlextResult::test_map -xvs --tb=long
poetry run pytest -m "unit and not slow" --tb=short -q
poetry run pytest tests/unit/ --cov=src/flext_core --cov-report=term-missing
poetry run pytest --lf --ff -x  # Last failed tests with fail-fast
poetry run pytest -n auto tests/unit/  # Parallel execution
poetry run pytest tests/unit/ -k "test_result" -v  # Tests matching pattern
```

## High-Level Architecture

### Core Pattern: FlextResult Railway

The foundation of error handling across the entire ecosystem - eliminates exceptions in business logic:

```python
from flext_core import FlextResult

def validate_user(data: dict) -> FlextResult[User]:
    """All operations return FlextResult for composability."""
    if not data.get("email"):
        return FlextResult[None].fail("Email required")
    return FlextResult[None].ok(User(**data))

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

### Module Organization

The library follows Clean Architecture with layered imports to avoid circular dependencies:

```
src/flext_core/
├── Foundation Layer (Core Patterns)
│   ├── result.py           # FlextResult[T] railway pattern with map/flat_map
│   ├── container.py        # Dependency injection container with FlextResult
│   ├── exceptions.py       # Exception hierarchy with error codes and metrics
│   ├── constants.py        # FlextConstants, enums, error codes, performance metrics
│   └── typings.py          # Type definitions and aliases (T, U, V, etc.)
│
├── Domain Layer (DDD Patterns)
│   ├── aggregate_root.py   # FlextAggregateRoot with domain events
│   ├── domain_services.py  # Domain service patterns and operations
│   ├── models.py           # Pydantic models and JSON schemas
│   └── root_models.py      # RootModel patterns for validation
│
├── Application Layer (CQRS/Handlers)
│   ├── commands.py         # FlextCommands pattern and CQRS foundation
│   ├── handlers.py         # Handler implementations and registry
│   ├── validation.py       # Validation framework with predicates
│   ├── payload.py          # Message/event patterns for integration
│   └── guards.py           # Type guards and validation decorators
│
├── Infrastructure Layer (Cross-cutting)
│   ├── config.py           # Configuration management with Pydantic Settings
│   ├── loggings.py         # Structured logging with structlog integration
│   ├── observability.py    # Metrics, tracing, monitoring abstractions
│   ├── protocols.py        # Interface definitions and contracts
│   └── context.py          # Request/operation context management
│
└── Support Modules (Utilities & Extensions)
    ├── mixins.py           # Reusable behavior patterns (timestamps, logging, etc.)
    ├── decorators.py       # Enterprise decorator patterns (validation, caching, etc.)
    ├── utilities.py        # Helper functions, generators, type guards
    ├── fields.py           # Field validation and metadata
    ├── services.py         # Service layer abstractions
    ├── delegation_system.py # Mixin delegation patterns
    ├── schema_processing.py # Schema validation and processing
    ├── type_adapters.py    # Type adaptation utilities
    └── legacy.py           # Backward compatibility layer
```

### Domain Modeling Pattern

DDD patterns that all ecosystem projects inherit:

```python
from flext_core import FlextEntity, FlextValueObject, FlextAggregateRoot

class Email(FlextValueObject):
    """Value objects are immutable and compared by value."""
    address: str

    def validate(self) -> FlextResult[None]:
        if "@" not in self.address:
            return FlextResult[None].fail("Invalid email")
        return FlextResult[None].ok(None)

class User(FlextEntity):
    """Entities have identity and lifecycle."""
    name: str
    email: Email  # Value object

    def activate(self) -> FlextResult[None]:
        """Business logic returns FlextResult."""
        if self.is_active:
            return FlextResult[None].fail("Already active")
        self.is_active = True
        # Domain events automatically tracked
        return FlextResult[None].ok(None)
```

## Development Patterns

### Mandatory Development Patterns

#### 1. Error Handling Pattern (MANDATORY)

Never raise exceptions in business logic - always return FlextResult:

```python
from flext_core import FlextResult

def business_operation(data: dict) -> FlextResult[ProcessedData]:
    """MANDATORY: All business operations must return FlextResult."""
    if not data:
        return FlextResult[None].fail("Data required", error_code="VALIDATION_ERROR")

    # Railway-oriented composition (PREFERRED)
    return (
        validate_data(data)
        .flat_map(lambda d: process_data(d))      # Chain operations
        .map(lambda d: enrich_data(d))            # Transform success
        .map_error(lambda e: f"Processing failed: {e}")  # Handle errors
    )

# Consumption pattern
result = business_operation(data)
if result.success:
    processed = result.unwrap()  # Safe unwrap after success check
else:
    logger.error(f"Operation failed: {result.error}")
```

#### 2. Dependency Injection Pattern (MANDATORY)

Always use the global container for service registration:

```python
from flext_core import get_flext_container

# Registration (typically in application startup)
container = get_flext_container()
container.register("database", DatabaseService())
container.register_factory("logger", lambda: create_logger())

# Consumption (in business logic)
def service_operation() -> FlextResult[Data]:
    db_result = container.get("database")
    if db_result.failure:
        return FlextResult[None].fail("Database service unavailable")
    
    db = db_result.unwrap()
    return db.fetch_data()
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

### Quality Requirements

- **MyPy Strict Mode**: Zero tolerance for type errors in src/
- **Runtime Type Checking**: All public APIs validate types
- **Generic Types**: Use TypeVar for proper generic constraints
- **No Any Types**: Explicitly type all parameters and returns
- **Coverage minimum**: 75% (enforced in Makefile)
- **Test markers**: Use appropriate markers (unit, integration, slow)
- **Fixtures**: Use provided fixtures from conftest.py
- **Shared domain**: Import test entities from shared_test_domain.py

### Code Style Requirements

- **Line length**: 79 characters maximum (PEP8 strict)
- **Naming**: FlextXxx prefix for all public exports
- **Imports**: Absolute imports from flext_core
- **Docstrings**: Required for all public APIs
- **Type hints**: Mandatory for all functions and methods
- **Error handling**: FlextResult return types for all business operations

## Ecosystem Integration

### Foundation Library Role

FLEXT Core is the architectural foundation for 32+ projects in the FLEXT data integration ecosystem:

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

## Common Issues and Solutions

### Import Errors (FREQUENT)

```bash
# FIRST: Always verify actual exports
Read src/flext_core/__init__.py

# SECOND: Check if class exists where expected
grep -r "class ClassName" src/

# THIRD: Test import directly
PYTHONPATH=src python -c "from flext_core import ClassName"
```

### API Signature Changes (FREQUENT)

```bash
# When examples fail, check current API:
Read src/flext_core/result.py  # Check actual method signatures
Read examples/shared_domain.py # Check current domain patterns

# Fix imports systematically:
# .shared_domain → shared_domain (relative to absolute)
# FlextProcessingUtils → direct JSON parsing implementation
```

### Type Checking Issues

```bash
# Check errors with specific codes:
mypy src/ --show-error-codes --show-error-context
pyright src/ --outputformat text

# Focus on src/ first (zero tolerance), then tests (pragmatic)
```

## Development Workflow

### Verification-First Development (CRITICAL)

**ALWAYS verify before asserting anything:**

```bash
# Before claiming something works:
Read file.py                    # Verify actual content
PYTHONPATH=src python file.py   # Test actual execution

# Before assuming imports exist:
Read src/flext_core/__init__.py  # Check actual exports
python -c "from flext_core import Class"  # Verify import works
```

**NEVER assume based on:**
- File names or "logical" patterns
- Previous session memory
- What "should" work

### Disciplined Development Approach

```bash
# 1. VERIFY FIRST (most important lesson)
Read file.py                    # Understand current state
PYTHONPATH=src python file.py   # Test current functionality

# 2. MAKE TARGETED CHANGES
# - One thing at a time
# - Test after each change
# - Mark todos as completed immediately

# 3. QUALITY GATES (only after verification)
make format          # Format code
make validate        # Run all quality gates

# 4. COMMIT WITH EVIDENCE
git add .
git commit -m "Fix: specific issue with evidence of resolution"
```

---

**FLEXT-CORE AUTHORITY**: These guidelines are specific to flext-core development.
**ECOSYSTEM STANDARDS**: See [../CLAUDE.md](../CLAUDE.md) for workspace-wide patterns.
**EVIDENCE-BASED**: All patterns here are proven through implementation and testing.