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
│   ├── result.py           # FlextResult[T] railway pattern with map/flat_map
│   ├── container.py        # Dependency injection container with FlextResult
│   ├── exceptions.py       # Exception hierarchy with error codes and metrics
│   ├── constants.py        # FlextConstants, enums, error codes, performance metrics
│   └── typings.py          # Type definitions and aliases (T, U, V, etc.)
│
├── Domain Layer (DDD Patterns)  
│   ├── entities.py         # FlextEntity base with identity and lifecycle
│   ├── value_objects.py    # FlextValueObject immutable patterns
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
├── Support Modules (Utilities & Extensions)
│   ├── mixins.py           # Reusable behavior patterns (timestamps, logging, etc.)
│   ├── decorators.py       # Enterprise decorator patterns (validation, caching, etc.)
│   ├── utilities.py        # Helper functions, generators, type guards
│   ├── fields.py           # Field validation and metadata
│   ├── semantic.py         # Semantic modeling and analysis
│   ├── testing_utilities.py # Test helpers and factories
│   ├── delegation_system.py # Mixin delegation patterns
│   ├── schema_processing.py # Schema validation and processing
│   ├── type_adapters.py    # Type adaptation utilities
│   └── legacy.py           # Backward compatibility layer
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

### Mandatory Development Patterns

#### 1. Error Handling Pattern (MANDATORY)

**Never raise exceptions in business logic - always return FlextResult:**

```python
from flext_core import FlextResult

def business_operation(data: dict) -> FlextResult[ProcessedData]:
    """MANDATORY: All business operations must return FlextResult."""
    if not data:
        return FlextResult.fail("Data required", error_code="VALIDATION_ERROR")
    
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

**Always use the global container for service registration:**

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
        return FlextResult.fail("Database service unavailable")
    
    db = db_result.unwrap()
    return db.fetch_data()
```

#### 3. Domain Modeling Pattern (MANDATORY)

**Use DDD patterns for all domain modeling:**

```python
from flext_core import FlextEntity, FlextValueObject, FlextAggregateRoot

class Email(FlextValueObject):
    """Value objects are immutable and compared by value."""
    address: str

    def validate(self) -> FlextResult[None]:
        if "@" not in self.address:
            return FlextResult.fail("Invalid email format")
        return FlextResult.ok(None)

class User(FlextEntity):
    """Entities have identity and lifecycle."""
    name: str
    email: Email

    def activate(self) -> FlextResult[None]:
        """Business logic returns FlextResult."""
        if self.is_active:
            return FlextResult.fail("User already active")
        self.is_active = True
        return FlextResult.ok(None)

class UserAggregate(FlextAggregateRoot):
    """Aggregate roots enforce consistency boundaries."""
    user: User
    
    def register_user(self, data: dict) -> FlextResult[User]:
        # Domain events are automatically tracked
        return self.create_user(data)
```

### Type Safety Requirements

- **MyPy Strict Mode**: Zero tolerance for type errors
- **Runtime Type Checking**: All public APIs validate types
- **Generic Types**: Use TypeVar for proper generic constraints
- **No Any Types**: Explicitly type all parameters and returns

### Error Handling Requirements

- **Always use FlextResult**: Never raise exceptions in business logic
- **Chain operations**: Use map/flat_map for composition
- **Handle all paths**: Both success and failure must be handled
- **Unwrap carefully**: Only unwrap when you've checked success

### Testing Requirements

- **Coverage minimum**: 75% (enforced in Makefile)
- **Test markers**: Use appropriate markers (unit, integration, slow)
- **Fixtures**: Use provided fixtures from conftest.py
- **Shared domain**: Import test entities from shared_test_domain.py

### Enhanced Testing Patterns

#### Test Execution Strategies

```bash
# Single test with maximum debugging
poetry run pytest tests/unit/core/test_result.py::TestFlextResult::test_map -xvs --tb=long

# Fast feedback loop (unit tests without slow tests)
poetry run pytest -m "unit and not slow" --tb=short -q

# Coverage with detailed missing lines report
poetry run pytest tests/unit/ --cov=src/flext_core --cov-report=term-missing

# Last failed tests with fail-fast
poetry run pytest --lf --ff -x

# Parallel execution for speed
poetry run pytest -n auto tests/unit/

# Specific test patterns
poetry run pytest tests/unit/ -k "test_result" -v      # Tests matching pattern
poetry run pytest tests/unit/ --collect-only -q       # Show available tests
```

#### Test Markers for Different Scenarios

```bash
# Core functionality tests
poetry run pytest -m unit              # Fast unit tests
poetry run pytest -m integration       # Integration tests
poetry run pytest -m core              # Core framework tests
poetry run pytest -m ddd               # Domain-driven design tests
poetry run pytest -m architecture      # Architectural pattern tests

# Performance and edge cases
poetry run pytest -m performance       # Performance benchmarks
poetry run pytest -m boundary          # Boundary condition tests
poetry run pytest -m error_path        # Error path scenarios
poetry run pytest -m happy_path        # Happy path scenarios

# Quality and compliance
poetry run pytest -m pep8              # PEP8 compliance tests
poetry run pytest -m parametrize_advanced  # Advanced parametrized tests
```

#### Test Development Best Practices

```python
# Use shared domain (MANDATORY)
from shared_test_domain import User, Order, Email  # NOT local models

# Test structure pattern
class TestBusinessOperation:
    """Test class following AAA pattern."""
    
    def test_operation_success_happy_path(self):
        # Arrange
        data = {"email": "test@example.com"}
        
        # Act
        result = business_operation(data)
        
        # Assert
        assert result.success
        assert result.data.email.address == "test@example.com"
    
    def test_operation_failure_invalid_data(self):
        # Arrange
        data = {}
        
        # Act
        result = business_operation(data)
        
        # Assert
        assert result.failure
        assert "Data required" in result.error
        assert result.error_code == "VALIDATION_ERROR"
```

### Quality Gates (MANDATORY)

#### Pre-Commit Requirements

**ALL quality gates must pass before commit:**

```bash
# MANDATORY: Run complete validation pipeline
make validate               # Must pass: 0 lint errors, 0 type errors, 75%+ coverage

# Individual quality checks with zero tolerance
make lint                   # 0 linting errors required (Ruff)
make type-check            # 0 type errors required (MyPy strict)
make test                  # 75% minimum coverage required
make security              # 0 security vulnerabilities (Bandit + pip-audit)
```

#### Specific Quality Thresholds

```bash
# Linting with comprehensive rules (must be 0 errors)
poetry run ruff check src/ tests/ --output-format=github

# Type checking in strict mode (must be 0 errors)
poetry run mypy src/ --strict --show-error-codes --show-error-context

# Coverage with minimum threshold (must be >= 75%)
poetry run pytest --cov=flext_core --cov-fail-under=75 --cov-report=term-missing

# Security scanning (must be 0 high/medium issues)
poetry run bandit -r src/ -ll
poetry run pip-audit --desc --format=json
```

#### Quality Standards Enforcement

- **Line Length**: 79 characters maximum (PEP8 strict) - enforced by Ruff
- **Type Coverage**: 100% type annotations required - enforced by MyPy strict
- **Test Coverage**: 75% minimum - enforced by pytest-cov
- **Security**: Zero high/medium vulnerabilities - enforced by Bandit
- **Dependencies**: No known vulnerabilities - enforced by pip-audit

### Code Style Requirements

- **Line length**: 79 characters maximum (PEP8 strict)
- **Naming**: FlextXxx prefix for all public exports
- **Imports**: Absolute imports from flext_core
- **Docstrings**: Required for all public APIs
- **Type hints**: Mandatory for all functions and methods
- **Error handling**: FlextResult return types for all business operations

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

## Working with Examples and Tests

### Example Structure

The project contains 21 comprehensive examples demonstrating all patterns:

```bash
# Run all examples sequentially
python examples/shared_example_helpers.py

# Run specific pattern examples
python examples/01_flext_result_railway_pattern.py    # FlextResult patterns
python examples/02_flext_container_dependency_injection.py  # DI patterns
python examples/06_flext_entity_valueobject_ddd_patterns.py # DDD patterns
```

### Testing Strategy

- **Shared Domain**: All tests use `shared_test_domain.py` for consistency
- **Test Markers**: Use `pytest -m unit`, `pytest -m integration`, etc.
- **Coverage**: Minimum 75% enforced by CI
- **Type Safety**: MyPy strict mode on all code

### Development Workflow

```bash
# Standard development cycle
make format          # Format code first
make validate        # Run all quality gates
git add .           # Stage changes
git commit -m "..."  # Commit with descriptive message
```

### Key Import Patterns

```python
# Foundation imports (always use these)
from flext_core import FlextResult, FlextContainer, get_flext_container

# Domain modeling
from flext_core import FlextEntity, FlextValueObject, FlextAggregateRoot

# Error handling
from flext_core import FlextError, FlextValidationError

# Testing (use shared domain)
from shared_test_domain import User, Order, Email  # NOT local models
```

## Architecture Principles

### Clean Architecture Implementation

**FLEXT Core follows Clean Architecture with strict dependency rules:**

#### Layer Dependencies (Inward Only)

```
┌─────────────────────────────────────────┐
│  Infrastructure Layer                    │  ← External concerns
│  (config.py, loggings.py, protocols.py) │
└─────────────────┬───────────────────────┘
                  │ depends on
┌─────────────────▼───────────────────────┐
│  Application Layer                       │  ← Use cases, handlers
│  (commands.py, handlers.py, payload.py) │
└─────────────────┬───────────────────────┘
                  │ depends on
┌─────────────────▼───────────────────────┐
│  Domain Layer                            │  ← Pure business logic
│  (entities.py, value_objects.py)        │
└─────────────────┬───────────────────────┘
                  │ depends on
┌─────────────────▼───────────────────────┐
│  Foundation Layer                        │  ← Core patterns
│  (result.py, container.py, typings.py)  │
└─────────────────────────────────────────┘
```

#### Layer Responsibilities

**Foundation Layer (Core Patterns):**
- Railway-oriented programming (`FlextResult`)
- Dependency injection (`FlextContainer`)
- Type definitions and core exceptions
- **Rule**: No dependencies on other layers

**Domain Layer (Business Logic):**
- Entities with identity and lifecycle
- Value objects for immutable data
- Aggregate roots for consistency boundaries
- Domain services for complex operations
- **Rule**: Only depends on Foundation layer

**Application Layer (Use Cases):**
- Commands and queries (CQRS)
- Handlers and orchestration
- Validation and business workflows
- **Rule**: Depends on Domain and Foundation layers only

**Infrastructure Layer (External Concerns):**
- Configuration management
- Logging and observability
- External service protocols
- **Rule**: Can depend on all inner layers

### Domain-Driven Design Patterns

#### Entity Pattern

```python
from flext_core import FlextEntity

class User(FlextEntity):
    """Entities have identity and mutable state."""
    
    # Identity (required)
    id: str
    
    # Mutable state
    name: str
    email: str
    is_active: bool = False
    
    def activate(self) -> FlextResult[None]:
        """Business operations return FlextResult."""
        if self.is_active:
            return FlextResult.fail("User already active")
        
        self.is_active = True
        # Domain events can be raised here
        return FlextResult.ok(None)
```

#### Value Object Pattern

```python
from flext_core import FlextValueObject

class Email(FlextValueObject):
    """Value objects are immutable and compared by value."""
    
    address: str
    
    def __post_init__(self) -> None:
        """Value objects validate themselves on creation."""
        result = self.validate()
        if result.failure:
            raise ValueError(result.error)
    
    def validate(self) -> FlextResult[None]:
        if "@" not in self.address:
            return FlextResult.fail("Invalid email format")
        return FlextResult.ok(None)
```

#### Aggregate Root Pattern

```python
from flext_core import FlextAggregateRoot

class UserAggregate(FlextAggregateRoot):
    """Aggregate roots enforce consistency boundaries."""
    
    user: User
    profile: UserProfile
    
    def change_email(self, new_email: str) -> FlextResult[None]:
        """Aggregate operations maintain consistency."""
        email_result = Email.create(new_email)
        if email_result.failure:
            return email_result.map_error(lambda e: f"Email change failed: {e}")
        
        # Business rule: email change requires profile update
        self.user.email = new_email
        self.profile.update_email_timestamp()
        
        # Raise domain event
        self.raise_event("EmailChanged", {"user_id": self.user.id, "new_email": new_email})
        
        return FlextResult.ok(None)
```

### CQRS Implementation

#### Command Pattern

```python
from flext_core import FlextResult

class RegisterUserCommand:
    """Commands represent intent to change state."""
    name: str
    email: str

class RegisterUserHandler:
    """Handlers execute commands."""
    
    def __init__(self, user_repo: UserRepository) -> None:
        self.user_repo = user_repo
    
    def handle(self, command: RegisterUserCommand) -> FlextResult[str]:
        """Commands return success/failure with minimal data."""
        return (
            self._validate_command(command)
            .flat_map(lambda _: self._create_user(command))
            .map(lambda user: user.id)
        )
    
    def _validate_command(self, command: RegisterUserCommand) -> FlextResult[None]:
        if not command.name:
            return FlextResult.fail("Name required")
        if not command.email:
            return FlextResult.fail("Email required")
        return FlextResult.ok(None)
    
    def _create_user(self, command: RegisterUserCommand) -> FlextResult[User]:
        user = User(
            id=generate_id(),
            name=command.name,
            email=command.email
        )
        return self.user_repo.save(user)
```

#### Query Pattern

```python
class GetUserQuery:
    """Queries represent requests for data."""
    user_id: str

class UserView:
    """Query results are read-only view models."""
    id: str
    name: str
    email: str
    is_active: bool

class GetUserHandler:
    """Query handlers return data without side effects."""
    
    def __init__(self, user_repo: UserRepository) -> None:
        self.user_repo = user_repo
    
    def handle(self, query: GetUserQuery) -> FlextResult[UserView]:
        """Queries return data or not found."""
        return (
            self.user_repo.find_by_id(query.user_id)
            .map(lambda user: UserView(
                id=user.id,
                name=user.name,
                email=user.email,
                is_active=user.is_active
            ))
        )
```

## Architecture Gaps (TODOs)

### Critical Missing Components

1. **Event Sourcing**: FlextAggregateRoot exists but event store missing
2. **CQRS Bus**: Command/Query bus not implemented (handlers exist)
3. **Plugin System**: No foundation for FlexCore plugin architecture
4. **Cross-Language Bridge**: Python-Go type mapping undefined
5. **Observability**: Correlation IDs exist but tracing incomplete

These gaps are documented but not blocking current development.
