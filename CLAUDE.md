# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FLEXT Core is the foundational library for the entire FLEXT ecosystem - a production-ready enterprise foundation framework built on Clean Architecture, Domain-Driven Design (DDD), and Python 3.13. This is a pure library (no CLI) that serves as the architectural base for 32+ FLEXT projects.

**Key Characteristics:**

- Python 3.13 only with modern type hints and PEP8 strict compliance
- Minimal runtime dependencies (pydantic, dependency-injector)
- Clean Architecture + DDD + CQRS patterns
- Enterprise-grade quality standards with 95%+ test coverage requirement
- Type-safe error handling with FlextResult pattern

## Essential Commands

### Quality Gates (MANDATORY)

```bash
# Full validation - ALL must pass before any commit
make validate              # Complete validation pipeline

# Quick checks
make check                 # lint + type-check (fast feedback)
make test                  # Run full test suite (95% coverage required)

# Individual quality gates
make lint                  # Ruff linting (ALL rules enabled)
make type-check            # MyPy strict mode (zero tolerance)
make security              # bandit + pip-audit
make format                # Code formatting (79 char lines)
```

### Development Setup

```bash
make setup                 # Complete development setup
make install-dev           # Install all dependencies with dev tools
make pre-commit-install    # Setup pre-commit hooks
```

### Testing

```bash
# Run tests
make test                  # Full test suite with coverage (95% minimum)
make test-unit             # Unit tests only (excludes integration tests)
make test-integration      # Integration tests only
make test-fast             # Tests without coverage (faster feedback)
make test-watch            # Watch mode for continuous testing
make coverage-html         # Generate HTML coverage report

# Run specific tests
poetry run pytest tests/test_result.py -v
poetry run pytest tests/test_container.py -v
poetry run pytest tests/unit/core/test_result_comprehensive.py -v

# Test with markers (configured in pyproject.toml)
poetry run pytest -m unit              # Unit tests (isolated components)
poetry run pytest -m integration       # Integration tests (component interaction)
poetry run pytest -m e2e               # End-to-end tests
poetry run pytest -m pep8              # PEP8 compliance validation
poetry run pytest -m core              # Core framework tests
poetry run pytest -m ddd               # Domain-driven design tests
poetry run pytest -m architecture      # Architectural pattern tests
poetry run pytest -m slow              # Slow-running tests
poetry run pytest -m performance       # Performance tests

# Exclude slow tests for fast feedback
poetry run pytest -m "not slow" -v
```

### Additional Development Tools

```bash
# Dependency management
make deps-show             # Show dependency tree
make deps-audit            # Security audit of dependencies
make deps-update           # Update all dependencies

# Development utilities
make shell                 # Open Python shell with project loaded
make notebook              # Start Jupyter notebook
make pre-commit            # Run pre-commit hooks manually
make fix                   # Auto-fix code issues

# Diagnostics
make doctor                # Complete health check
make diagnose              # Project diagnostics info

# Build and publish
make build                 # Build distribution packages
make publish-test          # Publish to test PyPI

# Documentation
make docs                  # Build documentation
make docs-serve            # Serve docs locally
```

## Architecture Overview

FLEXT Core implements Clean Architecture with Domain-Driven Design patterns:

### Core Module Structure

```
src/flext_core/
├── __init__.py              # Unified modern public API with comprehensive exports
├── result.py                # FlextResult[T] - type-safe error handling
├── container.py             # FlextContainer - enterprise DI system
├── config.py                # FlextBaseSettings - configuration management
├── constants.py             # Core enums and constants
├── flext_types.py           # Modern type definitions and type system
├── payload.py               # FlextPayload/FlextEvent/FlextMessage
├── exceptions.py            # FLEXT exception hierarchy
├── entities.py              # FlextEntity - DDD entities
├── value_objects.py         # FlextValueObject - DDD value objects
├── aggregate_root.py        # FlextAggregateRoot - DDD aggregates
├── domain_services.py       # FlextDomainService - domain services
├── commands.py              # Command pattern (CQRS)
├── handlers.py              # Handler patterns
├── validation.py            # Validation system
├── loggings.py              # Structured logging with structlog
├── fields.py                # Field metadata system
├── mixins.py                # Reusable behavior mixins
├── decorators.py            # Enterprise decorator patterns
├── interfaces.py            # Protocol definitions
├── guards.py                # Validation guards and builders
├── utilities.py             # Utility functions
├── core.py                  # FlextCore main class
├── version.py               # Version management and compatibility
├── py.typed                 # Type information marker
└── _*_base.py               # Base implementation modules (internal)
```

### Core Patterns

**FlextResult Pattern:** Type-safe error handling without exceptions

```python
from flext_core import FlextResult

def process_data(data: str) -> FlextResult[ProcessedData]:
    if not data:
        return FlextResult.fail("Empty data provided")
    return FlextResult.ok(ProcessedData(data))
```

**FlextContainer:** Enterprise dependency injection with type safety

```python
from flext_core import FlextContainer, get_flext_container

container = get_flext_container()
result = container.register("user_service", UserService())
service = container.get("user_service").unwrap()
```

**Domain-Driven Design:** Rich domain models with business logic

```python
from flext_core import FlextEntity, FlextValueObject

class User(FlextEntity):
    name: str
    email: str

    def activate(self) -> FlextResult[None]:
        # Business logic with domain events
        return FlextResult.ok(None)
```

## Quality Standards

### Code Quality Requirements

- **Line Length:** 79 characters maximum (PEP8 strict compliance)
- **Type Safety:** MyPy strict mode with zero tolerance for type errors (configured in pyproject.toml)
- **Test Coverage:** 95% minimum requirement (enforced in Makefile: `MIN_COVERAGE := 95`)
- **Security:** Bandit + pip-audit scanning for all dependencies
- **Linting:** Ruff with comprehensive rule set (extends shared config)
- **Line Length:** 79 characters maximum (PEP8 strict compliance: `PEP8_LINE_LENGTH := 79`)

### Testing Standards

Available test markers (use with `pytest -m <marker>`):

- `unit` - Unit tests (isolated components)
- `integration` - Integration tests (component interaction)
- `e2e` - End-to-end tests (full system testing)
- `slow` - Slow-running tests (deselect with `-m "not slow"`)
- `pep8` - PEP8 compliance validation tests
- `core` - Core framework functionality tests
- `architecture` - Architectural pattern tests
- `ddd` - Domain-driven design tests

### Development Guidelines

- **Python Version:** Python 3.13 only (no backward compatibility)
- **Runtime Dependencies:** Minimal dependencies (pydantic>=2.11.7, pydantic-settings>=2.10.1, structlog>=25.4.0)
- **Development Dependencies:** Rich toolset including ruff, mypy, pytest, bandit (see pyproject.toml)
- **Modern Python:** Use Python 3.13 features with full type annotations
- **Immutable by Default:** Use frozen Pydantic models and `@final` decorators
- **Error Handling:** Always use FlextResult pattern, never raise exceptions in business logic
- **Naming:** Use FlextXxx prefix for all exports to avoid namespace conflicts
- **Code Style:** 79 character line limit, strict PEP8 compliance

## Common Development Patterns

### Service Registration and Retrieval

```python
from flext_core import FlextContainer, get_flext_container

# Get global container
container = get_flext_container()

# Register services (returns FlextResult)
result = container.register("user_service", UserService())
if result.is_success:
    print("Service registered successfully")

# Retrieve services (returns FlextResult[T])
service_result = container.get("user_service")
if service_result.is_success:
    user_service = service_result.data
```

### Type-Safe Error Handling

```python
from flext_core import FlextResult

def validate_user(user_data: dict) -> FlextResult[User]:
    if not user_data.get("email"):
        return FlextResult.fail("Email is required")

    user = User(**user_data)
    return FlextResult.ok(user)

# Chain operations safely
result = (
    validate_user(data)
    .flat_map(lambda u: save_user(u))
    .map(lambda u: format_response(u))
)
```

### Domain Entity Implementation

```python
from flext_core import FlextEntity, FlextResult

class User(FlextEntity):
    name: str
    email: str
    is_active: bool = False

    def activate(self) -> FlextResult[None]:
        if self.is_active:
            return FlextResult.fail("User already active")

        self.is_active = True
        # Domain events can be added here
        return FlextResult.ok(None)
```

### Configuration Management

```python
from flext_core import FlextBaseSettings

class AppSettings(FlextBaseSettings):
    database_url: str = "sqlite:///app.db"
    log_level: str = "INFO"

    class Config:
        env_prefix = "APP_"

# Instantiate settings
settings = AppSettings()
```

### Testing with Fixtures

```python
import pytest
from flext_core import FlextContainer, FlextResult

def test_user_activation(clean_container: FlextContainer):
    # Use clean container fixture
    user_service = UserService()
    clean_container.register("user_service", user_service)

    # Test the service
    result = user_service.activate_user("user123")
    assert result.is_success
    assert result.data.is_active
```

### Shared Domain Pattern

FLEXT Core uses a shared domain pattern for examples and tests to avoid code duplication:

```python
# Use shared domain models in examples and tests
from shared_domain import (
    SharedDomainFactory,
    User as SharedUser,
    log_domain_operation,
)

# Never create local domain models in examples
# Always import from shared_domain
```

## Project Structure

### Test Organization

```
tests/
├── unit/                    # Unit tests (isolated components)
│   ├── core/               # Core framework tests
│   ├── domain/            # Domain model tests
│   ├── patterns/          # Pattern implementation tests
│   └── mixins/           # Mixin behavior tests
├── integration/            # Integration tests
│   ├── configs/          # Configuration integration
│   ├── containers/       # Container integration
│   └── services/         # Service integration
├── e2e/                   # End-to-end tests
├── conftest.py           # Test configuration and fixtures
├── conftest_integration.py # Integration test fixtures
└── shared_test_domain.py  # Shared domain models for testing
```

### Examples Structure

```
examples/
├── 01_flext_result_railway_pattern.py        # Railway-oriented programming
├── 02_flext_container_dependency_injection.py # DI container usage
├── 03_flext_commands_cqrs_pattern.py         # CQRS implementation
├── 04_flext_utilities_generation_formatting.py # Utility functions
├── 05_flext_validation_advanced_system.py    # Validation patterns
├── 06_flext_entity_valueobject_ddd_patterns.py # DDD modeling
├── 07_flext_mixins_multiple_inheritance.py   # Mixin patterns
├── 08_flext_config_enterprise_configuration.py # Configuration management
├── 09_flext_decorators_enterprise_patterns.py # Decorator patterns
├── 10_flext_payload_messaging_events.py      # Event messaging
├── 11_flext_handlers_enterprise_patterns.py  # Handler patterns
├── 12_flext_logging_structured_system.py     # Structured logging
├── 13_flext_interfaces_architecture_patterns.py # Interface patterns
├── 14_flext_exceptions_enterprise_handling.py # Exception handling
├── 15_flext_advanced_examples.py             # Advanced usage
├── 16_flext_integration_example.py           # Integration example
├── 17_flext_working_examples.py              # Working examples
└── shared_domain.py                          # Shared domain models
```

## Important Notes

- **Foundation Library:** No CLI entry points, pure library design (project type: core-library)
- **Type Safety:** All public APIs use FlextResult for error handling
- **Testing:** Use provided fixtures and markers for consistent testing (95% coverage required)
- **Dependencies:** Minimal runtime dependencies for maximum portability
- **Namespace:** All exports use FlextXxx prefix to avoid conflicts
- **Architecture:** Follows Clean Architecture and DDD principles strictly
- **Shared Domain:** Use shared_domain module in examples and tests, never create local models
- **Quality Gates:** All quality gates must pass before commits (validate, lint, type-check, test)
- **Documentation:** Comprehensive docs in docs/ directory with MkDocs
- **Python Version:** Python 3.13 only, no backward compatibility

## TODO: GAPS DE ARQUITETURA IDENTIFICADOS - PRIORIDADE CRÍTICA

### 🚨 GAP 1: Ecosystem Compatibility e Versionamento Strategy

**Status**: CRÍTICO - Foundation library sem garantias de compatibilidade
**Problema**:

- flext-core é dependência crítica de 31+ projetos mas sem compatibility matrix
- Breaking changes em core library podem quebrar ecosystem inteiro instantaneamente
- Sem semantic versioning definido especificamente para ecosystem dependencies
- Upgrade path não documentado para major version changes

**TODO**:

- [ ] Criar comprehensive compatibility testing matrix com todos projetos dependentes
- [ ] Implementar semantic versioning strategy específico para ecosystem foundation
- [ ] Documentar detailed migration guides para breaking changes em flext-core
- [ ] Setup automated compatibility testing pipeline contra todos projetos FLEXT
- [ ] Definir deprecation policy e timeline para breaking changes

### 🚨 GAP 2: Event Sourcing Architecture Foundation Missing

**Status**: CRÍTICO - Event Sourcing promises não delivered
**Problema**:

- Event Sourcing amplamente mencionado em workspace mas zero implementation em core
- FlextAggregateRoot exists mas sem event sourcing capabilities
- Domain Events pattern incomplete - apenas mentions, não working implementation
- Event Store patterns não definidos em foundation library

**TODO**:

- [ ] Implementar complete FlextEventStore com event persistence patterns
- [ ] Criar FlextDomainEvent base class com serialization support
- [ ] Refactor FlextAggregateRoot para full Event Sourcing support
- [ ] Documentar comprehensive Event Sourcing patterns e working examples
- [ ] Implementar event replay mechanisms e projection utilities
- [ ] Criar event versioning strategy para backward compatibility

### 🚨 GAP 3: Plugin Architecture Foundation Completely Missing

**Status**: CRÍTICO - Plugin system central to ecosystem mas zero foundation
**Problema**:

- FlexCore (Go) relies on plugin system mas flext-core has zero plugin foundation
- Plugin interfaces não defined em Python foundation layer
- Hot-swappable components mencionados mas sem base architecture
- Extensibility patterns não documented em core library

**TODO**:

- [ ] Design e implement comprehensive FlextPlugin base class hierarchy
- [ ] Criar plugin registry system com lifecycle management
- [ ] Implement plugin loading mechanisms com dependency resolution
- [ ] Document complete plugin development patterns e best practices
- [ ] Criar plugin testing utilities e validation framework
- [ ] Define plugin interface contracts para Go-Python bridge

### 🚨 GAP 4: CQRS Pattern Implementation Superficial

**Status**: CRÍTICO - CQRS mentioned extensively mas implementation inadequate
**Problema**:

- commands.py e handlers.py exist mas são basic stub implementations
- Command/Query separation não properly architected
- Command Bus e Query Bus critical components completely missing
- Pipeline behaviors (middleware) não implemented

**TODO**:

- [ ] Implement production-ready FlextCommandBus com message routing
- [ ] Criar comprehensive FlextQueryBus com query optimization patterns
- [ ] Document complete CQRS architectural patterns com working examples
- [ ] Implement handler registration e discovery mechanisms
- [ ] Create pipeline behaviors framework (validation, logging, metrics, caching)
- [ ] Add command/query serialization para cross-service communication

### 🚨 GAP 5: Cross-Language Integration Architecture Gap

**Status**: CRÍTICO - Python-Go bridge critical mas não architected
**Problema**:

- flext-core foundation para Go services via Python bridge mas integration undefined
- Type safety between Python-Go não guaranteed nem documented
- Serialization patterns para FlextResult cross-language não implemented
- Bridge performance e reliability concerns não addressed em foundation

**TODO**:

- [ ] Design comprehensive Python-Go type mapping system
- [ ] Implement FlextResult serialization/deserialization para Go integration
- [ ] Create cross-language error handling patterns e best practices
- [ ] Document complete bridge integration architecture patterns
- [ ] Implement performance monitoring para cross-language calls
- [ ] Define data contract versioning para bridge compatibility

### 🚨 GAP 6: Enterprise Observability Foundation Missing

**Status**: CRÍTICO - Enterprise platform mas observability não built into foundation
**Problema**:

- flext-observability project exists mas core library não integrated
- Structured logging exists mas correlation IDs e tracing não built-in
- Metrics collection patterns não defined em foundation level
- Health check patterns não standardized across ecosystem

**TODO**:

- [ ] Integrate observability patterns directly into flext-core foundation
- [ ] Implement correlation ID propagation throughout FlextResult chains
- [ ] Create standardized metrics collection interfaces em core library
- [ ] Define health check contracts para all ecosystem components
- [ ] Document observability best practices para ecosystem developers
- [ ] Implement distributed tracing support em core patterns
