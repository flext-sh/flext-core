# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FLEXT-Core is a standalone, foundational framework implementing Domain-Driven Design (DDD) and Clean Architecture principles in Python 3.13. It provides the essential building blocks for enterprise applications without dependencies on any other FLEXT modules.

**Key Independence Features:**

- Zero dependencies on other FLEXT modules
- Self-contained domain models and infrastructure  
- Can be used as a foundation for any Python application
- Other FLEXT modules depend on this, but not vice versa
- Only requires dynaconf and pyyaml as external dependencies

**Quality Standards:**

- **Zero Tolerance**: 100% strict compliance with all quality gates
- **Type Safety**: MyPy strict mode with complete type coverage
- **Code Quality**: Ruff with ALL rules enabled but sensible ignores for development
- **Test Coverage**: 95% minimum (configured in pyproject.toml)
- **Security**: Bandit + pip-audit + detect-secrets scanning

## Development Commands

### Environment Setup

```bash
# Complete development setup (recommended)
make setup                 # Install dependencies + pre-commit hooks

# Manual steps if needed
make install               # Install dependencies with dev group
poetry install --with dev,test,docs  # Alternative Poetry command

# Verify installation
make poetry-check          # Verify Poetry is available
make status               # Check quality status
```

### Testing

```bash
# Run tests
make test                  # Run all tests with coverage (95% minimum)
make test-unit            # Run only unit tests
make test-integration     # Run only integration tests  
make test-watch          # Run tests in watch mode

# Run specific test
poetry run pytest tests/unit/test_specific.py::TestClass::test_method -xvs

# Run with markers (defined in pyproject.toml)
poetry run pytest -m "unit" -v              # Unit tests only
poetry run pytest -m "integration" -v       # Integration tests only
poetry run pytest -m "core" -v              # Core framework tests
poetry run pytest -m "requires_db" -v       # Database tests
poetry run pytest -m "not slow" -v          # Skip slow tests
```

### Code Quality

```bash
# All quality checks (run this before committing)
make check                # Run ALL quality checks (zero tolerance)

# Individual checks
make lint                 # Ruff linting with sensible ignores
make format-check        # Check code formatting (Black + Ruff)
make type-check          # MyPy strict type checking
make security            # Bandit + pip-audit + detect-secrets
make complexity          # Radon complexity analysis with vulture
make docstring-check     # Documentation coverage (80% minimum)

# Fix issues automatically
make fix                 # Auto-fix formatting and imports
make format              # Format code with Black + Ruff
make sort-imports        # Sort imports with isort
make lint-fix           # Auto-fix linting issues where possible
```

### Development

```bash
# Development commands
make run                 # Run the application (python -m flext_core)
make shell              # Open Poetry shell
make watch              # Watch for changes and run checks

# Status and validation
make status             # Show current quality status with metrics
make validate           # Validate 100% strict compliance

# Build and publish
make build              # Build distribution packages
make clean              # Remove cache and build files

# Documentation  
make docs               # Build documentation with mkdocs
make docs-serve        # Serve documentation at localhost:8000

# Dependency management
make update             # Update dependencies  
make lock               # Update poetry.lock
make outdated          # Check for outdated dependencies
```

## Architecture

### Clean Architecture Layers

1. **Domain Layer** (`src/flext_core/domain/`)

    - Pure business logic, no external dependencies
    - Entities, Value Objects, Domain Events, Specifications
    - Base classes in `pydantic_base.py` - foundation for all domain objects
    - Business types in `types.py` using Python 3.13 features
    - Constants and shared models for framework-wide usage

2. **Application Layer** (`src/flext_core/application/`)

    - Use case orchestration via command/query handlers
    - Pipeline service implementation
    - Domain services for complex business logic
    - Application services coordinate between layers

3. **Infrastructure Layer** (`src/flext_core/infrastructure/`)

    - In-memory repository implementations
    - Database persistence base classes
    - External service adapters
    - Infrastructure concerns like caching and messaging

4. **Configuration System** (`src/flext_core/config/`)
    - Multi-environment configuration management
    - Adapters for CLI, Django, and Singer frameworks
    - Dependency injection container with Lato
    - Validation and settings hierarchy

### Key Design Patterns

- **CQRS**: Separate command and query responsibilities
- **Repository Pattern**: Abstract data access
- **Unit of Work**: Transaction boundary management
- **Domain Events**: Decoupled communication
- **Service Result**: Type-safe error handling without exceptions
- **Specifications**: Encapsulated business rules

### How Other Modules Use FLEXT-Core

FLEXT-Core is designed as a foundation that other modules build upon:

- **flext-api**: Uses command bus and domain models for REST endpoints
- **flext-grpc**: Uses handlers and repositories for RPC services
- **flext-web**: Uses Django integration utilities and domain entities
- **flext-cli**: Uses application services and command patterns
- **flext-auth**: Extends domain models for authentication
- **flext-meltano**: Uses pipeline entities and execution models

**Important**: FLEXT-Core has no knowledge of these modules and works independently

## Code Standards

### Type Safety

- MyPy strict mode enforced (`mypy --strict`)
- All functions must have type annotations
- Use `ServiceResult[T]` for operations that can fail
- Prefer domain types over primitives (e.g., `UserId` over `str`)

### Code Quality

- Ruff with ALL rules enabled but sensible development ignores
- Black formatting with 88 character line length  
- 95% minimum test coverage (configured in pyproject.toml)
- Complexity analysis with Radon and dead code detection with Vulture

### Testing Approach

- Test files mirror source structure in `tests/`
- Use pytest fixtures for test data (see `conftest.py`)
- Async tests use `pytest-asyncio`  
- Multiple test markers available: unit, integration, core, slow, requires_db, requires_redis
- Coverage reports generated in HTML and XML formats in `reports/` directory

### Domain Modeling

- All domain objects extend from base classes in `pydantic_base.py`
- Value objects are immutable
- Entities have identity (ID)
- Aggregates manage consistency boundaries
- Domain events record what happened

## Common Development Tasks

### Adding a New Domain Entity

1. Define entity in `domain/core.py` or `domain/pipeline.py` extending appropriate base class
2. Ensure it extends from base classes in `domain/pydantic_base.py`
3. Add repository interface if needed (using Protocol pattern)
4. Implement repository in `infrastructure/memory.py` or `infrastructure/persistence/`
5. Write comprehensive tests in `tests/domain/` and `tests/integration/`

### Adding a New Command/Query

1. Define command/query in the appropriate domain module
2. Create handler in `application/handlers.py` or `application/pipeline.py`
3. Register handler with the service or command bus
4. Add unit test in `tests/application/`
5. Add integration test verifying full flow

### Working with Configuration

1. All config through environment variables or `config/` module
2. Use `BaseConfig` and `BaseSettings` base classes for configuration objects
3. Adapters available for CLI, Django, and Singer frameworks  
4. See `.env.example` for all available configuration options
5. Dynaconf used for advanced configuration management with validation

### Performance Analysis

```bash
# Run performance profiling
make profile            # Generate profile.stats file

# Run benchmarks (if benchmark tests exist)
make benchmark         # Run pytest --benchmark-only

# Check for outdated dependencies
make outdated          # Show outdated packages
```

## Important Notes

- **Independence**: This module has zero dependencies on other FLEXT modules
- **Foundation Role**: Other modules depend on this, so maintain backward compatibility
- **Pure Domain**: Domain layer has minimal external dependencies (only Pydantic + Python stdlib)
- **Clean Boundaries**: Infrastructure implements domain ports, never the reverse
- **Extensibility**: Designed to be extended by other modules without modification
- **Self-Contained**: Can be used standalone in any Python project
- **Configuration**: All config via environment variables (see `.env.example`)
- **Dependencies**: Only dynaconf and pyyaml required; other services optional

## Project Structure

```
src/flext_core/
├── domain/          # Business logic and domain models
│   ├── constants.py     # Framework constants and enums
│   ├── core.py         # Core domain abstractions
│   ├── pipeline.py     # Pipeline domain entities
│   ├── pydantic_base.py # Base classes for all domain objects
│   ├── types.py        # Domain types and value objects
│   ├── mixins.py       # Reusable domain behaviors
│   └── shared_models.py # Shared domain models
├── application/     # Use cases and orchestration
│   ├── handlers.py  # Command/Query handlers
│   └── pipeline.py  # Pipeline service implementation
├── infrastructure/  # External integrations
│   ├── memory.py    # In-memory repository implementations
│   └── persistence/    # Database base classes
│       └── base.py     # Base repository patterns
└── config/          # Configuration management
    ├── base.py         # Base configuration classes
    ├── adapters/       # Framework adapters
    │   ├── cli.py      # CLI configuration
    │   ├── django.py   # Django integration
    │   └── singer.py   # Singer/Meltano integration
    ├── dynaconf_bridge.py # Dynaconf integration
    ├── flext_config.py    # Main config classes
    └── validators.py      # Configuration validation

tests/               # Test suite (mirrors src structure)
├── application/    # Application layer tests
├── config/         # Configuration tests
├── domain/         # Domain layer tests
├── infrastructure/ # Infrastructure tests
├── conftest.py     # Pytest configuration and fixtures
└── test_imports.py # Import validation tests
```

## Development Tools Integration

### Cursor AI
This project includes comprehensive `.cursorrules` that define:
- Python 3.13 best practices with type hints
- Pydantic v2 patterns and Clean Architecture enforcement  
- Testing patterns with Given-When-Then approach
- Error handling with ServiceResult pattern
- Formatting rules (100 char line length, Black + isort)

### Pre-commit Hooks
Setup with `make pre-commit` or `make setup`. Includes:
- Code formatting (Black, Ruff format)
- Import sorting (isort) 
- Linting (Ruff with development-friendly rules)
- Type checking (MyPy strict mode)
- Security scanning (Bandit, detect-secrets)

### IDE Support
- MyPy configuration in `pyproject.toml` for strict type checking
- Pytest markers for test organization
- Coverage configuration for accurate reporting
- MkDocs for documentation generation
