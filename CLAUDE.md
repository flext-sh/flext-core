# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FLEXT-Core is a standalone, foundational framework implementing Domain-Driven Design (DDD) and Clean Architecture principles in Python 3.13. It provides the essential building blocks for enterprise applications without dependencies on any other FLEXT modules.

**Key Independence Features:**
- Zero dependencies on other FLEXT modules
- Self-contained domain models and infrastructure
- Can be used as a foundation for any Python application
- Other FLEXT modules depend on this, but not vice versa

## Development Commands

### Environment Setup
```bash
# Initial setup
poetry install              # Install runtime dependencies
poetry install --with dev   # Install with development dependencies

# Using Make commands (preferred)
make install               # Install dependencies
make install-dev          # Install with dev dependencies
```

### Testing
```bash
# Run tests
make test                  # Run all tests
make test-coverage        # Run tests with coverage report (must be >= 95%)
make test-unit            # Run only unit tests  
make test-integration     # Run only integration tests
make test-watch          # Run tests in watch mode

# Run specific test
poetry run pytest tests/unit/test_specific.py::TestClass::test_method -xvs
```

### Code Quality
```bash
# All quality checks (run this before committing)
make check

# Individual checks
make lint                 # Ruff linting (ALL rules enabled - maximum strictness)
make format              # Format with Black + Ruff
make type-check          # MyPy strict type checking
make security            # Bandit security analysis
make complexity          # Radon complexity analysis
```

### Development
```bash
# Start development mode
make dev                 # Start with hot reload
make dev-test           # Quick test during development

# Build package
make build              # Build distribution packages
make publish-test       # Publish to TestPyPI
make publish           # Publish to PyPI (requires credentials)
```

## Architecture

### Clean Architecture Layers

1. **Domain Layer** (`src/flext_core/domain/`)
   - Pure business logic, no external dependencies
   - Entities, Value Objects, Domain Events, Specifications
   - Base classes in `pydantic_base.py` - foundation for all domain objects
   - Business types in `advanced_types.py` using Python 3.13 features

2. **Application Layer** (`src/flext_core/application/`)
   - Use case orchestration via command/query handlers
   - Command Bus pattern for business operations
   - Domain services for complex business logic
   - Application services coordinate between layers

3. **Infrastructure Layer** (`src/flext_core/infrastructure/`)
   - Database persistence with SQLAlchemy 2.0 async
   - Repository implementations following port interfaces
   - External service adapters
   - Unit of Work pattern for transaction management

4. **Events System** (`src/flext_core/events/`)
   - Event-driven architecture with domain events
   - Lato-based event bus for decoupling
   - Async event handling throughout

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
- Ruff with ALL rules enabled (maximum strictness)
- Black formatting with 88 character line length
- 95% minimum test coverage
- Complexity limits: cyclomatic < 10, cognitive < 15

### Testing Approach
- Test files mirror source structure in `tests/`
- Use pytest fixtures for test data
- Async tests use `pytest-asyncio`
- Integration tests use real PostgreSQL (via docker-compose)

### Domain Modeling
- All domain objects extend from base classes in `pydantic_base.py`
- Value objects are immutable
- Entities have identity (ID)
- Aggregates manage consistency boundaries
- Domain events record what happened

## Common Development Tasks

### Adding a New Domain Entity
1. Define entity in `domain/entities.py` extending `DomainEntity`
2. Add repository interface in `domain/ports.py`
3. Implement repository in `infrastructure/persistence/repositories/`
4. Add SQLAlchemy model in `infrastructure/persistence/models/`
5. Write comprehensive tests in `tests/unit/domain/` and `tests/integration/`

### Adding a New Command
1. Define command in `domain/commands.py`
2. Create handler in `application/handlers/`
3. Register handler with command bus
4. Add integration test verifying full flow

### Running Database Migrations
```bash
# Generate migration
poetry run alembic revision --autogenerate -m "Description"

# Apply migrations
poetry run alembic upgrade head

# Rollback
poetry run alembic downgrade -1
```

## Important Notes

- **Independence**: This module has zero dependencies on other FLEXT modules
- **Foundation Role**: Other modules depend on this, so maintain backward compatibility
- **Pure Domain**: Domain layer has no external dependencies (only Python stdlib + Pydantic)
- **Clean Boundaries**: Infrastructure implements domain ports, never the reverse
- **Extensibility**: Designed to be extended by other modules without modification
- **Self-Contained**: Can be used standalone in any Python project
- **Configuration**: All config via environment variables (see `.env.example`)
- **External Dependencies**: Only PostgreSQL and Redis for persistence/caching