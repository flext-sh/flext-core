# FLEXT Core

**Foundation framework for the FLEXT enterprise ecosystem**

Modern Python 3.13 • Pydantic v2 • Clean Architecture • Domain-Driven Design

> **Part of FLEXT Framework**: The foundational module that powers all other FLEXT components including flext-api, flext-web, flext-auth, and flext-meltano.

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-94%25-brightgreen.svg)]()
[![Architecture](https://img.shields.io/badge/architecture-Clean%2FDDD-purple.svg)]()

## Overview

FLEXT Core is the **foundational module** of the FLEXT Framework ecosystem. It provides the essential domain models, application services, and infrastructure abstractions that power all other FLEXT modules.

**Within FLEXT Framework:**
- **flext-api** (Go) uses Core's domain models via gRPC
- **flext-web** (Django) extends Core's entities for web interfaces
- **flext-auth** builds authentication on Core's user abstractions
- **flext-meltano** leverages Core's pipeline entities for ETL
- **flext-cli** uses Core's application services for command-line tools

**Why FLEXT Core?**
- **Ecosystem foundation** - Shared models used across all FLEXT modules
- **Type-safe** - Full mypy strict compliance for reliability
- **Well-tested** - 94% coverage with real-world scenarios
- **Minimal dependencies** - Only essential packages (Pydantic, Dynaconf)
- **Production-proven** - Powers enterprise FLEXT deployments
- **Framework agnostic** - Works standalone or with other FLEXT modules

**Core Features:**
- **Domain Foundation** - Rich business models shared across FLEXT
- **Clean Architecture** - Dependency inversion for all FLEXT modules
- **Type-Safe Operations** - ServiceResult pattern used framework-wide
- **Configuration System** - Unified config for entire FLEXT ecosystem
- **Repository Abstractions** - Data access patterns for all storage needs
- **Event System** - Domain events for inter-module communication

## Architecture

The project follows Clean Architecture with clear separation of concerns:

```
src/flext_core/
├── domain/          # Business logic and entities
│   ├── core.py      # Base domain abstractions
│   ├── pipeline.py  # Pipeline domain models
│   └── types.py     # Domain types and value objects
├── application/     # Use cases and services  
│   ├── handlers.py  # Command/Query handlers
│   └── pipeline.py  # Pipeline application service
├── infrastructure/ # External integrations
│   └── memory.py    # In-memory repository implementation
└── config/          # Configuration management
    ├── base.py      # Base configuration classes
    └── validators.py # Configuration validators
```

## Quick Start

### Installation

```bash
# Development installation with Poetry (recommended)
poetry install --with dev

# Or with pip for basic usage
pip install -e .
```

### Basic Usage

```python
import asyncio
from flext_core.domain.pipeline import Pipeline, PipelineName
from flext_core.application.pipeline import PipelineService, CreatePipelineCommand
from flext_core.infrastructure.memory import InMemoryRepository

async def main():
    # Setup service with repository
    repository = InMemoryRepository()
    service = PipelineService(pipeline_repo=repository)
    
    # Create a new pipeline
    command = CreatePipelineCommand(
        name="data-processing",
        description="Process customer data"
    )
    
    result = await service.create_pipeline(command)
    if result.is_success:
        pipeline = result.value
        print(f"✅ Created: {pipeline.pipeline_name.value}")
        
        # Execute the pipeline
        from flext_core.application.pipeline import ExecutePipelineCommand
        exec_cmd = ExecutePipelineCommand(
            pipeline_id=str(pipeline.pipeline_id.value)
        )
        
        exec_result = await service.execute_pipeline(exec_cmd)
        if exec_result.is_success:
            execution = exec_result.value
            print(f"🚀 Execution status: {execution.execution_status}")
    else:
        print(f"❌ Error: {result.error}")

asyncio.run(main())
```

### Configuration

```python
from flext_core.config.base import BaseSettings
from flext_core.config.validators import validate_url

class AppSettings(BaseSettings):
    api_url: str = "https://api.example.com"
    database_url: str = "sqlite:///app.db"
    debug: bool = False
    max_connections: int = 10
    
    def __post_init__(self):
        self.api_url = validate_url(self.api_url)

# Automatically loads from FLEXT_* environment variables
settings = AppSettings()
```

## Development

### Setup

```bash
# Clone and install
git clone <repository>
cd flext-core

# Complete development setup
make setup         # Install deps + pre-commit hooks

# Or manual installation
poetry install --with dev
make pre-commit

# Verify everything works
make check
```

### Testing

```bash
# Run all tests with coverage (Make)
make test

# Or with Poetry directly
poetry run pytest tests/ --cov=flext_core --cov-report=term-missing

# Run specific test categories
poetry run pytest tests/domain/ -v      # Domain tests
poetry run pytest tests/application/ -v # Application tests
poetry run pytest tests/config/ -v      # Configuration tests

# Watch mode for development
make test-watch
```

### Code Quality

```bash
# All quality checks (recommended)
make check

# Individual checks
make lint          # Ruff linting (ALL rules)
make type-check    # MyPy strict mode
make security      # Security scans
make format-check  # Check formatting

# Auto-fix issues
make fix           # Fix all auto-fixable issues
make format        # Format code
```

### Project Metrics

| Aspect | Status | Target |
|--------|--------|--------|
| Test Coverage | 94% | ≥95% |
| Type Safety | 100% | 100% |
| Dependencies | 1 (Pydantic) | Minimal |
| Python Version | 3.13+ | Latest |

## Key Concepts

### ServiceResult Pattern

Type-safe error handling without exceptions:

```python
from flext_core.domain.types import ServiceResult

def divide(a: float, b: float) -> ServiceResult[float]:
    if b == 0:
        return ServiceResult.fail("Cannot divide by zero")
    return ServiceResult.ok(a / b)

result = divide(10, 2)
if result.is_success:
    print(f"Result: {result.value}")  # 5.0
else:
    print(f"Error: {result.error}")
```

### Repository Pattern

Abstract data access with clean interfaces:

```python
from flext_core.infrastructure.memory import InMemoryRepository

# Works with any entity type
repository = InMemoryRepository()

# Type-safe operations
entity = await repository.save(pipeline)
found = await repository.get_by_id(pipeline_id)
deleted = await repository.delete(pipeline_id)
```

### Domain Events

Decouple business logic with events:

```python
# Domain events are automatically emitted
pipeline.create()  # Emits PipelineCreated event
execution = pipeline.execute()  # Emits PipelineExecuted event

# Access events for integration
events = pipeline.get_events()
for event in events:
    print(f"Event: {event.__class__.__name__}")
```

## FLEXT Ecosystem Integration

### How FLEXT Modules Use Core

**Direct Dependencies (import flext_core):**
- **flext-api** (Go): Implements Core's domain models in protobuf definitions
- **flext-web** (Django): Extends Core entities with Django model mixins
- **flext-cli** (Python): Uses Core's application services and commands
- **flext-auth** (Python): Extends Core's user/role domain models
- **flext-meltano** (Python): Built entirely on Core's pipeline entities

**Indirect Usage (via shared patterns):**
- **flext-grpc**: Implements Core's Repository patterns in gRPC services
- **flext-plugin**: Uses Core's configuration system for plugin management
- **flext-observability**: Monitors Core's domain events and ServiceResults

### Workspace Integration

```bash
# FLEXT workspace structure
/home/marlonsc/flext/
├── flext-core/          # 👑 Foundation (this module)
├── flext-api/           # → imports flext_core domain models
├── flext-web/           # → extends flext_core entities  
├── flext-auth/          # → builds on flext_core user models
├── flext-meltano/       # → uses flext_core pipeline framework
├── flext-cli/           # → leverages flext_core application layer
└── [other modules...]   # → all depend on flext-core patterns
```

**Design Principle**: Core has **zero dependencies** on other FLEXT modules but provides the foundation they all build upon.

## Configuration

FLEXT Core provides the configuration foundation used across all FLEXT modules:

```python
from flext_core.config.base import BaseSettings

class AppSettings(BaseSettings):
    database_url: str = "sqlite:///app.db"
    debug: bool = False
    log_level: str = "INFO"

# Environment variable support with FLEXT_ prefix
settings = AppSettings()  # Used by all FLEXT modules
```

## Testing

The project includes comprehensive tests with 94% coverage:

```python
# Example test structure
class TestPipelineService:
    async def test_create_pipeline_success(self) -> None:
        # Given
        command = CreatePipelineCommand(
            name="test-pipeline",
            description="Test pipeline"
        )
        
        # When
        result = await self.service.create_pipeline(command)
        
        # Then
        assert result.is_success
        pipeline = result.value
        assert pipeline.pipeline_name.value == "test-pipeline"
```

## Dependencies

**Runtime Dependencies** (shared across FLEXT ecosystem):

```toml
[tool.poetry.dependencies]
python = ">=3.13,<3.14"  # Modern Python for all FLEXT modules
dynaconf = ">=3.2.0"     # Configuration management
pyyaml = ">=6.0.0"       # Config file support

# Workspace-level dependencies (provided by parent):
# - pydantic>=2.11.0     # Validation (used by all modules)
# - pydantic-settings    # Settings management
# - structlog            # Structured logging
# - rich                 # Terminal formatting
```

**Development Dependencies:**

```toml
[tool.poetry.group.dev.dependencies]
# Testing framework (standards for all FLEXT modules)
pytest = ">=8.0.0"
pytest-asyncio = ">=0.23.0"
pytest-cov = ">=4.0.0"

# Code quality (enforced workspace-wide)
ruff = ">=0.8.0"         # Linting and formatting
mypy = ">=1.13.0"        # Type checking
bandit = ">=1.8.0"       # Security scanning

# Documentation (MkDocs stack used across FLEXT)
mkdocs = ">=1.6.0"
mkdocs-material = ">=9.5.0"
```

## Contributing

### FLEXT Core Development

1. **Setup FLEXT workspace environment**: `make setup`
2. **Run all quality checks**: `make check` (enforces FLEXT standards)
3. **Run tests with coverage**: `make test` (95% minimum)
4. **Validate strict compliance**: `make validate`

### FLEXT Ecosystem Impact

**When contributing to Core, consider:**
- Changes affect **all FLEXT modules** that import core components
- Domain model changes require updates in flext-api protobuf definitions
- Configuration changes impact all modules using BaseSettings
- Repository pattern changes affect flext-web and flext-auth persistence

**Test Across Ecosystem:**
```bash
# Test Core changes don't break dependent modules
cd ../flext-api && make test
cd ../flext-web && make test
cd ../flext-meltano && make test
```

## License

MIT License
