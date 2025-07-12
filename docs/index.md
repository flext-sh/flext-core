# FLEXT Core Documentation

**The foundation of the FLEXT Framework ecosystem**

FLEXT Core provides the essential domain models, application services, and infrastructure abstractions that power the entire FLEXT Framework. All other FLEXT modules (flext-api, flext-web, flext-auth, flext-meltano, etc.) build upon Core's foundations.

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-94%25-brightgreen.svg)]()
[![Architecture](https://img.shields.io/badge/architecture-Clean%2FDDD-purple.svg)]()

## Overview

FLEXT Core is the **foundational module** that enables the FLEXT Framework's modular architecture. It provides:

- **Shared Domain Models** used across all FLEXT modules
- **Configuration Foundation** with workspace-wide settings management  
- **Application Patterns** (CQRS, Repository, ServiceResult) used throughout FLEXT
- **Clean Architecture** foundation that other modules extend

## Quick Links

- [API Reference](API.md) - Complete API documentation for Core components
- [Usage Examples](examples.md) - How to use Core independently and within FLEXT
- [Configuration Guide](configuration.md) - Settings management for FLEXT ecosystem
- [Architecture Guide](ARCHITECTURAL_TRUTH.md) - Design principles and FLEXT integration

## Key Features

- 🏗️ **FLEXT Foundation** - Core patterns used by all FLEXT modules
- 🎯 **Domain Models** - Shared entities used across flext-api, flext-web, flext-meltano
- ⚡ **Modern Python** - Python 3.13 + Pydantic v2 for the entire ecosystem
- 🔒 **Type Safety** - 100% typed foundation ensures FLEXT-wide reliability
- 🧪 **Production Proven** - 94% test coverage, powers enterprise FLEXT deployments
- 🌐 **Framework Agnostic** - Works standalone or as part of FLEXT ecosystem

## Getting Started

### Installation

```bash
# Development installation with Poetry (recommended)
poetry install --with dev

# Or with pip for basic usage
pip install -e .
```

### Basic Usage

```python
from flext_core.domain.pipeline import Pipeline, PipelineName
from flext_core.application.pipeline import PipelineService
from flext_core.infrastructure.memory import InMemoryRepository

# Setup
service = PipelineService(pipeline_repo=InMemoryRepository())

# Create a pipeline
from flext_core.application.pipeline import CreatePipelineCommand

command = CreatePipelineCommand(
    name="data-processing",
    description="Process customer data"
)

result = await service.create_pipeline(command)
if result.is_success:
    pipeline = result.value
    print(f"Created: {pipeline.pipeline_name}")
```

## FLEXT Ecosystem Role

### Foundation for All FLEXT Modules

- **flext-api** (Go): Uses Core's domain models via gRPC interfaces
- **flext-web** (Django): Extends Core entities with web-specific functionality
- **flext-auth** (Python): Builds authentication on Core's user abstractions
- **flext-meltano** (Python): Leverages Core's pipeline entities for ETL operations
- **flext-cli** (Python): Uses Core's application services for command-line tools

### Shared Patterns Across FLEXT

- **ServiceResult** - Type-safe error handling used in all modules
- **Repository** - Data access abstraction for flext-web, flext-auth storage
- **BaseSettings** - Configuration foundation inherited by all FLEXT modules
- **Domain Events** - Inter-module communication within FLEXT ecosystem
- **Clean Architecture** - Dependency inversion enforced framework-wide

## Documentation Structure

```
docs/
├── index.md              # FLEXT Core overview and ecosystem role
├── API.md                # Core components used by all FLEXT modules
├── examples.md           # Standalone and FLEXT-integrated usage
├── configuration.md      # FLEXT workspace configuration patterns
├── ARCHITECTURAL_TRUTH.md # Clean Architecture and FLEXT integration
└── getting-started/      # Setup within FLEXT workspace
```

## FLEXT Workspace Context

```bash
# FLEXT development workspace
/home/marlonsc/flext/
├── flext-core/          # 👑 This module - Foundation
├── flext-api/           # Go API server using Core models
├── flext-web/           # Django web interface extending Core
├── flext-auth/          # Authentication built on Core patterns
├── flext-meltano/       # ETL framework using Core pipelines
├── flext-cli/           # CLI tools leveraging Core services
└── [other modules...]   # All modules depend on flext-core
```

## Project Status

- **Test Coverage**: 94% (foundation stability for entire FLEXT)
- **Type Safety**: 100% (mypy strict mode enforced workspace-wide)
- **FLEXT Integration**: Powers 9 active FLEXT modules
- **Python Version**: 3.13+ (modern foundation for all FLEXT development)
- **Production Status**: Deployed in enterprise FLEXT installations

## Contributing

1. Setup development environment: `make setup`
2. Run all quality checks: `make check`
3. Run tests with coverage: `make test`
4. Validate strict compliance: `make validate`

## License

MIT License
