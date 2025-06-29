# FLX Core - Enterprise Foundation Framework

**Status**: 🟡 Development (95% Complete)
**Based on**: Real implementation from `flx-meltano-enterprise/src/flx_core/`

## Overview

FLX Core is the foundational framework that powers the entire FLX platform. It provides the domain model, business logic, and architectural patterns used by all other FLX modules. Built with Clean Architecture and Domain-Driven Design principles.

## Architecture Overview

```mermaid
graph TB
    subgraph "External Interfaces"
        CLI[flx-cli]
        API[flx-api]
        WEB[flx-web]
        GRPC[flx-grpc]
    end

    subgraph "Core Framework"
        CORE[flx-core]
        AUTH[flx-auth]
        PLUGIN[flx-plugin]
    end

    subgraph "Integration Layer"
        MELTANO[flx-meltano]
        OBS[flx-observability]
    end

    CLI --> CORE
    API --> CORE
    WEB --> CORE
    GRPC --> CORE

    CORE --> AUTH
    CORE --> PLUGIN
    CORE --> MELTANO
    CORE --> OBS

    API --> AUTH
    WEB --> AUTH
    GRPC --> AUTH

    API --> OBS
    WEB --> OBS
    GRPC --> OBS
```

## Real Implementation Status

| Layer              | Lines of Code | Status           | Details                                 |
| ------------------ | ------------- | ---------------- | --------------------------------------- |
| **Domain**         | 3,721         | ✅ 95% Complete  | Entities, value objects, specifications |
| **Application**    | ~2,500        | ✅ 90% Complete  | Command handlers, services              |
| **Infrastructure** | ~3,000        | ✅ 85% Complete  | Repositories, adapters                  |
| **Configuration**  | ~800          | ✅ 100% Complete | Zero hardcoded values                   |

## Module Dependencies

### Core Integrations

#### 🔐 [flx-auth](../flx-auth/README.md)

- **Dependency**: Core provides domain models (User, Role, Permission)
- **Integration Points**:
  - `flx_core.domain.entities.User` → Authentication base
  - `flx_core.domain.value_objects.Email` → Email validation
  - `flx_core.application.services.ServiceResult` → Error handling pattern

#### 🌐 [flx-api](../flx-api/README.md)

- **Dependency**: Core provides business logic and command bus
- **Integration Points**:
  - `flx_core.application.CommandBus` → Command execution
  - `flx_core.application.QueryBus` → Query handling
  - `flx_core.domain.events.EventBus` → Event streaming

#### 🖥️ [flx-web](../flx-web/README.md)

- **Dependency**: Core provides Django integration and models
- **Integration Points**:
  - `flx_core.config.django_integration` → Django settings
  - `flx_core.infrastructure.persistence.models` → ORM models
  - `flx_core.application.services` → Business logic

#### 📡 [flx-grpc](../flx-grpc/README.md)

- **Dependency**: Core provides service implementation
- **Integration Points**:
  - `flx_core.application.handlers` → RPC handlers
  - `flx_core.domain.commands` → Command definitions
  - `flx_core.infrastructure.adapters` → Protocol adapters

#### 🔌 [flx-plugin](../flx-plugin/README.md)

- **Dependency**: Embedded in core, provides plugin infrastructure
- **Integration Points**:
  - `flx_core.plugins.manager` → Plugin lifecycle
  - `flx_core.plugins.discovery` → Plugin discovery
  - `flx_core.plugins.loader` → Dynamic loading

#### 📊 [flx-meltano](../flx-meltano/README.md)

- **Dependency**: Core provides execution framework
- **Integration Points**:
  - `flx_core.domain.entities.Pipeline` → Pipeline models
  - `flx_core.application.ExecutionEngine` → Execution control
  - `flx_core.infrastructure.StateManager` → State persistence

#### 📈 [flx-observability](../flx-observability/README.md)

- **Dependency**: Core integrates monitoring
- **Integration Points**:
  - `flx_core.events.EventBus` → Metric events
  - `flx_core.infrastructure.monitoring` → Health checks
  - `flx_core.application.middleware` → Request tracking

#### 🖥️ [flx-cli](../flx-cli/README.md)

- **Dependency**: Core provides CLI framework
- **Integration Points**:
  - `flx_core.application.cli` → CLI commands
  - `flx_core.config.CliConfig` → CLI configuration
  - `flx_core.application.CommandBus` → Command execution

## Features

### Clean Architecture Implementation

- **Domain Layer**: Pure business logic, no external dependencies
- **Application Layer**: Use cases and orchestration
- **Infrastructure Layer**: External integrations
- **Interface Layer**: API/CLI/Web adapters

### Domain-Driven Design

- **Entities**: User, Pipeline, Plugin, Project, Execution
- **Value Objects**: Email, ProjectName, PipelineConfig, PluginMetadata
- **Aggregates**: Project (root), Pipeline (root)
- **Domain Services**: ValidationService, ExecutionService
- **Specifications**: PipelineSpecification, UserSpecification

### Enterprise Patterns

- **Command/Query Segregation** (CQRS)
- **Event Sourcing** capabilities
- **Repository Pattern** with Unit of Work
- **Dependency Injection** with containers
- **Service Result Pattern** for error handling

## Quick Start

```bash
# Install dependencies
cd /home/marlonsc/pyauto/flx-core
poetry install

# Run tests
poetry run pytest

# Start development
poetry run python -m flx_core.cli
```

## Core Components

### Domain Models

```python
# From entities.py - Enterprise domain models
@dataclass
class Pipeline:
    """Core pipeline entity with business rules."""
    id: PipelineId
    name: PipelineName
    project: ProjectReference
    config: PipelineConfig
    schedule: Optional[Schedule]

    def validate(self) -> ValidationResult:
        """Business rule validation."""

    def can_execute(self) -> bool:
        """Execution authorization."""
```

### Application Services

```python
# From services.py - Business logic orchestration
class PipelineService:
    """Pipeline management with transaction boundaries."""

    async def create_pipeline(
        self,
        command: CreatePipelineCommand
    ) -> ServiceResult[Pipeline]:
        """Create pipeline with validation and events."""
```

### Infrastructure Adapters

```python
# From adapters.py - External integrations
class MeltanoAdapter:
    """Adapt core domain to Meltano SDK."""

    def to_meltano_project(self, project: Project) -> MeltanoProject:
        """Convert domain model to Meltano format."""
```

## Configuration

All configuration is centralized with zero hardcoded values:

```python
# Required environment variables
FLX_CORE_DATABASE_URL=postgresql://user:pass@localhost/flx_core
FLX_CORE_REDIS_URL=redis://localhost:6379/0
FLX_CORE_LOG_LEVEL=INFO

# Domain Configuration
FLX_PIPELINE_MAX_RETRIES=3
FLX_PIPELINE_TIMEOUT_SECONDS=3600
FLX_PROJECT_MAX_PIPELINES=100

# Infrastructure
FLX_REPOSITORY_CACHE_TTL=300
FLX_EVENT_BUS_BUFFER_SIZE=1000
```

## Integration Examples

### Using Core in flx-api

```python
# flx-api integrating with core
from flx_core.application import CommandBus, QueryBus
from flx_core.domain.commands import CreatePipelineCommand

class PipelineEndpoint:
    def __init__(self, command_bus: CommandBus):
        self.command_bus = command_bus

    async def create_pipeline(self, request):
        command = CreatePipelineCommand.from_request(request)
        result = await self.command_bus.execute(command)
        return result.to_response()
```

### Using Core in flx-cli

```python
# flx-cli using core commands
from flx_core.application.cli import CliApplication
from flx_core.domain.commands import ExecutePipelineCommand

@click.command()
@click.argument('pipeline_name')
def execute(pipeline_name):
    """Execute a pipeline."""
    app = CliApplication()
    command = ExecutePipelineCommand(name=pipeline_name)
    result = app.execute_command(command)
    click.echo(result)
```

## Testing

```bash
# Unit tests
poetry run pytest tests/unit/

# Integration tests
poetry run pytest tests/integration/

# Domain tests (no external dependencies)
poetry run pytest tests/domain/

# Coverage report
poetry run pytest --cov=flx_core --cov-report=html
```

## Architecture Decision Records

- [ADR-001: Modularization Strategy](docs/architecture/ADR-001-modularization-strategy.md)
- [ADR-002: Authentication Architecture](docs/architecture/ADR-002-authentication-architecture.md)
- [ADR-003: Plugin System Design](docs/architecture/ADR-003-plugin-system-design.md)
- [ADR-004: CLI Architecture](docs/architecture/ADR-004-cli-architecture.md)
- [ADR-005: Event-Driven Architecture](docs/architecture/ADR-005-event-architecture.md)

## Development Guidelines

### Adding New Features

1. Start with domain model changes
2. Add application layer use cases
3. Implement infrastructure adapters
4. Update interface layers (API/CLI/Web)
5. Add comprehensive tests

### Code Organization

```
src/flx_core/
├── domain/          # Pure business logic
├── application/     # Use cases and services
├── infrastructure/  # External integrations
└── interfaces/      # API/CLI/Web adapters
```

## Performance Considerations

- Domain models use immutable data classes
- Repository pattern enables caching
- Event bus supports async processing
- Command bus allows queuing
- Infrastructure adapters are pluggable

## Security

- No business logic in interfaces
- Domain validation on all inputs
- Repository pattern prevents SQL injection
- Event sourcing provides audit trail
- Service result pattern for error handling

## License

Part of the FLX Platform - Enterprise License
