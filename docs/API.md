# API Reference

## Domain Layer

### Pipeline

The core domain entity representing a data processing pipeline.

```python
from flext_core.domain.pipeline import Pipeline, PipelineName

pipeline = Pipeline(
    pipeline_name=PipelineName(value="data-processing"),
    pipeline_description="Process customer data"
)
```

**Methods:**

- `create()` - Initialize pipeline and emit creation event
- `execute()` - Execute pipeline and return execution entity
- `deactivate()` - Mark pipeline as inactive

### PipelineExecution

Represents a single execution of a pipeline.

```python
execution = pipeline.execute()
print(f"Status: {execution.execution_status}")
print(f"Started: {execution.started_at}")
```

## Application Layer

### PipelineService

Main application service for pipeline operations.

```python
from flext_core.application.pipeline import PipelineService, CreatePipelineCommand
from flext_core.infrastructure.memory import InMemoryRepository

service = PipelineService(pipeline_repo=InMemoryRepository())

# Create pipeline
command = CreatePipelineCommand(name="etl", description="ETL pipeline")
result = await service.create_pipeline(command)

if result.is_success:
    pipeline = result.value
```

**Methods:**

- `create_pipeline(command)` - Create new pipeline
- `execute_pipeline(command)` - Execute existing pipeline
- `get_pipeline(query)` - Retrieve pipeline by ID
- `deactivate_pipeline(pipeline_id)` - Deactivate pipeline

### Commands and Queries

**CreatePipelineCommand**

```python
command = CreatePipelineCommand(
    name="pipeline-name",
    description="Pipeline description"
)
```

**ExecutePipelineCommand**

```python
command = ExecutePipelineCommand(pipeline_id="uuid-string")
```

**GetPipelineQuery**

```python
query = GetPipelineQuery(pipeline_id="uuid-string")
```

## Infrastructure Layer

### InMemoryRepository

In-memory implementation of the repository pattern.

```python
from flext_core.infrastructure.memory import InMemoryRepository

repo = InMemoryRepository()
await repo.save(entity)
entity = await repo.get_by_id(entity_id)
success = await repo.delete(entity_id)
```

## Configuration

### BaseSettings

Base class for application settings with environment variable support.

```python
from flext_core.config.base import BaseSettings

class AppSettings(BaseSettings):
    database_url: str = "sqlite:///app.db"
    debug: bool = False

settings = AppSettings()  # Automatically reads FLEXT_* env vars
```

### Validators

Built-in validators for common configuration values.

```python
from flext_core.config.validators import validate_url, validate_port

# Validate URL format
url = validate_url("https://api.example.com")

# Validate port number
port = validate_port("8080")
```

## Error Handling

### ServiceResult

Type-safe result pattern for operations that can fail.

```python
from flext_core.domain.types import ServiceResult

result = await service.create_pipeline(command)

if result.is_success:
    pipeline = result.value
    print(f"Created: {pipeline.pipeline_name}")
else:
    print(f"Error: {result.error}")
```

**Methods:**

- `is_success` - Check if operation succeeded
- `value` - Get the successful result value
- `error` - Get the error message if failed
- `ServiceResult.ok(value)` - Create successful result
- `ServiceResult.fail(error)` - Create failed result
