# Usage Examples

## Basic Pipeline Management

### Creating and Managing Pipelines

```python
import asyncio
from flext_core.domain.pipeline import Pipeline, PipelineName
from flext_core.application.pipeline import PipelineService, CreatePipelineCommand
from flext_core.infrastructure.memory import InMemoryRepository

async def main():
    # Setup
    repository = InMemoryRepository()
    service = PipelineService(pipeline_repo=repository)
    
    # Create a pipeline
    command = CreatePipelineCommand(
        name="data-etl",
        description="Extract, transform, and load customer data"
    )
    
    result = await service.create_pipeline(command)
    if result.is_success:
        pipeline = result.value
        print(f"Created pipeline: {pipeline.pipeline_name}")
        
        # Execute the pipeline
        from flext_core.application.pipeline import ExecutePipelineCommand
        
        exec_command = ExecutePipelineCommand(
            pipeline_id=str(pipeline.pipeline_id.value)
        )
        
        exec_result = await service.execute_pipeline(exec_command)
        if exec_result.is_success:
            execution = exec_result.value
            print(f"Execution status: {execution.execution_status}")

asyncio.run(main())
```

## Configuration Management

### Application Settings

```python
from flext_core.config.base import BaseSettings
from flext_core.config.validators import validate_url, validate_port

class DatabaseConfig(BaseSettings):
    host: str = "localhost"
    port: int = 5432
    database: str = "myapp"
    
    def get_connection_url(self) -> str:
        return f"postgresql://{self.host}:{self.port}/{self.database}"

class AppConfig(BaseSettings):
    api_url: str = "https://api.example.com"
    debug: bool = False
    log_level: str = "INFO"
    
    # Custom validation
    def __post_init__(self):
        self.api_url = validate_url(self.api_url)

# Usage with environment variables
# Set FLEXT_API_URL=https://prod.api.com
# Set FLEXT_DEBUG=true
config = AppConfig()
print(f"API URL: {config.api_url}")
print(f"Debug mode: {config.debug}")
```

### Environment-Specific Configuration

```python
import os
from flext_core.config.base import BaseSettings

class Settings(BaseSettings):
    environment: str = "development"
    database_url: str = "sqlite:///dev.db"
    redis_url: str = "redis://localhost:6379"
    
    @classmethod
    def for_environment(cls, env: str):
        """Factory method for environment-specific settings."""
        if env == "production":
            return cls(
                environment="production",
                database_url=os.getenv("DATABASE_URL", "postgresql://..."),
                redis_url=os.getenv("REDIS_URL", "redis://..."),
            )
        elif env == "test":
            return cls(
                environment="test", 
                database_url="sqlite:///:memory:",
                redis_url="redis://localhost:6379/1",
            )
        return cls()

# Usage
env = os.getenv("ENVIRONMENT", "development")
settings = Settings.for_environment(env)
```

## Domain Modeling

### Rich Domain Objects

```python
from flext_core.domain.pipeline import Pipeline, PipelineName, ExecutionStatus
from flext_core.domain.pydantic_base import DomainEvent

# Custom domain events
class PipelineCompletedEvent(DomainEvent):
    pipeline_id: str
    execution_time: float
    records_processed: int

# Custom pipeline with business logic
class DataPipeline(Pipeline):
    """Specialized pipeline for data processing."""
    
    def process_batch(self, batch_size: int = 1000):
        """Process data in batches."""
        execution = self.execute()
        
        # Business logic here
        records_processed = 0
        
        # Emit custom event
        self.add_event(PipelineCompletedEvent(
            pipeline_id=str(self.pipeline_id.value),
            execution_time=30.5,
            records_processed=records_processed
        ))
        
        return execution

# Usage
pipeline = DataPipeline(
    pipeline_name=PipelineName(value="batch-processor"),
    pipeline_description="Process data in batches"
)

execution = pipeline.process_batch(batch_size=500)
```

## Testing Patterns

### Unit Testing Services

```python
import pytest
from unittest.mock import AsyncMock
from flext_core.application.pipeline import PipelineService, CreatePipelineCommand
from flext_core.domain.pipeline import Pipeline, PipelineName

class TestPipelineService:
    @pytest.fixture
    def mock_repository(self):
        """Create a mock repository for testing."""
        return AsyncMock()
    
    @pytest.fixture
    def service(self, mock_repository):
        """Create a service instance with mocked repository."""
        return PipelineService(pipeline_repo=mock_repository)
    
    async def test_create_pipeline_success(self, service, mock_repository):
        # Given
        command = CreatePipelineCommand(
            name="test-pipeline",
            description="Test pipeline"
        )
        
        expected_pipeline = Pipeline(
            pipeline_name=PipelineName(value="test-pipeline"),
            pipeline_description="Test pipeline"
        )
        mock_repository.save.return_value = expected_pipeline
        
        # When
        result = await service.create_pipeline(command)
        
        # Then
        assert result.is_success
        pipeline = result.value
        assert pipeline.pipeline_name.value == "test-pipeline"
        mock_repository.save.assert_called_once()
```

### Integration Testing

```python
import pytest
from flext_core.application.pipeline import PipelineService
from flext_core.infrastructure.memory import InMemoryRepository

class TestPipelineIntegration:
    @pytest.fixture
    def service(self):
        repository = InMemoryRepository()
        return PipelineService(pipeline_repo=repository)
    
    async def test_full_pipeline_lifecycle(self, service):
        # Create
        create_cmd = CreatePipelineCommand(
            name="integration-test",
            description="Full lifecycle test"
        )
        
        create_result = await service.create_pipeline(create_cmd)
        assert create_result.is_success
        
        pipeline = create_result.value
        pipeline_id = str(pipeline.pipeline_id.value)
        
        # Execute
        exec_cmd = ExecutePipelineCommand(pipeline_id=pipeline_id)
        exec_result = await service.execute_pipeline(exec_cmd)
        assert exec_result.is_success
        
        # Deactivate
        deactivate_result = await service.deactivate_pipeline(pipeline_id)
        assert deactivate_result.is_success
```

## Error Handling

### ServiceResult Pattern

```python
from flext_core.domain.types import ServiceResult

def divide_numbers(a: float, b: float) -> ServiceResult[float]:
    """Safe division using ServiceResult pattern."""
    if b == 0:
        return ServiceResult.fail("Division by zero")
    
    result = a / b
    return ServiceResult.ok(result)

# Usage
result = divide_numbers(10, 2)
if result.is_success:
    print(f"Result: {result.value}")
else:
    print(f"Error: {result.error}")

# Chaining operations
def process_calculation(a: float, b: float) -> ServiceResult[str]:
    division_result = divide_numbers(a, b)
    if not division_result.is_success:
        return ServiceResult.fail(f"Calculation failed: {division_result.error}")
    
    formatted = f"Result: {division_result.value:.2f}"
    return ServiceResult.ok(formatted)
```

## Advanced Patterns

### Custom Repository Implementation

```python
from typing import Protocol
from flext_core.domain.pipeline import Pipeline
from flext_core.domain.types import ServiceResult

class DatabaseRepository:
    """Example database repository implementation."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    async def save(self, pipeline: Pipeline) -> Pipeline:
        # Database save logic here
        return pipeline
    
    async def get_by_id(self, pipeline_id) -> Pipeline | None:
        # Database query logic here
        return None
    
    async def delete(self, pipeline_id) -> bool:
        # Database delete logic here
        return True

# Usage with service
repository = DatabaseRepository("postgresql://user:pass@localhost/db")
service = PipelineService(pipeline_repo=repository)

# Example usage
async def main():
    command = CreatePipelineCommand(name="db-pipeline", description="Database pipeline")
    result = await service.create_pipeline(command)
    if result.is_success:
        print(f"Pipeline created: {result.value.pipeline_name}")

asyncio.run(main())
```