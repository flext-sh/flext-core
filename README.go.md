# FLEXT Core - Go Implementation

This is the Go conversion of the FLEXT Core module, maintaining the same hexagonal architecture and domain-driven design patterns from the Python implementation.

## Architecture Overview

The Go implementation follows the same architectural principles as the Python version:

### Domain-Driven Design (DDD)
- **Entities**: Objects with identity (`Pipeline`, `PipelineExecution`)
- **Value Objects**: Immutable objects (`PipelineID`, `ExecutionStatus`, `Duration`)
- **Aggregates**: Cluster of entities (`Pipeline` as aggregate root)
- **Domain Events**: Events published when domain state changes
- **Specifications**: Encapsulated business rules

### Hexagonal Architecture (Ports & Adapters)
- **Domain Layer**: Core business logic (`pkg/domain/`)
- **Application Layer**: Use cases and orchestration (`pkg/application/`)
- **Infrastructure Layer**: External adapters (`pkg/infrastructure/`)
- **Ports**: Interfaces for external dependencies (`pkg/domain/ports/`)

## Project Structure

```
flext-core/
├── cmd/                          # Command-line applications
├── pkg/                          # Public API packages
│   ├── domain/                   # Domain layer
│   │   ├── entities/             # Domain entities
│   │   ├── valueobjects/         # Value objects
│   │   ├── specifications/       # Business rules
│   │   ├── ports/               # Port interfaces
│   │   ├── base.go              # Base domain types
│   │   └── result.go            # ServiceResult type
│   ├── application/             # Application layer
│   │   ├── commands/            # Command DTOs
│   │   ├── queries/             # Query DTOs
│   │   ├── handlers/            # Command/Query handlers
│   │   └── services/            # Application services
│   └── infrastructure/          # Infrastructure layer
│       ├── persistence/         # Database implementations
│       └── adapters/           # External service adapters
├── internal/                    # Private application packages
│   └── config/                 # Configuration management
├── docs/                       # Documentation
├── examples/                   # Usage examples
├── tests/                      # Test files
├── scripts/                    # Build and deployment scripts
├── go.mod                      # Go module definition
└── README.go.md               # This file
```

## Key Components

### Domain Types

#### Base Types
- `EntityID`: Unique identifier for domain entities
- `DomainEntity`: Base for all entities with identity
- `DomainValueObject`: Base for immutable value objects
- `DomainAggregateRoot`: Base for aggregate roots with domain events
- `ServiceResult[T]`: Result type for operations that can fail

#### Pipeline Domain
- `Pipeline`: Main aggregate root representing a data pipeline
- `PipelineExecution`: Entity representing a pipeline run
- `PipelineID`, `ExecutionID`: Strongly-typed identifiers
- `ExecutionStatus`: Enumeration of execution states
- `PipelineStep`: Value object representing pipeline steps

### Application Layer

#### Commands (Write Operations)
- `CreatePipelineCommand`: Create a new pipeline
- `UpdatePipelineCommand`: Update pipeline properties
- `ExecutePipelineCommand`: Start pipeline execution
- `CancelExecutionCommand`: Cancel running execution

#### Queries (Read Operations)
- `GetPipelineQuery`: Retrieve pipeline by ID
- `ListPipelinesQuery`: List pipelines with filtering
- `GetExecutionQuery`: Retrieve execution details
- `ListExecutionsQuery`: List executions with filtering

#### Handlers
- `PipelineCommandHandlers`: Handles pipeline-related commands
- `PipelineQueryHandlers`: Handles pipeline-related queries

### Infrastructure Ports

#### Repository Interfaces
- `PipelineRepository`: Pipeline persistence interface
- `PipelineExecutionRepository`: Execution persistence interface

#### Service Interfaces
- `EventBusPort`: Domain event publishing
- `LoggingPort`: Structured logging
- `MetricsPort`: Metrics collection
- `ConfigurationPort`: Configuration management

## Usage Examples

### Creating a Pipeline

```go
package main

import (
    "context"
    "fmt"

    "github.com/flext-sh/flext-core/pkg/application/commands"
    "github.com/flext-sh/flext-core/pkg/application/handlers"
    "github.com/flext-sh/flext-core/pkg/domain/valueobjects"
)

func main() {
    ctx := context.Background()
    
    // Create command
    cmd := commands.NewCreatePipelineCommand("my-pipeline", "A sample data pipeline")
    cmd.Tags = []string{"etl", "sample"}
    cmd.SetConfiguration("batch_size", 1000)
    
    // Handle command
    handler := handlers.NewPipelineCommandHandlers(...)
    result := handler.HandleCreatePipeline(ctx, cmd)
    
    if result.IsSuccess() {
        pipeline, _ := result.Value()
        fmt.Printf("Created pipeline: %s\n", pipeline.PipelineID.String())
    } else {
        fmt.Printf("Error: %v\n", result.Error())
    }
}
```

### Adding Pipeline Steps

```go
// Add a step to the pipeline
step := valueobjects.NewPipelineStep("extract-data", "extractor", 1)
step.SetConfiguration("source_table", "users")
step.SetConfiguration("connection_string", "postgres://...")

addStepCmd := commands.NewAddPipelineStepCommand(
    pipeline.PipelineID.String(),
    step.Name,
    step.StepType,
    step.Order,
)
addStepCmd.Configuration = step.Configuration

result := handler.HandleAddPipelineStep(ctx, addStepCmd)
```

### Executing a Pipeline

```go
// Execute the pipeline
executeCmd := commands.NewExecutePipelineCommand(
    pipeline.PipelineID.String(),
    "user@example.com",
    "manual",
)
executeCmd.InputData["param1"] = "value1"

executionResult := handler.HandleExecutePipeline(ctx, executeCmd)
if executionResult.IsSuccess() {
    execution, _ := executionResult.Value()
    fmt.Printf("Started execution: %s\n", execution.ExecutionID.String())
}
```

### Querying Pipelines

```go
// List all active pipelines
query := queries.NewListPipelinesQuery()
query.ActiveOnly = true
query.Limit = &[]int{10}[0]

queryHandler := handlers.NewPipelineQueryHandlers(...)
result := queryHandler.HandleListPipelines(ctx, query)

if result.IsSuccess() {
    pipelines, _ := result.Value()
    for _, pipeline := range pipelines {
        fmt.Printf("Pipeline: %s (%s)\n", 
            pipeline.Name.Value(), 
            pipeline.PipelineID.String())
    }
}
```

## Domain Events

The system publishes domain events for important state changes:

```go
// Pipeline events
- PipelineCreated
- PipelineActivated
- PipelineDeactivated
- StepAdded
- StepRemoved
- StepUpdated

// Execution events
- PipelineExecutionCreated
- PipelineExecutionStarted
- PipelineExecutionCompleted
- PipelineExecutionFailed
- PipelineExecutionCancelled
```

### Event Handling

```go
type PipelineEventHandler struct {
    logger ports.LoggingPort
}

func (h *PipelineEventHandler) Handle(ctx context.Context, event domain.DomainEvent) error {
    switch event.EventType {
    case "PipelineCreated":
        h.logger.Info(ctx, "Pipeline created", map[string]interface{}{
            "pipeline_id": event.EventData["pipeline_id"],
            "name":        event.EventData["name"],
        })
    case "PipelineExecutionStarted":
        h.logger.Info(ctx, "Pipeline execution started", map[string]interface{}{
            "execution_id": event.EventData["execution_id"],
            "pipeline_id":  event.EventData["pipeline_id"],
        })
    }
    return nil
}

func (h *PipelineEventHandler) CanHandle(eventType string) bool {
    return strings.HasPrefix(eventType, "Pipeline")
}
```

## Testing

### Unit Tests

```go
func TestCreatePipeline(t *testing.T) {
    // Arrange
    name, _ := valueobjects.NewPipelineName("test-pipeline")
    
    // Act
    pipeline, err := entities.NewPipeline(name, "Test description")
    
    // Assert
    assert.NoError(t, err)
    assert.Equal(t, "test-pipeline", pipeline.Name.Value())
    assert.True(t, pipeline.IsActive)
    assert.Equal(t, 0, len(pipeline.Steps))
    
    // Check domain events
    events := pipeline.DomainEvents()
    assert.Equal(t, 1, len(events))
    assert.Equal(t, "PipelineCreated", events[0].EventType)
}
```

### Integration Tests

```go
func TestPipelineRepository(t *testing.T) {
    // Setup test database
    db := setupTestDB(t)
    repo := NewPipelineRepository(db)
    
    // Create pipeline
    name, _ := valueobjects.NewPipelineName("integration-test")
    pipeline, _ := entities.NewPipeline(name, "Integration test")
    
    // Save and retrieve
    ctx := context.Background()
    saveResult := repo.Save(ctx, pipeline)
    assert.True(t, saveResult.IsSuccess())
    
    findResult := repo.FindByID(ctx, pipeline.PipelineID)
    assert.True(t, findResult.IsSuccess())
    
    found, _ := findResult.Value()
    assert.Equal(t, pipeline.PipelineID, found.PipelineID)
    assert.Equal(t, pipeline.Name.Value(), found.Name.Value())
}
```

## Building and Running

### Prerequisites

- Go 1.23 or later
- Dependencies specified in `go.mod`

### Build

```bash
# Build the module
go build ./...

# Run tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Run specific tests
go test ./pkg/domain/entities -v

# Build CLI (when available)
go build -o bin/flext-core ./cmd/flext-core
```

### Development

```bash
# Install dependencies
go mod download

# Run linter
golangci-lint run

# Format code
go fmt ./...

# Generate documentation
godoc -http=:6060
```

## Configuration

Configuration is handled through the `ConfigurationPort` interface, allowing different implementations:

```go
type Config struct {
    Database struct {
        URL             string `json:"url"`
        MaxConnections  int    `json:"max_connections"`
        MaxIdleTime     string `json:"max_idle_time"`
    } `json:"database"`
    
    EventBus struct {
        Provider string `json:"provider"`
        URL      string `json:"url"`
    } `json:"event_bus"`
    
    Logging struct {
        Level  string `json:"level"`
        Format string `json:"format"`
    } `json:"logging"`
}
```

## Migration from Python

This Go implementation maintains API compatibility with the Python version:

### Equivalent Concepts

| Python | Go |
|--------|-----|
| `DomainBaseModel` | `DomainBaseModel` |
| `DomainEntity` | `DomainEntity` |
| `DomainValueObject` | `DomainValueObject` |
| `ServiceResult[T]` | `ServiceResult[T]` |
| `PipelineId` | `PipelineID` |
| `ExecutionStatus` | `ExecutionStatus` |

### Key Differences

1. **Type Safety**: Go's type system provides compile-time safety
2. **Performance**: Better performance characteristics
3. **Concurrency**: Built-in goroutine support for concurrent operations
4. **Memory Management**: Automatic garbage collection
5. **Interfaces**: Go interfaces are satisfied implicitly

## Integration with FLEXT Ecosystem

This core module integrates with other FLEXT components:

- **flext-auth**: Authentication and authorization
- **flext-api**: REST API gateway
- **flext-grpc**: gRPC services
- **flext-web**: Web dashboard
- **flext-cli**: Command-line interface
- **flext-meltano**: Meltano integration
- **flext-observability**: Monitoring and metrics

The Go implementation maintains the same domain contracts, allowing seamless integration with other language implementations.

## Contributing

1. Follow Go conventions and best practices
2. Maintain test coverage above 80%
3. Use dependency injection for all external dependencies
4. Follow the hexagonal architecture principles
5. Write comprehensive documentation

## Performance Considerations

- Use Go's built-in concurrency primitives for parallel processing
- Implement connection pooling for database operations
- Use structured logging for observability
- Implement proper error handling and recovery
- Monitor memory usage and garbage collection

## Future Enhancements

- [ ] Add support for pipeline templating
- [ ] Implement circuit breaker pattern for external calls
- [ ] Add distributed tracing support
- [ ] Implement event sourcing for complete audit trail
- [ ] Add support for pipeline versioning
- [ ] Implement advanced scheduling capabilities

---

This Go implementation provides a solid foundation for the FLEXT platform while maintaining architectural consistency with the original Python design.