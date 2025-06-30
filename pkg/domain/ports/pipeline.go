package ports

import (
	"context"

	"github.com/flext-sh/flext-core/pkg/domain"
	"github.com/flext-sh/flext-core/pkg/domain/entities"
	"github.com/flext-sh/flext-core/pkg/domain/valueobjects"
)

// PipelineRepository defines the interface for pipeline persistence
// Following go-ddd principles: find vs get, soft deletion, read after write
type PipelineRepository interface {
	// Save persists a new pipeline
	// Following go-ddd: Always read after write to ensure data integrity
	Save(ctx context.Context, pipeline *entities.Pipeline) (*entities.Pipeline, error)
	
	// Update updates an existing pipeline
	// Following go-ddd: Always read after write to ensure data integrity
	Update(ctx context.Context, pipeline *entities.Pipeline) (*entities.Pipeline, error)
	
	// GetByID retrieves a pipeline by its ID
	// Following go-ddd: get methods must return a value or error
	GetByID(ctx context.Context, id valueobjects.PipelineID) (*entities.Pipeline, error)
	
	// FindByID retrieves a pipeline by its ID
	// Following go-ddd: find methods can return nil without error
	FindByID(ctx context.Context, id valueobjects.PipelineID) (*entities.Pipeline, error)
	
	// GetByName retrieves a pipeline by its name
	// Following go-ddd: get methods must return a value or error
	GetByName(ctx context.Context, name valueobjects.PipelineName) (*entities.Pipeline, error)
	
	// FindByName retrieves a pipeline by its name
	// Following go-ddd: find methods can return nil without error
	FindByName(ctx context.Context, name valueobjects.PipelineName) (*entities.Pipeline, error)
	
	// FindAll retrieves all non-deleted pipelines with optional pagination
	// Following go-ddd: find methods can return empty list
	FindAll(ctx context.Context, limit, offset int) ([]*entities.Pipeline, error)
	
	// FindByTag retrieves pipelines by tag
	FindByTag(ctx context.Context, tag string) ([]*entities.Pipeline, error)
	
	// FindActive retrieves all active, non-deleted pipelines
	FindActive(ctx context.Context) ([]*entities.Pipeline, error)
	
	// Delete performs soft deletion on a pipeline
	// Following go-ddd: Always use soft deletion to preserve history
	Delete(ctx context.Context, id valueobjects.PipelineID) error
	
	// Exists checks if a non-deleted pipeline exists
	Exists(ctx context.Context, id valueobjects.PipelineID) (bool, error)
	
	// ExistsByName checks if a non-deleted pipeline with the given name exists
	ExistsByName(ctx context.Context, name valueobjects.PipelineName) (bool, error)
	
	// Count returns the total number of non-deleted pipelines
	Count(ctx context.Context) (int64, error)
}

// PipelineExecutionRepository defines the interface for pipeline execution persistence
// Following go-ddd principles: find vs get, soft deletion, read after write
type PipelineExecutionRepository interface {
	// Save persists a new pipeline execution
	// Following go-ddd: Always read after write to ensure data integrity
	Save(ctx context.Context, execution *entities.PipelineExecution) (*entities.PipelineExecution, error)
	
	// Update updates an existing execution
	// Following go-ddd: Always read after write to ensure data integrity
	Update(ctx context.Context, execution *entities.PipelineExecution) (*entities.PipelineExecution, error)
	
	// GetByID retrieves a pipeline execution by its ID
	// Following go-ddd: get methods must return a value or error
	GetByID(ctx context.Context, id valueobjects.ExecutionID) (*entities.PipelineExecution, error)
	
	// FindByID retrieves a pipeline execution by its ID
	// Following go-ddd: find methods can return nil without error
	FindByID(ctx context.Context, id valueobjects.ExecutionID) (*entities.PipelineExecution, error)
	
	// FindByPipelineID retrieves all non-deleted executions for a pipeline
	FindByPipelineID(ctx context.Context, pipelineID valueobjects.PipelineID, limit, offset int) ([]*entities.PipelineExecution, error)
	
	// FindByStatus retrieves non-deleted executions by status
	FindByStatus(ctx context.Context, status valueobjects.ExecutionStatus, limit, offset int) ([]*entities.PipelineExecution, error)
	
	// FindRunning retrieves all running, non-deleted executions
	FindRunning(ctx context.Context) ([]*entities.PipelineExecution, error)
	
	// FindLatest retrieves the latest non-deleted execution for a pipeline
	FindLatest(ctx context.Context, pipelineID valueobjects.PipelineID) (*entities.PipelineExecution, error)
	
	// Delete performs soft deletion on an execution
	// Following go-ddd: Always use soft deletion to preserve history
	Delete(ctx context.Context, id valueobjects.ExecutionID) error
	
	// GetNextExecutionNumber gets the next execution number for a pipeline
	GetNextExecutionNumber(ctx context.Context, pipelineID valueobjects.PipelineID) (int, error)
	
	// Count returns the total number of non-deleted executions
	Count(ctx context.Context) (int64, error)
	
	// CountByPipeline returns the number of non-deleted executions for a specific pipeline
	CountByPipeline(ctx context.Context, pipelineID valueobjects.PipelineID) (int64, error)
}

// PipelineManagementPort defines the primary port for pipeline management use cases
type PipelineManagementPort interface {
	// CreatePipeline creates a new pipeline
	CreatePipeline(ctx context.Context, name valueobjects.PipelineName, description string) domain.ServiceResult[*entities.Pipeline]
	
	// UpdatePipeline updates an existing pipeline
	UpdatePipeline(ctx context.Context, pipeline *entities.Pipeline) domain.ServiceResult[*entities.Pipeline]
	
	// DeletePipeline deletes a pipeline
	DeletePipeline(ctx context.Context, id valueobjects.PipelineID) domain.ServiceResult[bool]
	
	// GetPipeline retrieves a pipeline by ID
	GetPipeline(ctx context.Context, id valueobjects.PipelineID) domain.ServiceResult[*entities.Pipeline]
	
	// GetPipelineByName retrieves a pipeline by name
	GetPipelineByName(ctx context.Context, name valueobjects.PipelineName) domain.ServiceResult[*entities.Pipeline]
	
	// ListPipelines lists all pipelines with pagination
	ListPipelines(ctx context.Context, limit, offset int) domain.ServiceResult[[]*entities.Pipeline]
	
	// ListActivePipelines lists all active pipelines
	ListActivePipelines(ctx context.Context) domain.ServiceResult[[]*entities.Pipeline]
	
	// ActivatePipeline activates a pipeline
	ActivatePipeline(ctx context.Context, id valueobjects.PipelineID) domain.ServiceResult[*entities.Pipeline]
	
	// DeactivatePipeline deactivates a pipeline
	DeactivatePipeline(ctx context.Context, id valueobjects.PipelineID) domain.ServiceResult[*entities.Pipeline]
	
	// AddPipelineStep adds a step to a pipeline
	AddPipelineStep(ctx context.Context, pipelineID valueobjects.PipelineID, step valueobjects.PipelineStep) domain.ServiceResult[*entities.Pipeline]
	
	// RemovePipelineStep removes a step from a pipeline
	RemovePipelineStep(ctx context.Context, pipelineID valueobjects.PipelineID, stepName string) domain.ServiceResult[*entities.Pipeline]
	
	// UpdatePipelineStep updates a step in a pipeline
	UpdatePipelineStep(ctx context.Context, pipelineID valueobjects.PipelineID, stepName string, step valueobjects.PipelineStep) domain.ServiceResult[*entities.Pipeline]
}

// PipelineExecutionPort defines the primary port for pipeline execution use cases
type PipelineExecutionPort interface {
	// ExecutePipeline starts a new pipeline execution
	ExecutePipeline(ctx context.Context, pipelineID valueobjects.PipelineID, triggeredBy, triggerType string) domain.ServiceResult[*entities.PipelineExecution]
	
	// GetExecution retrieves an execution by ID
	GetExecution(ctx context.Context, id valueobjects.ExecutionID) domain.ServiceResult[*entities.PipelineExecution]
	
	// ListExecutions lists executions for a pipeline
	ListExecutions(ctx context.Context, pipelineID valueobjects.PipelineID, limit, offset int) domain.ServiceResult[[]*entities.PipelineExecution]
	
	// ListRunningExecutions lists all running executions
	ListRunningExecutions(ctx context.Context) domain.ServiceResult[[]*entities.PipelineExecution]
	
	// CancelExecution cancels a running execution
	CancelExecution(ctx context.Context, id valueobjects.ExecutionID) domain.ServiceResult[*entities.PipelineExecution]
	
	// GetExecutionLogs retrieves logs for an execution
	GetExecutionLogs(ctx context.Context, id valueobjects.ExecutionID, limit int) domain.ServiceResult[[]string]
	
	// UpdateExecutionStatus updates the status of an execution
	UpdateExecutionStatus(ctx context.Context, id valueobjects.ExecutionID, status valueobjects.ExecutionStatus, errorMessage *string) domain.ServiceResult[*entities.PipelineExecution]
}

// EventBusPort defines the interface for event publishing
type EventBusPort interface {
	// Publish publishes a domain event
	Publish(ctx context.Context, event domain.DomainEvent) error
	
	// PublishBatch publishes multiple domain events
	PublishBatch(ctx context.Context, events []domain.DomainEvent) error
	
	// Subscribe subscribes to domain events of a specific type
	Subscribe(eventType string, handler EventHandler) error
	
	// Unsubscribe unsubscribes from domain events
	Unsubscribe(eventType string, handler EventHandler) error
}

// EventHandler defines the interface for handling domain events
type EventHandler interface {
	// Handle processes a domain event
	Handle(ctx context.Context, event domain.DomainEvent) error
	
	// CanHandle returns true if this handler can process the given event type
	CanHandle(eventType string) bool
}

// UnitOfWorkPort defines the interface for transaction management
type UnitOfWorkPort interface {
	// Begin starts a new transaction
	Begin(ctx context.Context) (UnitOfWork, error)
}

// UnitOfWork represents a transaction boundary
type UnitOfWork interface {
	// Commit commits the transaction
	Commit() error
	
	// Rollback rolls back the transaction
	Rollback() error
	
	// PipelineRepository returns the pipeline repository within this transaction
	PipelineRepository() PipelineRepository
	
	// PipelineExecutionRepository returns the execution repository within this transaction
	PipelineExecutionRepository() PipelineExecutionRepository
}

// ConfigurationPort defines the interface for configuration management
type ConfigurationPort interface {
	// GetString retrieves a string configuration value
	GetString(key string) (string, error)
	
	// GetInt retrieves an integer configuration value
	GetInt(key string) (int, error)
	
	// GetBool retrieves a boolean configuration value
	GetBool(key string) (bool, error)
	
	// GetDuration retrieves a duration configuration value
	GetDuration(key string) (valueobjects.Duration, error)
	
	// Set sets a configuration value
	Set(key string, value interface{}) error
	
	// Exists checks if a configuration key exists
	Exists(key string) bool
}

// LoggingPort defines the interface for structured logging
type LoggingPort interface {
	// Debug logs a debug message
	Debug(ctx context.Context, message string, fields map[string]interface{})
	
	// Info logs an info message
	Info(ctx context.Context, message string, fields map[string]interface{})
	
	// Warn logs a warning message
	Warn(ctx context.Context, message string, fields map[string]interface{})
	
	// Error logs an error message
	Error(ctx context.Context, message string, err error, fields map[string]interface{})
	
	// Fatal logs a fatal message and exits
	Fatal(ctx context.Context, message string, err error, fields map[string]interface{})
}

// MetricsPort defines the interface for metrics collection
type MetricsPort interface {
	// IncrementCounter increments a counter metric
	IncrementCounter(name string, tags map[string]string)
	
	// UpdateGauge updates a gauge metric
	UpdateGauge(name string, value float64, tags map[string]string)
	
	// RecordHistogram records a value in a histogram
	RecordHistogram(name string, value float64, tags map[string]string)
	
	// RecordTimer records a duration
	RecordTimer(name string, duration valueobjects.Duration, tags map[string]string)
}