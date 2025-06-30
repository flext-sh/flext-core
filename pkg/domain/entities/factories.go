package entities

import (
	"time"

	"github.com/flext-sh/flext-core/pkg/domain"
	"github.com/flext-sh/flext-core/pkg/domain/valueobjects"
)

// PipelineFactory creates pipeline entities following go-ddd principles
// Factory pattern ensures consistent entity creation with proper defaults
type PipelineFactory struct{}

// NewPipelineFactory creates a new pipeline factory
func NewPipelineFactory() *PipelineFactory {
	return &PipelineFactory{}
}

// CreatePipeline creates a new pipeline with proper defaults set in domain layer
// Following go-ddd principle: Domain layer sets defaults, not infrastructure/database
func (f *PipelineFactory) CreatePipeline(name valueobjects.PipelineName, description string) (*Pipeline, error) {
	// Validate inputs at domain level
	if name.Value() == "" {
		return nil, domain.NewInvalidInputError("name", name.Value(), "pipeline name cannot be empty")
	}

	if len(description) > 1000 {
		return nil, domain.NewInvalidInputError("description", description, "description cannot exceed 1000 characters")
	}

	// Create pipeline with domain defaults
	pipeline := &Pipeline{
		DomainAggregateRoot: domain.NewDomainAggregateRoot(),
		PipelineID:          valueobjects.NewPipelineID(),
		Name:                name,
		Description:         description,
		Steps:               make([]valueobjects.PipelineStep, 0),
		Configuration:       make(map[string]interface{}),
		IsActive:            true, // Default: new pipelines are active
		Tags:                make([]string, 0),
		Schedule:            nil, // Default: no schedule
	}

	// Generate domain event for pipeline creation
	// Following go-ddd: Domain events are part of domain logic
	event := domain.NewDomainEvent(
		pipeline.ID,
		"PipelineCreated",
		domain.DomainEventData{
			"pipeline_id":   pipeline.PipelineID.String(),
			"name":          pipeline.Name.Value(),
			"description":   pipeline.Description,
			"created_at":    pipeline.CreatedAt,
			"is_active":     pipeline.IsActive,
		},
	)
	pipeline.AddDomainEvent(event)

	return pipeline, nil
}

// PipelineExecutionFactory creates pipeline execution entities
type PipelineExecutionFactory struct{}

// NewPipelineExecutionFactory creates a new execution factory
func NewPipelineExecutionFactory() *PipelineExecutionFactory {
	return &PipelineExecutionFactory{}
}

// CreateExecution creates a new pipeline execution with proper defaults
// Following go-ddd principle: Domain layer sets defaults and validates
func (f *PipelineExecutionFactory) CreateExecution(
	pipelineID valueobjects.PipelineID,
	executionNumber int,
	triggeredBy, triggerType string,
) (*PipelineExecution, error) {
	// Validate inputs at domain level
	if pipelineID.Value().IsZero() {
		return nil, domain.NewInvalidInputError("pipeline_id", pipelineID.String(), "pipeline ID cannot be empty")
	}

	if executionNumber < 1 {
		return nil, domain.NewInvalidInputError("execution_number", executionNumber, "execution number must be positive")
	}

	if len(triggeredBy) == 0 {
		return nil, domain.NewInvalidInputError("triggered_by", triggeredBy, "triggered_by cannot be empty")
	}

	if len(triggerType) == 0 {
		triggerType = "manual" // Default trigger type
	}

	// Create execution with domain defaults
	execution := &PipelineExecution{
		DomainEntity:    domain.NewDomainEntity(),
		PipelineID:      pipelineID,
		ExecutionID:     valueobjects.NewExecutionID(),
		ExecutionNumber: executionNumber,
		Status:          valueobjects.ExecutionStatusPending, // Default: pending
		TriggeredBy:     triggeredBy,
		TriggerType:     triggerType,
		StartedAt:       nil, // Will be set when execution starts
		CompletedAt:     nil, // Will be set when execution completes
		InputData:       make(map[string]interface{}),
		OutputData:      make(map[string]interface{}),
		LogMessages:     make([]string, 0),
		ErrorMessage:    nil,
		CPUUsage:        nil,
		MemoryUsage:     nil,
		domainEvents:    make([]domain.DomainEvent, 0),
	}

	// Generate domain event for execution creation
	event := domain.NewDomainEvent(
		execution.ID,
		"PipelineExecutionCreated",
		domain.DomainEventData{
			"pipeline_id":      pipelineID.String(),
			"execution_id":     execution.ExecutionID.String(),
			"execution_number": executionNumber,
			"triggered_by":     triggeredBy,
			"trigger_type":     triggerType,
			"status":           string(execution.Status),
			"created_at":       execution.CreatedAt,
		},
	)
	execution.addDomainEvent(event)

	// Add initial log message
	execution.AddLogMessage("Pipeline execution created")

	return execution, nil
}

// RehydratePipeline creates a pipeline entity from stored data
// Following go-ddd principle: Don't validate on read, only on write
// This allows historical data to be read even if validation rules changed
func (f *PipelineFactory) RehydratePipeline(
	id domain.EntityID,
	pipelineID valueobjects.PipelineID,
	name valueobjects.PipelineName,
	description string,
	steps []valueobjects.PipelineStep,
	configuration map[string]interface{},
	isActive bool,
	tags []string,
	schedule *string,
	createdAt time.Time,
	updatedAt *time.Time,
	version int,
	aggregateVersion int,
) *Pipeline {
	// Create pipeline without validation - this is rehydration from storage
	pipeline := &Pipeline{
		DomainAggregateRoot: domain.DomainAggregateRoot{
			DomainEntity: domain.DomainEntity{
				ID:        id,
				CreatedAt: createdAt,
				UpdatedAt: updatedAt,
				Version:   version,
			},
			AggregateVersion: aggregateVersion,
		},
		PipelineID:    pipelineID,
		Name:          name,
		Description:   description,
		Steps:         steps,
		Configuration: configuration,
		IsActive:      isActive,
		Tags:          tags,
		Schedule:      schedule,
	}

	// No domain events on rehydration - these are historical entities
	return pipeline
}

// RehydrateExecution creates a pipeline execution entity from stored data
// Following go-ddd principle: Don't validate on read
func (f *PipelineExecutionFactory) RehydrateExecution(
	id domain.EntityID,
	pipelineID valueobjects.PipelineID,
	executionID valueobjects.ExecutionID,
	executionNumber int,
	status valueobjects.ExecutionStatus,
	triggeredBy, triggerType string,
	startedAt, completedAt *time.Time,
	inputData, outputData map[string]interface{},
	logMessages []string,
	errorMessage *string,
	cpuUsage, memoryUsage *float64,
	createdAt time.Time,
	updatedAt *time.Time,
	version int,
) *PipelineExecution {
	// Create execution without validation - this is rehydration from storage
	execution := &PipelineExecution{
		DomainEntity: domain.DomainEntity{
			ID:        id,
			CreatedAt: createdAt,
			UpdatedAt: updatedAt,
			Version:   version,
		},
		PipelineID:      pipelineID,
		ExecutionID:     executionID,
		ExecutionNumber: executionNumber,
		Status:          status,
		TriggeredBy:     triggeredBy,
		TriggerType:     triggerType,
		StartedAt:       startedAt,
		CompletedAt:     completedAt,
		InputData:       inputData,
		OutputData:      outputData,
		LogMessages:     logMessages,
		ErrorMessage:    errorMessage,
		CPUUsage:        cpuUsage,
		MemoryUsage:     memoryUsage,
		domainEvents:    make([]domain.DomainEvent, 0), // No events on rehydration
	}

	return execution
}