package commands

import (
	"github.com/flext-sh/flext-core/pkg/domain"
	"github.com/flext-sh/flext-core/pkg/domain/valueobjects"
)

// CreatePipelineCommand represents a command to create a new pipeline
type CreatePipelineCommand struct {
	domain.DomainCommand
	Name        string                 `json:"name" validate:"required,min=1,max=100"`
	Description string                 `json:"description" validate:"max=500"`
	Tags        []string               `json:"tags,omitempty"`
	Schedule    *string                `json:"schedule,omitempty"`
	IsActive    bool                   `json:"is_active"`
	Configuration map[string]interface{} `json:"configuration,omitempty"`
}

// NewCreatePipelineCommand creates a new CreatePipelineCommand
func NewCreatePipelineCommand(name, description string) CreatePipelineCommand {
	return CreatePipelineCommand{
		DomainCommand: domain.NewDomainCommand(),
		Name:          name,
		Description:   description,
		Tags:          make([]string, 0),
		IsActive:      true,
		Configuration: make(map[string]interface{}),
	}
}

// UpdatePipelineCommand represents a command to update an existing pipeline
type UpdatePipelineCommand struct {
	domain.DomainCommand
	PipelineID  string                 `json:"pipeline_id" validate:"required,uuid"`
	Name        *string                `json:"name,omitempty" validate:"omitempty,min=1,max=100"`
	Description *string                `json:"description,omitempty" validate:"omitempty,max=500"`
	Tags        []string               `json:"tags,omitempty"`
	Schedule    *string                `json:"schedule,omitempty"`
	IsActive    *bool                  `json:"is_active,omitempty"`
	Configuration map[string]interface{} `json:"configuration,omitempty"`
}

// NewUpdatePipelineCommand creates a new UpdatePipelineCommand
func NewUpdatePipelineCommand(pipelineID string) UpdatePipelineCommand {
	return UpdatePipelineCommand{
		DomainCommand: domain.NewDomainCommand(),
		PipelineID:    pipelineID,
		Tags:          make([]string, 0),
		Configuration: make(map[string]interface{}),
	}
}

// DeletePipelineCommand represents a command to delete a pipeline
type DeletePipelineCommand struct {
	domain.DomainCommand
	PipelineID string `json:"pipeline_id" validate:"required,uuid"`
}

// NewDeletePipelineCommand creates a new DeletePipelineCommand
func NewDeletePipelineCommand(pipelineID string) DeletePipelineCommand {
	return DeletePipelineCommand{
		DomainCommand: domain.NewDomainCommand(),
		PipelineID:    pipelineID,
	}
}

// AddPipelineStepCommand represents a command to add a step to a pipeline
type AddPipelineStepCommand struct {
	domain.DomainCommand
	PipelineID    string                 `json:"pipeline_id" validate:"required,uuid"`
	Name          string                 `json:"name" validate:"required,min=1,max=100"`
	StepType      string                 `json:"step_type" validate:"required"`
	Order         int                    `json:"order" validate:"min=0"`
	Configuration map[string]interface{} `json:"configuration,omitempty"`
	DependsOn     []string               `json:"depends_on,omitempty"`
}

// NewAddPipelineStepCommand creates a new AddPipelineStepCommand
func NewAddPipelineStepCommand(pipelineID, name, stepType string, order int) AddPipelineStepCommand {
	return AddPipelineStepCommand{
		DomainCommand: domain.NewDomainCommand(),
		PipelineID:    pipelineID,
		Name:          name,
		StepType:      stepType,
		Order:         order,
		Configuration: make(map[string]interface{}),
		DependsOn:     make([]string, 0),
	}
}

// RemovePipelineStepCommand represents a command to remove a step from a pipeline
type RemovePipelineStepCommand struct {
	domain.DomainCommand
	PipelineID string `json:"pipeline_id" validate:"required,uuid"`
	StepName   string `json:"step_name" validate:"required"`
}

// NewRemovePipelineStepCommand creates a new RemovePipelineStepCommand
func NewRemovePipelineStepCommand(pipelineID, stepName string) RemovePipelineStepCommand {
	return RemovePipelineStepCommand{
		DomainCommand: domain.NewDomainCommand(),
		PipelineID:    pipelineID,
		StepName:      stepName,
	}
}

// UpdatePipelineStepCommand represents a command to update a pipeline step
type UpdatePipelineStepCommand struct {
	domain.DomainCommand
	PipelineID    string                 `json:"pipeline_id" validate:"required,uuid"`
	StepName      string                 `json:"step_name" validate:"required"`
	NewName       *string                `json:"new_name,omitempty" validate:"omitempty,min=1,max=100"`
	StepType      *string                `json:"step_type,omitempty"`
	Order         *int                   `json:"order,omitempty" validate:"omitempty,min=0"`
	Configuration map[string]interface{} `json:"configuration,omitempty"`
	DependsOn     []string               `json:"depends_on,omitempty"`
}

// NewUpdatePipelineStepCommand creates a new UpdatePipelineStepCommand
func NewUpdatePipelineStepCommand(pipelineID, stepName string) UpdatePipelineStepCommand {
	return UpdatePipelineStepCommand{
		DomainCommand: domain.NewDomainCommand(),
		PipelineID:    pipelineID,
		StepName:      stepName,
		Configuration: make(map[string]interface{}),
		DependsOn:     make([]string, 0),
	}
}

// ActivatePipelineCommand represents a command to activate a pipeline
type ActivatePipelineCommand struct {
	domain.DomainCommand
	PipelineID string `json:"pipeline_id" validate:"required,uuid"`
}

// NewActivatePipelineCommand creates a new ActivatePipelineCommand
func NewActivatePipelineCommand(pipelineID string) ActivatePipelineCommand {
	return ActivatePipelineCommand{
		DomainCommand: domain.NewDomainCommand(),
		PipelineID:    pipelineID,
	}
}

// DeactivatePipelineCommand represents a command to deactivate a pipeline
type DeactivatePipelineCommand struct {
	domain.DomainCommand
	PipelineID string `json:"pipeline_id" validate:"required,uuid"`
}

// NewDeactivatePipelineCommand creates a new DeactivatePipelineCommand
func NewDeactivatePipelineCommand(pipelineID string) DeactivatePipelineCommand {
	return DeactivatePipelineCommand{
		DomainCommand: domain.NewDomainCommand(),
		PipelineID:    pipelineID,
	}
}

// ExecutePipelineCommand represents a command to execute a pipeline
type ExecutePipelineCommand struct {
	domain.DomainCommand
	PipelineID    string                 `json:"pipeline_id" validate:"required,uuid"`
	TriggeredBy   string                 `json:"triggered_by" validate:"required"`
	TriggerType   string                 `json:"trigger_type" validate:"required"`
	InputData     map[string]interface{} `json:"input_data,omitempty"`
}

// NewExecutePipelineCommand creates a new ExecutePipelineCommand
func NewExecutePipelineCommand(pipelineID, triggeredBy, triggerType string) ExecutePipelineCommand {
	return ExecutePipelineCommand{
		DomainCommand: domain.NewDomainCommand(),
		PipelineID:    pipelineID,
		TriggeredBy:   triggeredBy,
		TriggerType:   triggerType,
		InputData:     make(map[string]interface{}),
	}
}

// CancelExecutionCommand represents a command to cancel a pipeline execution
type CancelExecutionCommand struct {
	domain.DomainCommand
	ExecutionID string `json:"execution_id" validate:"required,uuid"`
	Reason      string `json:"reason,omitempty"`
}

// NewCancelExecutionCommand creates a new CancelExecutionCommand
func NewCancelExecutionCommand(executionID string) CancelExecutionCommand {
	return CancelExecutionCommand{
		DomainCommand: domain.NewDomainCommand(),
		ExecutionID:   executionID,
	}
}

// UpdateExecutionStatusCommand represents a command to update execution status
type UpdateExecutionStatusCommand struct {
	domain.DomainCommand
	ExecutionID  string                        `json:"execution_id" validate:"required,uuid"`
	Status       valueobjects.ExecutionStatus `json:"status" validate:"required"`
	ErrorMessage *string                       `json:"error_message,omitempty"`
	OutputData   map[string]interface{}        `json:"output_data,omitempty"`
	CPUUsage     *float64                      `json:"cpu_usage,omitempty" validate:"omitempty,min=0,max=100"`
	MemoryUsage  *float64                      `json:"memory_usage,omitempty" validate:"omitempty,min=0"`
}

// NewUpdateExecutionStatusCommand creates a new UpdateExecutionStatusCommand
func NewUpdateExecutionStatusCommand(executionID string, status valueobjects.ExecutionStatus) UpdateExecutionStatusCommand {
	return UpdateExecutionStatusCommand{
		DomainCommand: domain.NewDomainCommand(),
		ExecutionID:   executionID,
		Status:        status,
		OutputData:    make(map[string]interface{}),
	}
}

// AddExecutionLogCommand represents a command to add a log message to an execution
type AddExecutionLogCommand struct {
	domain.DomainCommand
	ExecutionID string `json:"execution_id" validate:"required,uuid"`
	Message     string `json:"message" validate:"required"`
}

// NewAddExecutionLogCommand creates a new AddExecutionLogCommand
func NewAddExecutionLogCommand(executionID, message string) AddExecutionLogCommand {
	return AddExecutionLogCommand{
		DomainCommand: domain.NewDomainCommand(),
		ExecutionID:   executionID,
		Message:       message,
	}
}