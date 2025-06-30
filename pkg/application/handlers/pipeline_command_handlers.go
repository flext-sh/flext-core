package handlers

import (
	"context"
	"fmt"

	"github.com/flext-sh/flext-core/pkg/application/commands"
	"github.com/flext-sh/flext-core/pkg/domain"
	"github.com/flext-sh/flext-core/pkg/domain/entities"
	"github.com/flext-sh/flext-core/pkg/domain/ports"
	"github.com/flext-sh/flext-core/pkg/domain/valueobjects"
)

// PipelineCommandHandlers contains handlers for pipeline-related commands
// Following go-ddd principles: Application layer orchestrates domain and infrastructure
type PipelineCommandHandlers struct {
	pipelineRepo ports.PipelineRepository
	executionRepo ports.PipelineExecutionRepository
	eventBus     ports.EventBusPort
	logger       ports.LoggingPort
	factory      *entities.PipelineFactory
}

// NewPipelineCommandHandlers creates a new instance of PipelineCommandHandlers
func NewPipelineCommandHandlers(
	pipelineRepo ports.PipelineRepository,
	executionRepo ports.PipelineExecutionRepository,
	eventBus ports.EventBusPort,
	logger ports.LoggingPort,
) *PipelineCommandHandlers {
	return &PipelineCommandHandlers{
		pipelineRepo:  pipelineRepo,
		executionRepo: executionRepo,
		eventBus:      eventBus,
		logger:        logger,
		factory:       entities.NewPipelineFactory(),
	}
}

// HandleCreatePipeline handles the CreatePipelineCommand
// Following go-ddd: Application layer orchestrates business operations
func (h *PipelineCommandHandlers) HandleCreatePipeline(ctx context.Context, cmd commands.CreatePipelineCommand) (*entities.Pipeline, error) {
	h.logger.Info(ctx, "Handling CreatePipelineCommand", map[string]interface{}{
		"command_id": cmd.CommandID.String(),
		"name":       cmd.Name,
	})

	// Create pipeline name value object (with domain validation)
	pipelineName, err := valueobjects.NewPipelineName(cmd.Name)
	if err != nil {
		h.logger.Error(ctx, "Invalid pipeline name", err, map[string]interface{}{
			"name": cmd.Name,
		})
		return nil, fmt.Errorf("invalid pipeline name: %w", err)
	}

	// Check business rule: pipeline name must be unique
	exists, err := h.pipelineRepo.ExistsByName(ctx, pipelineName)
	if err != nil {
		h.logger.Error(ctx, "Failed to check pipeline existence", err, nil)
		return nil, fmt.Errorf("failed to check pipeline existence: %w", err)
	}

	if exists {
		err := domain.NewAlreadyExistsError(fmt.Sprintf("pipeline with name '%s' already exists", cmd.Name))
		h.logger.Error(ctx, "Pipeline name conflict", err, nil)
		return nil, err
	}

	// Create pipeline entity using domain factory
	// Following go-ddd: Domain layer sets defaults and validates
	pipeline, err := h.factory.CreatePipeline(pipelineName, cmd.Description)
	if err != nil {
		h.logger.Error(ctx, "Failed to create pipeline entity", err, nil)
		return nil, fmt.Errorf("failed to create pipeline: %w", err)
	}

	// Apply additional business logic
	pipeline.IsActive = cmd.IsActive
	if cmd.Schedule != nil {
		pipeline.SetSchedule(*cmd.Schedule)
	}

	// Add tags
	for _, tag := range cmd.Tags {
		pipeline.AddTag(tag)
	}

	// Set configuration
	for key, value := range cmd.Configuration {
		pipeline.SetConfiguration(key, value)
	}

	// Persist pipeline
	// Following go-ddd: Repository reads after write for data integrity
	savedPipeline, err := h.pipelineRepo.Save(ctx, pipeline)
	if err != nil {
		h.logger.Error(ctx, "Failed to save pipeline", err, map[string]interface{}{
			"pipeline_id": pipeline.PipelineID.String(),
		})
		return nil, fmt.Errorf("failed to save pipeline: %w", err)
	}

	// Publish domain events
	events := savedPipeline.ClearDomainEvents()
	if len(events) > 0 {
		if err := h.eventBus.PublishBatch(ctx, events); err != nil {
			h.logger.Error(ctx, "Failed to publish domain events", err, map[string]interface{}{
				"event_count": len(events),
			})
			// Don't fail the command for event publishing errors
		}
	}

	h.logger.Info(ctx, "Pipeline created successfully", map[string]interface{}{
		"pipeline_id": savedPipeline.PipelineID.String(),
		"name":        savedPipeline.Name.Value(),
	})

	return savedPipeline, nil
}

// HandleUpdatePipeline handles the UpdatePipelineCommand
func (h *PipelineCommandHandlers) HandleUpdatePipeline(ctx context.Context, cmd commands.UpdatePipelineCommand) (*entities.Pipeline, error) {
	h.logger.Info(ctx, "Handling UpdatePipelineCommand", map[string]interface{}{
		"command_id":  cmd.CommandID.String(),
		"pipeline_id": cmd.PipelineID,
	})

	// Parse pipeline ID
	pipelineID, err := valueobjects.ParsePipelineID(cmd.PipelineID)
	if err != nil {
		return nil, domain.NewInvalidInputError("pipeline_id", cmd.PipelineID, "invalid pipeline ID format")
	}

	// Get existing pipeline using GetByID (must exist)
	pipeline, err := h.pipelineRepo.GetByID(ctx, pipelineID)
	if err != nil {
		h.logger.Error(ctx, "Failed to find pipeline", err, nil)
		return nil, fmt.Errorf("failed to find pipeline: %w", err)
	}

	// Update fields if provided
	if cmd.Name != nil {
		newName, err := valueobjects.NewPipelineName(*cmd.Name)
		if err != nil {
			return nil, fmt.Errorf("invalid pipeline name: %w", err)
		}

		// Check name uniqueness if changed
		if newName.Value() != pipeline.Name.Value() {
			exists, err := h.pipelineRepo.ExistsByName(ctx, newName)
			if err != nil {
				return nil, fmt.Errorf("failed to check name uniqueness: %w", err)
			}
			if exists {
				return nil, domain.NewAlreadyExistsError(fmt.Sprintf("pipeline with name '%s' already exists", *cmd.Name))
			}
		}

		pipeline.Name = newName
	}

	if cmd.Description != nil {
		pipeline.Description = *cmd.Description
	}

	if cmd.IsActive != nil {
		if *cmd.IsActive {
			pipeline.Activate()
		} else {
			pipeline.Deactivate()
		}
	}

	if cmd.Schedule != nil {
		if *cmd.Schedule == "" {
			pipeline.ClearSchedule()
		} else {
			pipeline.SetSchedule(*cmd.Schedule)
		}
	}

	// Update tags
	if len(cmd.Tags) > 0 {
		// Clear existing tags and add new ones
		pipeline.Tags = make([]string, 0)
		for _, tag := range cmd.Tags {
			pipeline.AddTag(tag)
		}
	}

	// Update configuration
	if len(cmd.Configuration) > 0 {
		for key, value := range cmd.Configuration {
			pipeline.SetConfiguration(key, value)
		}
	}

	// Save updated pipeline
	// Following go-ddd: Repository reads after write
	updatedPipeline, err := h.pipelineRepo.Update(ctx, pipeline)
	if err != nil {
		h.logger.Error(ctx, "Failed to update pipeline", err, nil)
		return nil, fmt.Errorf("failed to update pipeline: %w", err)
	}

	// Publish domain events
	events := updatedPipeline.ClearDomainEvents()
	if len(events) > 0 {
		if err := h.eventBus.PublishBatch(ctx, events); err != nil {
			h.logger.Error(ctx, "Failed to publish domain events", err, nil)
		}
	}

	h.logger.Info(ctx, "Pipeline updated successfully", map[string]interface{}{
		"pipeline_id": updatedPipeline.PipelineID.String(),
	})

	return updatedPipeline, nil
}

// HandleDeletePipeline handles the DeletePipelineCommand
func (h *PipelineCommandHandlers) HandleDeletePipeline(ctx context.Context, cmd commands.DeletePipelineCommand) error {
	h.logger.Info(ctx, "Handling DeletePipelineCommand", map[string]interface{}{
		"command_id":  cmd.CommandID.String(),
		"pipeline_id": cmd.PipelineID,
	})

	// Parse pipeline ID
	pipelineID, err := valueobjects.ParsePipelineID(cmd.PipelineID)
	if err != nil {
		return domain.NewInvalidInputError("pipeline_id", cmd.PipelineID, "invalid pipeline ID format")
	}

	// Check business rule: cannot delete pipeline with running executions
	runningExecutions, err := h.executionRepo.FindRunning(ctx)
	if err != nil {
		return fmt.Errorf("failed to check running executions: %w", err)
	}

	for _, execution := range runningExecutions {
		if execution.PipelineID == pipelineID {
			return domain.NewBusinessRuleError("cannot delete pipeline with running executions")
		}
	}

	// Perform soft deletion
	// Following go-ddd: Always use soft deletion to preserve history
	if err := h.pipelineRepo.Delete(ctx, pipelineID); err != nil {
		h.logger.Error(ctx, "Failed to delete pipeline", err, nil)
		return fmt.Errorf("failed to delete pipeline: %w", err)
	}

	h.logger.Info(ctx, "Pipeline deleted successfully", map[string]interface{}{
		"pipeline_id": cmd.PipelineID,
	})

	return nil
}

// HandleExecutePipeline handles the ExecutePipelineCommand
func (h *PipelineCommandHandlers) HandleExecutePipeline(ctx context.Context, cmd commands.ExecutePipelineCommand) (*entities.PipelineExecution, error) {
	h.logger.Info(ctx, "Handling ExecutePipelineCommand", map[string]interface{}{
		"command_id":   cmd.CommandID.String(),
		"pipeline_id":  cmd.PipelineID,
		"triggered_by": cmd.TriggeredBy,
	})

	// Parse pipeline ID
	pipelineID, err := valueobjects.ParsePipelineID(cmd.PipelineID)
	if err != nil {
		return nil, domain.NewInvalidInputError("pipeline_id", cmd.PipelineID, "invalid pipeline ID format")
	}

	// Check if pipeline exists and can execute
	pipeline, err := h.pipelineRepo.GetByID(ctx, pipelineID)
	if err != nil {
		return nil, fmt.Errorf("failed to find pipeline: %w", err)
	}

	if !pipeline.CanExecute() {
		return nil, domain.NewBusinessRuleError("pipeline cannot be executed (inactive or no steps)")
	}

	// Get next execution number
	executionNumber, err := h.executionRepo.GetNextExecutionNumber(ctx, pipelineID)
	if err != nil {
		return nil, fmt.Errorf("failed to get execution number: %w", err)
	}

	// Create new execution using factory
	executionFactory := entities.NewPipelineExecutionFactory()
	execution, err := executionFactory.CreateExecution(pipelineID, executionNumber, cmd.TriggeredBy, cmd.TriggerType)
	if err != nil {
		return nil, fmt.Errorf("failed to create execution: %w", err)
	}

	// Set input data
	for key, value := range cmd.InputData {
		execution.SetInputData(key, value)
	}

	// Save execution
	savedExecution, err := h.executionRepo.Save(ctx, execution)
	if err != nil {
		h.logger.Error(ctx, "Failed to save execution", err, nil)
		return nil, fmt.Errorf("failed to save execution: %w", err)
	}

	// Start execution
	if err := savedExecution.Start(); err != nil {
		h.logger.Error(ctx, "Failed to start execution", err, nil)
		return nil, fmt.Errorf("failed to start execution: %w", err)
	}

	// Update execution status
	updatedExecution, err := h.executionRepo.Update(ctx, savedExecution)
	if err != nil {
		h.logger.Error(ctx, "Failed to update execution", err, nil)
		return nil, fmt.Errorf("failed to update execution: %w", err)
	}

	// Publish domain events
	events := updatedExecution.ClearDomainEvents()
	if len(events) > 0 {
		if err := h.eventBus.PublishBatch(ctx, events); err != nil {
			h.logger.Error(ctx, "Failed to publish domain events", err, nil)
		}
	}

	h.logger.Info(ctx, "Pipeline execution started successfully", map[string]interface{}{
		"execution_id":     updatedExecution.ExecutionID.String(),
		"execution_number": updatedExecution.ExecutionNumber,
	})

	return updatedExecution, nil
}

// HandleCancelExecution handles the CancelExecutionCommand
func (h *PipelineCommandHandlers) HandleCancelExecution(ctx context.Context, cmd commands.CancelExecutionCommand) (*entities.PipelineExecution, error) {
	h.logger.Info(ctx, "Handling CancelExecutionCommand", map[string]interface{}{
		"command_id":   cmd.CommandID.String(),
		"execution_id": cmd.ExecutionID,
	})

	// Parse execution ID
	executionID, err := valueobjects.ParseExecutionID(cmd.ExecutionID)
	if err != nil {
		return nil, domain.NewInvalidInputError("execution_id", cmd.ExecutionID, "invalid execution ID format")
	}

	// Get execution
	execution, err := h.executionRepo.GetByID(ctx, executionID)
	if err != nil {
		return nil, fmt.Errorf("failed to find execution: %w", err)
	}

	// Cancel execution
	if err := execution.Cancel(); err != nil {
		h.logger.Error(ctx, "Failed to cancel execution", err, nil)
		return nil, fmt.Errorf("failed to cancel execution: %w", err)
	}

	// Update execution
	updatedExecution, err := h.executionRepo.Update(ctx, execution)
	if err != nil {
		h.logger.Error(ctx, "Failed to update execution", err, nil)
		return nil, fmt.Errorf("failed to update execution: %w", err)
	}

	// Publish domain events
	events := updatedExecution.ClearDomainEvents()
	if len(events) > 0 {
		if err := h.eventBus.PublishBatch(ctx, events); err != nil {
			h.logger.Error(ctx, "Failed to publish domain events", err, nil)
		}
	}

	h.logger.Info(ctx, "Execution cancelled successfully", map[string]interface{}{
		"execution_id": cmd.ExecutionID,
	})

	return updatedExecution, nil
}