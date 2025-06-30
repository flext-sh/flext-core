package usecases

import (
	"context"
	"fmt"

	"github.com/flext-sh/flext-core/pkg/domain"
	"github.com/flext-sh/flext-core/pkg/domain/entities"
	"github.com/flext-sh/flext-core/pkg/domain/ports"
	"github.com/flext-sh/flext-core/pkg/domain/valueobjects"
)

// CreatePipelineUseCase represents the use case for creating a pipeline
// Following go-ddd principle: Application layer orchestrates domain operations
type CreatePipelineUseCase struct {
	pipelineRepo ports.PipelineRepository
	eventBus     ports.EventBusPort
	logger       ports.LoggingPort
	factory      *entities.PipelineFactory
}

// CreatePipelineRequest represents the input for creating a pipeline
type CreatePipelineRequest struct {
	Name          string                 `json:"name" validate:"required,min=3,max=100"`
	Description   string                 `json:"description" validate:"max=1000"`
	Tags          []string               `json:"tags,omitempty"`
	Schedule      *string                `json:"schedule,omitempty"`
	IsActive      bool                   `json:"is_active"`
	Configuration map[string]interface{} `json:"configuration,omitempty"`
}

// CreatePipelineResponse represents the output of creating a pipeline
type CreatePipelineResponse struct {
	PipelineID  string `json:"pipeline_id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	IsActive    bool   `json:"is_active"`
	CreatedAt   string `json:"created_at"`
}

// NewCreatePipelineUseCase creates a new instance of CreatePipelineUseCase
func NewCreatePipelineUseCase(
	pipelineRepo ports.PipelineRepository,
	eventBus ports.EventBusPort,
	logger ports.LoggingPort,
) *CreatePipelineUseCase {
	return &CreatePipelineUseCase{
		pipelineRepo: pipelineRepo,
		eventBus:     eventBus,
		logger:       logger,
		factory:      entities.NewPipelineFactory(),
	}
}

// Execute executes the create pipeline use case
// Following go-ddd principle: Use cases orchestrate domain logic and infrastructure
func (uc *CreatePipelineUseCase) Execute(ctx context.Context, req CreatePipelineRequest) (*CreatePipelineResponse, error) {
	uc.logger.Info(ctx, "Executing CreatePipelineUseCase", map[string]interface{}{
		"name": req.Name,
	})

	// Step 1: Create domain value objects (with validation)
	pipelineName, err := valueobjects.NewPipelineName(req.Name)
	if err != nil {
		uc.logger.Error(ctx, "Invalid pipeline name", err, map[string]interface{}{
			"name": req.Name,
		})
		return nil, fmt.Errorf("invalid pipeline name: %w", err)
	}

	// Step 2: Check business rules - pipeline name must be unique
	exists, err := uc.pipelineRepo.ExistsByName(ctx, pipelineName)
	if err != nil {
		uc.logger.Error(ctx, "Failed to check pipeline name uniqueness", err, nil)
		return nil, fmt.Errorf("failed to check pipeline name uniqueness: %w", err)
	}

	if exists {
		err := domain.NewAlreadyExistsError(fmt.Sprintf("pipeline with name '%s' already exists", req.Name))
		uc.logger.Error(ctx, "Pipeline name already exists", err, map[string]interface{}{
			"name": req.Name,
		})
		return nil, err
	}

	// Step 3: Create pipeline entity using domain factory
	pipeline, err := uc.factory.CreatePipeline(pipelineName, req.Description)
	if err != nil {
		uc.logger.Error(ctx, "Failed to create pipeline entity", err, nil)
		return nil, fmt.Errorf("failed to create pipeline: %w", err)
	}

	// Step 4: Apply additional business logic
	if !req.IsActive {
		pipeline.Deactivate()
	}

	if req.Schedule != nil && *req.Schedule != "" {
		pipeline.SetSchedule(*req.Schedule)
	}

	// Add tags if provided
	for _, tag := range req.Tags {
		pipeline.AddTag(tag)
	}

	// Set configuration if provided
	for key, value := range req.Configuration {
		pipeline.SetConfiguration(key, value)
	}

	// Step 5: Persist the pipeline
	// Following go-ddd: Repository reads after write for data integrity
	savedPipeline, err := uc.pipelineRepo.Save(ctx, pipeline)
	if err != nil {
		uc.logger.Error(ctx, "Failed to save pipeline", err, map[string]interface{}{
			"pipeline_id": pipeline.PipelineID.String(),
		})
		return nil, fmt.Errorf("failed to save pipeline: %w", err)
	}

	// Step 6: Publish domain events
	events := savedPipeline.ClearDomainEvents()
	if len(events) > 0 {
		if err := uc.eventBus.PublishBatch(ctx, events); err != nil {
			uc.logger.Error(ctx, "Failed to publish domain events", err, map[string]interface{}{
				"event_count": len(events),
				"pipeline_id": savedPipeline.PipelineID.String(),
			})
			// Don't fail the use case for event publishing errors - log and continue
		}
	}

	uc.logger.Info(ctx, "Pipeline created successfully", map[string]interface{}{
		"pipeline_id": savedPipeline.PipelineID.String(),
		"name":        savedPipeline.Name.Value(),
	})

	// Step 7: Return response (don't leak domain objects)
	// Following go-ddd principle: Don't leak domain objects to the outside world
	return &CreatePipelineResponse{
		PipelineID:  savedPipeline.PipelineID.String(),
		Name:        savedPipeline.Name.Value(),
		Description: savedPipeline.Description,
		IsActive:    savedPipeline.IsActive,
		CreatedAt:   savedPipeline.CreatedAt.Format("2006-01-02T15:04:05Z"),
	}, nil
}

// GetPipelineUseCase represents the use case for retrieving a pipeline
type GetPipelineUseCase struct {
	pipelineRepo ports.PipelineRepository
	logger       ports.LoggingPort
}

// GetPipelineRequest represents the input for getting a pipeline
type GetPipelineRequest struct {
	PipelineID string `json:"pipeline_id" validate:"required,uuid"`
}

// GetPipelineResponse represents the output of getting a pipeline
type GetPipelineResponse struct {
	PipelineID    string                 `json:"pipeline_id"`
	Name          string                 `json:"name"`
	Description   string                 `json:"description"`
	IsActive      bool                   `json:"is_active"`
	StepCount     int                    `json:"step_count"`
	Tags          []string               `json:"tags"`
	Schedule      *string                `json:"schedule,omitempty"`
	Configuration map[string]interface{} `json:"configuration"`
	CreatedAt     string                 `json:"created_at"`
	UpdatedAt     *string                `json:"updated_at,omitempty"`
	Version       int                    `json:"version"`
}

// NewGetPipelineUseCase creates a new instance of GetPipelineUseCase
func NewGetPipelineUseCase(
	pipelineRepo ports.PipelineRepository,
	logger ports.LoggingPort,
) *GetPipelineUseCase {
	return &GetPipelineUseCase{
		pipelineRepo: pipelineRepo,
		logger:       logger,
	}
}

// Execute executes the get pipeline use case
func (uc *GetPipelineUseCase) Execute(ctx context.Context, req GetPipelineRequest) (*GetPipelineResponse, error) {
	uc.logger.Info(ctx, "Executing GetPipelineUseCase", map[string]interface{}{
		"pipeline_id": req.PipelineID,
	})

	// Step 1: Parse and validate pipeline ID
	pipelineID, err := valueobjects.ParsePipelineID(req.PipelineID)
	if err != nil {
		uc.logger.Error(ctx, "Invalid pipeline ID format", err, map[string]interface{}{
			"pipeline_id": req.PipelineID,
		})
		return nil, domain.NewInvalidInputError("pipeline_id", req.PipelineID, "invalid pipeline ID format")
	}

	// Step 2: Retrieve pipeline from repository
	// Using GetByID because we expect the pipeline to exist
	pipeline, err := uc.pipelineRepo.GetByID(ctx, pipelineID)
	if err != nil {
		if domain.IsNotFoundError(err) {
			uc.logger.Info(ctx, "Pipeline not found", map[string]interface{}{
				"pipeline_id": req.PipelineID,
			})
			return nil, domain.NewNotFoundError(fmt.Sprintf("pipeline with ID '%s' not found", req.PipelineID))
		}
		uc.logger.Error(ctx, "Failed to retrieve pipeline", err, map[string]interface{}{
			"pipeline_id": req.PipelineID,
		})
		return nil, fmt.Errorf("failed to retrieve pipeline: %w", err)
	}

	uc.logger.Info(ctx, "Pipeline retrieved successfully", map[string]interface{}{
		"pipeline_id": pipeline.PipelineID.String(),
		"name":        pipeline.Name.Value(),
	})

	// Step 3: Convert to response (don't leak domain objects)
	response := &GetPipelineResponse{
		PipelineID:    pipeline.PipelineID.String(),
		Name:          pipeline.Name.Value(),
		Description:   pipeline.Description,
		IsActive:      pipeline.IsActive,
		StepCount:     len(pipeline.Steps),
		Tags:          pipeline.Tags,
		Schedule:      pipeline.Schedule,
		Configuration: pipeline.Configuration,
		CreatedAt:     pipeline.CreatedAt.Format("2006-01-02T15:04:05Z"),
		Version:       pipeline.Version,
	}

	if pipeline.UpdatedAt != nil {
		updatedAt := pipeline.UpdatedAt.Format("2006-01-02T15:04:05Z")
		response.UpdatedAt = &updatedAt
	}

	return response, nil
}

