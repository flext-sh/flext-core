package entities

import (
	"testing"

	"github.com/flext-sh/flext-core/pkg/domain"
	"github.com/flext-sh/flext-core/pkg/domain/valueobjects"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewPipeline(t *testing.T) {
	tests := []struct {
		name        string
		pipelineName string
		description string
		wantErr     bool
	}{
		{
			name:        "valid pipeline creation",
			pipelineName: "test-pipeline",
			description: "A test pipeline",
			wantErr:     false,
		},
		{
			name:        "empty name should fail",
			pipelineName: "",
			description: "A test pipeline",
			wantErr:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pipelineName, err := valueobjects.NewPipelineName(tt.pipelineName)
			if tt.wantErr && err != nil {
				return // Expected error in pipeline name creation
			}
			require.NoError(t, err)

			pipeline, err := NewPipeline(pipelineName, tt.description)

			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, pipeline)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, pipeline)
				assert.Equal(t, tt.pipelineName, pipeline.Name.Value())
				assert.Equal(t, tt.description, pipeline.Description)
				assert.True(t, pipeline.IsActive)
				assert.Equal(t, 0, len(pipeline.Steps))
				assert.Equal(t, 0, len(pipeline.Tags))
				assert.Nil(t, pipeline.Schedule)
				
				// Check that entity ID was generated
				assert.False(t, pipeline.ID.IsZero())
				assert.False(t, pipeline.PipelineID.Value().IsZero())
				
				// Check domain events
				events := pipeline.DomainEvents()
				assert.Equal(t, 1, len(events))
				assert.Equal(t, "PipelineCreated", events[0].EventType)
			}
		})
	}
}

func TestPipeline_AddStep(t *testing.T) {
	// Setup
	pipelineName, _ := valueobjects.NewPipelineName("test-pipeline")
	pipeline, _ := NewPipeline(pipelineName, "Test pipeline")
	
	step1 := valueobjects.NewPipelineStep("extract", "extractor", 1)
	step2 := valueobjects.NewPipelineStep("transform", "transformer", 2)
	duplicateStep := valueobjects.NewPipelineStep("extract", "extractor", 3)

	tests := []struct {
		name    string
		step    valueobjects.PipelineStep
		wantErr bool
	}{
		{
			name:    "add first step",
			step:    step1,
			wantErr: false,
		},
		{
			name:    "add second step",
			step:    step2,
			wantErr: false,
		},
		{
			name:    "add duplicate step name should fail",
			step:    duplicateStep,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			initialStepCount := len(pipeline.Steps)
			initialVersion := pipeline.Version
			
			err := pipeline.AddStep(tt.step)

			if tt.wantErr {
				assert.Error(t, err)
				assert.Equal(t, initialStepCount, len(pipeline.Steps))
				assert.Equal(t, initialVersion, pipeline.Version)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, initialStepCount+1, len(pipeline.Steps))
				assert.Greater(t, pipeline.Version, initialVersion)
				
				// Check that step was added
				found := false
				for _, step := range pipeline.Steps {
					if step.Name == tt.step.Name {
						found = true
						assert.Equal(t, tt.step.StepType, step.StepType)
						assert.Equal(t, tt.step.Order, step.Order)
						break
					}
				}
				assert.True(t, found, "Step should be found in pipeline")
				
				// Check domain events - find the latest StepAdded event
				events := pipeline.DomainEvents()
				var latestStepAddedEvent *domain.DomainEvent
				for i := len(events) - 1; i >= 0; i-- {
					if events[i].EventType == "StepAdded" {
						latestStepAddedEvent = &events[i]
						break
					}
				}
				assert.NotNil(t, latestStepAddedEvent, "StepAdded event should be present")
				if latestStepAddedEvent != nil {
					assert.Equal(t, tt.step.Name, latestStepAddedEvent.EventData["step_name"])
				}
			}
		})
	}
}

func TestPipeline_RemoveStep(t *testing.T) {
	// Setup
	pipelineName, _ := valueobjects.NewPipelineName("test-pipeline")
	pipeline, _ := NewPipeline(pipelineName, "Test pipeline")
	
	step1 := valueobjects.NewPipelineStep("extract", "extractor", 1)
	step2 := valueobjects.NewPipelineStep("transform", "transformer", 2)
	step2.AddDependency("extract")
	
	pipeline.AddStep(step1)
	pipeline.AddStep(step2)
	pipeline.ClearDomainEvents() // Clear events from setup

	tests := []struct {
		name     string
		stepName string
		wantErr  bool
	}{
		{
			name:     "remove step with dependencies should fail",
			stepName: "extract",
			wantErr:  true,
		},
		{
			name:     "remove step without dependencies",
			stepName: "transform",
			wantErr:  false,
		},
		{
			name:     "remove non-existent step should fail",
			stepName: "non-existent",
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			initialStepCount := len(pipeline.Steps)
			initialVersion := pipeline.Version
			
			err := pipeline.RemoveStep(tt.stepName)

			if tt.wantErr {
				assert.Error(t, err)
				assert.Equal(t, initialStepCount, len(pipeline.Steps))
				assert.Equal(t, initialVersion, pipeline.Version)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, initialStepCount-1, len(pipeline.Steps))
				assert.Greater(t, pipeline.Version, initialVersion)
				
				// Check that step was removed
				for _, step := range pipeline.Steps {
					assert.NotEqual(t, tt.stepName, step.Name)
				}
				
				// Check domain events
				events := pipeline.DomainEvents()
				stepRemovedFound := false
				for _, event := range events {
					if event.EventType == "StepRemoved" {
						stepRemovedFound = true
						assert.Equal(t, tt.stepName, event.EventData["step_name"])
						break
					}
				}
				assert.True(t, stepRemovedFound, "StepRemoved event should be present")
			}
		})
	}
}

func TestPipeline_ActivateDeactivate(t *testing.T) {
	// Setup
	pipelineName, _ := valueobjects.NewPipelineName("test-pipeline")
	pipeline, _ := NewPipeline(pipelineName, "Test pipeline")
	pipeline.ClearDomainEvents() // Clear creation events
	
	// Test deactivation
	t.Run("deactivate active pipeline", func(t *testing.T) {
		assert.True(t, pipeline.IsActive)
		initialVersion := pipeline.Version
		
		pipeline.Deactivate()
		
		assert.False(t, pipeline.IsActive)
		assert.Greater(t, pipeline.Version, initialVersion)
		
		// Check domain events
		events := pipeline.DomainEvents()
		deactivatedFound := false
		for _, event := range events {
			if event.EventType == "PipelineDeactivated" {
				deactivatedFound = true
				break
			}
		}
		assert.True(t, deactivatedFound, "PipelineDeactivated event should be present")
	})
	
	// Test activation
	t.Run("activate inactive pipeline", func(t *testing.T) {
		assert.False(t, pipeline.IsActive)
		pipeline.ClearDomainEvents() // Clear previous events
		initialVersion := pipeline.Version
		
		pipeline.Activate()
		
		assert.True(t, pipeline.IsActive)
		assert.Greater(t, pipeline.Version, initialVersion)
		
		// Check domain events
		events := pipeline.DomainEvents()
		activatedFound := false
		for _, event := range events {
			if event.EventType == "PipelineActivated" {
				activatedFound = true
				break
			}
		}
		assert.True(t, activatedFound, "PipelineActivated event should be present")
	})
	
	// Test idempotency
	t.Run("activate already active pipeline", func(t *testing.T) {
		assert.True(t, pipeline.IsActive)
		pipeline.ClearDomainEvents() // Clear previous events
		initialVersion := pipeline.Version
		
		pipeline.Activate()
		
		assert.True(t, pipeline.IsActive)
		assert.Equal(t, initialVersion, pipeline.Version) // Version should not change
		
		// No events should be generated
		events := pipeline.DomainEvents()
		assert.Equal(t, 0, len(events))
	})
}

func TestPipeline_ScheduleManagement(t *testing.T) {
	// Setup
	pipelineName, _ := valueobjects.NewPipelineName("test-pipeline")
	pipeline, _ := NewPipeline(pipelineName, "Test pipeline")
	pipeline.ClearDomainEvents()
	
	t.Run("set schedule", func(t *testing.T) {
		assert.False(t, pipeline.HasSchedule())
		
		schedule := "0 0 * * *" // Daily at midnight
		pipeline.SetSchedule(schedule)
		
		assert.True(t, pipeline.HasSchedule())
		assert.Equal(t, schedule, *pipeline.Schedule)
		
		// Check domain events
		events := pipeline.DomainEvents()
		scheduleUpdatedFound := false
		for _, event := range events {
			if event.EventType == "PipelineScheduleUpdated" {
				scheduleUpdatedFound = true
				assert.Equal(t, schedule, event.EventData["schedule"])
				break
			}
		}
		assert.True(t, scheduleUpdatedFound, "PipelineScheduleUpdated event should be present")
	})
	
	t.Run("clear schedule", func(t *testing.T) {
		assert.True(t, pipeline.HasSchedule())
		pipeline.ClearDomainEvents()
		
		pipeline.ClearSchedule()
		
		assert.False(t, pipeline.HasSchedule())
		assert.Nil(t, pipeline.Schedule)
		
		// Check domain events
		events := pipeline.DomainEvents()
		scheduleClearedFound := false
		for _, event := range events {
			if event.EventType == "PipelineScheduleCleared" {
				scheduleClearedFound = true
				break
			}
		}
		assert.True(t, scheduleClearedFound, "PipelineScheduleCleared event should be present")
	})
}

func TestPipeline_TagManagement(t *testing.T) {
	// Setup
	pipelineName, _ := valueobjects.NewPipelineName("test-pipeline")
	pipeline, _ := NewPipeline(pipelineName, "Test pipeline")
	pipeline.ClearDomainEvents()
	
	t.Run("add tags", func(t *testing.T) {
		assert.Equal(t, 0, len(pipeline.Tags))
		
		pipeline.AddTag("etl")
		pipeline.AddTag("production")
		
		assert.Equal(t, 2, len(pipeline.Tags))
		assert.Contains(t, pipeline.Tags, "etl")
		assert.Contains(t, pipeline.Tags, "production")
		
		// Test duplicate tag (should be ignored)
		pipeline.AddTag("etl")
		assert.Equal(t, 2, len(pipeline.Tags))
	})
	
	t.Run("remove tags", func(t *testing.T) {
		assert.Equal(t, 2, len(pipeline.Tags))
		
		pipeline.RemoveTag("production")
		
		assert.Equal(t, 1, len(pipeline.Tags))
		assert.Contains(t, pipeline.Tags, "etl")
		assert.NotContains(t, pipeline.Tags, "production")
		
		// Test removing non-existent tag (should be ignored)
		pipeline.RemoveTag("non-existent")
		assert.Equal(t, 1, len(pipeline.Tags))
	})
}

func TestPipeline_CanExecute(t *testing.T) {
	// Setup
	pipelineName, _ := valueobjects.NewPipelineName("test-pipeline")
	pipeline, _ := NewPipeline(pipelineName, "Test pipeline")
	
	t.Run("active pipeline without steps cannot execute", func(t *testing.T) {
		assert.True(t, pipeline.IsActive)
		assert.Equal(t, 0, len(pipeline.Steps))
		assert.False(t, pipeline.CanExecute())
	})
	
	t.Run("active pipeline with steps can execute", func(t *testing.T) {
		step := valueobjects.NewPipelineStep("extract", "extractor", 1)
		pipeline.AddStep(step)
		
		assert.True(t, pipeline.IsActive)
		assert.Equal(t, 1, len(pipeline.Steps))
		assert.True(t, pipeline.CanExecute())
	})
	
	t.Run("inactive pipeline with steps cannot execute", func(t *testing.T) {
		pipeline.Deactivate()
		
		assert.False(t, pipeline.IsActive)
		assert.Equal(t, 1, len(pipeline.Steps))
		assert.False(t, pipeline.CanExecute())
	})
}