package entities

import (
	"fmt"
	"time"

	"github.com/flext-sh/flext-core/pkg/domain"
	"github.com/flext-sh/flext-core/pkg/domain/valueobjects"
)

// Pipeline represents a data pipeline aggregate root
type Pipeline struct {
	domain.DomainAggregateRoot
	PipelineID    valueobjects.PipelineID     `json:"pipeline_id"`
	Name          valueobjects.PipelineName   `json:"name"`
	Description   string                      `json:"description"`
	Steps         []valueobjects.PipelineStep `json:"steps"`
	Configuration map[string]interface{}      `json:"configuration"`
	IsActive      bool                        `json:"is_active"`
	Tags          []string                    `json:"tags"`
	Schedule      *string                     `json:"schedule,omitempty"`
}

// NewPipeline creates a new pipeline using the factory pattern
// Deprecated: Use PipelineFactory.CreatePipeline instead for better DDD compliance
func NewPipeline(name valueobjects.PipelineName, description string) (*Pipeline, error) {
	factory := NewPipelineFactory()
	return factory.CreatePipeline(name, description)
}

// AddStep adds a step to the pipeline
// Following go-ddd principle: Business logic and validation in domain layer
func (p *Pipeline) AddStep(step valueobjects.PipelineStep) error {
	// Business rule: Step name must be unique within pipeline
	for _, existingStep := range p.Steps {
		if existingStep.Name == step.Name {
			return domain.NewBusinessRuleError(fmt.Sprintf("step with name '%s' already exists in pipeline", step.Name))
		}
	}

	// Business rule: Step order must be unique
	for _, existingStep := range p.Steps {
		if existingStep.Order == step.Order {
			return domain.NewBusinessRuleError(fmt.Sprintf("step with order %d already exists in pipeline", step.Order))
		}
	}

	// Business rule: Validate step dependencies exist
	if err := p.validateStepDependencies(step); err != nil {
		return err
	}

	p.Steps = append(p.Steps, step)
	p.UpdateVersion()

	// Add domain event for step addition
	event := domain.NewDomainEvent(
		p.ID,
		"StepAdded",
		domain.DomainEventData{
			"pipeline_id": p.PipelineID.String(),
			"step_name":   step.Name,
			"step_type":   step.StepType,
			"order":       step.Order,
			"dependencies": step.DependsOn,
		},
	)
	p.AddDomainEvent(event)

	return nil
}

// validateStepDependencies validates that all step dependencies exist
func (p *Pipeline) validateStepDependencies(step valueobjects.PipelineStep) error {
	for _, dependency := range step.DependsOn {
		found := false
		for _, existingStep := range p.Steps {
			if existingStep.Name == dependency {
				found = true
				break
			}
		}
		if !found {
			return domain.NewBusinessRuleError(fmt.Sprintf("dependency '%s' does not exist in pipeline", dependency))
		}
	}
	return nil
}

// RemoveStep removes a step from the pipeline
func (p *Pipeline) RemoveStep(stepName string) error {
	stepIndex := -1
	for i, step := range p.Steps {
		if step.Name == stepName {
			stepIndex = i
			break
		}
	}

	if stepIndex == -1 {
		return fmt.Errorf("step with name %s not found", stepName)
	}

	// Check if other steps depend on this step
	for _, step := range p.Steps {
		for _, dep := range step.DependsOn {
			if dep == stepName {
				return fmt.Errorf("cannot remove step %s: step %s depends on it", stepName, step.Name)
			}
		}
	}

	// Remove the step
	p.Steps = append(p.Steps[:stepIndex], p.Steps[stepIndex+1:]...)
	p.UpdateVersion()

	// Add domain event for step removal
	event := domain.NewDomainEvent(
		p.ID,
		"StepRemoved",
		domain.DomainEventData{
			"pipeline_id": p.PipelineID.String(),
			"step_name":   stepName,
		},
	)
	p.AddDomainEvent(event)

	return nil
}

// UpdateStep updates an existing step
func (p *Pipeline) UpdateStep(stepName string, updatedStep valueobjects.PipelineStep) error {
	stepIndex := -1
	for i, step := range p.Steps {
		if step.Name == stepName {
			stepIndex = i
			break
		}
	}

	if stepIndex == -1 {
		return fmt.Errorf("step with name %s not found", stepName)
	}

	// If the name is being changed, check for conflicts
	if updatedStep.Name != stepName {
		for _, existingStep := range p.Steps {
			if existingStep.Name == updatedStep.Name {
				return fmt.Errorf("step with name %s already exists", updatedStep.Name)
			}
		}
	}

	p.Steps[stepIndex] = updatedStep
	p.UpdateVersion()

	// Add domain event for step update
	event := domain.NewDomainEvent(
		p.ID,
		"StepUpdated",
		domain.DomainEventData{
			"pipeline_id":   p.PipelineID.String(),
			"old_step_name": stepName,
			"new_step_name": updatedStep.Name,
			"step_type":     updatedStep.StepType,
		},
	)
	p.AddDomainEvent(event)

	return nil
}

// Activate activates the pipeline
func (p *Pipeline) Activate() {
	if !p.IsActive {
		p.IsActive = true
		p.UpdateVersion()

		event := domain.NewDomainEvent(
			p.ID,
			"PipelineActivated",
			domain.DomainEventData{
				"pipeline_id":  p.PipelineID.String(),
				"activated_at": time.Now().UTC(),
			},
		)
		p.AddDomainEvent(event)
	}
}

// Deactivate deactivates the pipeline
func (p *Pipeline) Deactivate() {
	if p.IsActive {
		p.IsActive = false
		p.UpdateVersion()

		event := domain.NewDomainEvent(
			p.ID,
			"PipelineDeactivated",
			domain.DomainEventData{
				"pipeline_id":    p.PipelineID.String(),
				"deactivated_at": time.Now().UTC(),
			},
		)
		p.AddDomainEvent(event)
	}
}

// SetSchedule sets the pipeline schedule
func (p *Pipeline) SetSchedule(schedule string) {
	p.Schedule = &schedule
	p.UpdateVersion()

	event := domain.NewDomainEvent(
		p.ID,
		"PipelineScheduleUpdated",
		domain.DomainEventData{
			"pipeline_id": p.PipelineID.String(),
			"schedule":    schedule,
		},
	)
	p.AddDomainEvent(event)
}

// ClearSchedule removes the pipeline schedule
func (p *Pipeline) ClearSchedule() {
	p.Schedule = nil
	p.UpdateVersion()

	event := domain.NewDomainEvent(
		p.ID,
		"PipelineScheduleCleared",
		domain.DomainEventData{
			"pipeline_id": p.PipelineID.String(),
		},
	)
	p.AddDomainEvent(event)
}

// AddTag adds a tag to the pipeline
func (p *Pipeline) AddTag(tag string) {
	// Check if tag already exists
	for _, existingTag := range p.Tags {
		if existingTag == tag {
			return // Tag already exists
		}
	}

	p.Tags = append(p.Tags, tag)
	p.UpdateVersion()

	event := domain.NewDomainEvent(
		p.ID,
		"PipelineTagAdded",
		domain.DomainEventData{
			"pipeline_id": p.PipelineID.String(),
			"tag":         tag,
		},
	)
	p.AddDomainEvent(event)
}

// RemoveTag removes a tag from the pipeline
func (p *Pipeline) RemoveTag(tag string) {
	tagIndex := -1
	for i, existingTag := range p.Tags {
		if existingTag == tag {
			tagIndex = i
			break
		}
	}

	if tagIndex == -1 {
		return // Tag doesn't exist
	}

	p.Tags = append(p.Tags[:tagIndex], p.Tags[tagIndex+1:]...)
	p.UpdateVersion()

	event := domain.NewDomainEvent(
		p.ID,
		"PipelineTagRemoved",
		domain.DomainEventData{
			"pipeline_id": p.PipelineID.String(),
			"tag":         tag,
		},
	)
	p.AddDomainEvent(event)
}

// SetConfiguration sets a configuration value
func (p *Pipeline) SetConfiguration(key string, value interface{}) {
	if p.Configuration == nil {
		p.Configuration = make(map[string]interface{})
	}
	p.Configuration[key] = value
	p.UpdateVersion()
}

// GetConfiguration gets a configuration value
func (p *Pipeline) GetConfiguration(key string) (interface{}, bool) {
	if p.Configuration == nil {
		return nil, false
	}
	value, exists := p.Configuration[key]
	return value, exists
}

// GetStepByName returns a step by its name
func (p *Pipeline) GetStepByName(name string) (*valueobjects.PipelineStep, bool) {
	for i, step := range p.Steps {
		if step.Name == name {
			return &p.Steps[i], true
		}
	}
	return nil, false
}

// GetStepCount returns the number of steps in the pipeline
func (p *Pipeline) GetStepCount() int {
	return len(p.Steps)
}

// HasSchedule returns true if the pipeline has a schedule
func (p *Pipeline) HasSchedule() bool {
	return p.Schedule != nil
}

// CanExecute checks if the pipeline can be executed
func (p *Pipeline) CanExecute() bool {
	// Basic checks for execution readiness
	return p.IsActive && len(p.Steps) > 0
}
