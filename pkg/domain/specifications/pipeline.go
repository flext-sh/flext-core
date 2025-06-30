package specifications

import (
	"github.com/flext-sh/flext-core/pkg/domain"
	"github.com/flext-sh/flext-core/pkg/domain/valueobjects"
)

// CanExecuteSpecification checks if a pipeline can be executed
type CanExecuteSpecification struct {
	domain.DomainSpecification
}

// NewCanExecuteSpecification creates a new CanExecuteSpecification
func NewCanExecuteSpecification() CanExecuteSpecification {
	return CanExecuteSpecification{
		DomainSpecification: domain.NewDomainSpecification("can_execute_pipeline"),
	}
}

// IsSatisfiedBy checks if the pipeline can be executed
func (s *CanExecuteSpecification) IsSatisfiedBy(candidate interface{}) bool {
	// Type assertion to check if candidate is a pipeline entity
	// In a real implementation, this would check pipeline status, dependencies, etc.
	
	// For now, we'll implement basic validation
	if candidate == nil {
		return false
	}
	
	// Here we would typically check:
	// - Pipeline is not already running
	// - All dependencies are satisfied
	// - Configuration is valid
	// - Resources are available
	
	return true // Simplified implementation
}

// HasValidDependenciesSpecification checks if pipeline dependencies are valid
type HasValidDependenciesSpecification struct {
	domain.DomainSpecification
}

// NewHasValidDependenciesSpecification creates a new HasValidDependenciesSpecification
func NewHasValidDependenciesSpecification() HasValidDependenciesSpecification {
	return HasValidDependenciesSpecification{
		DomainSpecification: domain.NewDomainSpecification("has_valid_dependencies"),
	}
}

// IsSatisfiedBy checks if the pipeline has valid dependencies
func (s *HasValidDependenciesSpecification) IsSatisfiedBy(candidate interface{}) bool {
	// In a real implementation, this would:
	// - Check for circular dependencies
	// - Verify all referenced steps exist
	// - Validate dependency order
	
	return true // Simplified implementation
}

// PipelineNameUniqueSpecification checks if pipeline name is unique
type PipelineNameUniqueSpecification struct {
	domain.DomainSpecification
	existingNames map[string]bool
}

// NewPipelineNameUniqueSpecification creates a new PipelineNameUniqueSpecification
func NewPipelineNameUniqueSpecification(existingNames []string) PipelineNameUniqueSpecification {
	nameMap := make(map[string]bool)
	for _, name := range existingNames {
		nameMap[name] = true
	}
	
	return PipelineNameUniqueSpecification{
		DomainSpecification: domain.NewDomainSpecification("pipeline_name_unique"),
		existingNames:       nameMap,
	}
}

// IsSatisfiedBy checks if the pipeline name is unique
func (s *PipelineNameUniqueSpecification) IsSatisfiedBy(candidate interface{}) bool {
	// Type assertion to get pipeline name
	if pipelineName, ok := candidate.(valueobjects.PipelineName); ok {
		return !s.existingNames[pipelineName.Value()]
	}
	
	// If it's a string, check directly
	if name, ok := candidate.(string); ok {
		return !s.existingNames[name]
	}
	
	return false
}

// ValidPipelineStepsSpecification checks if pipeline steps are valid
type ValidPipelineStepsSpecification struct {
	domain.DomainSpecification
}

// NewValidPipelineStepsSpecification creates a new ValidPipelineStepsSpecification
func NewValidPipelineStepsSpecification() ValidPipelineStepsSpecification {
	return ValidPipelineStepsSpecification{
		DomainSpecification: domain.NewDomainSpecification("valid_pipeline_steps"),
	}
}

// IsSatisfiedBy checks if the pipeline steps are valid
func (s *ValidPipelineStepsSpecification) IsSatisfiedBy(candidate interface{}) bool {
	// Type assertion to get pipeline steps
	if steps, ok := candidate.([]valueobjects.PipelineStep); ok {
		return s.validateSteps(steps)
	}
	
	return false
}

func (s *ValidPipelineStepsSpecification) validateSteps(steps []valueobjects.PipelineStep) bool {
	if len(steps) == 0 {
		return false // Pipeline must have at least one step
	}
	
	// Check for duplicate step names
	stepNames := make(map[string]bool)
	for _, step := range steps {
		if stepNames[step.Name] {
			return false // Duplicate step name
		}
		stepNames[step.Name] = true
	}
	
	// Validate dependencies exist
	for _, step := range steps {
		for _, dep := range step.DependsOn {
			if !stepNames[dep] {
				return false // Dependency doesn't exist
			}
		}
	}
	
	// Check for circular dependencies (simplified check)
	// In a real implementation, this would use a more sophisticated algorithm
	return true
}

// PluginAvailableSpecification checks if a plugin is available for use
type PluginAvailableSpecification struct {
	domain.DomainSpecification
	availablePlugins map[string]bool
}

// NewPluginAvailableSpecification creates a new PluginAvailableSpecification
func NewPluginAvailableSpecification(availablePlugins []string) PluginAvailableSpecification {
	pluginMap := make(map[string]bool)
	for _, plugin := range availablePlugins {
		pluginMap[plugin] = true
	}
	
	return PluginAvailableSpecification{
		DomainSpecification: domain.NewDomainSpecification("plugin_available"),
		availablePlugins:    pluginMap,
	}
}

// IsSatisfiedBy checks if the plugin is available
func (s *PluginAvailableSpecification) IsSatisfiedBy(candidate interface{}) bool {
	// Type assertion to get plugin ID
	if pluginID, ok := candidate.(valueobjects.PluginID); ok {
		return s.availablePlugins[pluginID.String()]
	}
	
	// If it's a string, check directly
	if pluginName, ok := candidate.(string); ok {
		return s.availablePlugins[pluginName]
	}
	
	return false
}