package valueobjects

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/flext-sh/flext-core/pkg/domain"
	"github.com/google/uuid"
)

// PipelineID represents a unique identifier for a pipeline
type PipelineID struct {
	domain.DomainValueObject
	value domain.EntityID
}

// NewPipelineID creates a new pipeline ID
func NewPipelineID() PipelineID {
	return PipelineID{
		value: domain.NewEntityID(),
	}
}

// ParsePipelineID creates a pipeline ID from a string
func ParsePipelineID(s string) (PipelineID, error) {
	id, err := uuid.Parse(s)
	if err != nil {
		return PipelineID{}, fmt.Errorf("invalid pipeline ID: %w", err)
	}
	return PipelineID{value: domain.EntityID(id)}, nil
}

// String returns the string representation
func (id PipelineID) String() string {
	return id.value.String()
}

// Value returns the underlying EntityID
func (id PipelineID) Value() domain.EntityID {
	return id.value
}

// MarshalJSON implements json.Marshaler
func (id PipelineID) MarshalJSON() ([]byte, error) {
	return json.Marshal(id.value.String())
}

// UnmarshalJSON implements json.Unmarshaler
func (id *PipelineID) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	parsed, err := ParsePipelineID(s)
	if err != nil {
		return err
	}
	*id = parsed
	return nil
}

// PipelineName represents a pipeline name with validation
type PipelineName struct {
	domain.DomainValueObject
	value string
}

// NewPipelineName creates a new pipeline name with validation
// Following go-ddd principle: Domain layer sets defaults and validates
func NewPipelineName(name string) (PipelineName, error) {
	name = strings.TrimSpace(name)
	
	// Validation rules - enforced at creation time
	if name == "" {
		return PipelineName{}, domain.NewInvalidInputError("name", name, "pipeline name cannot be empty")
	}
	if len(name) > 100 {
		return PipelineName{}, domain.NewInvalidInputError("name", name, "pipeline name cannot exceed 100 characters")
	}
	if len(name) < 3 {
		return PipelineName{}, domain.NewInvalidInputError("name", name, "pipeline name must be at least 3 characters")
	}
	
	// Basic validation for pipeline names (alphanumeric, dashes, underscores, spaces)
	for _, r := range name {
		if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || 
			 (r >= '0' && r <= '9') || r == '-' || r == '_' || r == ' ') {
			return PipelineName{}, domain.NewInvalidInputError("name", name, "pipeline name contains invalid characters (only alphanumeric, dashes, underscores, and spaces allowed)")
		}
	}
	
	return PipelineName{value: name}, nil
}

// String returns the string representation
func (n PipelineName) String() string {
	return n.value
}

// Value returns the underlying string value
func (n PipelineName) Value() string {
	return n.value
}

// ExecutionID represents a unique identifier for a pipeline execution
type ExecutionID struct {
	domain.DomainValueObject
	value domain.EntityID
}

// NewExecutionID creates a new execution ID
func NewExecutionID() ExecutionID {
	return ExecutionID{
		value: domain.NewEntityID(),
	}
}

// ParseExecutionID creates an execution ID from a string
func ParseExecutionID(s string) (ExecutionID, error) {
	id, err := uuid.Parse(s)
	if err != nil {
		return ExecutionID{}, fmt.Errorf("invalid execution ID: %w", err)
	}
	return ExecutionID{value: domain.EntityID(id)}, nil
}

// String returns the string representation
func (id ExecutionID) String() string {
	return id.value.String()
}

// Value returns the underlying EntityID
func (id ExecutionID) Value() domain.EntityID {
	return id.value
}

// ExecutionStatus represents the status of a pipeline execution
type ExecutionStatus string

const (
	ExecutionStatusPending   ExecutionStatus = "pending"
	ExecutionStatusRunning   ExecutionStatus = "running"
	ExecutionStatusCompleted ExecutionStatus = "completed"
	ExecutionStatusFailed    ExecutionStatus = "failed"
	ExecutionStatusCancelled ExecutionStatus = "cancelled"
)

// IsValid checks if the execution status is valid
func (s ExecutionStatus) IsValid() bool {
	switch s {
	case ExecutionStatusPending, ExecutionStatusRunning, ExecutionStatusCompleted, 
		 ExecutionStatusFailed, ExecutionStatusCancelled:
		return true
	default:
		return false
	}
}

// String returns the string representation
func (s ExecutionStatus) String() string {
	return string(s)
}

// IsTerminal returns true if the status is terminal (won't change)
func (s ExecutionStatus) IsTerminal() bool {
	return s == ExecutionStatusCompleted || s == ExecutionStatusFailed || s == ExecutionStatusCancelled
}

// Duration represents a time duration value object
type Duration struct {
	domain.DomainValueObject
	value time.Duration
}

// NewDuration creates a new duration
func NewDuration(d time.Duration) Duration {
	return Duration{value: d}
}

// NewDurationFromMillis creates a duration from milliseconds
func NewDurationFromMillis(millis int64) Duration {
	return Duration{value: time.Duration(millis) * time.Millisecond}
}

// Value returns the underlying duration
func (d Duration) Value() time.Duration {
	return d.value
}

// Milliseconds returns the duration in milliseconds
func (d Duration) Milliseconds() int64 {
	return d.value.Milliseconds()
}

// String returns the string representation
func (d Duration) String() string {
	return d.value.String()
}

// PipelineStep represents a step in a pipeline
type PipelineStep struct {
	domain.DomainValueObject
	Name         string                 `json:"name" validate:"required"`
	StepType     string                 `json:"step_type" validate:"required"`
	Configuration map[string]interface{} `json:"configuration"`
	Order        int                    `json:"order" validate:"min=0"`
	DependsOn    []string               `json:"depends_on"`
}

// NewPipelineStep creates a new pipeline step
func NewPipelineStep(name, stepType string, order int) PipelineStep {
	return PipelineStep{
		Name:          name,
		StepType:      stepType,
		Order:         order,
		Configuration: make(map[string]interface{}),
		DependsOn:     make([]string, 0),
	}
}

// AddDependency adds a dependency to the step
func (s *PipelineStep) AddDependency(stepName string) {
	for _, dep := range s.DependsOn {
		if dep == stepName {
			return // Already exists
		}
	}
	s.DependsOn = append(s.DependsOn, stepName)
}

// SetConfiguration sets a configuration value
func (s *PipelineStep) SetConfiguration(key string, value interface{}) {
	if s.Configuration == nil {
		s.Configuration = make(map[string]interface{})
	}
	s.Configuration[key] = value
}

// PluginID represents a unique identifier for a plugin
type PluginID struct {
	domain.DomainValueObject
	value domain.EntityID
}

// NewPluginID creates a new plugin ID
func NewPluginID() PluginID {
	return PluginID{
		value: domain.NewEntityID(),
	}
}

// ParsePluginID creates a plugin ID from a string
func ParsePluginID(s string) (PluginID, error) {
	id, err := uuid.Parse(s)
	if err != nil {
		return PluginID{}, fmt.Errorf("invalid plugin ID: %w", err)
	}
	return PluginID{value: domain.EntityID(id)}, nil
}

// String returns the string representation
func (id PluginID) String() string {
	return id.value.String()
}

// Value returns the underlying EntityID
func (id PluginID) Value() domain.EntityID {
	return id.value
}

// PluginType represents the type of a plugin
type PluginType string

const (
	PluginTypeExtractor PluginType = "extractor"
	PluginTypeLoader    PluginType = "loader"
	PluginTypeTransform PluginType = "transform"
	PluginTypeUtility   PluginType = "utility"
)

// IsValid checks if the plugin type is valid
func (t PluginType) IsValid() bool {
	switch t {
	case PluginTypeExtractor, PluginTypeLoader, PluginTypeTransform, PluginTypeUtility:
		return true
	default:
		return false
	}
}

// String returns the string representation
func (t PluginType) String() string {
	return string(t)
}

// PluginConfiguration represents plugin configuration
type PluginConfiguration struct {
	domain.DomainValueObject
	Settings map[string]interface{} `json:"settings"`
}

// NewPluginConfiguration creates a new plugin configuration
func NewPluginConfiguration() PluginConfiguration {
	return PluginConfiguration{
		Settings: make(map[string]interface{}),
	}
}

// Set sets a configuration value
func (c *PluginConfiguration) Set(key string, value interface{}) {
	if c.Settings == nil {
		c.Settings = make(map[string]interface{})
	}
	c.Settings[key] = value
}

// Get gets a configuration value
func (c *PluginConfiguration) Get(key string) (interface{}, bool) {
	if c.Settings == nil {
		return nil, false
	}
	value, exists := c.Settings[key]
	return value, exists
}

// GetString gets a string configuration value
func (c *PluginConfiguration) GetString(key string) (string, bool) {
	value, exists := c.Get(key)
	if !exists {
		return "", false
	}
	if s, ok := value.(string); ok {
		return s, true
	}
	return "", false
}