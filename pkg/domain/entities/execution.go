package entities

import (
	"fmt"
	"time"

	"github.com/flext-sh/flext-core/pkg/domain"
	"github.com/flext-sh/flext-core/pkg/domain/valueobjects"
)

// PipelineExecution represents a single run of a Pipeline
type PipelineExecution struct {
	domain.DomainEntity
	PipelineID      valueobjects.PipelineID      `json:"pipeline_id"`
	ExecutionID     valueobjects.ExecutionID     `json:"execution_id"`
	ExecutionNumber int                          `json:"execution_number"`
	Status          valueobjects.ExecutionStatus `json:"status"`
	TriggeredBy     string                       `json:"triggered_by"`
	TriggerType     string                       `json:"trigger_type"`
	StartedAt       *time.Time                   `json:"started_at,omitempty"`
	CompletedAt     *time.Time                   `json:"completed_at,omitempty"`
	InputData       map[string]interface{}       `json:"input_data"`
	OutputData      map[string]interface{}       `json:"output_data"`
	LogMessages     []string                     `json:"log_messages"`
	ErrorMessage    *string                      `json:"error_message,omitempty"`
	CPUUsage        *float64                     `json:"cpu_usage,omitempty"`
	MemoryUsage     *float64                     `json:"memory_usage,omitempty"`
	domainEvents    []domain.DomainEvent         `json:"-"`
}

// NewPipelineExecution creates a new pipeline execution
func NewPipelineExecution(pipelineID valueobjects.PipelineID, executionNumber int, triggeredBy, triggerType string) *PipelineExecution {
	execution := &PipelineExecution{
		DomainEntity:    domain.NewDomainEntity(),
		PipelineID:      pipelineID,
		ExecutionID:     valueobjects.NewExecutionID(),
		ExecutionNumber: executionNumber,
		Status:          valueobjects.ExecutionStatusPending,
		TriggeredBy:     triggeredBy,
		TriggerType:     triggerType,
		InputData:       make(map[string]interface{}),
		OutputData:      make(map[string]interface{}),
		LogMessages:     make([]string, 0),
		domainEvents:    make([]domain.DomainEvent, 0),
	}

	// Add domain event for execution creation
	event := domain.NewDomainEvent(
		execution.ID,
		"PipelineExecutionCreated",
		domain.DomainEventData{
			"pipeline_id":      pipelineID.String(),
			"execution_id":     execution.ExecutionID.String(),
			"execution_number": executionNumber,
			"triggered_by":     triggeredBy,
			"trigger_type":     triggerType,
			"created_at":       execution.CreatedAt,
		},
	)
	execution.addDomainEvent(event)

	return execution
}

// Start starts the pipeline execution
func (pe *PipelineExecution) Start() error {
	if pe.Status != valueobjects.ExecutionStatusPending {
		return fmt.Errorf("cannot start execution: current status is %s", pe.Status)
	}

	pe.Status = valueobjects.ExecutionStatusRunning
	now := time.Now().UTC()
	pe.StartedAt = &now
	pe.UpdateVersion()

	event := domain.NewDomainEvent(
		pe.ID,
		"PipelineExecutionStarted",
		domain.DomainEventData{
			"pipeline_id":  pe.PipelineID.String(),
			"execution_id": pe.ExecutionID.String(),
			"started_at":   now,
		},
	)
	pe.addDomainEvent(event)

	pe.AddLogMessage(fmt.Sprintf("Pipeline execution started at %s", now.Format(time.RFC3339)))

	return nil
}

// Complete marks the execution as completed successfully
func (pe *PipelineExecution) Complete() error {
	if pe.Status != valueobjects.ExecutionStatusRunning {
		return fmt.Errorf("cannot complete execution: current status is %s", pe.Status)
	}

	pe.Status = valueobjects.ExecutionStatusCompleted
	now := time.Now().UTC()
	pe.CompletedAt = &now
	pe.UpdateVersion()

	event := domain.NewDomainEvent(
		pe.ID,
		"PipelineExecutionCompleted",
		domain.DomainEventData{
			"pipeline_id":  pe.PipelineID.String(),
			"execution_id": pe.ExecutionID.String(),
			"completed_at": now,
			"duration":     pe.GetDuration().Milliseconds(),
		},
	)
	pe.addDomainEvent(event)

	pe.AddLogMessage(fmt.Sprintf("Pipeline execution completed successfully at %s", now.Format(time.RFC3339)))

	return nil
}

// Fail marks the execution as failed
func (pe *PipelineExecution) Fail(errorMessage string) error {
	if pe.Status.IsTerminal() {
		return fmt.Errorf("cannot fail execution: current status is %s", pe.Status)
	}

	pe.Status = valueobjects.ExecutionStatusFailed
	pe.ErrorMessage = &errorMessage
	now := time.Now().UTC()
	pe.CompletedAt = &now
	pe.UpdateVersion()

	event := domain.NewDomainEvent(
		pe.ID,
		"PipelineExecutionFailed",
		domain.DomainEventData{
			"pipeline_id":   pe.PipelineID.String(),
			"execution_id":  pe.ExecutionID.String(),
			"failed_at":     now,
			"error_message": errorMessage,
			"duration":      pe.GetDuration().Milliseconds(),
		},
	)
	pe.addDomainEvent(event)

	pe.AddLogMessage(fmt.Sprintf("Pipeline execution failed at %s: %s", now.Format(time.RFC3339), errorMessage))

	return nil
}

// Cancel marks the execution as cancelled
func (pe *PipelineExecution) Cancel() error {
	if pe.Status.IsTerminal() {
		return fmt.Errorf("cannot cancel execution: current status is %s", pe.Status)
	}

	pe.Status = valueobjects.ExecutionStatusCancelled
	now := time.Now().UTC()
	pe.CompletedAt = &now
	pe.UpdateVersion()

	event := domain.NewDomainEvent(
		pe.ID,
		"PipelineExecutionCancelled",
		domain.DomainEventData{
			"pipeline_id":  pe.PipelineID.String(),
			"execution_id": pe.ExecutionID.String(),
			"cancelled_at": now,
			"duration":     pe.GetDuration().Milliseconds(),
		},
	)
	pe.addDomainEvent(event)

	pe.AddLogMessage(fmt.Sprintf("Pipeline execution cancelled at %s", now.Format(time.RFC3339)))

	return nil
}

// AddLogMessage adds a log message to the execution record
func (pe *PipelineExecution) AddLogMessage(message string) {
	timestamp := time.Now().UTC().Format(time.RFC3339)
	logEntry := fmt.Sprintf("[%s] %s", timestamp, message)
	pe.LogMessages = append(pe.LogMessages, logEntry)
}

// SetInputData sets input data for the execution
func (pe *PipelineExecution) SetInputData(key string, value interface{}) {
	if pe.InputData == nil {
		pe.InputData = make(map[string]interface{})
	}
	pe.InputData[key] = value
	pe.UpdateVersion()
}

// GetInputData gets input data from the execution
func (pe *PipelineExecution) GetInputData(key string) (interface{}, bool) {
	if pe.InputData == nil {
		return nil, false
	}
	value, exists := pe.InputData[key]
	return value, exists
}

// SetOutputData sets output data for the execution
func (pe *PipelineExecution) SetOutputData(key string, value interface{}) {
	if pe.OutputData == nil {
		pe.OutputData = make(map[string]interface{})
	}
	pe.OutputData[key] = value
	pe.UpdateVersion()
}

// GetOutputData gets output data from the execution
func (pe *PipelineExecution) GetOutputData(key string) (interface{}, bool) {
	if pe.OutputData == nil {
		return nil, false
	}
	value, exists := pe.OutputData[key]
	return value, exists
}

// SetResourceUsage sets CPU and memory usage metrics
func (pe *PipelineExecution) SetResourceUsage(cpuUsage, memoryUsage float64) {
	pe.CPUUsage = &cpuUsage
	pe.MemoryUsage = &memoryUsage
	pe.UpdateVersion()

	event := domain.NewDomainEvent(
		pe.ID,
		"PipelineExecutionResourceUsageUpdated",
		domain.DomainEventData{
			"pipeline_id":  pe.PipelineID.String(),
			"execution_id": pe.ExecutionID.String(),
			"cpu_usage":    cpuUsage,
			"memory_usage": memoryUsage,
		},
	)
	pe.addDomainEvent(event)
}

// GetDuration returns the execution duration
func (pe *PipelineExecution) GetDuration() valueobjects.Duration {
	if pe.StartedAt == nil {
		return valueobjects.NewDuration(0)
	}

	endTime := time.Now().UTC()
	if pe.CompletedAt != nil {
		endTime = *pe.CompletedAt
	}

	duration := endTime.Sub(*pe.StartedAt)
	return valueobjects.NewDuration(duration)
}

// IsRunning returns true if the execution is currently running
func (pe *PipelineExecution) IsRunning() bool {
	return pe.Status == valueobjects.ExecutionStatusRunning
}

// IsCompleted returns true if the execution completed successfully
func (pe *PipelineExecution) IsCompleted() bool {
	return pe.Status == valueobjects.ExecutionStatusCompleted
}

// IsFailed returns true if the execution failed
func (pe *PipelineExecution) IsFailed() bool {
	return pe.Status == valueobjects.ExecutionStatusFailed
}

// IsCancelled returns true if the execution was cancelled
func (pe *PipelineExecution) IsCancelled() bool {
	return pe.Status == valueobjects.ExecutionStatusCancelled
}

// IsTerminal returns true if the execution is in a terminal state
func (pe *PipelineExecution) IsTerminal() bool {
	return pe.Status.IsTerminal()
}

// GetLogMessageCount returns the number of log messages
func (pe *PipelineExecution) GetLogMessageCount() int {
	return len(pe.LogMessages)
}

// GetLatestLogMessages returns the latest N log messages
func (pe *PipelineExecution) GetLatestLogMessages(count int) []string {
	if count <= 0 {
		return make([]string, 0)
	}

	totalMessages := len(pe.LogMessages)
	if count >= totalMessages {
		result := make([]string, totalMessages)
		copy(result, pe.LogMessages)
		return result
	}

	startIndex := totalMessages - count
	result := make([]string, count)
	copy(result, pe.LogMessages[startIndex:])
	return result
}

// addDomainEvent adds a domain event to the execution
func (pe *PipelineExecution) addDomainEvent(event domain.DomainEvent) {
	pe.domainEvents = append(pe.domainEvents, event)
}

// GetDomainEvents returns a copy of uncommitted domain events
func (pe *PipelineExecution) GetDomainEvents() []domain.DomainEvent {
	events := make([]domain.DomainEvent, len(pe.domainEvents))
	copy(events, pe.domainEvents)
	return events
}

// ClearDomainEvents clears and returns domain events
func (pe *PipelineExecution) ClearDomainEvents() []domain.DomainEvent {
	events := make([]domain.DomainEvent, len(pe.domainEvents))
	copy(events, pe.domainEvents)
	pe.domainEvents = pe.domainEvents[:0]
	return events
}
