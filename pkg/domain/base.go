// Package domain provides the core domain model of the FLEXT platform.
//
// This module contains the fundamental building blocks for domain-driven design,
// including base types for entities, value objects, aggregates, commands, queries,
// events, and specifications that maintain the same architectural patterns as
// the Python implementation.
package domain

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
)

// EntityID represents a unique identifier for domain entities
type EntityID uuid.UUID

// NewEntityID creates a new entity ID
func NewEntityID() EntityID {
	return EntityID(uuid.New())
}

// String returns the string representation of the entity ID
func (id EntityID) String() string {
	return uuid.UUID(id).String()
}

// IsZero checks if the entity ID is zero value
func (id EntityID) IsZero() bool {
	return uuid.UUID(id) == uuid.Nil
}

// MarshalJSON implements json.Marshaler
func (id EntityID) MarshalJSON() ([]byte, error) {
	return json.Marshal(uuid.UUID(id).String())
}

// UnmarshalJSON implements json.Unmarshaler
func (id *EntityID) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	u, err := uuid.Parse(s)
	if err != nil {
		return err
	}
	*id = EntityID(u)
	return nil
}

// DomainEventData represents event payload data
type DomainEventData map[string]interface{}

// MetadataDict represents metadata information
type MetadataDict map[string]interface{}

// ConfigurationValue represents a configuration value
type ConfigurationValue interface{}

// DomainBaseModel provides the foundation for all domain models
type DomainBaseModel struct {
	// Base functionality for validation and serialization
}

// Validate performs validation on the model
func (m *DomainBaseModel) Validate() error {
	// Base validation logic
	return nil
}

// ToJSON serializes the model to JSON
func (m *DomainBaseModel) ToJSON() ([]byte, error) {
	return json.Marshal(m)
}

// DomainValueObject represents an immutable value object
type DomainValueObject struct {
	DomainBaseModel
}

// Equals compares two value objects for equality
func (vo *DomainValueObject) Equals(other *DomainValueObject) bool {
	// Value-based equality implementation
	return fmt.Sprintf("%+v", vo) == fmt.Sprintf("%+v", other)
}

// DomainEntity represents an entity with identity-based equality
type DomainEntity struct {
	DomainBaseModel
	ID        EntityID  `json:"id" validate:"required"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt *time.Time `json:"updated_at,omitempty"`
	Version   int        `json:"version" validate:"min=1"`
}

// NewDomainEntity creates a new domain entity
func NewDomainEntity() DomainEntity {
	now := time.Now().UTC()
	return DomainEntity{
		ID:        NewEntityID(),
		CreatedAt: now,
		Version:   1,
	}
}

// Equals compares two entities based on their ID
func (e *DomainEntity) Equals(other *DomainEntity) bool {
	return e.ID == other.ID
}

// UpdateVersion increments the entity version and sets updated timestamp
func (e *DomainEntity) UpdateVersion() {
	e.Version++
	now := time.Now().UTC()
	e.UpdatedAt = &now
}

// DomainEvent represents an immutable domain event
type DomainEvent struct {
	DomainValueObject
	EventID       EntityID           `json:"event_id"`
	AggregateID   EntityID           `json:"aggregate_id" validate:"required"`
	EventType     string             `json:"event_type" validate:"required"`
	EventVersion  int                `json:"event_version" validate:"min=1"`
	OccurredAt    time.Time          `json:"occurred_at"`
	CorrelationID *EntityID          `json:"correlation_id,omitempty"`
	CausationID   *EntityID          `json:"causation_id,omitempty"`
	EventData     DomainEventData    `json:"event_data"`
	Metadata      MetadataDict       `json:"metadata"`
}

// NewDomainEvent creates a new domain event
func NewDomainEvent(aggregateID EntityID, eventType string, eventData DomainEventData) DomainEvent {
	return DomainEvent{
		EventID:      NewEntityID(),
		AggregateID:  aggregateID,
		EventType:    eventType,
		EventVersion: 1,
		OccurredAt:   time.Now().UTC(),
		EventData:    eventData,
		Metadata:     make(MetadataDict),
	}
}

// EventStreamID generates event stream identifier
func (e *DomainEvent) EventStreamID() string {
	return fmt.Sprintf("%s-%s", e.EventType, e.AggregateID.String())
}

// DomainAggregateRoot represents an aggregate root with domain events
type DomainAggregateRoot struct {
	DomainEntity
	domainEvents     []DomainEvent `json:"-"`
	AggregateVersion int           `json:"aggregate_version" validate:"min=1"`
}

// NewDomainAggregateRoot creates a new aggregate root
func NewDomainAggregateRoot() DomainAggregateRoot {
	return DomainAggregateRoot{
		DomainEntity:     NewDomainEntity(),
		domainEvents:     make([]DomainEvent, 0),
		AggregateVersion: 1,
	}
}

// AddDomainEvent adds a domain event to the aggregate
func (ar *DomainAggregateRoot) AddDomainEvent(event DomainEvent) {
	ar.domainEvents = append(ar.domainEvents, event)
	ar.AggregateVersion++
}

// ClearDomainEvents clears and returns domain events
func (ar *DomainAggregateRoot) ClearDomainEvents() []DomainEvent {
	events := make([]DomainEvent, len(ar.domainEvents))
	copy(events, ar.domainEvents)
	ar.domainEvents = ar.domainEvents[:0]
	return events
}

// DomainEvents returns a copy of uncommitted domain events
func (ar *DomainAggregateRoot) DomainEvents() []DomainEvent {
	events := make([]DomainEvent, len(ar.domainEvents))
	copy(events, ar.domainEvents)
	return events
}

// DomainCommand represents a command with validation and metadata
type DomainCommand struct {
	DomainBaseModel
	CommandID     EntityID     `json:"command_id"`
	CorrelationID *EntityID    `json:"correlation_id,omitempty"`
	IssuedAt      time.Time    `json:"issued_at"`
	IssuedBy      *string      `json:"issued_by,omitempty"`
	Metadata      MetadataDict `json:"metadata"`
}

// NewDomainCommand creates a new domain command
func NewDomainCommand() DomainCommand {
	return DomainCommand{
		CommandID: NewEntityID(),
		IssuedAt:  time.Now().UTC(),
		Metadata:  make(MetadataDict),
	}
}

// DomainQuery represents a query with pagination and metadata
type DomainQuery struct {
	DomainBaseModel
	QueryID       EntityID     `json:"query_id"`
	CorrelationID *EntityID    `json:"correlation_id,omitempty"`
	IssuedAt      time.Time    `json:"issued_at"`
	IssuedBy      *string      `json:"issued_by,omitempty"`
	Limit         *int         `json:"limit,omitempty" validate:"omitempty,min=1,max=1000"`
	Offset        *int         `json:"offset,omitempty" validate:"omitempty,min=0"`
}

// NewDomainQuery creates a new domain query
func NewDomainQuery() DomainQuery {
	return DomainQuery{
		QueryID:  NewEntityID(),
		IssuedAt: time.Now().UTC(),
	}
}

// DomainSpecification represents a business rule specification
type DomainSpecification struct {
	DomainBaseModel
	SpecificationName string     `json:"specification_name" validate:"required"`
	Description       *string    `json:"description,omitempty"`
	CreatedAt         time.Time  `json:"created_at"`
}

// NewDomainSpecification creates a new domain specification
func NewDomainSpecification(name string) DomainSpecification {
	return DomainSpecification{
		SpecificationName: name,
		CreatedAt:         time.Now().UTC(),
	}
}

// IsSatisfiedBy checks if candidate satisfies specification
// This is the base implementation that should be overridden by concrete specifications
func (s *DomainSpecification) IsSatisfiedBy(candidate interface{}) bool {
	// Base implementation provides a functional default for development
	return true
}

// AndSpecification represents AND composition of specifications
type AndSpecification struct {
	DomainSpecification
	Left  DomainSpecification `json:"left"`
	Right DomainSpecification `json:"right"`
}

// NewAndSpecification creates an AND specification
func NewAndSpecification(left, right DomainSpecification) AndSpecification {
	return AndSpecification{
		DomainSpecification: NewDomainSpecification("and_specification"),
		Left:                left,
		Right:               right,
	}
}

// IsSatisfiedBy checks if candidate satisfies both specifications
func (s *AndSpecification) IsSatisfiedBy(candidate interface{}) bool {
	return s.Left.IsSatisfiedBy(candidate) && s.Right.IsSatisfiedBy(candidate)
}

// OrSpecification represents OR composition of specifications
type OrSpecification struct {
	DomainSpecification
	Left  DomainSpecification `json:"left"`
	Right DomainSpecification `json:"right"`
}

// NewOrSpecification creates an OR specification
func NewOrSpecification(left, right DomainSpecification) OrSpecification {
	return OrSpecification{
		DomainSpecification: NewDomainSpecification("or_specification"),
		Left:                left,
		Right:               right,
	}
}

// IsSatisfiedBy checks if candidate satisfies either specification
func (s *OrSpecification) IsSatisfiedBy(candidate interface{}) bool {
	return s.Left.IsSatisfiedBy(candidate) || s.Right.IsSatisfiedBy(candidate)
}

// NotSpecification represents NOT negation of specification
type NotSpecification struct {
	DomainSpecification
	Specification DomainSpecification `json:"specification"`
}

// NewNotSpecification creates a NOT specification
func NewNotSpecification(spec DomainSpecification) NotSpecification {
	return NotSpecification{
		DomainSpecification: NewDomainSpecification("not_specification"),
		Specification:       spec,
	}
}

// IsSatisfiedBy checks if candidate does NOT satisfy specification
func (s *NotSpecification) IsSatisfiedBy(candidate interface{}) bool {
	return !s.Specification.IsSatisfiedBy(candidate)
}