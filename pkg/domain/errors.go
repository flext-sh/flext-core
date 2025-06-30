package domain

import (
	"errors"
	"fmt"
)

// Domain error types following go-ddd principles
var (
	// ErrNotFound represents a not found error
	ErrNotFound = errors.New("not found")
	
	// ErrAlreadyExists represents a conflict error
	ErrAlreadyExists = errors.New("already exists")
	
	// ErrInvalidInput represents invalid input
	ErrInvalidInput = errors.New("invalid input")
	
	// ErrBusinessRule represents a business rule violation
	ErrBusinessRule = errors.New("business rule violation")
	
	// ErrConcurrency represents a concurrency conflict
	ErrConcurrency = errors.New("concurrency conflict")
)

// DomainError represents a domain-specific error with context
type DomainError struct {
	Type    error
	Message string
	Field   string
	Value   interface{}
}

// Error implements the error interface
func (e DomainError) Error() string {
	if e.Field != "" {
		return fmt.Sprintf("%s: %s (field: %s, value: %v)", e.Type.Error(), e.Message, e.Field, e.Value)
	}
	return fmt.Sprintf("%s: %s", e.Type.Error(), e.Message)
}

// Is implements error unwrapping for errors.Is()
func (e DomainError) Is(target error) bool {
	return errors.Is(e.Type, target)
}

// NewNotFoundError creates a not found error
func NewNotFoundError(message string) DomainError {
	return DomainError{
		Type:    ErrNotFound,
		Message: message,
	}
}

// NewAlreadyExistsError creates an already exists error
func NewAlreadyExistsError(message string) DomainError {
	return DomainError{
		Type:    ErrAlreadyExists,
		Message: message,
	}
}

// NewInvalidInputError creates an invalid input error
func NewInvalidInputError(field string, value interface{}, message string) DomainError {
	return DomainError{
		Type:    ErrInvalidInput,
		Message: message,
		Field:   field,
		Value:   value,
	}
}

// NewBusinessRuleError creates a business rule violation error
func NewBusinessRuleError(message string) DomainError {
	return DomainError{
		Type:    ErrBusinessRule,
		Message: message,
	}
}

// NewConcurrencyError creates a concurrency conflict error
func NewConcurrencyError(message string) DomainError {
	return DomainError{
		Type:    ErrConcurrency,
		Message: message,
	}
}

// Helper functions for error checking

// IsNotFoundError checks if an error is a not found error
func IsNotFoundError(err error) bool {
	var domainErr DomainError
	if errors.As(err, &domainErr) {
		return errors.Is(domainErr.Type, ErrNotFound)
	}
	return false
}

// IsAlreadyExistsError checks if an error is an already exists error
func IsAlreadyExistsError(err error) bool {
	var domainErr DomainError
	if errors.As(err, &domainErr) {
		return errors.Is(domainErr.Type, ErrAlreadyExists)
	}
	return false
}

// IsInvalidInputError checks if an error is an invalid input error
func IsInvalidInputError(err error) bool {
	var domainErr DomainError
	if errors.As(err, &domainErr) {
		return errors.Is(domainErr.Type, ErrInvalidInput)
	}
	return false
}

// IsBusinessRuleError checks if an error is a business rule violation error
func IsBusinessRuleError(err error) bool {
	var domainErr DomainError
	if errors.As(err, &domainErr) {
		return errors.Is(domainErr.Type, ErrBusinessRule)
	}
	return false
}

// IsConcurrencyError checks if an error is a concurrency conflict error
func IsConcurrencyError(err error) bool {
	var domainErr DomainError
	if errors.As(err, &domainErr) {
		return errors.Is(domainErr.Type, ErrConcurrency)
	}
	return false
}