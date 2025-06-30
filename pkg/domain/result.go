package domain

import (
	"fmt"
)

// ServiceResult represents a result that can be either success or failure
// This is equivalent to the Python ServiceResult[T] type
type ServiceResult[T any] struct {
	value T
	err   error
	isErr bool
}

// Success creates a successful result
func Success[T any](value T) ServiceResult[T] {
	return ServiceResult[T]{
		value: value,
		isErr: false,
	}
}

// Failure creates a failed result
func Failure[T any](err error) ServiceResult[T] {
	var zero T
	return ServiceResult[T]{
		value: zero,
		err:   err,
		isErr: true,
	}
}

// FailureMsg creates a failed result from a message
func FailureMsg[T any](message string) ServiceResult[T] {
	return Failure[T](fmt.Errorf("%s", message))
}

// IsSuccess returns true if the result represents success
func (r ServiceResult[T]) IsSuccess() bool {
	return !r.isErr
}

// IsFailure returns true if the result represents failure
func (r ServiceResult[T]) IsFailure() bool {
	return r.isErr
}

// Value returns the success value and whether it's valid
func (r ServiceResult[T]) Value() (T, bool) {
	if r.isErr {
		var zero T
		return zero, false
	}
	return r.value, true
}

// Error returns the error if the result is a failure
func (r ServiceResult[T]) Error() error {
	if !r.isErr {
		return nil
	}
	return r.err
}

// Unwrap returns the value or panics if it's an error
func (r ServiceResult[T]) Unwrap() T {
	if r.isErr {
		panic(fmt.Sprintf("ServiceResult unwrap failed: %v", r.err))
	}
	return r.value
}

// UnwrapOr returns the value or the provided default if it's an error
func (r ServiceResult[T]) UnwrapOr(defaultValue T) T {
	if r.isErr {
		return defaultValue
	}
	return r.value
}

// UnwrapOrElse returns the value or calls the provided function if it's an error
func (r ServiceResult[T]) UnwrapOrElse(fn func() T) T {
	if r.isErr {
		return fn()
	}
	return r.value
}

// Map transforms the success value using the provided function
func Map[T, U any](r ServiceResult[T], fn func(T) U) ServiceResult[U] {
	if r.isErr {
		return Failure[U](r.err)
	}
	return Success(fn(r.value))
}

// FlatMap transforms the success value using a function that returns a ServiceResult
func FlatMap[T, U any](r ServiceResult[T], fn func(T) ServiceResult[U]) ServiceResult[U] {
	if r.isErr {
		return Failure[U](r.err)
	}
	return fn(r.value)
}

// MapError transforms the error using the provided function
func (r ServiceResult[T]) MapError(fn func(error) error) ServiceResult[T] {
	if !r.isErr {
		return r
	}
	return Failure[T](fn(r.err))
}