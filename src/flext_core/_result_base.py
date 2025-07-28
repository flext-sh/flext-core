"""FLEXT Core Result Base Module.

Comprehensive railway-oriented programming implementation providing the foundational
Result pattern for type-safe error handling across the FLEXT Core library. Implements
consolidated architecture with factory patterns and functional operations.

Architecture:
    - Railway-oriented programming patterns for type-safe error handling
    - Single source of truth pattern eliminating base module duplication
    - Factory pattern support for result creation with validation
    - Functional programming patterns with monadic operations (map, flat_map, filter)
    - Exception-safe operations with automatic error handling and recovery
    - Immutable result objects ensuring thread safety and consistency

Result System Components:
    - _BaseResult: Core result implementation with comprehensive railway operations
    - _BaseResultFactory: Factory patterns for result creation with validation
    - _BaseResultOperations: Utility operations for result chaining and validation
    - Monadic operations: map, flat_map, filter for functional composition
    - Error recovery: recover and tap operations for error handling and side effects
    - Result combination: combine operations for multi-result aggregation

Maintenance Guidelines:
    - Maintain immutability of result objects for thread safety
    - Use factory methods for consistent result creation patterns
    - Implement comprehensive error handling in all operations
    - Preserve error context through operation chains
    - Keep operations pure without side effects except for tap operation
    - Follow monadic laws for functional programming compliance

Design Decisions:
    - Immutable result objects preventing state corruption
    - Generic type support with proper variance handling
    - Exception-safe operations with contextual error messages
    - Factory pattern for consistent result creation
    - Monadic operations following functional programming principles
    - Error data preservation through operation chains

Railway-Oriented Programming Features:
    - Type-safe error handling without exception propagation
    - Functional composition through map and flat_map operations
    - Error short-circuiting preventing unnecessary computation
    - Context preservation through error data and codes
    - Recovery patterns for graceful error handling
    - Pipeline composition for complex operation chains

Error Handling Patterns:
    - Structured error information with codes and contextual data
    - Exception capture with type and message preservation
    - Error propagation through operation chains
    - Recovery mechanisms for alternative execution paths
    - Side effect execution with error isolation

Dependencies:
    - typing: Type annotations and generic type support
    - contextlib: Exception suppression for side effect operations
    - Standard library: No external runtime dependencies

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, TypeVar

# Define T and U locally for runtime use
T = TypeVar("T")
U = TypeVar("U")

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core.types import TErrorCode, TErrorMessage
else:
    # Runtime type aliases
    TErrorCode = str
    TErrorMessage = str


# =============================================================================
# BASE RESULT - CONSOLIDATED single source of truth for ALL Result functionality
# =============================================================================


class _BaseResultFactory:
    """Comprehensive factory for result creation providing standardized instantiation.

    Factory implementation providing multiple creation patterns for _BaseResult
    with consistent error handling, validation, and exception safety. Eliminates code
    duplication by consolidating all result creation logic in a single location.

    Architecture:
        - Static factory methods for different result creation patterns
        - Exception-safe creation with automatic error handling
        - Validation-based creation with predicate checking
        - Conditional creation based on boolean logic
        - Exception capture with comprehensive error context

    Factory Creation Patterns:
        - Success creation: create_success for successful operation results
        - Failure creation: create_failure with error codes and contextual data
        - Exception creation: from_exception for automatic exception handling
        - Callable creation: create_from_callable for function execution
        - Conditional creation: create_conditional for predicate-based results

    Usage Patterns:
        # Basic success result
        result = _BaseResultFactory.create_success("operation completed")

        # Detailed failure result
        failure = _BaseResultFactory.create_failure(
            "Validation failed",
            error_code="VALIDATION_ERROR",
            error_data={"field": "email", "value": "invalid"}
        )

        # Exception handling
        try:
            risky_operation()
        except Exception as e:
            result = _BaseResultFactory.from_exception(e)

        # Conditional result creation
        result = _BaseResultFactory.create_conditional(
            condition=user.is_active,
            success_data=user,
            failure_message="User is not active",
            failure_code="USER_INACTIVE"
        )

    """

    @staticmethod
    def create_success(data: T) -> _BaseResult[T]:
        """Create successful result with provided data.

        Creates a successful result instance containing the provided data
        with no error information set.

        Args:
            data: Data to store in successful result

        Returns:
            _BaseResult[T] instance marked as successful with provided data

        Usage:
            result = _BaseResultFactory.create_success("operation completed")
            assert result.is_success
            assert result.data == "operation completed"

        """
        return _BaseResult(data=data)

    @staticmethod
    def create_failure(
        error: TErrorMessage,
        error_code: TErrorCode | None = None,
        error_data: dict[str, object] | None = None,
    ) -> _BaseResult[T]:
        """Create failure result with comprehensive error information.

        Creates a failure result with detailed error context including
        error message, optional error code, and additional error data.

        Args:
            error: Human-readable error message
            error_code: Optional structured error code for programmatic handling
            error_data: Optional additional context data for debugging

        Returns:
            _BaseResult[T] instance marked as failure with error details

        Usage:
            failure = _BaseResultFactory.create_failure(
                "Validation failed",
                error_code="VALIDATION_ERROR",
                error_data={"field": "email", "value": "invalid@"}
            )

        """
        return _BaseResult(error=error, error_code=error_code, error_data=error_data)

    @staticmethod
    def from_exception(exception: Exception) -> _BaseResult[T]:
        """Create failure result from exception with automatic error extraction.

        Converts Python exceptions into structured failure results preserving
        exception information and type for debugging and error handling.

        Args:
            exception: Python exception to convert to result

        Returns:
            _BaseResult[T] failure with exception message and type information

        Usage:
            try:
                risky_operation()
            except ValueError as e:
                result = _BaseResultFactory.from_exception(e)
                # result contains exception message and type

        """
        return _BaseResult(
            error=str(exception),
            error_data={"exception_type": type(exception).__name__},
        )

    @staticmethod
    def create_from_callable(
        func: object,
        error_message: TErrorMessage = "Operation failed",
    ) -> _BaseResult[T]:
        """Create result from callable execution with comprehensive exception handling.

        Executes callable function safely, capturing exceptions and converting
        them to structured failure results with detailed error context.

        Args:
            func: Callable to execute (validated for callability)
            error_message: Custom error message for exceptions

        Returns:
            _BaseResult[T] with function result or exception details

        Usage:
            def risky_operation():
                return "success"

            result = _BaseResultFactory.create_from_callable(risky_operation)
            if result.is_success:
                print(result.data)

        """
        try:
            if callable(func):
                result = func()
                return _BaseResult(data=result)
            return _BaseResult(error="Provided argument is not callable")
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            return _BaseResult(
                error=error_message,
                error_data={"exception": str(e), "exception_type": type(e).__name__},
            )

    @staticmethod
    def create_conditional(
        *,
        condition: bool,
        success_data: T,
        failure_message: TErrorMessage,
        failure_code: TErrorCode | None = None,
    ) -> _BaseResult[T]:
        """Create result based on boolean condition evaluation.

        Creates success or failure result based on condition evaluation,
        providing a clean way to convert boolean logic to railway patterns.

        Args:
            condition: Boolean condition to evaluate
            success_data: Data to include in success result
            failure_message: Error message for failure result
            failure_code: Optional error code for failure result

        Returns:
            _BaseResult[T] success if condition is True, failure otherwise

        Usage:
            result = _BaseResultFactory.create_conditional(
                condition=user.is_active,
                success_data=user,
                failure_message="User is not active",
                failure_code="USER_INACTIVE"
            )

        """
        if condition:
            return _BaseResult(data=success_data)
        return _BaseResult(error=failure_message, error_code=failure_code)


class _BaseResultOperations:
    """Comprehensive utility operations for result chaining and validation patterns.

    Utility class providing advanced operations for working with multiple results,
    validation patterns, and complex operation chains. Eliminates code duplication
    by consolidating all utility operations in a single location.

    Architecture:
        - Static utility methods for result aggregation and chaining
        - Validation and conversion patterns with predicate support
        - Exception-safe operation execution with automatic error handling
        - Multi-result combination with early failure detection
        - Functional composition utilities for complex workflows

    Operation Categories:
        - Result chaining: chain_results for aggregating multiple results
        - Validation operations: validate_and_convert for predicate-based validation
        - Safe execution: try_operation for exception-safe function execution
        - Multi-step workflows: Complex operation composition patterns

    Usage Patterns:
        # Chain multiple results
        results = _BaseResultOperations.chain_results(
            result1, result2, result3
        )

        # Validate and convert in one step
        validated = _BaseResultOperations.validate_and_convert(
            value="123",
            validator=str.isdigit,
            converter=int,
            error_message="Must be numeric"
        )

        # Safe operation execution
        result = _BaseResultOperations.try_operation(
            lambda: expensive_computation(),
            error_message="Computation failed"
        )

    """

    @staticmethod
    def chain_results(*results: _BaseResult[object]) -> _BaseResult[list[object]]:
        """Chain multiple results together with early failure detection.

        Aggregates multiple results into a single result containing all
        successful data. Returns failure immediately upon encountering
        the first failed result in the chain.

        Args:
            *results: Variable number of results to chain together

        Returns:
            _BaseResult[list[object]] with all data or first failure encountered

        Usage:
            result1 = _BaseResult.ok("first")
            result2 = _BaseResult.ok("second")
            result3 = _BaseResult.ok("third")

            chained = _BaseResultOperations.chain_results(result1, result2, result3)
            # chained.data == ["first", "second", "third"]

        """
        data_list = []
        for result in results:
            if result.is_failure:
                return _BaseResult.fail(
                    result.error or "Operation in chain failed",
                    result.error_code,
                    result.error_data,
                )
            if result.data is not None:
                data_list.append(result.data)
        return _BaseResult.ok(data_list)

    @staticmethod
    def validate_and_convert(
        value: object,
        validator: Callable[[object], bool],
        converter: Callable[[object], T],
        error_message: str = "Validation failed",
    ) -> _BaseResult[T]:
        """Validate and convert value in atomic operation with error handling.

        Performs validation and conversion as a single atomic operation,
        ensuring data integrity through predicate validation followed by
        type conversion with exception safety.

        Args:
            value: Value to validate and convert
            validator: Predicate function for validation
            converter: Function to convert validated value
            error_message: Custom error message for validation failure

        Returns:
            _BaseResult[T] with converted value or validation/conversion error

        Usage:
            # Convert string to integer with validation
            result = _BaseResultOperations.validate_and_convert(
                value="123",
                validator=str.isdigit,
                converter=int,
                error_message="Must be numeric string"
            )

        """
        try:
            if not validator(value):
                return _BaseResult.fail(error_message)
            converted = converter(value)
            return _BaseResult.ok(converted)
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            return _BaseResult.fail(f"Validation/conversion failed: {e}")

    @staticmethod
    def try_operation(
        operation: Callable[[], T],
        error_message: str = "Operation failed",
    ) -> _BaseResult[T]:
        """Execute operation with comprehensive exception handling and error context.

        Safely executes callable operation capturing exceptions and converting
        them to structured failure results with detailed error information.

        Args:
            operation: Zero-argument callable to execute safely
            error_message: Custom error message for exception cases

        Returns:
            _BaseResult[T] with operation result or exception details

        Usage:
            # Safe execution of potentially failing operation
            result = _BaseResultOperations.try_operation(
                lambda: complex_computation(),
                error_message="Complex computation failed"
            )

        """
        try:
            result = operation()
            return _BaseResult.ok(result)
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            return _BaseResult.fail(
                error_message,
                error_data={"exception": str(e), "exception_type": type(e).__name__},
            )


# =============================================================================
# MAIN BASE RESULT CLASS
# =============================================================================


class _BaseResult[T]:
    """Comprehensive railway-oriented programming implementation with monadic ops.

    Core result implementation providing type-safe error handling through railway
    programming patterns. Supports functional composition, error recovery, and effects
    management with immutable state and comprehensive error context preservation.

    Architecture:
        - Immutable result objects ensuring thread safety and consistency
        - Generic type support with proper variance and type safety
        - Monadic operations following functional programming laws
        - Comprehensive error context with codes and additional data
        - Exception-safe operations with automatic error propagation
        - Side effect isolation through tap operation

    Railway Programming Features:
        - Success/failure track separation with automatic error propagation
        - Functional composition through map and flat_map operations
        - Error short-circuiting preventing unnecessary computation
        - Error recovery mechanisms with alternative execution paths
        - Context preservation through monadic operation chains
        - Type-safe unwrapping with optional default values

    Monadic Operations:
        - map: Transform successful data with pure functions
        - flat_map: Chain result-returning operations (monadic bind)
        - filter: Conditional success based on predicate evaluation
        - recover: Error recovery with alternative result generation
        - tap: Side effect execution without result modification
        - combine: Multi-result aggregation with failure propagation

    Error Management:
        - Structured error information with human-readable messages
        - Optional error codes for programmatic error handling
        - Additional error data for debugging and context
        - Exception capture with type and message preservation
        - Error propagation through operation chains

    Usage Patterns:
        # Basic result creation and checking
        result = _BaseResult.ok("success")
        if result.is_success:
            data = result.unwrap()

        # Functional composition
        final_result = (
            _BaseResult.ok("input")
            .map(str.upper)
            .flat_map(lambda s: process_string(s))
            .filter(lambda s: len(s) > 5, "String too short")
        )

        # Error recovery
        result_with_fallback = (
            risky_operation()
            .recover(lambda error: _BaseResult.ok("default_value"))
        )

        # Side effects
        result.tap(lambda data: logger.info(f"Processing: {data}"))

    Thread Safety:
        - Immutable result objects safe for concurrent access
        - No shared mutable state between result instances
        - Thread-safe operation chaining and composition
        - Exception-safe side effect execution

    """

    def __init__(
        self,
        *,
        data: T | None = None,
        error: TErrorMessage | None = None,
        error_code: TErrorCode | None = None,
        error_data: dict[str, object] | None = None,
    ) -> None:
        """Initialize Result with data or error."""
        self._data = data
        self._error = error
        self._error_code = error_code
        self._error_data = error_data or {}

    @property
    def data(self) -> T | None:
        """Get result data."""
        return self._data

    @property
    def error(self) -> TErrorMessage | None:
        """Get error message."""
        return self._error

    @property
    def error_code(self) -> TErrorCode | None:
        """Get error code."""
        return self._error_code

    @property
    def error_data(self) -> dict[str, object]:
        """Get error data."""
        return self._error_data

    @property
    def is_success(self) -> bool:
        """Check if result is successful."""
        return self._error is None

    @property
    def is_failure(self) -> bool:
        """Check if result is failure."""
        return self._error is not None

    def __bool__(self) -> bool:
        """Boolean conversion returns success status."""
        return self.is_success

    def __eq__(self, other: object) -> bool:
        """Check equality based on success state and data/error."""
        if not isinstance(other, _BaseResult):
            return False
        if self.is_success != other.is_success:
            return False
        if self.is_success:
            return self.data == other.data
        return (
            self.error == other.error
            and self.error_code == other.error_code
            and self.error_data == other.error_data
        )

    def __hash__(self) -> int:
        """Hash based on result state and data/error."""
        if self.is_success:
            return hash((True, self.data))
        return hash(
            (
                False,
                self.error,
                self.error_code,
                tuple(sorted(self.error_data.items())),
            ),
        )

    def __repr__(self) -> str:
        """Return string representation showing result state."""
        if self.is_success:
            return f"_BaseResult(data={self.data!r}, is_success=True)"
        error_parts = [f"error={self.error!r}", "is_success=False"]
        if self.error_code:
            error_parts.append(f"error_code={self.error_code!r}")
        if self.error_data:
            error_parts.append(f"error_data={self.error_data!r}")
        return f"_BaseResult({', '.join(error_parts)})"

    @classmethod
    def ok(cls, data: T) -> _BaseResult[T]:
        """Create success result."""
        return cls(data=data)

    @classmethod
    def fail(
        cls,
        error: TErrorMessage,
        error_code: TErrorCode | None = None,
        error_data: dict[str, object] | None = None,
    ) -> _BaseResult[T]:
        """Create failure result."""
        return cls(error=error, error_code=error_code, error_data=error_data)

    def unwrap(self) -> T:
        """Unwrap result data or raise exception."""
        if self.is_failure:
            error_msg = f"Result failed: {self.error}"
            raise ValueError(error_msg)
        if self._data is None:
            msg = "Result has no data"
            raise ValueError(msg)
        return self._data

    def unwrap_or(self, default: T) -> T:
        """Unwrap result data or return default."""
        if self.is_failure or self._data is None:
            return default
        return self._data

    def map(self, transform_func: Callable[[T], U]) -> _BaseResult[U]:
        """Transform successful result data."""
        if self.is_failure:
            return _BaseResult.fail(
                self.error or "Previous operation failed",
                self.error_code,
                self.error_data,
            )

        try:
            if self._data is None:
                return _BaseResult.fail("Cannot transform None data")
            transformed_data = transform_func(self._data)
            return _BaseResult.ok(transformed_data)
        except (TypeError, ValueError, AttributeError) as e:
            return _BaseResult.fail(f"Transformation failed: {e}")

    def flat_map(self, transform_func: Callable[[T], _BaseResult[U]]) -> _BaseResult[U]:
        """Chain result-returning transformations."""
        if self.is_failure:
            return _BaseResult.fail(
                self.error or "Previous operation failed",
                self.error_code,
                self.error_data,
            )

        try:
            if self._data is None:
                return _BaseResult.fail("Cannot chain transform None data")
            return transform_func(self._data)
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            return _BaseResult.fail(f"Chain transformation failed: {e}")

    def filter(
        self,
        predicate: Callable[[T], bool],
        error_message: str = "Filter condition not met",
    ) -> _BaseResult[T]:
        """Filter result based on predicate."""
        if self.is_failure:
            return self

        try:
            if self._data is None:
                return _BaseResult.fail("Cannot filter None data")
            if predicate(self._data):
                return self
            return _BaseResult.fail(error_message)
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            return _BaseResult.fail(f"Filter evaluation failed: {e}")

    def recover(
        self,
        recovery_func: Callable[[str], _BaseResult[T]],
    ) -> _BaseResult[T]:
        """Recover from failure with alternative result."""
        if self.is_success:
            return self

        try:
            return recovery_func(self.error or "Unknown error")
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            return _BaseResult.fail(f"Recovery failed: {e}")

    def tap(self, side_effect: Callable[[T], None]) -> _BaseResult[T]:
        """Execute side effect without changing result."""
        if self.is_success and self._data is not None:
            with contextlib.suppress(
                TypeError,
                ValueError,
                AttributeError,
                RuntimeError,
            ):
                side_effect(self._data)
        return self

    @staticmethod
    def combine(
        result1: _BaseResult[T],
        result2: _BaseResult[U],
        combiner: Callable[[T, U], object],
    ) -> _BaseResult[object]:
        """Combine two results if both are successful."""
        if not result1.is_success:
            return _BaseResult.fail(
                result1.error or "First operation failed",
                result1.error_code,
                result1.error_data,
            )

        if not result2.is_success:
            return _BaseResult.fail(
                result2.error or "Second operation failed",
                result2.error_code,
                result2.error_data,
            )

        try:
            if result1.data is None or result2.data is None:
                return _BaseResult.fail("Cannot combine with None data")
            combined_data = combiner(result1.data, result2.data)
            return _BaseResult.ok(combined_data)
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            return _BaseResult.fail(f"Combination failed: {e}")


# Export API
__all__ = ["_BaseResult", "_BaseResultFactory", "_BaseResultOperations"]
