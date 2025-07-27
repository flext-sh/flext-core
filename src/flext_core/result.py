"""FLEXT Core Result Module.

Railway-oriented programming implementation for the FLEXT Core library providing
comprehensive error handling through consolidated base orchestration patterns.

Architecture:
    - Railway-oriented programming patterns for error handling
    - Complex orchestration combining multiple base implementations
    - Type-safe error handling without exception propagation
    - Functional programming patterns with monadic operations
    - No underscore prefixes on public objects

Result System Components:
    - FlextResult[T]: Main result type with success/failure states
    - Factory methods: Create results from various sources
    - Transformation methods: Map, chain, filter operations
    - Railway operations: Bind, recover, tap patterns
    - Combination methods: Sequence and combine multiple results

Maintenance Guidelines:
    - Maintain type safety across all transformation operations
    - Preserve railway patterns for consistent error handling
    - Use orchestration patterns combining multiple base implementations
    - Avoid exception-based error handling in favor of result patterns
    - Keep transformation chains composable and predictable

Design Decisions:
    - Inheritance from _BaseResult for core functionality
    - Orchestration patterns combining factory and railway bases
    - Type-safe transformations preserving generic constraints
    - Functional programming patterns with pure functions
    - Complex operations impossible with single base implementation

Railway Programming Patterns:
    - Success path: Data flows through transformations
    - Failure path: Errors bypass transformations and propagate
    - Composition: Chain operations without nested error checking
    - Recovery: Handle failures with alternative success paths

Dependencies:
    - _railway_base: Core railway programming patterns
    - _result_base: Basic result functionality and factory methods
    - types: Type definitions for generic programming

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from flext_core._railway_base import _BaseRailway
from flext_core._result_base import (
    _BaseResult,
    _BaseResultFactory,
)

# _BaseResultTransforms functionality now in _BaseResult
from flext_core.types import (
    T,
    TAnyDict,
    TErrorCode,
    TErrorMessage,
    TFactory,
    U,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# =============================================================================
# FLEXT RESULT - FUNCIONALIDADES COMPLEXAS que bases sozinhas NÃO podem fazer
# =============================================================================


class FlextResult(_BaseResult[T]):
    """Railway-oriented programming result type with complex orchestration patterns.

    Implements comprehensive error handling through railway programming patterns,
    combining multiple base implementations to provide functionality impossible
    with single base classes.

    Architecture:
        - Inherits core functionality from _BaseResult[T]
        - Orchestrates factory and railway patterns from multiple bases
        - Type-safe generic operations preserving type constraints
        - Functional composition without exception propagation

    Result States:
        - Success: Contains data of type T with is_success = True
        - Failure: Contains error information with is_success = False
        - Immutable: All operations return new result instances

    Core Operations:
        - Creation: ok(data) for success, fail(error) for failure
        - Transformation: map, chain, where for data processing
        - Railway: then, recover, tap for control flow
        - Combination: combine, sequence for multiple results

    Usage Patterns:
        # Basic usage
        result = FlextResult.ok("valid data")
        failed = FlextResult.fail("error occurred")

        # Chaining operations
        final = (
            FlextResult.ok(5)
            .map(lambda x: x * 2)
            .chain(lambda x: process_value(x))
            .recover(lambda err: FlextResult.ok(default_value))
        )

        # Combining results
        combined = FlextResult.combine(
            result1, result2, lambda a, b: (a, b)
        )

        # Sequencing multiple operations
        sequence = FlextResult.sequence(result1, result2, result3)
    """

    @classmethod
    def ok(cls, data: T) -> FlextResult[T]:
        """Create successful result containing the provided data.

        Factory method for creating success results in railway programming patterns.
        The result will have is_success=True and contain the provided data.

        Args:
            data: The successful value to wrap in the result

        Returns:
            FlextResult[T] in success state containing the data

        Usage:
            result = FlextResult.ok("operation succeeded")
            user_result = FlextResult.ok(User(id="123", name="John"))

        """
        # Use inherited method directly and cast to FlextResult
        super().ok(data)
        return cls(data=data)

    @classmethod
    def fail(
        cls,
        error: TErrorMessage,
        error_code: TErrorCode | None = None,
        error_data: TAnyDict | None = None,
    ) -> FlextResult[T]:
        """Create failure result containing error information.

        Factory method for creating failure results in railway programming patterns.
        The result will have is_success=False and contain structured error information.

        Args:
            error: Human-readable error message
            error_code: Machine-readable error categorization code
            error_data: Additional structured error context

        Returns:
            FlextResult[T] in failure state containing error information

        Usage:
            result = FlextResult.fail("Validation failed")
            detailed = FlextResult.fail(
                "Invalid email format",
                error_code="VALIDATION_ERROR",
                error_data={"field": "email", "value": "invalid"}
            )

        """
        # Use inherited method directly
        return cls(error=error, error_code=error_code, error_data=error_data)

    # =========================================================================
    # COMPLEX FUNCTIONALITY: Factory + Railway patterns combined
    # =========================================================================

    @classmethod
    def from_callable(
        cls,
        func: TFactory[T],
        error_message: TErrorMessage = "Operation failed",
    ) -> FlextResult[T]:
        """Create result from callable with automatic exception handling.

        Complex orchestration pattern combining factory and exception handling.
        Executes the provided callable and captures any exceptions as failure results.

        Args:
            func: Callable that returns value of type T
            error_message: Error message to use if callable raises exception

        Returns:
            FlextResult[T] with success data or captured exception as failure

        Usage:
            # Safe database operation
            result = FlextResult.from_callable(
                lambda: database.get_user(user_id),
                "Failed to retrieve user"
            )

            # Safe file operation
            config_result = FlextResult.from_callable(
                lambda: json.load(open("config.json")),
                "Failed to load configuration"
            )

        """
        # Use inherited factory with direct instantiation
        return _BaseResultFactory.create_from_callable(func, error_message)

    @classmethod
    def conditional(
        cls,
        *,
        condition: bool,
        success_data: T,
        failure_message: TErrorMessage,
        failure_code: TErrorCode | None = None,
    ) -> FlextResult[T]:
        """Create result based on boolean condition evaluation.

        Complex conditional factory pattern for branching logic without explicit
        if/else constructs in railway programming style.

        Args:
            condition: Boolean condition to evaluate
            success_data: Data to use if condition is True
            failure_message: Error message if condition is False
            failure_code: Optional error code for failure case

        Returns:
            FlextResult[T] with success or failure based on condition

        Usage:
            # Conditional validation
            result = FlextResult.conditional(
                condition=user.is_active,
                success_data=user,
                failure_message="User account is inactive",
                failure_code="INACTIVE_USER"
            )

            # Authorization check
            auth_result = FlextResult.conditional(
                condition=has_permission(user, resource),
                success_data=resource,
                failure_message="Access denied"
            )

        """
        # Use inherited factory with direct instantiation
        return _BaseResultFactory.create_conditional(
            condition=condition,
            success_data=success_data,
            failure_message=failure_message,
            failure_code=failure_code,
        )

    # =========================================================================
    # COMPLEX TRANSFORMATIONS: Transform + Railway orchestration
    # =========================================================================

    def map(self, transform_func: Callable[[T], U]) -> FlextResult[U]:
        """Transform success data with function, preserving failure state.

        Railway programming pattern for data transformation. If this result is
        successful,
        applies the transformation function to the data. If this result is a failure,
        bypasses the transformation and propagates the failure.

        Args:
            transform_func: Function to transform data from type T to type U

        Returns:
            FlextResult[U] with transformed data or propagated failure

        Usage:
            # Transform successful data
            result = FlextResult.ok(5).map(lambda x: x * 2)  # FlextResult.ok(10)

            # Chain transformations
            user_email = (
                get_user_result()
                .map(lambda user: user.email)
                .map(lambda email: email.lower())
            )

        """
        # Use inherited map method directly
        base_result = super().map(transform_func)
        return FlextResult._from_base(base_result)

    def chain(self, transform_func: Callable[[T], FlextResult[U]]) -> FlextResult[U]:
        """Chain operations that return results, flattening nested results.

        Complex railway orchestration for chaining operations that themselves return
        FlextResult instances. Prevents nested FlextResult[FlextResult[U]] structures.

        Args:
            transform_func: Function that takes T and returns FlextResult[U]

        Returns:
            FlextResult[U] with chained operation result or propagated failure

        Usage:
            # Chain operations that return results
            result = (
                FlextResult.ok(user_id)
                .chain(lambda id: database.get_user(id))  # Returns FlextResult[User]
                .chain(lambda user: validate_user(user))  # Returns FlextResult[User]
            )

        """

        def base_transform(value: T) -> _BaseResult[U]:
            result = transform_func(value)
            return result.to_base()

        base_result = super().flat_map(base_transform)
        return FlextResult._from_base(base_result)

    def where(
        self,
        predicate: Callable[[T], bool],
        error_message: str = "Filter condition not met",
    ) -> FlextResult[T]:
        """Filter result data with predicate, converting to failure if predicate fails.

        Complex filter orchestration with error handling. If this result is successful
        and the predicate returns True, preserves the success. Otherwise, converts
        to failure.

        Args:
            predicate: Function that returns True if data should be preserved
            error_message: Error message to use if predicate returns False

        Returns:
            FlextResult[T] preserving data if predicate passes, or failure if not

        Usage:
            adult_user = (
                get_user_result()
                .where(lambda user: user.age >= 18, "User must be adult")
            )

        """
        if self.is_failure:
            return self

        try:
            if self.data is None:
                return FlextResult.fail("Cannot filter None data")
            if predicate(self.data):
                return self
            return FlextResult.fail(error_message)
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            return FlextResult.fail(f"Filter evaluation failed: {e}")

    # =========================================================================
    # COMPLEX RAILWAY PATTERNS: Multiple base orchestration
    # =========================================================================

    def then(self, func: Callable[[T], FlextResult[object]]) -> FlextResult[object]:
        """Railway bind operation with type conversion orchestration.

        Complex railway pattern for sequential operations with type flexibility.
        Similar to chain but allows type conversion and more flexible return types.

        Args:
            func: Function that takes T and returns FlextResult[object]

        Returns:
            FlextResult[object] with operation result or propagated failure

        Usage:
            result = (
                FlextResult.ok(user_data)
                .then(lambda data: process_user(data))
                .then(lambda user: send_notification(user))
            )

        """

        def base_func(value: T) -> _BaseResult[object]:
            result = func(value)
            return result.to_base()

        base_result = _BaseRailway.bind(self, base_func)
        return FlextResult._from_base(base_result)

    def recover(
        self,
        recovery_func: Callable[[str], FlextResult[T]],
    ) -> FlextResult[T]:
        """Recover from failure with alternative success path.

        Complex recovery orchestration allowing failure handling with alternative
        operations. If this result is successful, returns it unchanged. If failure,
        applies recovery function to the error message.

        Args:
            recovery_func: Function that takes error string and returns FlextResult[T]

        Returns:
            FlextResult[T] with original success or recovery result

        Usage:
            result = (
                risky_operation()
                .recover(lambda err: FlextResult.ok(default_value))
                .recover(lambda err: load_from_cache())
            )

        """
        if self.is_success:
            return self

        try:
            return recovery_func(self.error or "Unknown error")
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            return FlextResult.fail(f"Recovery failed: {e}")

    def tap(self, side_effect: Callable[[T], None]) -> FlextResult[T]:
        """Execute side effect preserving original result.

        Complex side effect orchestration that executes the provided function
        for successful results without affecting the result value or state.
        Useful for logging, metrics, or other observability operations.

        Args:
            side_effect: Function to execute for side effects (returns None)

        Returns:
            FlextResult[T] unchanged from original result

        Usage:
            result = (
                process_data()
                .tap(lambda data: log.info(f"Processed: {data}"))
                .tap(lambda data: metrics.increment("processed"))
                .map(lambda data: transform_data(data))
            )

        """
        if self.is_success and self.data is not None:
            with contextlib.suppress(
                TypeError,
                ValueError,
                AttributeError,
                RuntimeError,
            ):
                side_effect(self.data)
        return self

    # =========================================================================
    # COMPLEX STATIC OPERATIONS: Multi-base orchestration
    # =========================================================================

    @staticmethod
    def combine(
        result1: FlextResult[T],
        result2: FlextResult[U],
        combiner: Callable[[T, U], object],
    ) -> FlextResult[object]:
        """Combine two results with combiner function if both are successful.

        Complex multi-result combination orchestration that applies the combiner
        function to both result values only if both results are successful.
        If either result is a failure, returns the first failure encountered.

        Args:
            result1: First result of type FlextResult[T]
            result2: Second result of type FlextResult[U]
            combiner: Function that combines T and U into final result

        Returns:
            FlextResult[object] with combined value or first failure

        Usage:
            # Combine two database queries
            combined = FlextResult.combine(
                get_user_result(),
                get_posts_result(),
                lambda user, posts: {"user": user, "posts": posts}
            )

            # Combine validation results
            final = FlextResult.combine(
                validate_email(email),
                validate_password(password),
                lambda e, p: User(email=e, password=p)
            )

        """
        if not result1.is_success:
            return FlextResult.fail(
                result1.error or "First operation failed",
                result1.error_code,
                result1.error_data,
            )

        if not result2.is_success:
            return FlextResult.fail(
                result2.error or "Second operation failed",
                result2.error_code,
                result2.error_data,
            )

        try:
            if result1.data is None or result2.data is None:
                return FlextResult.fail("Cannot combine with None data")
            combined_data = combiner(result1.data, result2.data)
            return FlextResult.ok(combined_data)
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            return FlextResult.fail(f"Combination failed: {e}")

    @staticmethod
    def sequence(*results: FlextResult[object]) -> FlextResult[list[object]]:
        """Sequence multiple results into a single result containing all values.

        Complex multiple result sequencing orchestration that combines multiple
        results into a single result containing a list of all values. If any
        result is a failure, returns the first failure encountered.

        Args:
            *results: Variable number of FlextResult instances

        Returns:
            FlextResult[list[object]] with all values or first failure

        Usage:
            # Sequence multiple operations
            all_data = FlextResult.sequence(
                get_user_data(),
                get_settings_data(),
                get_preferences_data()
            )

            # Process only if all succeed
            if all_data.is_success:
                user_data, settings, preferences = all_data.data

        """
        if not results:
            return FlextResult.ok([])

        # Orchestrate multiple base operations
        accumulated_data = []
        for result in results:
            if not result.is_success:
                return FlextResult.fail(
                    result.error or "Sequence operation failed",
                    result.error_code,
                    result.error_data,
                )
            accumulated_data.append(result.data)

        return FlextResult.ok(accumulated_data)

    # =========================================================================
    # INTERNAL HELPERS: Base conversion orchestration
    # =========================================================================

    @classmethod
    def _from_base(cls, base: _BaseResult[T]) -> FlextResult[T]:
        """Convert base to FlextResult - orchestration helper."""
        return cls(
            data=base.data,
            error=base.error,
            error_code=base.error_code,
            error_data=base.error_data,
        )

    def to_base(self) -> _BaseResult[T]:
        """Convert to base result - public orchestration helper."""
        if self.is_success and self.data is not None:
            return _BaseResult.ok(self.data)
        return _BaseResult.fail(
            self.error or "Unknown error",
            self.error_code,
            self.error_data,
        )


# =============================================================================
# PUBLIC API - Complex orchestration exports
# =============================================================================

__all__ = ["FlextResult"]
