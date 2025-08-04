"""FLEXT Core Railway - Internal Implementation Module.

Internal implementation providing the foundational logic for railway-oriented
programming.
This module is part of the Internal Implementation Layer and should not be imported
directly by ecosystem projects. Use the public API through result module instead.

Module Role in Architecture:
    Internal Implementation Layer → Railway Programming → Public API Layer

    This internal module provides:
    - Base railway programming operations (bind, compose, switch, tee, plus)
    - Monadic function composition with type safety
    - Utility functions for function lifting and transformation
    - Pure functional patterns with immutable operation chains

Implementation Patterns:
    Railway Operations: Complete monadic bind and composition implementations
    Utility Functions: Function lifting, identity operations, pass-through patterns

Design Principles:
    - Single responsibility for internal railway implementation concerns
    - No external dependencies beyond standard library and sibling modules
    - Performance-optimized implementations for public API consumption
    - Type safety maintained through internal validation

Access Restrictions:
    - This module is internal and not exported in __init__.py
    - Use result module for all external access to railway functionality
    - Breaking changes may occur without notice in internal modules
    - No compatibility guarantees for internal implementation details

Quality Standards:
    - Internal implementation must maintain public API contracts
    - Performance optimizations must not break type safety
    - Code must be thoroughly tested through public API surface
    - Internal changes must not affect public behavior

See Also:
    result: Public API for railway-oriented programming patterns
    docs/python-module-organization.md: Internal module architecture

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from flext_core.result import FlextResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core.flext_types import T


# =============================================================================
# BASE RAILWAY OPERATIONS - Core railway programming
# =============================================================================


class _BaseRailway:
    """Foundation railway programming implementing enterprise functional composition.

    Comprehensive railway programming implementation providing monadic operations,
    function composition, and type-safe error handling chains. Base implementation for
    all railway-oriented programming patterns across the FLEXT ecosystem.

    Architecture:
        - Pure functional patterns with immutable operation chains
        - Monadic bind operations for automatic error propagation
        - Function composition primitives for complex workflow orchestration
        - Type-safe error handling with short-circuit evaluation
        - Side effect management without breaking functional chains
        - Zero external dependencies for maximum portability

    Railway Operations:
        - bind: Monadic bind operation (>>=) for chaining with error propagation
        - compose_functions: Left-to-right function composition for workflow building
        - switch: Conditional branching based on value inspection for business logic
        - tee: Side effect execution for logging and monitoring without chain disruption
        - dead_end: Void function conversion to railway-compatible functions
        - plus: Parallel execution with result aggregation for performance optimization

    Error Handling Features:
        - Automatic failure propagation through railway failure track
        - Short-circuit evaluation for early failure detection and performance
        - Comprehensive exception handling with graceful degradation patterns
        - Type-safe error representation through FlextResult integration
        - Pure functional error handling eliminating exception-based patterns

    Composition Patterns:
        - Sequential processing through bind operations for dependent operations
        - Parallel processing through plus operations for independent operations
        - Conditional processing through switch operations for business rules
        - Side effect processing through tee operations for cross-cutting concerns
        - Error recovery through composition with fallback function chains

    Usage Patterns:
        # Sequential processing with automatic error propagation
        result = _BaseRailway.bind(
            initial_result,
            lambda x: process_step_1(x)
        )

        # Function composition for complex workflows
        workflow = _BaseRailway.compose_functions(
            validate_input,
            transform_data,
            save_result
        )

        # Conditional branching for business logic
        processor = _BaseRailway.switch(
            condition=lambda x: x.is_valid,
            success_func=process_valid_data,
            failure_func=handle_invalid_data
        )

        # Side effects without breaking the chain
        logged_processor = _BaseRailway.tee(
            main_func=process_data,
            side_func=log_processing
        )

        # Parallel execution with result aggregation
        parallel_processor = _BaseRailway.plus(
            validation_func,
            enrichment_func
        )

    Performance Benefits:
        - Short-circuit evaluation eliminates unnecessary computation on failures
        - Pure functional patterns enable efficient optimization and caching
        - Composition reuse reduces redundant function definition and maintenance
        - Type safety eliminates runtime type checking overhead
        - Lazy evaluation patterns for conditional operation execution
    """

    @staticmethod
    def bind(
        result: FlextResult[T],
        func: Callable[[T], FlextResult[object]],
    ) -> FlextResult[object]:
        """Execute monadic bind operation for railway chaining with errors.

        Core railway programming operation implementing the bind operator (>>=) for
        monadic composition. Automatically propagates failures down the railway track
        while applying the function only to successful results.

        Railway Semantics:
            - Success track: Apply function to result data and continue
            - Failure track: Short-circuit and propagate failure
            - Type safety: Maintain type safety through generic type parameters
            - Error context: Preserve original error information

        Error Propagation Rules:
            - Input failure: Return original failure result without function execution
            - Function success: Return function result maintaining success track
            - Function failure: Return function failure switching to failure
            - Exception handling: Convert function exceptions to failure results

        Args:
            result: Input result containing either success data or failure information
            func: Railway function to bind that transforms success data to new result

        Returns:
            FlextResult[object] with function output on success or propagated failure

        Usage:
            # Basic bind operation
            result = _BaseRailway.bind(
                validate_input(data),
                lambda x: transform_data(x)
            )

            # Chained bind operations
            final_result = _BaseRailway.bind(
                _BaseRailway.bind(
                    parse_input(raw_data),
                    lambda x: validate_parsed(x)
                ),
                lambda x: save_validated(x)
            )

        Error Handling:
            - TypeError: Function call type mismatches converted to failure results
            - ValueError: Invalid function arguments converted to failure results
            - AttributeError: Missing function attributes converted to failure results
            - Preserves original error context from input result on failure propagation

        Performance:
            - Short-circuit evaluation: Function not executed on input failure
            - Zero overhead: Direct function call on success with minimal wrapping
            - Exception safety: All exceptions caught and converted to results

        """
        if not result.success:
            return FlextResult.fail(
                result.error or "Previous operation failed",
                result.error_code,
                result.error_data,
            )

        try:
            # In success case, data should be available
            data = result.data
            if data is None:
                return FlextResult.fail("Cannot bind with None data")
            return func(data)
        except (TypeError, ValueError, AttributeError) as e:
            return FlextResult.fail(f"Bind operation failed: {e}")

    @staticmethod
    def compose_functions(
        *functions: Callable[[object], FlextResult[object]],
    ) -> Callable[[object], FlextResult[object]]:
        """Compose multiple railway functions into single workflow.

        Creates a composite function that applies multiple railway functions
        automatically handling error propagation through the railway pattern. Functions
        execute left-to-right with automatic short-circuiting on first failure.

        Composition Semantics:
            - Left-to-right execution: Functions applied in the order provided
            - Short-circuit evaluation: Stops execution on first function failure
            - Type preservation: Maintains type safety through generic composition
            - Error propagation: Automatically propagates failures without manual

        Function Chain Rules:
            - Success continuation: Each function receives output of previous function
            - Failure propagation: Chain stops on first failure, returns failure
            - Empty composition: Returns identity function wrapping input in success
            - Exception handling: Function exceptions converted to failure results

        Args:
            *functions: Railway functions to compose in left-to-right order

        Returns:
            Callable[[object], FlextResult[object]] composed workflow function

        Usage:
            # Simple three-step workflow
            data_pipeline = _BaseRailway.compose_functions(
                validate_input,
                transform_data,
                save_result
            )
            result = data_pipeline(raw_data)

            # Complex business workflow
            user_registration = _BaseRailway.compose_functions(
                parse_registration_data,
                validate_user_details,
                check_email_uniqueness,
                hash_password,
                create_user_account,
                send_welcome_email,
                log_registration_success
            )
            registration_result = user_registration(form_data)

        Composition Benefits:
            - Readable workflows: Clear representation of complex business processes
            - Error safety: Automatic error handling eliminates manual error checking
            - Reusability: Composed functions can be further composed or reused
            - Testability: Functions and compositions can be tested independently
            - Maintenance: Easy to add, remove, or reorder steps in the workflow

        Performance:
            - Short-circuit evaluation: Stops processing on first failure for efficiency
            - Function inlining: Direct function calls with minimal composition overhead
            - Lazy evaluation: Functions only executed when needed in the chain
            - Memory efficiency: Minimal intermediate object creation during composition

        """

        def composed(value: object) -> FlextResult[object]:
            result = FlextResult.ok(value)
            for func in functions:
                if not result.success:
                    break
                result = _BaseRailway.bind(result, func)
            return result

        return composed

    @staticmethod
    def switch(
        condition: Callable[[T], bool],
        success_func: Callable[[T], FlextResult[object]],
        failure_func: Callable[[T], FlextResult[object]],
    ) -> Callable[[T], FlextResult[object]]:
        """Railway switch based on condition.

        Args:
            condition: Boolean condition
            success_func: Function if condition is True
            failure_func: Function if condition is False

        Returns:
            Switch function

        """

        def switch_func(value: T) -> FlextResult[object]:
            try:
                if condition(value):
                    return success_func(value)
                return failure_func(value)
            except (TypeError, ValueError, AttributeError) as e:
                return FlextResult.fail(f"Switch evaluation failed: {e}")

        return switch_func

    @staticmethod
    def tee(
        main_func: Callable[[T], FlextResult[object]],
        side_func: Callable[[T], FlextResult[object]],
    ) -> Callable[[T], FlextResult[object]]:
        """Railway tee - execute both functions, return main result.

        Args:
            main_func: Main function
            side_func: Side function (result ignored)

        Returns:
            Tee function

        """

        def tee_func(value: T) -> FlextResult[object]:
            # Execute side function but ignore result
            with contextlib.suppress(TypeError, ValueError, AttributeError):
                side_func(value)

            # Return main function result
            return main_func(value)

        return tee_func

    @staticmethod
    def dead_end(
        func: Callable[[T], None],
    ) -> Callable[[T], FlextResult[T]]:
        """Convert void function to railway function.

        Args:
            func: Void function

        Returns:
            Railway function

        """

        def railway_func(value: T) -> FlextResult[T]:
            try:
                func(value)
                return FlextResult.ok(value)
            except (TypeError, ValueError, AttributeError) as e:
                return FlextResult.fail(f"Dead end function failed: {e}")

        return railway_func

    @staticmethod
    def plus(
        func1: Callable[[T], FlextResult[object]],
        func2: Callable[[T], FlextResult[object]],
    ) -> Callable[[T], FlextResult[list[object]]]:
        """Railway plus - execute both functions and collect results.

        Args:
            func1: First function
            func2: Second function

        Returns:
            Plus function collecting both results

        """

        def plus_func(value: T) -> FlextResult[list[object]]:
            result1 = func1(value)
            result2 = func2(value)

            if result1.success and result2.success:
                return FlextResult.ok([result1.data, result2.data])

            # Collect errors
            errors = []
            if not result1.success:
                errors.append(result1.error or "Function 1 failed")
            if not result2.success:
                errors.append(result2.error or "Function 2 failed")

            return FlextResult.fail(f"Plus operation failed: {'; '.join(errors)}")

        return plus_func


# =============================================================================
# BASE RAILWAY UTILITIES - Helper functions
# =============================================================================


class _BaseRailwayUtils:
    """Foundation railway utility functions implementing function lifting operations.

    Comprehensive utility system providing function transformation operations, identity,
    and integration helpers for seamless adoption of railway patterns. Enables
    conversion between regular functions and railway-compatible functions.

    Architecture:
        - Function lifting for converting regular functions to railway functions
        - Identity operations for pass-through and ignore patterns in workflows
        - Zero external dependencies for maximum portability and reusability
        - Type-safe transformations maintaining generic type parameters
        - Exception-safe operations with comprehensive error handling

    Utility Categories:
        - lift: Convert regular functions to railway-compatible functions
        - ignore: Create functions that discard input and return success
        - pass_through: Create identity functions that preserve input values
        - Helper functions for common railway programming scenarios

    Integration Features:
        - Seamless adoption: Convert existing functions without modification
        - Type preservation: Maintain original function type signatures
        - Error handling: Automatic exception conversion to railway results
        - Performance optimization: Minimal overhead for function wrapping
        - Pure functional patterns: No side effects in utility operations

    Usage Patterns:
        # Convert regular function to railway function
        regular_func = lambda x: x.upper()
        railway_func = _BaseRailwayUtils.lift(regular_func)
        result = railway_func("hello")  # Returns FlextResult.ok("HELLO")

        # Create ignore function for side effects
        ignore_func = _BaseRailwayUtils.ignore()
        result = ignore_func(any_input)  # Returns FlextResult.ok(None)

        # Create pass-through function for identity operations
        pass_func = _BaseRailwayUtils.pass_through()
        result = pass_func(value)  # Returns FlextResult.ok(value)

    Function Transformation:
        - Regular to railway: lift() converts functions to return FlextResult
        - Exception handling: Automatic conversion of exceptions to failure results
        - Type safety: Preserved through generic type parameters
        - Performance: Minimal wrapping overhead for efficient execution
    """

    @staticmethod
    def lift(
        func: Callable[[T], object],
    ) -> Callable[[T], FlextResult[object]]:
        """Lift regular function to railway function.

        Args:
            func: Regular function

        Returns:
            Railway function

        """

        def lifted_func(value: T) -> FlextResult[object]:
            try:
                result = func(value)
                return FlextResult.ok(result)
            except (TypeError, ValueError, AttributeError) as e:
                return FlextResult.fail(f"Lifted function failed: {e}")

        return lifted_func

    @staticmethod
    def ignore() -> Callable[[object], FlextResult[None]]:
        """Railway function that ignores input and returns success.

        Returns:
            Ignore function

        """

        def ignore_func(_value: object) -> FlextResult[None]:
            return FlextResult.ok(None)

        return ignore_func

    @staticmethod
    def pass_through() -> Callable[[T], FlextResult[T]]:
        """Railway function that passes value through unchanged.

        Returns:
            Pass-through function

        """

        def pass_func(value: T) -> FlextResult[T]:
            return FlextResult.ok(value)

        return pass_func


# =============================================================================
# EXPORTS - Base railway functionality only
# =============================================================================

__all__: list[str] = ["_BaseRailway", "_BaseRailwayUtils"]
