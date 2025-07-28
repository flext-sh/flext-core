"""FLEXT Core Decorators Module.

Comprehensive decorator system for the FLEXT Core library providing consolidated
functionality through multiple inheritance patterns and orchestration.

Architecture:
    - Multiple inheritance from specialized decorator base classes
    - Complex orchestration patterns combining multiple decorator types
    - Direct base exposure eliminating nested class overhead
    - FlextResult integration for error handling patterns
    - No underscore prefixes on public objects

Decorator Categories:
    - Validation decorators: Pydantic-based input validation
    - Error handling decorators: Safe execution with error capture
    - Performance decorators: Caching and timing functionality
    - Logging decorators: Structured logging integration
    - Immutability decorators: Frozen and read-only patterns
    - Functional decorators: Pure function and composition patterns

Maintenance Guidelines:
    - Add new decorator types to appropriate base classes first
    - Use multiple inheritance for decorator combination patterns
    - Maintain backward compatibility through function aliases
    - Implement complex orchestration in FlextDecorators main class
    - Keep individual decorators focused and composable

Design Decisions:
    - Multiple inheritance pattern for maximum functionality reuse
    - Complex orchestration methods combining multiple decorator types
    - Direct base exposure for specialized use cases
    - FlextResult integration for consistent error handling
    - Function aliases for backward compatibility

Orchestration Patterns:
    - safe_result: Exception handling with FlextResult returns
    - validated_with_result: Pydantic validation with FlextResult
    - cached_with_timing: Performance optimization with metrics
    - complete_decorator: Full-featured decorator orchestration

Dependencies:
    - _decorators_base: Foundation decorator implementations
    - result: FlextResult pattern for error handling
    - pydantic: Validation and model support

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from flext_core._decorators_base import (
    _BaseErrorHandlingDecorators,
    _BaseFunctionalDecorators,
    _BaseImmutabilityDecorators,
    _BaseLoggingDecorators,
    _BasePerformanceDecorators,
    _BaseValidationDecorators,
)
from flext_core.result import FlextResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core.types import F

# =============================================================================
# FLEXT DECORATORS - Consolidados com herança múltipla + funcionalidades específicas
# =============================================================================


class FlextDecorators(
    _BaseValidationDecorators,
    _BaseErrorHandlingDecorators,
    _BasePerformanceDecorators,
    _BaseLoggingDecorators,
    _BaseImmutabilityDecorators,
    _BaseFunctionalDecorators,
):
    """Consolidated decorators with multiple inheritance and orchestration capabilities.

    Provides comprehensive decorator functionality through multiple inheritance from
    six specialized base classes, adding complex orchestration patterns impossible
    with single inheritance.

    Architecture:
        - Multiple inheritance from six specialized decorator bases
        - Complex orchestration methods combining multiple decorator types
        - FlextResult integration for consistent error handling
        - Pydantic validation integration for input validation

    Inherited Decorator Categories:
        - Validation: Input validation, type checking, constraint validation
        - Error Handling: Exception capture, safe execution, error recovery
        - Performance: Caching, timing, optimization, profiling
        - Logging: Structured logging, call tracing, debug information
        - Immutability: Frozen objects, read-only patterns, data protection
        - Functional: Pure functions, composition, pipeline patterns

    Orchestration Methods:
        - safe_result: Exception handling with FlextResult returns
        - validated_with_result: Pydantic validation with FlextResult
        - cached_with_timing: Performance optimization with metrics
        - complete_decorator: Full-featured decorator with all capabilities

    Usage:
        # Simple safe execution
        @FlextDecorators.safe_result
        def risky_operation(data):
            return process_data(data)

        # Validation with result handling
        @FlextDecorators.validated_with_result(UserModel)
        def create_user(**kwargs):
            return User(**kwargs)

        # Complete orchestration
        @FlextDecorators.complete_decorator(
            UserModel, cache_size=256,
            with_timing=True, with_logging=True
        )
        def complex_operation(**kwargs):
            return perform_complex_task(**kwargs)
    """

    # =========================================================================
    # FUNCIONALIDADES ESPECÍFICAS (combinam múltiplas bases)
    # =========================================================================

    @classmethod
    def safe_result(cls, func: F) -> F:
        """Execute function safely with automatic exception handling and Result return.

        Complex orchestration pattern combining inherited error handling capabilities
        with FlextResult patterns for comprehensive safe execution. Automatically
        captures common exceptions and converts them to FlextResult failures.

        Architecture:
            - Uses inherited error handling patterns from _BaseErrorHandlingDecorators
            - Integrates FlextResult for consistent error handling across the system
            - Captures specific exception types relevant to business logic
            - Preserves original function signature while changing return type

        Exception Handling:
            - TypeError: Invalid argument types or incompatible operations
            - ValueError: Invalid argument values or constraints
            - AttributeError: Missing object attributes or methods
            - RuntimeError: General runtime failures and state errors

        Args:
            func: Function to wrap with safe execution and FlextResult return

        Returns:
            Decorated function that returns FlextResult[T] instead of T

        Usage:
            @FlextDecorators.safe_result
            def risky_database_query(user_id: str) -> User:
                return database.get_user(user_id)  # May raise exceptions

            result = risky_database_query("123")
            if result.is_success:
                user = result.data
            else:
                error_message = result.error

        """

        def wrapper(*args: object, **kwargs: object) -> object:
            try:
                result = func(*args, **kwargs)  # type: ignore[operator]
                return FlextResult.ok(result)
            except (TypeError, ValueError, AttributeError, RuntimeError) as e:
                return FlextResult.fail(str(e))

        return wrapper  # type: ignore[return-value]

    @classmethod
    def validated_with_result(cls, model_class: type[BaseModel]) -> Callable[[F], F]:
        """Validate using Pydantic + return FlextResult.

        Combines validation + error handling using inherited methods.
        """

        def decorator(func: F) -> F:
            def wrapper(*args: object, **kwargs: object) -> object:
                try:
                    # Use inherited validation methods
                    validated_data = model_class(**kwargs)
                    result = func(*args, **validated_data.model_dump())  # type: ignore[operator]
                    return FlextResult.ok(result)
                except ValidationError as e:
                    return FlextResult.fail(f"Validation failed: {e}")
                except (TypeError, ValueError, AttributeError, RuntimeError) as e:
                    return FlextResult.fail(f"Execution failed: {e}")

            return wrapper  # type: ignore[return-value]

        return decorator

    @classmethod
    def cached_with_timing(cls, max_size: int = 128) -> Callable[[F], F]:
        """Combine caching and timing measurement using inherited performance methods.

        Complex orchestration pattern that layers multiple performance optimizations
        by combining inherited caching and timing capabilities from base classes.
        Provides both performance optimization and performance monitoring.

        Architecture:
            - Uses _BasePerformanceDecorators.create_cache_decorator for result caching
            - Uses _BasePerformanceDecorators.get_timing_decorator for execution timing
            - Applies decorators in optimal order: cache first, then timing
            - Maintains function signature while adding performance capabilities

        Performance Benefits:
            - Caching: Eliminates redundant expensive computations
            - Timing: Provides execution time metrics for performance monitoring
            - Combined: Cached calls show near-zero execution times
            - Monitoring: Identifies performance bottlenecks and cache effectiveness

        Args:
            max_size: Maximum number of results to cache (default: 128)

        Returns:
            Decorator function that applies caching and timing to target function

        Usage:
            @FlextDecorators.cached_with_timing(max_size=256)
            def expensive_calculation(data: ComplexData) -> ProcessedResult:
                return perform_heavy_computation(data)

            # First call: measured execution time, result cached
            result1 = expensive_calculation(data)

            # Subsequent calls: near-zero execution time, cached result
            result2 = expensive_calculation(data)

        """

        def decorator(func: F) -> F:
            # Use inherited performance methods
            cached_func = cls.create_cache_decorator(max_size)(func)  # type: ignore[arg-type]
            return cls.get_timing_decorator()(cached_func)  # type: ignore[return-value]

        return decorator

    @classmethod
    def safe_cached(cls, max_size: int = 128) -> Callable[[F], F]:
        """Combine safe execution with caching using inherited error handling methods.

        Complex orchestration pattern that provides both exception safety and
        performance optimization by combining inherited capabilities from multiple
        base classes in optimal layering order.

        Architecture:
            - Uses _BaseErrorHandlingDecorators.get_safe_decorator for exception safety
            - Uses _BasePerformanceDecorators.create_cache_decorator for result caching
            - Applies safety first, then caching to ensure cache consistency
            - Maintains function signature while adding safety and performance

        Safety and Performance Benefits:
            - Exception Safety: Prevents crashes from unexpected errors
            - Result Caching: Eliminates redundant computations for same inputs
            - Cache Consistency: Only successful results are cached
            - Error Isolation: Exceptions don't corrupt cache state

        Args:
            max_size: Maximum number of results to cache (default: 128)

        Returns:
            Decorator function that applies safe execution and caching to function

        Usage:
            @FlextDecorators.safe_cached(max_size=64)
            def risky_expensive_operation(input_data: str) -> ProcessedData:
                # May raise exceptions, expensive to compute
                return process_data_with_external_api(input_data)

            # Safe execution with caching - exceptions handled, results cached
            result = risky_expensive_operation("test_data")

        """

        def decorator(func: F) -> F:
            # Use inherited error handling + performance methods
            safe_func = cls.get_safe_decorator()(func)  # type: ignore[arg-type]
            return cls.create_cache_decorator(max_size)(safe_func)  # type: ignore[return-value]

        return decorator

    @classmethod
    def validated_cached(
        cls,
        model_class: type[BaseModel],
        max_size: int = 128,
    ) -> Callable[[F], F]:
        """Combine validation, caching, and safe execution using inherited methods.

        Most comprehensive orchestration pattern combining validation, performance
        optimization, and error handling through complex layering of inherited
        capabilities from multiple base classes.

        Architecture:
            - Uses validated_with_result for Pydantic input validation
            - Uses _BasePerformanceDecorators.create_cache_decorator for result caching
            - Integrates FlextResult pattern for consistent error handling
            - Applies validation first, then caching for validated inputs only

        Comprehensive Benefits:
            - Input Validation: Ensures data integrity through Pydantic models
            - Result Caching: Optimizes performance for validated inputs
            - FlextResult Integration: Consistent error handling patterns
            - Cache Efficiency: Only valid inputs cached, invalid inputs rejected early

        Validation Process:
            1. Input validation using Pydantic model
            2. Function execution with validated data
            3. Result caching for successful operations
            4. FlextResult return for consistent error handling

        Args:
            model_class: Pydantic model class for input validation
            max_size: Maximum number of results to cache (default: 128)

        Returns:
            Decorator function that applies validation, caching, and safe execution

        Usage:
            @FlextDecorators.validated_cached(UserCreateModel, max_size=100)
            def create_user_with_validation(**user_data) -> FlextResult[User]:
                return User.create(user_data)

            # Input validated, results cached, FlextResult returned
            result = create_user_with_validation(
                email="user@example.com",
                name="John Doe"
            )

        """

        def decorator(func: F) -> F:
            # Combine validation + performance + error handling
            validated_func = cls.validated_with_result(model_class)(func)
            return cls.create_cache_decorator(max_size)(validated_func)  # type: ignore[return-value, arg-type]

        return decorator

    @classmethod
    def complete_decorator(
        cls,
        model_class: type[BaseModel] | None = None,
        cache_size: int = 128,
        *,
        with_timing: bool = False,
        with_logging: bool = False,
    ) -> Callable[[F], F]:
        """Complete decorator orchestrating all inherited base methods.

        Ultimate orchestration pattern that combines all available decorator
        capabilities from six different base classes in optimal layering order.
        Provides comprehensive
        function enhancement impossible to achieve with single inheritance.

        Architecture:
            - Orchestrates capabilities from all six base decorator classes
            - Applies decorators in optimal order for maximum effectiveness
            - Optional validation, timing, and logging based on configuration
            - Maintains function signature while adding comprehensive capabilities

        Decorator Layering Order (innermost to outermost):
            1. Validation (if model_class provided)
            2. Safe execution (always applied)
            3. Caching (always applied)
            4. Timing (if with_timing=True)
            5. Logging (if with_logging=True)

        Complete Feature Set:
            - Optional Input Validation: Pydantic model validation with FlextResult
            - Exception Safety: Comprehensive error handling and recovery
            - Performance Caching: Intelligent result caching with configurable size
            - Optional Timing: Execution time measurement and reporting
            - Optional Logging: Structured call logging with context

        Args:
            model_class: Optional Pydantic model for input validation
            cache_size: Maximum cache entries (default: 128)
            with_timing: Enable execution timing measurement (default: False)
            with_logging: Enable structured call logging (default: False)

        Returns:
            Fully orchestrated decorator function with all requested capabilities

        Usage:
            # Maximum functionality - all features enabled
            @FlextDecorators.complete_decorator(
                UserCreateModel,
                cache_size=256,
                with_timing=True,
                with_logging=True
            )
            def create_enterprise_user(**user_data) -> FlextResult[User]:
                return enterprise_user_service.create(user_data)

            # Minimal functionality - just safety and caching
            @FlextDecorators.complete_decorator()
            def simple_calculation(x: int, y: int) -> int:
                return x * y + complex_computation()

        """

        def decorator(func: F) -> F:
            current_func: object = func

            # Apply validation if model provided
            if model_class:
                current_func = cls.validated_with_result(model_class)(current_func)

            # Apply safe execution (inherited)
            current_func = cls.get_safe_decorator()(current_func)  # type: ignore[arg-type]

            # Apply caching (inherited)
            current_func = cls.create_cache_decorator(cache_size)(current_func)

            # Apply timing if requested (inherited)
            if with_timing:
                current_func = cls.get_timing_decorator()(current_func)

            # Apply logging if requested (inherited)
            if with_logging:
                current_func = cls.log_calls_decorator(current_func)

            return current_func  # type: ignore[return-value]

        return decorator


# =============================================================================
# EXPOSIÇÃO DIRETA DAS BASES ÚTEIS (aliases limpos sem herança vazia)
# =============================================================================

# Expose useful base classes directly with clean names
FlextValidationDecorators = _BaseValidationDecorators
FlextErrorHandlingDecorators = _BaseErrorHandlingDecorators
FlextPerformanceDecorators = _BasePerformanceDecorators
FlextLoggingDecorators = _BaseLoggingDecorators
FlextImmutabilityDecorators = _BaseImmutabilityDecorators
FlextFunctionalDecorators = _BaseFunctionalDecorators

# =============================================================================
# ESSENTIAL COMPATIBILITY FUNCTION (mantém apenas interface crítica)
# =============================================================================


# Mantém apenas safe_call como função essencial mais usada
def safe_call(func: F) -> F:
    """Safely call function with FlextResult return pattern.

    Essential function providing direct access to safe execution.

    Args:
        func: Function to wrap with safe execution

    Returns:
        Function that returns FlextResult instead of raising exceptions

    """
    return FlextDecorators.safe_result(func)


def cache_decorator(max_size: int = 128) -> object:
    """Cache decorator for function results.
    
    Args:
        max_size: Maximum cache size
        
    Returns:
        Decorator function

    """
    return FlextDecorators.create_cache_decorator(max_size)


def safe_decorator() -> object:
    """Safe execution decorator.
    
    Returns:
        Decorator function

    """
    return FlextDecorators.get_safe_decorator()


def timing_decorator(func: object) -> object:
    """Timing decorator for performance measurement.
    
    Args:
        func: Function to wrap with timing
        
    Returns:
        Wrapped function with timing

    """
    return FlextDecorators.get_timing_decorator()(func)


# =============================================================================
# EXPORTS - Clean public API seguindo diretrizes
# =============================================================================

__all__ = [
    # Main consolidated class with multiple inheritance
    "FlextDecorators",
    "FlextErrorHandlingDecorators",
    "FlextFunctionalDecorators",
    "FlextImmutabilityDecorators",
    "FlextLoggingDecorators",
    "FlextPerformanceDecorators",
    # Direct base exports (no inheritance overhead)
    "FlextValidationDecorators",
    # Essential compatibility functions
    "safe_call",
    "cache_decorator",
    "safe_decorator",
    "timing_decorator",
]
