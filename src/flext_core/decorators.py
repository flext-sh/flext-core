"""FLEXT Core Decorators - Extension Layer Function Enhancement.

Comprehensive decorator system providing function enhancement patterns for validation,
error handling, performance optimization, and cross-cutting concerns across the
32-project FLEXT ecosystem. Foundation for aspect-oriented programming and functional
composition in data integration and business logic components.

Module Role in Architecture:
    Extension Layer → Function Enhancement → Aspect-Oriented Programming

    This module provides decorator patterns used throughout FLEXT projects:
    - Validation decorators for Pydantic-based input validation
    - Error handling decorators for safe execution with FlextResult integration
    - Performance decorators for caching and timing functionality
    - Logging decorators for structured logging with context management

Decorator Architecture Patterns:
    Single Responsibility: Focused decorators for specific concerns
    Composition Orchestration: Combining multiple decorator types through delegation
    FlextResult Integration: Consistent error handling without exception propagation
    Performance Optimization: Minimal overhead with direct delegation patterns

Development Status (v0.9.0 → 1.0.0):
    ✅ Production Ready: Validation, error handling, performance, logging decorators
    🚧 Active Development: Functional composition patterns (Enhancement 4 - Med)
    📋 TODO Integration: Async decorator support for pipeline optimization (Priority 3)

Decorator System Categories:
    Validation Decorators: Pydantic-based input validation with automatic error handling
    Error Handling Decorators: Safe execution with error capture and FlextResult returns
    Performance Decorators: Caching and timing functionality with metrics collection
    Logging Decorators: Structured logging integration with context preservation
    Immutability Decorators: Frozen and read-only patterns for data integrity

Ecosystem Usage Patterns:
    # FLEXT Service Functions
    @FlextDecorators.safe_result
    @FlextDecorators.validated_input
    def create_user(user_data: UserCreateRequest) -> FlextResult[User]:
        return user_service.create(user_data)

    # Singer Tap/Target Functions
    @FlextDecorators.cached_with_timing("oracle_extract")
    @FlextDecorators.logged("data_extraction")
    def extract_oracle_table(connection: str, table: str) -> FlextResult[list]:
        return oracle_client.extract_data(connection, table)

    # ALGAR Migration Functions
    @FlextDecorators.complete_decorator(cache_key="ldap_migration")
    def migrate_ldap_users(source_dn: str, target_dn: str) -> FlextResult[int]:
        return ldap_service.migrate_users(source_dn, target_dn)

Decorator Orchestration Features:
    - safe_result: Exception handling with FlextResult returns and error context
    - validated_with_result: Pydantic validation with comprehensive error reporting
    - cached_with_timing: Performance optimization with metrics and cache management
    - complete_decorator: Full-featured decorator orchestration with all features

Quality Standards:
    - All decorators must preserve function signatures and type annotations
    - Error handling decorators must use FlextResult for consistent error propagation
    - Performance decorators must minimize overhead and provide accurate metrics
    - Validation decorators must provide actionable error messages for debugging

See Also:
    docs/TODO.md: Enhancement 4 - Functional composition pattern development
    _decorators_base.py: Foundation decorator implementations
    result.py: FlextResult pattern for error handling integration

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

from pydantic import BaseModel, ValidationError

from flext_core._decorators_base import (
    _BaseErrorHandlingDecorators,
    _BaseFunctionalDecorators,
    _BaseImmutabilityDecorators,
    _BaseLoggingDecorators,
    _BasePerformanceDecorators,
    _BaseValidationDecorators,
    _DecoratedFunction,
)
from flext_core.result import FlextResult, safe_call

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Protocol

    from flext_core.flext_types import F

    class CallableProtocol(Protocol):
        def __call__(self, *args: object, **kwargs: object) -> object: ...


T = TypeVar("T")

# =============================================================================
# FLEXT DECORATORS - Consolidados com herança múltipla + funcionalidades específicas
# =============================================================================


class FlextDecorators:
    """Consolidated decorators with composition-based orchestration capabilities.

    Provides comprehensive decorator functionality through composition and delegation
    to specialized base classes, offering orchestration patterns that combine
    multiple decorator types without multiple inheritance complexity.

    Architecture:
        - Composition-based delegation to specialized decorator bases
        - Orchestration methods combining multiple decorator types through composition
        - FlextResult integration for consistent error handling patterns
        - Performance-optimized delegation with minimal overhead
        - Clean separation between orchestration and base functionality

    Decorator Categories (accessed through composition):
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

        Delegates to the single source of truth safe_call implementation from result.py
        eliminating code duplication following DRY principles and architectural
        guidelines.

        Architecture:
            - Delegates to result.py safe_call for single source of truth pattern
            - Maintains decorator interface while eliminating implementation duplication
            - Integrates FlextResult for consistent error handling across the system
            - Follows "deliver more with much less" by reusing existing implementations

        Args:
            func: Function to wrap with safe execution and FlextResult return

        Returns:
            Decorated function that returns FlextResult[T] instead of T

        Usage:
            @FlextDecorators.safe_result
            def risky_database_query(user_id: str) -> User:
                return database.get_user(user_id)  # May raise exceptions

            result = risky_database_query("123")
            if result.success:
                user = result.data
            else:
                error_message = result.error

        """
        # Delegate to result.py single source of truth - eliminates duplication

        def wrapper(*args: object, **kwargs: object) -> object:
            def call_func() -> object:
                return cast("CallableProtocol", func)(*args, **kwargs)

            return safe_call(call_func)

        return cast("F", wrapper)

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
                    result = cast("CallableProtocol", func)(
                        *args,
                        **validated_data.model_dump(),
                    )
                    return FlextResult.ok(result)
                except ValidationError as e:
                    return FlextResult.fail(f"Validation failed: {e}")
                except (TypeError, ValueError, AttributeError, RuntimeError) as e:
                    return FlextResult.fail(f"Execution failed: {e}")

            return cast("F", wrapper)

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
            # Use composition to access base functionality
            cached_func = _BasePerformanceDecorators.create_cache_decorator(
                max_size,
            )(cast("_DecoratedFunction", func))
            return cast(
                "F",
                _BasePerformanceDecorators.get_timing_decorator()(cached_func),
            )

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
            # Use composition to access base functionality - cast types for
            # compatibility between F and _DecoratedFunction protocols
            safe_func = _BaseErrorHandlingDecorators.get_safe_decorator()(
                cast("_DecoratedFunction", func),
            )
            cached_func = _BasePerformanceDecorators.create_cache_decorator(
                max_size,
            )(safe_func)
            return cast("F", cached_func)

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
            # Combine validation + performance using composition - cast types for
            # compatibility between F and _DecoratedFunction protocols
            validated_func = cls.validated_with_result(model_class)(func)
            cached_func = _BasePerformanceDecorators.create_cache_decorator(
                max_size,
            )(cast("_DecoratedFunction", validated_func))
            return cast("F", cached_func)

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
            # Cast types for compatibility with base decorators
            current_func: _DecoratedFunction = cast("_DecoratedFunction", func)

            # Apply validation if model provided
            if model_class:
                validated_func = cls.validated_with_result(model_class)(
                    cast("F", current_func),
                )
                current_func = cast("_DecoratedFunction", validated_func)

            # Apply safe execution using composition
            current_func = _BaseErrorHandlingDecorators.get_safe_decorator()(
                current_func,
            )

            # Apply caching using composition
            current_func = _BasePerformanceDecorators.create_cache_decorator(
                cache_size,
            )(current_func)

            # Apply timing if requested using composition
            if with_timing:
                current_func = _BasePerformanceDecorators.get_timing_decorator()(
                    current_func,
                )

            # Apply logging if requested using composition
            if with_logging:
                current_func = _BaseLoggingDecorators.log_calls_decorator(current_func)

            return cast("F", current_func)

        return decorator


# =============================================================================
# EXPOSIÇÃO DIRETA DAS BASES ÚTEIS (aliases limpos sem herança vazia)
# =============================================================================

# Direct exposure with clean names - eliminates inheritance overhead
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
def flext_safe_call(func: F) -> F:
    """Safely call function with FlextResult return pattern.

    Essential function providing direct access to safe execution.

    Args:
        func: Function to wrap with safe execution

    Returns:
        Function that returns FlextResult instead of raising exceptions

    """
    return FlextDecorators.safe_result(func)


def flext_cache_decorator(
    max_size: int = 128,
) -> Callable[[F], F]:
    """Cache decorator for function results.

    Args:
        max_size: Maximum cache size

    Returns:
        Decorator function

    """
    return cast(
        "Callable[[F], F]", 
        _BasePerformanceDecorators.create_cache_decorator(max_size)
    )


def flext_safe_decorator() -> Callable[[F], F]:
    """Safe execution decorator.

    Returns:
        Decorator function

    """
    return cast("Callable[[F], F]", _BaseErrorHandlingDecorators.get_safe_decorator())


def flext_timing_decorator(func: F) -> F:
    """Apply timing decorator for performance measurement.

    Args:
        func: Function to wrap with timing

    Returns:
        Wrapped function with timing

    """
    return cast(
        "F", 
        _BasePerformanceDecorators.get_timing_decorator()(
            cast("_DecoratedFunction", func)
        )
    )


# =============================================================================
# EXPORTS - Clean public API seguindo diretrizes
# =============================================================================

__all__: list[str] = [
    "FlextDecorators",
    "FlextErrorHandlingDecorators",
    "FlextFunctionalDecorators",
    "FlextImmutabilityDecorators",
    "FlextLoggingDecorators",
    "FlextPerformanceDecorators",
    "FlextValidationDecorators",
    "flext_cache_decorator",
    "flext_safe_call",
    "flext_safe_decorator",
    "flext_timing_decorator",
]
