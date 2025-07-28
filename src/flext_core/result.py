"""FLEXT Core Result Module.

Comprehensive railway-oriented programming implementation for the FLEXT Core library
providing type-safe error handling through inheritance from specialized result base
classes.

Architecture:
    - Inheritance from specialized result base classes (_BaseResult, _BaseResultFactory)
    - Single source of truth pattern with _result_base.py as internal definitions
    - Railway-oriented programming with monadic operations and functional composition
    - Type-safe error handling without exception propagation across entire system
    - Clean public API with Flext* prefixed classes eliminating underscore prefixes

Result System Components:
    - FlextResult[T]: Main result type inheriting from _BaseResult with all
      functionality
    - FlextResultFactory: Factory methods inherited from _BaseResultFactory
    - FlextResultOperations: Utility operations inherited from _BaseResultOperations
    - Railway operations: bind, compose, switch, tee patterns for workflow orchestration
    - Transformation methods: map, flat_map, filter for functional data processing
    - Combination methods: combine for multi-result aggregation

Maintenance Guidelines:
    - Add new result patterns to _result_base.py first following established patterns
    - Use inheritance from base classes for consistent functionality
    - Maintain type safety across all transformation and composition operations
    - Follow railway programming patterns for error handling and control flow
    - Keep transformation chains pure and composable for predictable behavior

Design Decisions:
    - Single source of truth with _result_base.py for internal definitions
    - Direct inheritance from base classes eliminating code duplication
    - Clean public API with Flext* prefixed classes
    - Type-safe transformations preserving generic constraints
    - Factory patterns consolidated from base implementations

Railway Programming Features:
    - Success track: Data flows through transformations with type preservation
    - Failure track: Errors bypass transformations and propagate with full context
    - Monadic composition: Chain operations without nested error checking boilerplate
    - Function composition: Build complex workflows from simple operation primitives
    - Error recovery: Handle failures with alternative success paths and fallbacks
    - Side effects: Execute logging and monitoring without breaking functional chains

Dependencies:
    - _result_base: Foundation result implementations with all core functionality
    - types: Type definitions for generic programming and TYPE_CHECKING

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from flext_core._result_base import (
    _BaseResult,
    _BaseResultFactory,
    _BaseResultOperations,
)

# Local type definitions for runtime use
T = TypeVar("T")
U = TypeVar("U")

if TYPE_CHECKING:
    from flext_core.types import TFactory

# =============================================================================
# FLEXT RESULT - Direct inheritance from base eliminating code duplication
# =============================================================================

# Direct exposure with clean names - completely eliminates empty inheritance
FlextResult = _BaseResult

# =============================================================================
# FLEXT RESULT FACTORY - Direct exposure from base with clean names
# =============================================================================

# Direct exposure eliminating inheritance overhead
FlextResultFactory = _BaseResultFactory
FlextResultOperations = _BaseResultOperations

# =============================================================================
# RAILWAY OPERATIONS - Module level functions for backward compatibility
# =============================================================================


def chain(*results: FlextResult[object]) -> FlextResult[list[object]]:
    """Chain multiple results together with early failure detection.

    Module-level function providing backward compatibility and convenient
    access to result chaining functionality from operations class.

    Args:
        *results: Variable number of results to chain together

    Returns:
        FlextResult[list[object]] with all data or first failure encountered

    """
    return _BaseResultOperations.chain_results(*results)


def safe_call[T](func: TFactory[T]) -> FlextResult[T]:
    """Safely call function with FlextResult error handling.

    Convenience function providing direct access to safe execution patterns
    with comprehensive exception handling and error context preservation.

    Args:
        func: Function to execute safely

    Returns:
        FlextResult[T] with function result or captured exception

    """
    return _BaseResultFactory.create_from_callable(func)


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__ = [
    # Main result class
    "FlextResult",
    # Factory and operations
    "FlextResultFactory",
    "FlextResultOperations",
    # Convenience functions
    "chain",
    "safe_call",
]
