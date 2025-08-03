"""FLEXT Core Result - Internal Implementation Module.

Internal implementation providing the foundational logic for result handling patterns.
This module is part of the Internal Implementation Layer and should not be imported
directly by ecosystem projects. Use the public API through result module instead.

Module Role in Architecture:
    Internal Implementation Layer → Result Operations → Public API Layer

    This internal module provides:
    - Backward compatibility aliases for result types
    - Base result operations and chaining utilities
    - Internal result composition patterns
    - Foundation operations for result handling workflows

Implementation Patterns:
    Backward Compatibility: _BaseResult alias maintains test compatibility
    Chaining Operations: Early failure detection with efficient composition

Design Principles:
    - Single responsibility for internal result implementation concerns
    - No external dependencies beyond core result module
    - Performance-optimized implementations for public API consumption
    - Type safety maintained through internal validation

Access Restrictions:
    - This module is internal and not exported in __init__.py
    - Use result module for all external access to result functionality
    - Breaking changes may occur without notice in internal modules
    - No compatibility guarantees for internal implementation details

Quality Standards:
    - Internal implementation must maintain public API contracts
    - Performance optimizations must not break type safety
    - Code must be thoroughly tested through public API surface
    - Internal changes must not affect public behavior

See Also:
    result: Public API for result handling and railway patterns
    docs/python-module-organization.md: Internal module architecture

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import TypeVar

from flext_core.result import FlextResult

# Type variables
T = TypeVar("T")

# Create alias for backward compatibility with tests
_BaseResult = FlextResult


class _BaseResultOperations:
    """Operations for result handling and composition."""

    @staticmethod
    def chain_results(*results: _BaseResult[object]) -> _BaseResult[list[object]]:
        """Chain multiple results together with early failure detection.

        Args:
            *results: Variable number of results to chain together

        Returns:
            _BaseResult[list[object]] with all data or first failure encountered

        """
        if not results:
            return _BaseResult.ok([])

        data: list[object] = []
        for result in results:
            if result.is_failure:
                return _BaseResult.fail(result.error or "Chain failed")
            if result.data is not None:
                data.append(result.data)
        return _BaseResult.ok(data)

    @staticmethod
    def combine_results(*results: _BaseResult[object]) -> _BaseResult[list[object]]:
        """Combine multiple results - alias for chain_results."""
        return _BaseResultOperations.chain_results(*results)
