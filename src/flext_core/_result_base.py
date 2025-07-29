"""FLEXT Core Result Base Module.

Internal base classes and operations for result handling.
This module provides the foundation for FlextResult operations.

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
