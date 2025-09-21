"""Concurrency utilities for FlextResult.

This module provides concurrency utilities for FlextResult,
extracted from the main FlextResult class to reduce complexity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.typings import TConcurrent

if TYPE_CHECKING:
    from flext_core.result import FlextResult


class FlextResultConcurrency:
    """Concurrency utilities for FlextResult.

    Provides parallel processing and concurrent operations
    extracted from the main FlextResult class.
    """

    @staticmethod
    def concurrent_sequence[TConcurrent](
        results: list[FlextResult[TConcurrent]], *, fail_fast: bool = True
    ) -> FlextResult[list[TConcurrent]]:
        """Sequence results with concurrent semantics.

        Arguments:
            results: The results to sequence.
            fail_fast: Whether to fail fast.

        Returns:
            A FlextResult containing the sequenced results.

        """
        if fail_fast:
            return FlextResultConcurrency._sequence_typed(results)
        return FlextResultConcurrency._accumulate_errors_typed(results)

    @staticmethod
    def _sequence_typed[TConcurrent](
        results: list[FlextResult[TConcurrent]],
    ) -> FlextResult[list[TConcurrent]]:
        """Type-safe sequence for concurrent operations.

        Arguments:
            results: The results to sequence.

        Returns:
            A FlextResult containing the sequenced results.

        """
        from flext_core.result import FlextResult  # noqa: PLC0415

        successes = []
        for result in results:
            if result.is_failure:
                error_msg = result.error or "Concurrent sequence operation failed"
                return FlextResult[list[TConcurrent]].fail(error_msg)
            successes.append(result.unwrap())
        return FlextResult[list[TConcurrent]].ok(successes)

    @staticmethod
    def _accumulate_errors_typed[TConcurrent](
        results: list[FlextResult[TConcurrent]],
    ) -> FlextResult[list[TConcurrent]]:
        """Type-safe error accumulation for concurrent operations.

        Arguments:
            results: The results to accumulate errors from.

        Returns:
            A FlextResult containing the accumulated errors.

        """
        from flext_core.result import FlextResult  # noqa: PLC0415

        successes = []
        errors = []
        for result in results:
            if result.is_failure:
                error_msg = result.error or "Concurrent operation failed"
                errors.append(error_msg)
            else:
                successes.append(result.unwrap())

        if errors:
            return FlextResult[list[TConcurrent]].fail("; ".join(errors))
        return FlextResult[list[TConcurrent]].ok(successes)
