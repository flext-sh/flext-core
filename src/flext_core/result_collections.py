"""Collection operations utilities for FlextResult.

This module provides collection processing utilities for FlextResult,
extracted from the main FlextResult class to reduce complexity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from flext_core.typings import TAccumulate, TItem, TParallel, TResult, TUtil, UParallel

if TYPE_CHECKING:
    from flext_core.result import FlextResult


class FlextResultCollections:
    """Collection processing utilities for FlextResult.

    OPTIMIZATION ANALYSIS:
    - Traverse operation: MEDIUM USAGE (5+ occurrences) - Core collection pattern
    - Collect operations: MEDIUM USAGE (3-5 occurrences each) - Data extraction
    - Accumulate errors: LOW USAGE (2+ occurrences) - Error collection pattern
    - Parallel operations: LOW USAGE (1-2 occurrences) - Advanced patterns
    - Validation operations: LOW USAGE (1-3 occurrences) - Validation patterns

    Provides batch processing, error accumulation, and parallel operations
    extracted from the main FlextResult class.

    OPTIMIZATION: Consolidated duplicate implementations, streamlined error handling
    """

    # =========================================================================
    # CORE TRAVERSAL OPERATIONS - MEDIUM USAGE (5+ occurrences)
    # =========================================================================

    @staticmethod
    def traverse[TItem, TResult](
        items: list[TItem],
        func: Callable[[TItem], FlextResult[TResult]],
    ) -> FlextResult[list[TResult]]:
        """Traverse operation from Category Theory - ADVANCED FUNCTIONAL PATTERN.

        USAGE: 5+ occurrences - Core collection processing pattern
        OPTIMIZATION: Streamlined iteration with early termination

        Arguments:
            items: The items to traverse.
            func: The function to traverse.

        Returns:
            A FlextResult containing the traversed results.

        """
        from flext_core.result import FlextResult  # noqa: PLC0415

        results = []
        for item in items:
            result = func(item)
            if result.is_failure:
                return FlextResult[list[TResult]].fail(
                    result.error or "Traverse operation failed"
                )
            results.append(result.unwrap())
        return FlextResult[list[TResult]].ok(results)

    # =========================================================================
    # ERROR ACCUMULATION OPERATIONS - LOW USAGE (2+ occurrences)
    # =========================================================================

    @staticmethod
    def accumulate_errors[TAccumulate](
        *results: FlextResult[TAccumulate],
    ) -> FlextResult[list[TAccumulate]]:
        """Accumulate all errors instead of early termination.

        USAGE: 2+ occurrences - Error collection pattern
        OPTIMIZATION: Streamlined error collection with metadata

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
                errors.append(result.error or "Unknown error")
            else:
                successes.append(result.unwrap())

        if errors:
            return FlextResult[list[TAccumulate]].fail(
                f"Accumulated errors: {'; '.join(errors)}",
                error_code="ACCUMULATED_ERRORS",
                error_data={"error_count": len(errors), "errors": errors},
            )

        return FlextResult[list[TAccumulate]].ok(successes)

    # =========================================================================
    # COLLECTION EXTRACTION OPERATIONS - MEDIUM USAGE (3-5 occurrences each)
    # =========================================================================

    @staticmethod
    def collect_successes[TUtil](results: list[FlextResult[TUtil]]) -> list[TUtil]:
        """Collect successful values from results.

        USAGE: 3+ occurrences - Success data extraction pattern
        OPTIMIZATION: Streamlined list comprehension

        Arguments:
            results: The results to collect successes from.

        Returns:
            A list of successful values.

        """
        return [result.unwrap() for result in results if result.is_success]

    @staticmethod
    def collect_failures[TUtil](results: list[FlextResult[TUtil]]) -> list[str]:
        """Collect error messages from failures.

        USAGE: 2+ occurrences - Error message extraction pattern
        OPTIMIZATION: Streamlined list comprehension with default error message

        Arguments:
            results: The results to collect failures from.

        Returns:
            A list of error messages.

        """
        return [
            result.error or "Unknown error" for result in results if result.is_failure
        ]

    # =========================================================================
    # PARALLEL OPERATIONS - LOW USAGE (1-2 occurrences)
    # OPTIMIZATION: These are advanced patterns with minimal usage
    # =========================================================================

    @staticmethod
    def parallel_map[TParallel, UParallel](
        items: list[TParallel],
        mapper: Callable[[TParallel], FlextResult[UParallel]],
        *,
        fail_fast: bool = True,
    ) -> FlextResult[list[UParallel]]:
        """Map function over items with parallel semantics.

        USAGE: 1+ occurrences - Parallel processing pattern
        OPTIMIZATION: Delegated to appropriate sequence/accumulate methods

        Arguments:
            items: The items to map.
            mapper: The function to map.
            fail_fast: Whether to fail fast.

        Returns:
            A FlextResult containing the mapped results.

        """
        if fail_fast:
            return FlextResultCollections._sequence_typed([
                mapper(item) for item in items
            ])
        return FlextResultCollections._accumulate_errors_typed([
            mapper(item) for item in items
        ])

    @staticmethod
    def _sequence_typed[U](results: list[FlextResult[U]]) -> FlextResult[list[U]]:
        """Type-safe sequence for parallel operations.

        USAGE: Internal - Type-safe sequencing
        OPTIMIZATION: Streamlined error handling with early termination

        Arguments:
            results: The results to sequence.

        Returns:
            A FlextResult containing the sequenced results.

        """
        from flext_core.result import FlextResult  # noqa: PLC0415

        successes = []
        for result in results:
            if result.is_failure:
                error_msg = result.error or "Sequence operation failed"
                return FlextResult[list[U]].fail(error_msg)
            successes.append(result.unwrap())
        return FlextResult[list[U]].ok(successes)

    @staticmethod
    def _accumulate_errors_typed[U](
        results: list[FlextResult[U]],
    ) -> FlextResult[list[U]]:
        """Type-safe error accumulation for parallel operations.

        USAGE: Internal - Type-safe error accumulation
        OPTIMIZATION: Streamlined error collection with metadata

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
                error_msg = result.error or "Operation failed"
                errors.append(error_msg)
            else:
                successes.append(result.unwrap())

        if errors:
            return FlextResult[list[U]].fail("; ".join(errors))
        return FlextResult[list[U]].ok(successes)

    # =========================================================================
    # VALIDATION OPERATIONS - LOW USAGE (1-3 occurrences)
    # =========================================================================

    @staticmethod
    def validate_all[TValidate](
        value: TValidate,
        *validators: Callable[[TValidate], FlextResult[None]],
    ) -> FlextResult[TValidate]:
        """Run all validators on a value, accumulating errors.

        USAGE: 2+ occurrences - Multi-validator pattern
        OPTIMIZATION: Streamlined validation with early termination

        Arguments:
            value: The value to validate.
            validators: The validators to run.

        Returns:
            A FlextResult containing the validated value.

        """
        from flext_core.result import FlextResult  # noqa: PLC0415

        for validator in validators:
            try:
                result = validator(value)
                if result.is_failure:
                    return FlextResult[TValidate].fail(
                        result.error or "Validation failed",
                        error_code=result.error_code,
                        error_data=result.error_data,
                    )
            except Exception as e:
                return FlextResult[TValidate].fail(f"Validator execution failed: {e}")

        return FlextResult[TValidate].ok(value)
