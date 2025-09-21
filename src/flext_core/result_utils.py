"""Utilities for FlextResult.

This module provides utilities for FlextResult,
extracted from the main FlextResult class to reduce complexity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flext_core.result import FlextResult


class FlextResultUtils:
    """Utility class for FlextResult static operations.

    OPTIMIZATION ANALYSIS:
    - Core utilities (chain_results, combine, all_success): HIGH USAGE (5-15 occurrences each)
    - Collection utilities (collect_successes, collect_failures): MEDIUM USAGE (2-5 occurrences each)
    - Batch processing: LOW USAGE (1-3 occurrences each)
    - Safe execution: MEDIUM USAGE (3-5 occurrences each)

    This class contains static/class methods extracted from FlextResult
    to improve maintainability while preserving API compatibility.
    """

    # =========================================================================
    # CORE UTILITY METHODS - HIGH USAGE (5-15 occurrences each)
    # =========================================================================

    @staticmethod
    def chain_results[TChain](
        results: list[FlextResult[TChain]],
    ) -> FlextResult[list[TChain]]:
        """Chain multiple results into a single result containing a list.

        USAGE: 5+ occurrences - Result chaining pattern
        OPTIMIZATION: Streamlined iteration with early termination

        Arguments:
            results: The results to chain.

        Returns:
            A FlextResult containing the chained results.

        """
        from flext_core.result import FlextResult

        successful_results = []
        for result in results:
            if not result.is_success:
                return FlextResult[list[TChain]].fail(
                    f"Chain failed at result: {result.error}"
                )
            successful_results.append(result.value)
        return FlextResult[list[TChain]].ok(successful_results)

    @staticmethod
    def combine[TCombine](
        *results: FlextResult[TCombine],
    ) -> FlextResult[list[TCombine]]:
        """Combine multiple results into a single result.

        USAGE: 8+ occurrences - Result combination pattern
        OPTIMIZATION: Streamlined iteration with early termination, filters out None values

        Arguments:
            results: The results to combine.

        Returns:
            A FlextResult containing the combined results.

        """
        from flext_core.result import FlextResult

        values = []
        for result in results:
            if result.is_failure:
                # result.error is guaranteed to be str when is_failure is True
                error_msg = result.error or "Unknown error"
                return FlextResult[list[TCombine]].fail(error_msg)
            # Only include non-None values to match test expectations
            if result.value is not None:
                values.append(result.value)
        return FlextResult[list[TCombine]].ok(values)

    @staticmethod
    def all_success[TAny](*results: FlextResult[TAny]) -> bool:
        """Check if all results are successful.

        USAGE: 5+ occurrences - Success validation pattern
        OPTIMIZATION: Early return for empty results, streamlined iteration

        Arguments:
            results: The results to check for success.

        Returns:
            A boolean indicating if all results are successful.

        """
        if not results:
            return True
        return all(result.is_success for result in results)

    @staticmethod
    def any_success[TAny](*results: FlextResult[TAny]) -> bool:
        """Check if any result is successful.

        Arguments:
            results: The results to check for success.

        Returns:
            A boolean indicating if any result is successful.

        USAGE: 3+ occurrences - Success checking pattern
        OPTIMIZATION: Early return for empty results, streamlined iteration

        Arguments:
            results: The results to check for success.

        Returns:
            A boolean indicating if any result is successful.

        """
        return any(result.is_success for result in results) if results else False

    @classmethod
    def first_success[TFirst](
        cls, *results: FlextResult[TFirst]
    ) -> FlextResult[TFirst]:
        """Return the first successful result.

        USAGE: 2+ occurrences - First success pattern
        OPTIMIZATION: Streamlined iteration with early return

        Arguments:
            results: The results to get the first successful result from.

        Returns:
            A FlextResult containing the first successful result.

        """
        from flext_core.result import FlextResult

        for result in results:
            if result.is_success:
                return result
        return FlextResult[TFirst].fail("No successful results found")

    @classmethod
    def sequence[TSeq](
        cls, results: list[FlextResult[TSeq]]
    ) -> FlextResult[list[TSeq]]:
        """Sequence a list of results into a result of list.

        USAGE: 8+ occurrences - Result sequencing pattern
        OPTIMIZATION: Streamlined iteration with early termination and error preservation

        Arguments:
            results: The results to sequence.

        Returns:
            A FlextResult containing the sequenced results.

        """
        from flext_core.result import FlextResult

        values = []
        for result in results:
            if result.is_failure:
                return FlextResult[list[TSeq]].fail(
                    result.error or "Sequence failed",
                    error_code=result.error_code,
                    error_data=result.error_data,
                )
            values.append(result.unwrap())
        return FlextResult[list[TSeq]].ok(values)

    @classmethod
    def try_all[TTry](
        cls, *operations: Callable[[], FlextResult[TTry]]
    ) -> FlextResult[TTry]:
        """Try operations in sequence until one succeeds.

        USAGE: 2+ occurrences - Function trying pattern
        OPTIMIZATION: Streamlined exception handling with error collection

        Arguments:
            operations: The operations to try.

        Returns:
            A FlextResult containing the result of the first successful operation.

        """
        from flext_core.result import FlextResult

        errors = []
        for operation in operations:
            try:
                result = operation()
                if result.is_success:
                    return result
                errors.append(result.error)
            except Exception as e:
                errors.append(str(e))
        return FlextResult[TTry].fail(f"All operations failed: {errors}")

    # =========================================================================
    # COLLECTION UTILITY METHODS - MEDIUM USAGE (2-5 occurrences each)
    # =========================================================================

    @classmethod
    def collect_successes[TCollect](
        cls, results: list[FlextResult[TCollect]]
    ) -> list[TCollect]:
        """Collect all successful values from results.

        USAGE: 3+ occurrences - Success collection pattern
        OPTIMIZATION: Delegated to FlextResultCollections for consistency

        Arguments:
            results: The results to collect successes from.

        Returns:
            A list of successful values.

        """
        from flext_core.result_collections import (
            FlextResultCollections,
        )

        return FlextResultCollections.collect_successes(results)

    @classmethod
    def collect_failures[TCollectFail](
        cls, results: list[FlextResult[TCollectFail]]
    ) -> list[str]:
        """Collect all error messages from failed results.

        USAGE: 2+ occurrences - Failure collection pattern
        OPTIMIZATION: Delegated to FlextResultCollections for consistency

        Arguments:
            results: The results to collect failures from.

        Returns:
            A list of error messages.

        """
        from flext_core.result_collections import (
            FlextResultCollections,
        )

        return FlextResultCollections.collect_failures(results)

    @classmethod
    def success_rate[TRate](cls, results: list[FlextResult[TRate]]) -> float:
        """Calculate success rate of results.

        USAGE: 1+ occurrences - Success rate calculation
        OPTIMIZATION: Direct calculation with early return for empty list

        Arguments:
            results: The results to calculate the success rate of.

        Returns:
            A float representing the success rate of the results.

        """
        if not results:
            return 0.0
        successful = sum(1 for result in results if result.is_success)
        return successful / len(results)

    # =========================================================================
    # BATCH PROCESSING METHODS - LOW USAGE (1-3 occurrences each)
    # =========================================================================

    @classmethod
    def batch_process[TBatch, UBatch](
        cls,
        items: list[TBatch],
        processor: Callable[[TBatch], FlextResult[UBatch]],
    ) -> tuple[list[UBatch], list[str]]:
        """Process batch and separate successes from failures.

        USAGE: 2+ occurrences - Batch processing pattern
        OPTIMIZATION: Streamlined processing with delegated collection

        Arguments:
            items: The items to process.
            processor: The processor to use.

        Returns:
            A tuple of lists of successes and failures.

        """
        results = [processor(item) for item in items]
        successes = cls.collect_successes(results)
        failures = cls.collect_failures(results)
        return successes, failures

    # =========================================================================
    # SAFE EXECUTION METHODS - MEDIUM USAGE (3-5 occurrences each)
    # =========================================================================

    @classmethod
    def safe_call[TSafe](
        cls, func: Callable[[], TSafe], error_message: str = "Operation failed"
    ) -> FlextResult[TSafe]:
        """Safely call a function and wrap result.

        USAGE: 3+ occurrences - Safe function execution pattern
        OPTIMIZATION: Streamlined exception handling with custom error messages

        Arguments:
            func: The function to call.
            error_message: The error message to use.

        Returns:
            A FlextResult containing the result of the function call.

        """
        from flext_core.result import FlextResult

        try:
            result = func()
            return FlextResult[TSafe].ok(result)
        except Exception as e:
            return FlextResult[TSafe].fail(f"{error_message}: {e}")


__all__: list[str] = [
    "FlextResultUtils",  # Utility class for static operations
]
