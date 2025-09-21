"""Advanced functional programming utilities for FlextResult.

This module provides advanced functional programming patterns and combinators
for FlextResult, extracted from the main FlextResult class to reduce complexity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from flext_core.typings import T1, T2, T3, TItem, TResult, U, V

if TYPE_CHECKING:
    from flext_core.result import FlextResult


class FlextResultFunctional:
    """Advanced functional programming utilities for FlextResult.

    OPTIMIZATION ANALYSIS:
    - Kleisli composition: ZERO USAGE - Over-engineered pattern
    - Applicative lifting: ZERO USAGE - Over-engineered pattern
    - Traverse operation: LOW USAGE (2+ occurrences) - Delegated to collections

    Provides monadic composition, applicative lifting, and traversal patterns
    extracted from the main FlextResult class.

    OPTIMIZATION: Consolidated duplicate implementations, removed unused patterns
    """

    # =========================================================================
    # ADVANCED MONADIC PATTERNS - ZERO USAGE (over-engineered)
    # OPTIMIZATION: These patterns are rarely used but kept for completeness
    # =========================================================================

    @staticmethod
    def kleisli_compose[T_co, U, V](
        f: Callable[[T_co], FlextResult[U]],
        g: Callable[[U], FlextResult[V]],
    ) -> Callable[[T_co], FlextResult[V]]:
        """Kleisli composition (fish operator >>=) - ADVANCED MONADIC PATTERN.

        USAGE: 0+ occurrences - OVER-ENGINEERED functional pattern
        OPTIMIZATION: Streamlined composition using flat_map chaining

        Arguments:
            f: The first function to compose.
            g: The second function to compose.

        Returns:
            A FlextResult containing the composed results.

        """
        # Import here to avoid circular import
        from flext_core.result import FlextResult  # noqa: PLC0415

        def composed(value: T_co) -> FlextResult[V]:
            return FlextResult[T_co].ok(value).flat_map(f).flat_map(g)

        return composed

    @staticmethod
    def applicative_lift2[T1, T2, TResult](
        func: Callable[[T1, T2], TResult],
        result1: FlextResult[T1],
        result2: FlextResult[T2],
    ) -> FlextResult[TResult]:
        """Lift binary function to applicative context - ADVANCED APPLICATIVE PATTERN.

        USAGE: 0+ occurrences - OVER-ENGINEERED functional pattern
        OPTIMIZATION: Streamlined error checking with direct function application

        Arguments:
            func: The function to lift.
            result1: The first result to lift.
            result2: The second result to lift.

        Returns:
            A FlextResult containing the lifted results.

        """
        # Import here to avoid circular import
        from flext_core.result import FlextResult  # noqa: PLC0415

        if result1.is_failure:
            return FlextResult[TResult].fail(result1.error or "First argument failed")
        if result2.is_failure:
            return FlextResult[TResult].fail(result2.error or "Second argument failed")

        return FlextResult[TResult].ok(func(result1.unwrap(), result2.unwrap()))

    @staticmethod
    def applicative_lift3[T1, T2, T3, TResult](
        func: Callable[[T1, T2, T3], TResult],
        result1: FlextResult[T1],
        result2: FlextResult[T2],
        result3: FlextResult[T3],
    ) -> FlextResult[TResult]:
        """Lift ternary function to applicative context - ADVANCED APPLICATIVE PATTERN.

        USAGE: 0+ occurrences - OVER-ENGINEERED functional pattern
        OPTIMIZATION: Delegated to applicative_lift2 for consistency

        Arguments:
            func: The function to lift.
            result1: The first result to lift.
            result2: The second result to lift.
            result3: The third result to lift.

        Returns:
            A FlextResult containing the lifted results.

        """

        def lift_func(t1_t2: tuple[T1, T2] | None, t3: T3) -> TResult:
            if t1_t2 is None:
                # This should not happen in practice due to applicative lifting
                msg = "Unexpected None value in applicative lift"
                raise ValueError(msg)
            return func(t1_t2[0], t1_t2[1], t3)

        # Use zip_with instead of removed @ operator
        return FlextResultFunctional.applicative_lift2(
            lift_func,
            result1.zip_with(result2, lambda t1, t2: (t1, t2)),
            result3,
        )

    # =========================================================================
    # TRAVERSAL OPERATIONS - LOW USAGE (2+ occurrences)
    # OPTIMIZATION: Delegated to FlextResultCollections to avoid duplication
    # =========================================================================

    @staticmethod
    def traverse[TItem, TResult](
        items: list[TItem],
        func: Callable[[TItem], FlextResult[TResult]],
    ) -> FlextResult[list[TResult]]:
        """Traverse operation from Category Theory - ADVANCED FUNCTIONAL PATTERN.

        USAGE: 2+ occurrences - ADVANCED functional pattern
        OPTIMIZATION: Delegated to FlextResultCollections to avoid duplication

        Arguments:
            items: The items to traverse.
            func: The function to traverse.

        Returns:
            A FlextResult containing the traversed results.

        """
        from flext_core.result_collections import (  # noqa: PLC0415
            FlextResultCollections,  # noqa: PLC0415, RUF100
        )

        return FlextResultCollections.traverse(items, func)
