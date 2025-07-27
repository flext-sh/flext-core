"""FLEXT Railway Base - Railway-oriented programming patterns.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Base railway programming patterns without external dependencies.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from flext_core._result_base import _BaseResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core.types import T


# =============================================================================
# BASE RAILWAY OPERATIONS - Core railway programming
# =============================================================================


class _BaseRailway:
    """Base railway programming operations without external dependencies."""

    @staticmethod
    def bind(
        result: _BaseResult[T],
        func: Callable[[T], _BaseResult[object]],
    ) -> _BaseResult[object]:
        """Railway bind operation (>>=).

        Args:
            result: Input result
            func: Function to bind

        Returns:
            Bound result

        """
        if not result.is_success:
            return _BaseResult.fail(
                result.error or "Previous operation failed",
                result.error_code,
                result.error_data,
            )

        try:
            # In success case, data should be available
            data = result.data
            if data is None:
                return _BaseResult.fail("Cannot bind with None data")
            return func(data)
        except (TypeError, ValueError, AttributeError) as e:
            return _BaseResult.fail(f"Bind operation failed: {e}")

    @staticmethod
    def compose_functions(
        *functions: Callable[[object], _BaseResult[object]],
    ) -> Callable[[object], _BaseResult[object]]:
        """Compose multiple railway functions (left to right).

        Args:
            *functions: Functions to compose

        Returns:
            Composed function

        """

        def composed(value: object) -> _BaseResult[object]:
            result = _BaseResult.ok(value)
            for func in functions:
                if not result.is_success:
                    break
                result = _BaseRailway.bind(result, func)
            return result

        return composed

    @staticmethod
    def switch(
        condition: Callable[[T], bool],
        success_func: Callable[[T], _BaseResult[object]],
        failure_func: Callable[[T], _BaseResult[object]],
    ) -> Callable[[T], _BaseResult[object]]:
        """Railway switch based on condition.

        Args:
            condition: Boolean condition
            success_func: Function if condition is True
            failure_func: Function if condition is False

        Returns:
            Switch function

        """

        def switch_func(value: T) -> _BaseResult[object]:
            try:
                if condition(value):
                    return success_func(value)
                return failure_func(value)
            except (TypeError, ValueError, AttributeError) as e:
                return _BaseResult.fail(f"Switch evaluation failed: {e}")

        return switch_func

    @staticmethod
    def tee(
        main_func: Callable[[T], _BaseResult[object]],
        side_func: Callable[[T], _BaseResult[object]],
    ) -> Callable[[T], _BaseResult[object]]:
        """Railway tee - execute both functions, return main result.

        Args:
            main_func: Main function
            side_func: Side function (result ignored)

        Returns:
            Tee function

        """

        def tee_func(value: T) -> _BaseResult[object]:
            # Execute side function but ignore result
            with contextlib.suppress(TypeError, ValueError, AttributeError):
                side_func(value)

            # Return main function result
            return main_func(value)

        return tee_func

    @staticmethod
    def dead_end(
        func: Callable[[T], None],
    ) -> Callable[[T], _BaseResult[T]]:
        """Convert void function to railway function.

        Args:
            func: Void function

        Returns:
            Railway function

        """

        def railway_func(value: T) -> _BaseResult[T]:
            try:
                func(value)
                return _BaseResult.ok(value)
            except (TypeError, ValueError, AttributeError) as e:
                return _BaseResult.fail(f"Dead end function failed: {e}")

        return railway_func

    @staticmethod
    def plus(
        func1: Callable[[T], _BaseResult[object]],
        func2: Callable[[T], _BaseResult[object]],
    ) -> Callable[[T], _BaseResult[list[object]]]:
        """Railway plus - execute both functions and collect results.

        Args:
            func1: First function
            func2: Second function

        Returns:
            Plus function collecting both results

        """

        def plus_func(value: T) -> _BaseResult[list[object]]:
            result1 = func1(value)
            result2 = func2(value)

            if result1.is_success and result2.is_success:
                return _BaseResult.ok([result1.data, result2.data])

            # Collect errors
            errors = []
            if not result1.is_success:
                errors.append(result1.error or "Function 1 failed")
            if not result2.is_success:
                errors.append(result2.error or "Function 2 failed")

            return _BaseResult.fail(f"Plus operation failed: {'; '.join(errors)}")

        return plus_func


# =============================================================================
# BASE RAILWAY UTILITIES - Helper functions
# =============================================================================


class _BaseRailwayUtils:
    """Base railway utility functions."""

    @staticmethod
    def lift(
        func: Callable[[T], object],
    ) -> Callable[[T], _BaseResult[object]]:
        """Lift regular function to railway function.

        Args:
            func: Regular function

        Returns:
            Railway function

        """

        def lifted_func(value: T) -> _BaseResult[object]:
            try:
                result = func(value)
                return _BaseResult.ok(result)
            except (TypeError, ValueError, AttributeError) as e:
                return _BaseResult.fail(f"Lifted function failed: {e}")

        return lifted_func

    @staticmethod
    def ignore() -> Callable[[object], _BaseResult[None]]:
        """Railway function that ignores input and returns success.

        Returns:
            Ignore function

        """

        def ignore_func(_value: object) -> _BaseResult[None]:
            return _BaseResult.ok(None)

        return ignore_func

    @staticmethod
    def pass_through() -> Callable[[T], _BaseResult[T]]:
        """Railway function that passes value through unchanged.

        Returns:
            Pass-through function

        """

        def pass_func(value: T) -> _BaseResult[T]:
            return _BaseResult.ok(value)

        return pass_func


# =============================================================================
# EXPORTS - Base railway functionality only
# =============================================================================

__all__ = ["_BaseRailway", "_BaseRailwayUtils"]
