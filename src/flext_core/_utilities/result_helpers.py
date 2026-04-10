"""Result and boolean composition helpers for dispatcher workflows.

Provides utility functions for working with Result types, boolean logic,
string validation, and exception handling. All functions use p.Result for consistent
error handling where applicable.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable

from flext_core import T, p, r, t


class FlextUtilitiesResultHelpers:
    """Result composition and boolean logic helpers.

    Provides utilities for working with Result types, boolean composition,
    string validation, and exception handling with p.Result-based error handling.
    Pure namespace class with only @staticmethod members.
    """

    @staticmethod
    def any_(*values: t.RecursiveContainer) -> bool:
        """Check if any value is truthy.

        Args:
            *values: Variable number of values to check.

        Returns:
            True if any value is truthy, False otherwise.

        """
        return any(bool(v) for v in values)

    @staticmethod
    def try_(
        func: Callable[[], T],
        *,
        default: T | None = None,
        catch: type[Exception] | tuple[type[Exception], ...] = Exception,
    ) -> r[T]:
        """Call function and catch exceptions, returning result or default.

        Args:
            func: Callable to execute.
            default: Default value if function fails.
            catch: Exception type(s) to catch (default: all Exceptions).

        Returns:
            p.Result[T] with function result, default, or failure message.

        Raises:
            Exception: If raised exception is not in catch types.

        """
        func_result = r[T].create_from_callable(func)
        if func_result.is_success:
            return r[T].ok(func_result.value)
        exc = getattr(func_result, "_exception", None)
        if exc is not None and not isinstance(exc, catch):
            raise exc
        if default is not None:
            return r[T].ok(default)
        return r[T].fail(func_result.error or "Callable failed")

    @staticmethod
    def expect_success[TValue](
        result: p.ResultLike[TValue],
        *,
        message: str | None = None,
    ) -> TValue:
        """Return the success payload or raise AssertionError.

        Keeps call sites terse while preserving strong generic typing.
        """
        if result.is_failure:
            if message is None:
                error_message = str(result.error)
            else:
                error_message = f"{message}: {result.error}"
            raise AssertionError(error_message)
        return result.value


__all__ = ["FlextUtilitiesResultHelpers"]
