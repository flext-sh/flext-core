"""Result and boolean composition helpers for dispatcher workflows.

Provides utility functions for working with Result types, boolean logic,
string validation, and exception handling. All functions use r for consistent
error handling where applicable.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from flext_core import FlextUtilitiesGuards, T, p, r, t


class FlextUtilitiesResultHelpers:
    """Result composition and boolean logic helpers.

    Provides utilities for working with Result types, boolean composition,
    string validation, and exception handling with r-based error handling.
    Pure namespace class with only @staticmethod members.
    """

    @staticmethod
    def any_(*values: t.NormalizedValue) -> bool:
        """Check if any value is truthy.

        Args:
            *values: Variable number of values to check.

        Returns:
            True if any value is truthy, False otherwise.

        """
        return any(bool(v) for v in values)

    @staticmethod
    def empty(items: t.NormalizedValue | None) -> bool:
        """Check if items are empty or None.

        Args:
            items: Items to check (can be None, non-container, or container).

        Returns:
            True if items is None, not a container, or empty container.

        """
        if items is None:
            return True
        if not FlextUtilitiesGuards.is_container(items):
            return True
        return not bool(items)

    @staticmethod
    def ends(value: str, suffix: str, *suffixes: str) -> bool:
        """Check if string ends with any of the given suffixes.

        Args:
            value: String to check.
            suffix: First suffix to check.
            *suffixes: Additional suffixes to check.

        Returns:
            True if string ends with any suffix.

        """
        return any(value.endswith(s) for s in (suffix, *suffixes))

    @staticmethod
    def err(result: p.Result[T], *, default: str = "Unknown error") -> str:
        """Extract error message from result.

        Args:
            result: Result to extract error from.
            default: Default message if result has no error.

        Returns:
            Error message string or default if result is success.

        """
        if result.is_failure and result.error:
            return str(result.error)
        return default

    @staticmethod
    def not_(value: t.NormalizedValue) -> bool:
        """Check if value is falsy.

        Args:
            value: Value to check.

        Returns:
            True if value is falsy, False if truthy.

        """
        return not bool(value)

    @staticmethod
    def or_(*values: T | None, default: T | None = None) -> r[T]:
        """Return first non-None value or default.

        Args:
            *values: Variable number of potential values.
            default: Default value if all inputs are None.

        Returns:
            r[T] with first non-None value, default, or failure if none found.

        """
        for value in values:
            if value is not None:
                return r[T].ok(value)
        if default is not None:
            return r[T].ok(default)
        return r[T].fail("No non-None value found")

    @staticmethod
    def starts(value: str, prefix: str, *prefixes: str) -> bool:
        """Check if string starts with any of the given prefixes.

        Args:
            value: String to check.
            prefix: First prefix to check.
            *prefixes: Additional prefixes to check.

        Returns:
            True if string starts with any prefix.

        """
        return any(value.startswith(p) for p in (prefix, *prefixes))

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
            r[T] with function result, default, or failure message.

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
    def val(result: p.Result[T], *, default: T | None = None) -> r[T]:
        """Extract value from result, using default on failure.

        Args:
            result: Result to extract value from.
            default: Default value if result is failure.

        Returns:
            r[T] with result value, default, or failure message.

        """
        if result.is_success:
            return r[T].ok(result.value)
        if default is not None:
            return r[T].ok(default)
        return r[T].fail(result.error or "Failed to extract result value")

    @staticmethod
    def vals(
        items: Mapping[str, T] | r[Mapping[str, T]],
        *,
        default: Sequence[T] | None = None,
    ) -> r[Sequence[T]]:
        """Extract list of values from mapping or result mapping.

        Args:
            items: Mapping or result containing mapping.
            default: Default list if no values available.

        Returns:
            r[Sequence[T]] with list of values, default, or failure.

        """
        if isinstance(items, r):
            if items.is_failure:
                if default is not None:
                    return r[Sequence[T]].ok(default)
                return r[Sequence[T]].fail(
                    items.error or "Failed to extract values from result",
                )
            value_mapping = items.value
            return r[Sequence[T]].ok(list(value_mapping.values()))
        if items:
            return r[Sequence[T]].ok(list(items.values()))
        if default is not None:
            return r[Sequence[T]].ok(default)
        return r[Sequence[T]].fail("No values available")

    @staticmethod
    def vals_sequence(results: Sequence[p.Result[T]]) -> Sequence[T]:
        """Extract values from sequence of results, filtering successes only.

        Args:
            results: Sequence of Result objects.

        Returns:
            List of successful result values.

        """
        return [result.value for result in results if result.is_success]

    @staticmethod
    def ensure_result(
        value: t.ValueOrModel,
    ) -> r[t.ValueOrModel]:
        """Wrap value in result type if not already a Result.

        Convenience method for ensuring a value is wrapped in r[T].ok().

        Args:
            value: Value to wrap.

        Returns:
            r[ValueOrModel] with value wrapped as success.

        """
        return r[t.ValueOrModel].ok(value)


__all__ = ["FlextUtilitiesResultHelpers"]
