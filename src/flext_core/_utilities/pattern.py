"""FlextUtilitiesPattern - Pattern matching utilities for FLEXT.

Provides flexible pattern matching functionality with predicates and callbacks.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flext_core.typings import T, U


@runtime_checkable
class _Predicate(Protocol[T]):
    """Protocol for callable predicates that accept a value and return bool."""

    def __call__(self, value: T) -> bool: ...


@runtime_checkable
class _Handler(Protocol[T, U]):
    """Protocol for callable handlers that transform a value."""

    def __call__(self, value: T) -> U: ...


class FlextUtilitiesPattern:
    """Pattern matching utilities for flexible conditional dispatch.

    Provides a match() function that tests predicates in sequence and executes
    the corresponding callback when a predicate returns True.

    Example:
        result = u.Pattern.match(
            value,
            (lambda x: x > 10, lambda x: f"{x} is large"),
            (lambda x: x > 0, lambda x: f"{x} is positive"),
            default=lambda x: "zero or negative",
        )

    """

    @staticmethod
    def match[T, U](
        value: T,
        *patterns: tuple[_Predicate[T], _Handler[T, U]],
        default: _Handler[T, U] | None = None,
    ) -> U:
        """Match value against patterns and execute corresponding handler.

        Tests each predicate in sequence. When a predicate returns True,
        executes and returns the result of its corresponding handler function.
        If no pattern matches and default is provided, executes default handler.

        Args:
            value: The value to match against patterns
            *patterns: Variable number of (predicate, handler) tuples where:
                - predicate: Callable that returns True/False
                - handler: Callable that executes when predicate matches
            default: Optional handler to execute if no pattern matches

        Returns:
            Result of the matching handler or default handler

        Raises:
            ValueError: If no pattern matches and no default is provided

        """
        for predicate, handler in patterns:
            result: bool = predicate(value)
            if result:
                return handler(value)

        if default is not None:
            return default(value)

        msg = f"No pattern matched for value: {value!r}"
        raise ValueError(msg)
