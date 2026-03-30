"""Reliability helpers aligned with dispatcher-centric CQRS flows.

Utilities extracted from ``flext_core.utilities`` to keep retry and
timeout behaviors modular. All helpers return ``p.Result``-compatible
implementations so dispatcher handlers can compose reliability policies
without raising exceptions or leaking thread-local state.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import ClassVar

from flext_core import (
    FlextModelFoundation,
    FlextProtocolsLogging,
    FlextRuntime,
    c,
    r,
)


class FlextUtilitiesReliability:
    """Reliability patterns for resilient, dispatcher-safe operations."""

    _V: ClassVar[type[FlextModelFoundation.Validators]] = (
        FlextModelFoundation.Validators
    )

    @property
    def logger(self) -> FlextProtocolsLogging.Logger:
        """Get structlog logger via FlextRuntime (infrastructure-level, no FlextLogger)."""
        return FlextRuntime.get_logger(__name__)

    @staticmethod
    def _calculate_retry_delay(
        attempt: int,
        delay_seconds: float,
        backoff_multiplier: float,
    ) -> float:
        """Calculate delay for retry attempt with exponential backoff."""
        current_delay = delay_seconds * backoff_multiplier**attempt
        return min(current_delay, c.DEFAULT_MAX_DELAY_SECONDS)

    @staticmethod
    def flow_result[T](result: r[T], *funcs: Callable[[T], r[T]]) -> r[T]:
        """Chain multiple operations on p.Result.

        Applies each function in sequence, short-circuiting on failure.
        Railway-oriented programming pattern for composing result-returning operations.

        Args:
            result: Initial p.Result to chain
            *funcs: Functions that take a value and return p.Result[T]

        Returns:
            Final p.Result after all operations or first failure

        Example:
            result = u.flow_result(
                r[T].ok(user),
                validate_user,
                enrich_profile,
                save_to_db,
            )

        """
        current: r[T] = result
        for func in funcs:
            if current.is_failure:
                return current
            current = func(current.value)
        return current

    @staticmethod
    def retry[TResult](
        operation: Callable[[], r[TResult]],
        max_attempts: int | None = None,
        delay: float | None = None,
        delay_seconds: float | None = None,
        backoff_multiplier: float | None = None,
        retry_on: tuple[type[Exception], ...] | None = None,
    ) -> r[TResult]:
        """Execute an operation with retry logic using railway patterns.

        Args:
            operation: Operation to execute. Must return RuntimeResult.
            max_attempts: Maximum number of attempts
            delay: Delay between retries in seconds (alias for delay_seconds)
            delay_seconds: Delay between retries in seconds
            backoff_multiplier: Multiplier for exponential backoff
            retry_on: Tuple of exception types to retry on. If None, retries on all exceptions.

        Returns:
            p.Result[TResult]: Result of operation with retry logic applied

        Fast fail: explicit default values instead of 'or' fallback.

        """
        delay_value: float | None = delay if delay is not None else delay_seconds
        max_attempts_value: int = (
            max_attempts if max_attempts is not None else c.MAX_RETRY_ATTEMPTS
        )
        delay_seconds_value: float = (
            delay_value if delay_value is not None else c.DEFAULT_RETRY_DELAY_SECONDS
        )
        backoff_multiplier_value: float = (
            backoff_multiplier
            if backoff_multiplier is not None
            else c.DEFAULT_BACKOFF_MULTIPLIER
        )
        if max_attempts_value < c.DEFAULT_RETRY_DELAY_SECONDS:
            return r[TResult].fail(
                f"Max attempts must be at least {c.DEFAULT_RETRY_DELAY_SECONDS}",
            )
        last_error: str | None = None
        for attempt in range(max_attempts_value):
            try:
                result = operation()
                if result.is_success:
                    return result
                last_error = result.error or "Unknown error"
                if attempt < max_attempts_value - 1:
                    current_delay = FlextUtilitiesReliability._calculate_retry_delay(
                        attempt,
                        delay_seconds_value,
                        backoff_multiplier_value,
                    )
                    if current_delay > 0:
                        time.sleep(current_delay)
            except (
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
                KeyError,
                OSError,
            ) as e:
                if retry_on is not None and (not isinstance(e, retry_on)):
                    raise
                last_error = str(e)
                if attempt < max_attempts_value - 1:
                    current_delay = FlextUtilitiesReliability._calculate_retry_delay(
                        attempt,
                        delay_seconds_value,
                        backoff_multiplier_value,
                    )
                    if current_delay > 0:
                        time.sleep(current_delay)
        return r[TResult].fail(
            f"Operation failed after {max_attempts_value} attempts: {last_error}",
        )


__all__ = ["FlextUtilitiesReliability"]
