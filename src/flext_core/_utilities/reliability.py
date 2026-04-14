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

from pydantic import Field

from flext_core import FlextUtilitiesArgs, c, m, p, r, t


class FlextUtilitiesReliability:
    """Reliability patterns for resilient, dispatcher-safe operations."""

    class RetryOptions(m.FlexibleInternalModel):
        """Configuration options for retry logic."""

        max_attempts: int | None = Field(
            default=None,
            description="Maximum number of retry attempts",
        )
        delay: float | None = Field(
            default=None,
            description="Initial delay between retries in seconds",
        )
        delay_seconds: float | None = Field(
            default=None,
            description="Alias for delay in seconds",
        )
        backoff_multiplier: float | None = Field(
            default=None,
            description="Multiplier for exponential backoff",
        )
        retry_on: tuple[type[Exception], ...] | None = Field(
            default=None,
            description="Exception types to retry on",
        )

    @staticmethod
    def flow_result[T](
        result: p.Result[T],
        *funcs: Callable[[T], p.Result[T]],
    ) -> p.Result[T]:
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
        current: p.Result[T] = result
        for func in funcs:
            if current.failure:
                return current
            current = func(current.value)
        return current

    _RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
        AttributeError,
        TypeError,
        ValueError,
        RuntimeError,
        KeyError,
        OSError,
    )

    @staticmethod
    def retry[TResult](
        operation: Callable[[], p.Result[TResult]],
        options: FlextUtilitiesReliability.RetryOptions | None = None,
        **kwargs: t.ValueOrModel,
    ) -> p.Result[TResult]:
        """Execute an operation with retry logic using railway patterns.

        Args:
            operation: Operation to execute. Must return RuntimeResult.
            options: Optional retry configuration using RetryOptions model.
            **kwargs: Inline retry configuration parsed into RetryOptions automatically.

        Returns:
            p.Result[TResult]: Result of operation with retry logic applied

        Fast fail: explicit default values instead of 'or' fallback.

        """
        opts_res = FlextUtilitiesArgs.resolve_options(
            options,
            kwargs,
            FlextUtilitiesReliability.RetryOptions,
        )
        if opts_res.failure:
            return r[TResult].fail(opts_res.error)
        opts = opts_res.value
        delay_value = opts.delay if opts.delay is not None else opts.delay_seconds
        max_att = (
            opts.max_attempts if opts.max_attempts is not None else c.MAX_RETRY_ATTEMPTS
        )
        delay_sec = (
            delay_value if delay_value is not None else c.DEFAULT_RETRY_DELAY_SECONDS
        )
        backoff = (
            opts.backoff_multiplier
            if opts.backoff_multiplier is not None
            else c.DEFAULT_BACKOFF_MULTIPLIER
        )
        if max_att < c.DEFAULT_RETRY_DELAY_SECONDS:
            return r[TResult].fail(
                f"Max attempts must be at least {c.DEFAULT_RETRY_DELAY_SECONDS}",
            )
        last_error: str | None = None
        for attempt in range(max_att):
            try:
                result = operation()
                if result.success:
                    return result
                last_error = result.error or "Unknown error"
            except FlextUtilitiesReliability._RETRYABLE_EXCEPTIONS as e:
                if opts.retry_on is not None and (not isinstance(e, opts.retry_on)):
                    raise
                last_error = str(e)
            if attempt < max_att - 1:
                current_delay = min(
                    delay_sec * backoff**attempt,
                    c.DEFAULT_MAX_DELAY_SECONDS,
                )
                if current_delay > 0:
                    time.sleep(current_delay)
        return r[TResult].fail(
            f"Operation failed after {max_att} attempts: {last_error}",
        )


__all__: list[str] = ["FlextUtilitiesReliability"]
