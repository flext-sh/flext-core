"""Reliability helpers aligned with dispatcher-centric CQRS flows.

Utilities extracted from ``flext_core.utilities`` to keep retry and
timeout behaviors modular. All helpers return ``p.Result``-compatible
implementations so dispatcher handlers can compose reliability policies
without raising exceptions or leaking thread-local state.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable

from pydantic import Field

from flext_core import c, r, t
from flext_core._models.base import FlextModelsBase
from flext_core._utilities.args import FlextUtilitiesArgs


class FlextUtilitiesReliability:
    """Reliability patterns for resilient, dispatcher-safe operations."""

    class RetryOptions(FlextModelsBase.FlexibleInternalModel):
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

    _RETRIABLE_ERROR_PATTERNS: tuple[str, ...] = (
        "Temporary failure",
        "timeout",
        "transient",
        "temporarily unavailable",
        "try again",
    )

    @staticmethod
    def apply_jitter(base_delay: float, jitter_factor: float) -> float:
        """Apply symmetric jitter variance to a delay value."""
        if base_delay <= 0.0 or jitter_factor <= 0.0:
            return base_delay
        secure_random = secrets.SystemRandom()
        variance = (2.0 * secure_random.random() - 1.0) * jitter_factor
        jittered = base_delay * (1.0 + variance)
        return max(0.0, jittered)

    @staticmethod
    def is_retriable_error_message(error: str | None) -> bool:
        """Return whether an error message matches transient failure heuristics."""
        if error is None:
            return False
        return any(
            pattern.lower() in error.lower()
            for pattern in FlextUtilitiesReliability._RETRIABLE_ERROR_PATTERNS
        )

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
            if current.failure:
                return current
            current = func(current.value)
        return current

    @staticmethod
    def _resolve_retry_config(
        max_attempts: int | None,
        delay: float | None,
        delay_seconds: float | None,
        backoff_multiplier: float | None,
    ) -> tuple[int, float, float]:
        """Resolve retry configuration with defaults."""
        delay_value = delay if delay is not None else delay_seconds
        return (
            max_attempts if max_attempts is not None else c.MAX_RETRY_ATTEMPTS,
            delay_value if delay_value is not None else c.DEFAULT_RETRY_DELAY_SECONDS,
            backoff_multiplier
            if backoff_multiplier is not None
            else c.DEFAULT_BACKOFF_MULTIPLIER,
        )

    @staticmethod
    def _should_retry(
        exc: Exception,
        retry_on: tuple[type[Exception], ...] | None,
    ) -> bool:
        """Check if the exception is retryable."""
        return retry_on is None or isinstance(exc, retry_on)

    @staticmethod
    def _sleep_between_retries(
        attempt: int,
        max_attempts: int,
        delay_seconds: float,
        backoff_multiplier: float,
    ) -> None:
        """Sleep between retry attempts with exponential backoff."""
        if attempt < max_attempts - 1:
            current_delay = FlextUtilitiesReliability._calculate_retry_delay(
                attempt,
                delay_seconds,
                backoff_multiplier,
            )
            if current_delay > 0:
                time.sleep(current_delay)

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
        operation: Callable[[], r[TResult]],
        options: FlextUtilitiesReliability.RetryOptions | None = None,
        **kwargs: t.ValueOrModel,
    ) -> r[TResult]:
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
        delay_val = opts.delay if opts.delay is not None else opts.delay_seconds

        max_att, delay_sec, backoff = FlextUtilitiesReliability._resolve_retry_config(
            opts.max_attempts,
            delay_val,
            delay_val,
            opts.backoff_multiplier,
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
                if not FlextUtilitiesReliability._should_retry(e, opts.retry_on):
                    raise
                last_error = str(e)
            FlextUtilitiesReliability._sleep_between_retries(
                attempt,
                max_att,
                delay_sec,
                backoff,
            )
        return r[TResult].fail(
            f"Operation failed after {max_att} attempts: {last_error}",
        )


__all__ = ["FlextUtilitiesReliability"]
