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
from typing import Annotated

from pydantic import Field

from flext_core import FlextModelsBase as m, FlextUtilitiesArgs, c, p, r, t


class FlextUtilitiesReliability:
    """Reliability patterns for resilient, dispatcher-safe operations."""

    class RetryOptions(m.FlexibleInternalModel):
        """Configuration options for retry logic."""

        max_attempts: Annotated[
            int | None,
            Field(
                description="Maximum number of retry attempts",
            ),
        ] = None
        delay: Annotated[
            float | None,
            Field(
                description="Initial delay between retries in seconds",
            ),
        ] = None
        delay_seconds: Annotated[
            float | None,
            Field(
                description="Alias for delay in seconds",
            ),
        ] = None
        backoff_multiplier: Annotated[
            float | None,
            Field(
                description="Multiplier for exponential backoff",
            ),
        ] = None
        retry_on: Annotated[
            tuple[type[Exception], ...] | None,
            Field(
                description="Exception types to retry on",
            ),
        ] = None

    _RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
        AttributeError,
        TypeError,
        ValueError,
        RuntimeError,
        KeyError,
        OSError,
    )

    @staticmethod
    def try_[TResult](
        operation: Callable[[], TResult],
        catch: type[Exception] | tuple[type[Exception], ...] | None = None,
    ) -> p.Result[TResult]:
        """Execute a callable and capture configured exceptions as failed results."""
        if catch is None:
            handled = FlextUtilitiesReliability._RETRYABLE_EXCEPTIONS
        elif isinstance(catch, type):
            handled = (catch,)
        else:
            handled = catch
        try:
            return r[TResult].ok(operation())
        except handled as exc:
            return r[TResult].fail_op("execute guarded operation", exc)

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
