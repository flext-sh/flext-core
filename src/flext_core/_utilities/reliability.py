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
from collections.abc import (
    Callable,
)
from typing import Annotated, no_type_check

from pydantic import Field

from flext_core import (
    FlextConstants as c,
    FlextModelsBase as m,
    FlextProtocols as p,
    FlextResult as r,
    FlextTypes as t,
    FlextUtilitiesArgs,
)


@no_type_check
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
        delay_seconds: Annotated[
            float | None,
            Field(
                description="Initial delay between retries in seconds",
            ),
        ] = None

    _RETRYABLE_EXCEPTIONS: t.VariadicTuple[type[Exception]] = (
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
        catch: type[Exception] | t.VariadicTuple[type[Exception]] | None = None,
        *,
        op_name: str = "execute guarded operation",
    ) -> p.Result[TResult]:
        """Execute a callable and capture configured exceptions as failed results.

        Wraps the callable's return value into ``r[T].ok(...)`` and translates
        any matched exception into ``r[T].fail_op(op_name, exc)``. Use
        ``op_name`` to give the failure a meaningful operation label without
        wrapping in a custom try/except block.
        """
        if catch is None:
            handled = FlextUtilitiesReliability._RETRYABLE_EXCEPTIONS
        elif isinstance(catch, type):
            handled = (catch,)
        else:
            handled = catch
        try:
            return r[TResult].ok(operation())
        except handled as exc:
            return r[TResult].fail_op(op_name, exc)

    @staticmethod
    def guard_result[TResult](
        operation: Callable[[], p.Result[TResult]],
        *,
        catch: type[Exception] | t.VariadicTuple[type[Exception]] | None = None,
        op_name: str = "execute guarded operation",
    ) -> p.Result[TResult]:
        """Boundary-guard a Result-returning operation.

        Wraps any callable that already returns ``p.Result[T]`` with an
        exception boundary translator. Eliminates the canonical
        ``try/except + r.fail_op`` boilerplate at every adapter boundary
        (LDAP/HTTP/DB/etc.) where library calls may raise but the domain
        contract is ``r[T]``.

        On exception: returns ``r[T].fail_op(op_name, exc)``.
        On Result outcome: propagates the original Result unchanged.
        """
        if catch is None:
            handled = FlextUtilitiesReliability._RETRYABLE_EXCEPTIONS
        elif isinstance(catch, type):
            handled = (catch,)
        else:
            handled = catch
        try:
            return operation()
        except handled as exc:
            return r[TResult].fail_op(op_name, exc)

    @staticmethod
    def retry[TResult](
        operation: Callable[[], p.Result[TResult]],
        options: FlextUtilitiesReliability.RetryOptions | None = None,
        **kwargs: t.JsonPayload,
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
        max_att = (
            opts.max_attempts if opts.max_attempts is not None else c.MAX_RETRY_ATTEMPTS
        )
        delay_sec = (
            opts.delay_seconds
            if opts.delay_seconds is not None
            else c.DEFAULT_RETRY_DELAY_SECONDS
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
                last_error = str(e)
            if attempt < max_att - 1:
                current_delay = min(
                    delay_sec * c.DEFAULT_BACKOFF_MULTIPLIER**attempt,
                    c.DEFAULT_MAX_DELAY_SECONDS,
                )
                if current_delay > 0:
                    time.sleep(current_delay)
        return r[TResult].fail(
            f"Operation failed after {max_att} attempts: {last_error}",
        )


__all__: t.MutableSequenceOf[str] = ["FlextUtilitiesReliability"]
