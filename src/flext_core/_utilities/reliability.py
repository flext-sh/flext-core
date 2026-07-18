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
from typing import Annotated, TYPE_CHECKING

from pydantic import Field

from flext_core import c, p, r, t
from flext_core._models.base import FlextModelsBase
from flext_core._utilities.args import FlextUtilitiesArgs


if TYPE_CHECKING:
    from collections.abc import Callable


type _HandledExceptions = tuple[type[Exception], ...]


class FlextUtilitiesReliability:
    """Reliability patterns for resilient, dispatcher-safe operations."""

    class RetryOptions(FlextModelsBase.FlexibleInternalModel):
        """Configuration options for retry logic."""

        max_attempts: Annotated[
            int | None, Field(ge=1, description="Maximum number of retry attempts")
        ] = None
        delay_seconds: Annotated[
            float | None,
            Field(ge=0, description="Initial delay between retries in seconds"),
        ] = None

    _RETRYABLE_EXCEPTIONS: _HandledExceptions = (
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
        catch: type[Exception] | _HandledExceptions | None = None,
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
        catch: type[Exception] | _HandledExceptions | None = None,
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
            options, kwargs, FlextUtilitiesReliability.RetryOptions
        )
        if opts_res.failure:
            # Preserve the validated options failure metadata for every consumer.
            return r[TResult].from_failure(opts_res)
        opts = opts_res.value
        max_att = (
            opts.max_attempts if opts.max_attempts is not None else c.MAX_RETRY_ATTEMPTS
        )
        delay_sec = (
            opts.delay_seconds
            if opts.delay_seconds is not None
            else c.DEFAULT_RETRY_DELAY_SECONDS
        )
        last_failure: p.Result[TResult] | None = None
        for attempt in range(max_att):
            try:
                result = operation()
            except FlextUtilitiesReliability._RETRYABLE_EXCEPTIONS as exc:
                result = r[TResult].fail_op("execute retry operation", exc)
            if result.success:
                return result
            last_failure = result
            if attempt < max_att - 1:
                current_delay = min(
                    delay_sec * c.DEFAULT_BACKOFF_MULTIPLIER**attempt,
                    c.DEFAULT_MAX_DELAY_SECONDS,
                )
                if current_delay > 0:
                    time.sleep(current_delay)
        if last_failure is None:
            return r[TResult].fail(c.ERR_RUNTIME_RETRY_LOOP_ENDED_WITHOUT_RESULT)
        return (
            r[TResult]
            .from_failure(last_failure)
            .map_error(
                lambda error: f"Operation failed after {max_att} attempts: {error}"
            )
        )


__all__: list[str] = ["FlextUtilitiesReliability"]
