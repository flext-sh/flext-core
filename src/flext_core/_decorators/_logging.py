"""Operation logging and correlation decorators.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from functools import wraps
from typing import TYPE_CHECKING, Annotated

from flext_core import c, u
from flext_core._decorators._logging_payloads import FlextDecoratorsLoggingPayloads
from flext_core._models import FlextModelsPydantic
from flext_core._models.pydantic import Field
from flext_core._protocols import FlextProtocolsLogging
from flext_core._typings.base import FlextTypingBase

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextDecoratorsLogging(FlextDecoratorsLoggingPayloads):
    """Decorators that bind operation logging and correlation context."""

    class FlextLoggedCallSpec(FlextModelsPydantic.BaseModel):
        """Frozen parameters for one logged call execution.

        Groups the logging-related arguments of
        ``FlextDecoratorsLogging._execute_logged_call`` so the executor keeps a
        narrow signature.
        """

        model_config = FlextModelsPydantic.ConfigDict(
            frozen=True,
            arbitrary_types_allowed=True,
            extra="forbid",
        )

        func_name: Annotated[str, Field(description="Wrapped callable name")]
        func_module: Annotated[str, Field(description="Wrapped callable module")]
        op_name: Annotated[str, Field(description="Declared operation name")]
        logger: Annotated[
            FlextProtocolsLogging.Logger,
            Field(description="Structured logger bound to the call"),
        ]
        correlation_id: Annotated[
            str | None,
            Field(description="Correlation id ensured for the call scope"),
        ]
        track_perf: Annotated[
            bool,
            Field(description="Whether the call measures its duration"),
        ]
        start_time: Annotated[
            float,
            Field(description="Perf-counter start instant (0.0 when not tracking)"),
        ]

    @classmethod
    def log_operation[**PCallback, TResult](
        cls,
        operation_name: str | None = None,
        *,
        track_perf: bool = False,
        ensure_correlation: bool = True,
    ) -> Callable[[Callable[PCallback, TResult]], Callable[PCallback, TResult]]:
        """Log operation execution with structured context.

        Returns:
            The resulting ``Callable[[Callable[PCallback, TResult]], Callable[PCallback,
                TResult]]``.

        """

        def decorator(
            func: Callable[PCallback, TResult],
        ) -> Callable[PCallback, TResult]:
            @wraps(func)
            def wrapper(*args: PCallback.args, **kwargs: PCallback.kwargs) -> TResult:
                op_name = (
                    operation_name if operation_name is not None else func.__name__
                )
                logger_carrier: cls._LoggerCarrier | None = None
                if args and cls._is_logger_carrier(args[0]):
                    logger_carrier = args[0]
                logger = cls._resolve_logger(
                    logger_carrier,
                    func_module=func.__module__,
                )
                correlation_id = cls._resolve_correlation_id(
                    ensure_correlation=ensure_correlation,
                )
                cls._context_type.apply_operation_name(op_name)
                binding_result = u.bind_context(
                    c.ContextScope.OPERATION,
                    operation=op_name,
                )
                if binding_result.failure:
                    binding_result.unwrap()
                start_time = time.perf_counter() if track_perf else 0.0
                try:
                    return cls._execute_logged_call(
                        lambda: func(*args, **kwargs),
                        spec=cls.FlextLoggedCallSpec(
                            func_name=func.__name__,
                            func_module=func.__module__,
                            op_name=op_name,
                            logger=logger,
                            correlation_id=correlation_id,
                            track_perf=track_perf,
                            start_time=start_time,
                        ),
                    )
                finally:
                    u.clear_scope(c.ContextScope.OPERATION).unwrap()

            return wrapper

        return decorator

    @classmethod
    def _resolve_correlation_id(cls, *, ensure_correlation: bool) -> str | None:
        """Resolve or ensure the current correlation id.

        Returns:
            The resulting ``str | None``.

        """
        if ensure_correlation:
            return cls._context_type.ensure_correlation_id()
        current_id = u.CORRELATION_ID.get()
        return current_id if isinstance(current_id, str) else None

    @classmethod
    def _execute_logged_call[TResult](
        cls,
        call: Callable[[], TResult],
        *,
        spec: FlextDecoratorsLogging.FlextLoggedCallSpec,
    ) -> TResult:
        """Execute the wrapped callable and emit success/failure logs.

        Returns:
            The resulting ``TResult``.

        """
        try:
            spec.logger.debug(
                "%s_started",
                spec.op_name,
                **cls._start_log_payload(
                    func_name=spec.func_name,
                    func_module=spec.func_module,
                    correlation_id=spec.correlation_id,
                ),
            )
            result = call()
        except cls._CAUGHT_EXCEPTIONS as exc:
            tracked_duration = (
                time.perf_counter() - spec.start_time if spec.track_perf else 0.0
            )
            exc_kw: FlextTypingBase.MutableJsonMapping = {
                "function": spec.func_name,
                "success": False,
                "error": str(exc),
                "error_type": exc.__class__.__name__,
                "operation": spec.op_name,
            }
            if spec.correlation_id is not None:
                exc_kw[c.ContextKey.CORRELATION_ID] = spec.correlation_id
            if spec.track_perf:
                exc_kw["duration_ms"] = tracked_duration * c.MS_PER_SECOND
                exc_kw[c.MetadataKey.DURATION_SECONDS] = tracked_duration
            spec.logger.exception(spec.op_name, exception=exc, **exc_kw)
            raise
        else:
            spec.logger.debug(
                "%s_completed",
                spec.op_name,
                **cls._success_log_payload(
                    func_name=spec.func_name,
                    correlation_id=spec.correlation_id,
                    track_perf=spec.track_perf,
                    start_time=spec.start_time,
                ),
            )
            return result

    @classmethod
    def with_correlation[**PCallback, TResult](
        cls,
    ) -> Callable[[Callable[PCallback, TResult]], Callable[PCallback, TResult]]:
        """Ensure a correlation ID exists during the wrapped operation.

        Returns:
            The resulting ``Callable[[Callable[PCallback, TResult]], Callable[PCallback,
                TResult]]``.

        """

        def decorator(
            func: Callable[PCallback, TResult],
        ) -> Callable[PCallback, TResult]:
            @wraps(func)
            def wrapper(*args: PCallback.args, **kwargs: PCallback.kwargs) -> TResult:
                _ = cls._context_type.ensure_correlation_id()
                return func(*args, **kwargs)

            return wrapper

        return decorator


__all__: list[str] = ["FlextDecoratorsLogging"]
