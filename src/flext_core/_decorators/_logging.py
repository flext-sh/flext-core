"""Operation logging and correlation decorators.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from functools import wraps

from flext_core import FlextUtilities as u
from flext_core._constants.base import FlextConstantsBase as cb
from flext_core._constants.infrastructure import FlextConstantsInfrastructure as ci
from flext_core._decorators._logging_payloads import FlextDecoratorsLoggingPayloads


from typing import TYPE_CHECKING

# mro-cqxy (ADR-011): runtime import — used in a runtime-evaluated annotation
from flext_core._typings.base import FlextTypingBase as tb

if TYPE_CHECKING:
    from flext_core._protocols.logging import FlextProtocolsLogging as pl
    from collections.abc import Callable


class FlextDecoratorsLogging(FlextDecoratorsLoggingPayloads):
    """Decorators that bind operation logging and correlation context."""

    @classmethod
    def log_operation[**PCallback, TResult](
        cls,
        operation_name: str | None = None,
        *,
        track_perf: bool = False,
        ensure_correlation: bool = True,
    ) -> Callable[[Callable[PCallback, TResult]], Callable[PCallback, TResult]]:
        """Log operation execution with structured context."""

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
                    logger_carrier, func_module=func.__module__
                )
                correlation_id = cls._resolve_correlation_id(
                    ensure_correlation=ensure_correlation
                )
                cls._context_type.apply_operation_name(op_name)
                binding_result = u.bind_context(
                    ci.ContextScope.OPERATION, operation=op_name
                )
                if binding_result.failure:
                    binding_result.unwrap()
                start_time = time.perf_counter() if track_perf else 0.0
                try:
                    return cls._execute_logged_call(
                        lambda: func(*args, **kwargs),
                        func_name=func.__name__,
                        func_module=func.__module__,
                        op_name=op_name,
                        logger=logger,
                        correlation_id=correlation_id,
                        track_perf=track_perf,
                        start_time=start_time,
                    )
                finally:
                    u.clear_scope(ci.ContextScope.OPERATION).unwrap()

            return wrapper

        return decorator

    @classmethod
    def _resolve_correlation_id(cls, *, ensure_correlation: bool) -> str | None:
        """Resolve or ensure the current correlation id."""
        if ensure_correlation:
            return cls._context_type.ensure_correlation_id()
        current_id = u.CORRELATION_ID.get()
        return current_id if isinstance(current_id, str) else None

    @classmethod
    def _execute_logged_call[TResult](
        cls,
        call: Callable[[], TResult],
        *,
        func_name: str,
        func_module: str,
        op_name: str,
        logger: pl.Logger,
        correlation_id: str | None,
        track_perf: bool,
        start_time: float,
    ) -> TResult:
        """Execute the wrapped callable and emit success/failure logs."""
        try:
            logger.debug(
                "%s_started",
                op_name,
                **cls._start_log_payload(
                    func_name=func_name,
                    func_module=func_module,
                    correlation_id=correlation_id,
                ),
            )
            result = call()
            logger.debug(
                "%s_completed",
                op_name,
                **cls._success_log_payload(
                    func_name=func_name,
                    correlation_id=correlation_id,
                    track_perf=track_perf,
                    start_time=start_time,
                ),
            )
            return result
        except cls._CAUGHT_EXCEPTIONS as exc:
            tracked_duration = time.perf_counter() - start_time if track_perf else 0.0
            exc_kw: tb.MutableJsonMapping = {
                "function": func_name,
                "success": False,
                "error": str(exc),
                "error_type": exc.__class__.__name__,
                "operation": op_name,
            }
            if correlation_id is not None:
                exc_kw[ci.ContextKey.CORRELATION_ID] = correlation_id
            if track_perf:
                exc_kw["duration_ms"] = tracked_duration * cb.DEFAULT_SIZE
                exc_kw[ci.MetadataKey.DURATION_SECONDS] = tracked_duration
            logger.exception(op_name, exception=exc, **exc_kw)
            raise

    @classmethod
    def with_correlation[**PCallback, TResult](
        cls,
    ) -> Callable[[Callable[PCallback, TResult]], Callable[PCallback, TResult]]:
        """Ensure a correlation ID exists during the wrapped operation."""

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
