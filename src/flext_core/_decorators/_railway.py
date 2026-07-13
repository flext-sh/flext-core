"""Railway and retry decorators.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from functools import wraps
from typing import TYPE_CHECKING

from flext_core._constants.errors import FlextConstantsErrors as ce
from flext_core._constants.infrastructure import FlextConstantsInfrastructure as ci
from flext_core._constants.validation import FlextConstantsValidation as cv
from flext_core._decorators._logging import FlextDecoratorsLogging
from flext_core._exceptions.types import FlextExceptionsTypes as et
from flext_core._models.settings import FlextModelsSettings as ms
from flext_core.result import FlextResult as r

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core._protocols.logging import FlextProtocolsLogging as pl
    from flext_core._protocols.result import FlextProtocolsResult as pr


class FlextDecoratorsRailway(FlextDecoratorsLogging):
    """Decorators that convert failures to results or retry operations."""

    @classmethod
    def railway[**PCallback, TValue](
        cls, error_code: str | None = None
    ) -> Callable[
        [Callable[PCallback, TValue]], Callable[PCallback, pr.Result[TValue]]
    ]:
        """Wrap a callable in the FLEXT railway result pattern."""

        def decorator(
            func: Callable[PCallback, TValue],
        ) -> Callable[PCallback, pr.Result[TValue]]:
            @wraps(func)
            def wrapper(
                *args: PCallback.args, **kwargs: PCallback.kwargs
            ) -> pr.Result[TValue]:
                try:
                    result = func(*args, **kwargs)
                    return r[TValue].ok(result)
                except cls._CAUGHT_EXCEPTIONS as exc:
                    effective_error_code = (
                        error_code if error_code is not None else "OPERATION_ERROR"
                    )
                    error_msg = f"{func.__name__} failed: {type(exc).__name__}: {exc}"
                    return r[TValue].fail(error_msg, error_code=effective_error_code)

            return wrapper

        return decorator

    @classmethod
    def retry[**PCallback, TResult](
        cls,
        max_attempts: int | None = None,
        delay_seconds: float | None = None,
        backoff_strategy: str | None = None,
        error_code: str | None = None,
    ) -> Callable[[Callable[PCallback, TResult]], Callable[PCallback, TResult]]:
        """Retry failed operations using configured backoff."""
        attempts = max_attempts if max_attempts is not None else ci.MAX_RETRY_ATTEMPTS
        delay = (
            delay_seconds
            if delay_seconds is not None
            else float(ci.DEFAULT_RETRY_DELAY_SECONDS)
        )
        strategy = (
            backoff_strategy
            if backoff_strategy is not None
            else ci.DEFAULT_BACKOFF_STRATEGY
        )

        def decorator(
            func: Callable[PCallback, TResult],
        ) -> Callable[PCallback, TResult]:
            @wraps(func)
            def wrapper(*args: PCallback.args, **kwargs: PCallback.kwargs) -> TResult:
                logger_carrier: cls._LoggerCarrier | None = None
                if args:
                    first_arg_raw = args[0]
                    if cls._is_logger_carrier(first_arg_raw):
                        logger_carrier = first_arg_raw
                logger = cls._resolve_logger(
                    logger_carrier, func_module=func.__module__
                )
                retry_settings = ms.RetryConfiguration.model_validate({
                    "max_retries": attempts,
                    "initial_delay_seconds": delay,
                    "exponential_backoff": strategy == ci.DEFAULT_BACKOFF_STRATEGY,
                    "retry_on_exceptions": [],
                    "retry_on_status_codes": [],
                })
                retry_result = cls._execute_retry_loop(
                    lambda: func(*args, **kwargs),
                    func.__name__,
                    logger,
                    retry_settings=retry_settings,
                )
                if isinstance(retry_result, Exception):
                    logger.error(
                        "operation_failed_all_retries_exhausted",
                        function=func.__name__,
                        attempts=attempts,
                        error=str(retry_result),
                        error_type=retry_result.__class__.__name__,
                    )
                    effective_error_code = (
                        error_code
                        if error_code is not None
                        else cv.ErrorCode.TIMEOUT_ERROR.value
                    )
                    timeout_message = (
                        f"Operation {func.__name__} failed after {attempts} attempts"
                    )
                    raise et.TimeoutError(
                        timeout_message,
                        error_code=effective_error_code,
                        operation=func.__name__,
                        attempts=attempts,
                        original_error=str(retry_result),
                    ) from retry_result
                return retry_result

            return wrapper

        return decorator

    @classmethod
    def _execute_retry_loop[TResult](
        cls,
        call: Callable[[], TResult],
        func_name: str,
        logger: pl.Logger,
        *,
        retry_settings: ms.RetryConfiguration,
    ) -> TResult | Exception:
        """Execute retry loop with closure; return last exception on exhaustion."""
        attempts = retry_settings.max_retries
        delay = retry_settings.initial_delay_seconds
        strategy = (
            ci.DEFAULT_BACKOFF_STRATEGY
            if retry_settings.exponential_backoff
            else ci.BackoffStrategy.LINEAR
        )
        last_exception: Exception | None = None
        current_delay = delay
        for attempt in range(1, attempts + 1):
            try:
                if attempt > 1:
                    logger.info(
                        "retry_attempt",
                        function=func_name,
                        attempt=attempt,
                        max_attempts=attempts,
                        delay_seconds=current_delay,
                    )
                    time.sleep(current_delay)
                return call()
            except cls._CAUGHT_EXCEPTIONS as exc:
                last_exception = exc
                logger.warning(
                    "operation_failed_retrying",
                    function=func_name,
                    attempt=attempt,
                    max_attempts=attempts,
                    error=str(exc),
                    error_type=exc.__class__.__name__,
                )
                if strategy == ci.DEFAULT_BACKOFF_STRATEGY:
                    current_delay *= 2
                elif strategy == ci.BackoffStrategy.LINEAR:
                    current_delay += delay
                if attempt == attempts:
                    break
        if last_exception is None:
            msg = ce.ERR_RUNTIME_RETRY_LOOP_ENDED_WITHOUT_RESULT
            return RuntimeError(msg)
        return last_exception


__all__: list[str] = ["FlextDecoratorsRailway"]
