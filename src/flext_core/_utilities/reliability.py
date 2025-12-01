"""Reliability helpers aligned with dispatcher-centric CQRS flows.

Utilities extracted from ``flext_core.utilities`` to keep retry and
timeout behaviors modular. All helpers return ``FlextResult`` so
dispatcher handlers can compose reliability policies without raising
exceptions or leaking thread-local state.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextvars
import threading
import time
from collections.abc import Callable

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime, StructlogLogger
from flext_core.typings import FlextTypes


class FlextUtilitiesReliability:
    """Reliability patterns for resilient, dispatcher-safe operations."""

    @property
    def logger(self) -> StructlogLogger:
        """Get logger instance using FlextRuntime (avoids circular imports).

        Returns structlog logger instance with all logging methods (debug, info, warning, error, etc).
        Uses same structure/config as FlextLogger but without circular import.
        """
        return FlextRuntime.get_logger(__name__)

    @staticmethod
    def with_timeout[TTimeout](
        operation: Callable[[], FlextResult[TTimeout]],
        timeout_seconds: float,
    ) -> FlextResult[TTimeout]:
        """Execute an operation with a hard timeout using railway patterns."""
        if timeout_seconds <= FlextConstants.INITIAL_TIME:
            return FlextResult[TTimeout].fail("Timeout must be positive")

        # Use proper typing for containers
        result_container: list[FlextResult[TTimeout] | None] = [None]
        exception_container: list[Exception | None] = [None]

        def run_operation() -> None:
            try:
                result_container[0] = operation()
            except (
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
                KeyError,
            ) as e:
                exception_container[0] = e

        # Copy current context to the new thread
        context = contextvars.copy_context()
        thread = threading.Thread(target=context.run, args=(run_operation,))
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)

        if thread.is_alive():
            # Thread is still running, timeout occurred
            return FlextResult[TTimeout].fail(
                f"Operation timed out after {timeout_seconds} seconds",
            )

        if exception_container[0]:
            return FlextResult[TTimeout].fail(
                f"Operation failed with exception: {exception_container[0]}",
            )

        if result_container[0] is None:
            return FlextResult[TTimeout].fail(
                "Operation completed but returned no result",
            )

        return result_container[0]

    @staticmethod
    def retry[TResult](
        operation: Callable[[], FlextResult[TResult]],
        max_attempts: int | None = None,
        delay_seconds: float | None = None,
        backoff_multiplier: float | None = None,
    ) -> FlextResult[TResult]:
        """Execute an operation with retry logic using railway patterns.

        Fast fail: explicit default values instead of 'or' fallback.
        """
        # Fast fail: explicit default values instead of 'or' fallback
        max_attempts_value: int = (
            max_attempts
            if max_attempts is not None
            else FlextConstants.Reliability.MAX_RETRY_ATTEMPTS
        )
        delay_seconds_value: float = (
            delay_seconds
            if delay_seconds is not None
            else FlextConstants.Reliability.DEFAULT_RETRY_DELAY_SECONDS
        )
        backoff_multiplier_value: float = (
            backoff_multiplier
            if backoff_multiplier is not None
            else FlextConstants.Reliability.RETRY_BACKOFF_BASE
        )

        if max_attempts_value < FlextConstants.Reliability.RETRY_COUNT_MIN:
            return FlextResult[TResult].fail(
                f"Max attempts must be at least {FlextConstants.Reliability.RETRY_COUNT_MIN}",
            )

        last_error: str | None = None

        for attempt in range(max_attempts_value):
            try:
                result = operation()
                if result.is_success:
                    return result

                # Fast fail: explicit error message instead of 'or' fallback
                last_error = (
                    result.error if result.error is not None else "Unknown error"
                )

                # Don't delay on the last attempt
                if attempt == max_attempts_value - 1:
                    break

                # Calculate delay with exponential backoff
                current_delay = delay_seconds_value * (
                    backoff_multiplier_value**attempt
                )
                current_delay = min(
                    current_delay,
                    FlextConstants.Reliability.RETRY_BACKOFF_MAX,
                )

                # Sleep before retry
                time.sleep(current_delay)

            except (
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
                KeyError,
            ) as e:
                last_error = str(e)

                # Don't delay on the last attempt
                if attempt == max_attempts_value - 1:
                    break

                # Calculate delay with exponential backoff
                current_delay = delay_seconds_value * (
                    backoff_multiplier_value**attempt
                )
                current_delay = min(
                    current_delay,
                    FlextConstants.Reliability.RETRY_BACKOFF_MAX,
                )

                # Sleep before retry
                time.sleep(current_delay)

        return FlextResult[TResult].fail(
            f"Operation failed after {max_attempts_value} attempts: {last_error}",
        )

    @staticmethod
    def calculate_delay(
        attempt: int,
        config: dict[str, FlextTypes.GeneralValueType] | None,
    ) -> float:
        """Calculate delay for retry attempt using configuration.

        Args:
            attempt: Current attempt number (0-based)
            config: Retry configuration object

        Returns:
            float: Delay in seconds for next attempt

        """
        # Extract configuration values safely with proper type conversion
        initial_delay = float(getattr(config, "initial_delay_seconds", 0.1))
        max_delay = float(getattr(config, "max_delay_seconds", 60.0))
        exponential_backoff = bool(getattr(config, "exponential_backoff", False))
        backoff_multiplier = getattr(config, "backoff_multiplier", None)
        if backoff_multiplier is not None:
            backoff_multiplier = float(backoff_multiplier)

        # Calculate base delay
        if exponential_backoff:
            delay = min(initial_delay * (2**attempt), max_delay)
        else:
            delay = min(initial_delay * (attempt + 1), max_delay)

        # Apply backoff multiplier if specified
        if backoff_multiplier is not None:
            delay *= backoff_multiplier
            delay = min(delay, max_delay)

        return float(delay)

    @staticmethod
    def with_retry[TResult](
        operation: Callable[[], FlextResult[TResult]],
        max_attempts: int = 3,
        should_retry_func: Callable[[int, str | None], bool] | None = None,
        cleanup_func: Callable[[], None] | None = None,
    ) -> FlextResult[TResult]:
        """Execute operation with retry logic using railway patterns.

        Args:
            operation: Operation to execute (should return FlextResult)
            max_attempts: Maximum number of attempts
            should_retry_func: Function to determine if retry should happen
            cleanup_func: Function to call for cleanup after each attempt

        Returns:
            FlextResult[TResult]: Result of operation with retry

        """
        for attempt in range(max_attempts):
            try:
                result = operation()
                if result.is_success:
                    return result

                # Check if we should retry
                should_retry = (
                    should_retry_func(attempt, result.error)
                    if should_retry_func
                    else False
                )

                if not should_retry or attempt >= max_attempts - 1:
                    return result

                # Call cleanup function if provided
                if cleanup_func:
                    cleanup_func()

            except Exception as e:
                # If operation throws exception, consider it a failure
                if attempt >= max_attempts - 1:
                    return FlextResult[TResult].fail(f"Operation failed: {e}")

                if cleanup_func:
                    cleanup_func()

        return FlextResult[TResult].fail("Max retries exceeded")


__all__ = ["FlextUtilitiesReliability"]
