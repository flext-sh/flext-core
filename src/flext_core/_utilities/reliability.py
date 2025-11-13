"""Utilities module - FlextUtilitiesReliability.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from collections.abc import Callable

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult

# Module constants
MAX_PORT_NUMBER: int = 65535
MIN_PORT_NUMBER: int = 1
_logger = logging.getLogger(__name__)


class FlextUtilitiesReliability:
    """Reliability patterns for resilient operations."""

    @staticmethod
    def with_timeout[TTimeout](
        operation: Callable[[], FlextResult[TTimeout]],
        timeout_seconds: float,
    ) -> FlextResult[TTimeout]:
        """Execute operation with timeout using railway patterns."""
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
        """Execute operation with retry logic using railway patterns."""
        max_attempts = max_attempts or FlextConstants.Reliability.MAX_RETRY_ATTEMPTS
        delay_seconds = (
            delay_seconds or FlextConstants.Reliability.DEFAULT_RETRY_DELAY_SECONDS
        )
        backoff_multiplier = (
            backoff_multiplier or FlextConstants.Reliability.RETRY_BACKOFF_BASE
        )

        if max_attempts < FlextConstants.Reliability.RETRY_COUNT_MIN:
            return FlextResult[TResult].fail(
                f"Max attempts must be at least {FlextConstants.Reliability.RETRY_COUNT_MIN}"
            )

        last_error: str | None = None

        for attempt in range(max_attempts):
            try:
                result = operation()
                if result.is_success:
                    return result

                last_error = result.error or "Unknown error"

                # Don't delay on the last attempt
                if attempt == max_attempts - 1:
                    break

                # Calculate delay with exponential backoff
                current_delay = delay_seconds * (backoff_multiplier**attempt)
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
                if attempt == max_attempts - 1:
                    break

                # Calculate delay with exponential backoff
                current_delay = delay_seconds * (backoff_multiplier**attempt)
                current_delay = min(
                    current_delay,
                    FlextConstants.Reliability.RETRY_BACKOFF_MAX,
                )

                # Sleep before retry
                time.sleep(current_delay)

        return FlextResult[TResult].fail(
            f"Operation failed after {max_attempts} attempts: {last_error}"
        )

    @staticmethod
    def calculate_delay(attempt: int, config: object) -> float:
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
