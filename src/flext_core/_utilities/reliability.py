"""Reliability helpers aligned with dispatcher-centric CQRS flows.

Utilities extracted from ``flext_core.utilities`` to keep retry and
timeout behaviors modular. All helpers return ``p.Result``-compatible
implementations so dispatcher handlers can compose reliability policies
without raising exceptions or leaking thread-local state.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextvars
import threading
import time
from collections.abc import Callable, MutableSequence
from typing import ClassVar, TypeIs

from flext_core import (
    FlextModelFoundation,
    FlextProtocolsLogging,
    FlextRuntime,
    FlextUtilitiesGuards,
    c,
    r,
    t,
)


class FlextUtilitiesReliability:
    """Reliability patterns for resilient, dispatcher-safe operations."""

    _V: ClassVar[type[FlextModelFoundation.Validators]] = (
        FlextModelFoundation.Validators
    )

    @property
    def logger(self) -> FlextProtocolsLogging.Logger:
        """Get structlog logger via FlextRuntime (infrastructure-level, no FlextLogger)."""
        return FlextRuntime.get_logger(__name__)

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
    def _is_match_predicate(
        value: type | t.NormalizedValue | Callable[[t.NormalizedValue], bool],
    ) -> TypeIs[Callable[[t.NormalizedValue], bool]]:
        return callable(value) and (not isinstance(value, type))

    @staticmethod
    def _resolve_match_output(
        candidate: t.NormalizedValue | Callable[[t.NormalizedValue], t.NormalizedValue],
        value: t.NormalizedValue,
    ) -> t.Container:
        if not callable(candidate):
            if FlextUtilitiesGuards.is_container(candidate):
                return candidate
            return str(candidate)
        raw_mapped: t.NormalizedValue = candidate(value)
        if FlextUtilitiesGuards.is_container(raw_mapped):
            return raw_mapped
        return str(raw_mapped)

    @staticmethod
    def calculate_delay(attempt: int, config: t.ScalarMapping | None) -> float:
        """Calculate delay for retry attempt using configuration.

        Args:
            attempt: Current attempt number (0-based)
            config: Retry configuration t.NormalizedValue

        Returns:
            float: Delay in seconds for next attempt

        """
        cfg: t.ScalarMapping = config if config is not None else {}
        initial_delay_raw = cfg.get("initial_delay_seconds", 0.1)
        max_delay_raw = cfg.get("max_delay_seconds", 60.0)
        exponential_backoff_raw = cfg.get("exponential_backoff", False)
        backoff_multiplier_raw = cfg.get("backoff_multiplier")
        initial_delay = (
            float(initial_delay_raw)
            if isinstance(initial_delay_raw, (int, float))
            and (not isinstance(initial_delay_raw, bool))
            else 0.1
        )
        max_delay = (
            float(max_delay_raw)
            if isinstance(max_delay_raw, (int, float))
            and (not isinstance(max_delay_raw, bool))
            else 60.0
        )
        exponential_backoff = (
            bool(exponential_backoff_raw)
            if isinstance(exponential_backoff_raw, bool)
            else False
        )
        backoff_multiplier: float | None = None
        if (
            backoff_multiplier_raw is not None
            and isinstance(backoff_multiplier_raw, (int, float))
            and (not isinstance(backoff_multiplier_raw, bool))
        ):
            backoff_multiplier = float(backoff_multiplier_raw)
        if exponential_backoff:
            delay = min(initial_delay * 2**attempt, max_delay)
        else:
            delay = min(initial_delay * (attempt + 1), max_delay)
        if backoff_multiplier is not None:
            delay *= backoff_multiplier
            delay = min(delay, max_delay)
        return float(delay)

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
            if current.is_failure:
                return current
            current = func(current.value)
        return current

    @staticmethod
    def fold_result[T, U](
        result: r[T],
        on_failure: Callable[[str], U],
        on_success: Callable[[T], U],
    ) -> U:
        """Fold p.Result into single value (catamorphism).

        Allows handling both success and failure cases uniformly.

        Args:
            result: p.Result to fold
            on_failure: Handler for failure case (receives error message)
            on_success: Handler for success case (receives value)

        Returns:
            Result of applying appropriate handler

        Example:
            response = u.fold_result(
                user_result,
                on_failure=lambda e: {"error": e, "status": 400},
                on_success=lambda u: {"user": u.model_dump(), "status": 200},
            )

        """
        return result.fold(
            on_failure=lambda e: on_failure(e or "Unknown error"),
            on_success=on_success,
        )

    @staticmethod
    def match(
        value: t.NormalizedValue,
        *cases: tuple[
            type | t.NormalizedValue | Callable[[t.NormalizedValue], bool],
            t.NormalizedValue | Callable[[t.NormalizedValue], t.NormalizedValue],
        ],
        default: t.NormalizedValue
        | Callable[[t.NormalizedValue], t.NormalizedValue]
        | None = None,
    ) -> r[t.Container]:
        """Pattern match on a value with type, value, or predicate matching.

        Supports three matching modes:
         1. Type matching: `(str, lambda s: s.upper())` - match by class
        2. Value matching: `("REDACTED_LDAP_BIND_PASSWORD", "is_REDACTED_LDAP_BIND_PASSWORD")` - match if value == "REDACTED_LDAP_BIND_PASSWORD"
        3. Predicate matching: `(lambda x: x > 10, "big")` - match if predicate returns True

        Args:
            value: Value to match against
            *cases: (pattern, result) tuples where pattern can be:
                - A type (matches via isinstance)
                - A value (matches via equality)
                - A callable predicate (matches if returns True)
            default: Default value/callable if no match (optional)

        Returns:
            The result from the first matching case, or default, or None

        Example:
            >>> u.match(
            ...     "REDACTED_LDAP_BIND_PASSWORD",
            ...     (str, lambda s: s.upper()),  # type match
            ...     (
            ...         "REDACTED_LDAP_BIND_PASSWORD",
            ...         "is_REDACTED_LDAP_BIND_PASSWORD",
            ...     ),  # value match
            ...     default="unknown",
            ... )
            'ADMIN'

            >>> u.match(
            ...     15,
            ...     (lambda x: x > 10, "big"),  # predicate match
            ...     (lambda x: x > 5, "medium"),
            ...     default="small",
            ... )
            'big'

        """
        input_value: t.NormalizedValue = value
        for pattern, result in cases:
            if isinstance(pattern, type) and isinstance(input_value, pattern):
                return r[t.Container].ok(
                    FlextUtilitiesReliability._resolve_match_output(result, value),
                )
            if pattern == input_value:
                return r[t.Container].ok(
                    FlextUtilitiesReliability._resolve_match_output(
                        result,
                        input_value,
                    ),
                )
            if FlextUtilitiesReliability._is_match_predicate(pattern):
                try:
                    pred_result = pattern(input_value)
                    if pred_result:
                        return r[t.Container].ok(
                            FlextUtilitiesReliability._resolve_match_output(
                                result,
                                input_value,
                            ),
                        )
                except (ValueError, TypeError, AttributeError):
                    pass
        if default is not None:
            return r[t.Container].ok(
                FlextUtilitiesReliability._resolve_match_output(default, input_value),
            )
        return r[t.Container].fail("No match found and no default provided")

    @staticmethod
    def retry[TResult](
        operation: Callable[[], r[TResult]],
        max_attempts: int | None = None,
        delay: float | None = None,
        delay_seconds: float | None = None,
        backoff_multiplier: float | None = None,
        retry_on: tuple[type[Exception], ...] | None = None,
    ) -> r[TResult]:
        """Execute an operation with retry logic using railway patterns.

        Args:
            operation: Operation to execute. Must return RuntimeResult.
            max_attempts: Maximum number of attempts
            delay: Delay between retries in seconds (alias for delay_seconds)
            delay_seconds: Delay between retries in seconds
            backoff_multiplier: Multiplier for exponential backoff
            retry_on: Tuple of exception types to retry on. If None, retries on all exceptions.

        Returns:
            p.Result[TResult]: Result of operation with retry logic applied

        Fast fail: explicit default values instead of 'or' fallback.

        """
        delay_value: float | None = delay if delay is not None else delay_seconds
        max_attempts_value: int = (
            max_attempts if max_attempts is not None else c.MAX_RETRY_ATTEMPTS
        )
        delay_seconds_value: float = (
            delay_value if delay_value is not None else c.DEFAULT_RETRY_DELAY_SECONDS
        )
        backoff_multiplier_value: float = (
            backoff_multiplier
            if backoff_multiplier is not None
            else c.DEFAULT_BACKOFF_MULTIPLIER
        )
        if max_attempts_value < c.DEFAULT_RETRY_DELAY_SECONDS:
            return r[TResult].fail(
                f"Max attempts must be at least {c.DEFAULT_RETRY_DELAY_SECONDS}",
            )
        last_error: str | None = None
        for attempt in range(max_attempts_value):
            try:
                result = operation()
                if result.is_success:
                    return result
                last_error = result.error or "Unknown error"
                if attempt < max_attempts_value - 1:
                    current_delay = FlextUtilitiesReliability._calculate_retry_delay(
                        attempt,
                        delay_seconds_value,
                        backoff_multiplier_value,
                    )
                    if current_delay > 0:
                        time.sleep(current_delay)
            except (
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
                KeyError,
                OSError,
            ) as e:
                if retry_on is not None and (not isinstance(e, retry_on)):
                    raise
                last_error = str(e)
                if attempt < max_attempts_value - 1:
                    current_delay = FlextUtilitiesReliability._calculate_retry_delay(
                        attempt,
                        delay_seconds_value,
                        backoff_multiplier_value,
                    )
                    if current_delay > 0:
                        time.sleep(current_delay)
        return r[TResult].fail(
            f"Operation failed after {max_attempts_value} attempts: {last_error}",
        )

    @staticmethod
    def tap_result[T](result: r[T], func: Callable[[T], None]) -> r[T]:
        """Execute side effect on success without changing result.

        Useful for logging, metrics, or other side effects.

        Args:
            result: p.Result to tap
            func: Side effect function (return value ignored)

        Returns:
            Original result unchanged

        Example:
            result = u.tap_result(
                user_result,
                lambda u: logger.info("User created", user_id=u.id)
            )

        """
        if result.is_success:
            func(result.value)
        return result

    @staticmethod
    def then[T, U](result: r[T], func: Callable[[T], r[U]]) -> r[U]:
        """Chain single operation on p.Result (monadic bind).

        Also known as flatMap or bind in other languages.

        Args:
            result: p.Result to chain
            func: Function that takes value and returns new p.Result

        Returns:
            Result of applying func, or original failure

        Example:
            user_result = u.then(
                validate_input(data),
                lambda data: create_user(data)
            )

        """
        return result.fold(
            on_failure=lambda e: r[U].fail(e or "Unknown error"),
            on_success=func,
        )

    @staticmethod
    def with_retry[TResult](
        operation: Callable[[], r[TResult]],
        max_attempts: int = c.MAX_RETRY_ATTEMPTS,
        should_retry_func: Callable[[int, str | None], bool] | None = None,
        cleanup_func: Callable[[], None] | None = None,
    ) -> r[TResult]:
        """Execute operation with retry logic using railway patterns.

        Args:
            operation: Operation to execute (should return RuntimeResult)
            max_attempts: Maximum number of attempts
            should_retry_func: Function to determine if retry should happen
            cleanup_func: Function to call for cleanup after each attempt

        Returns:
            p.Result[TResult]: Result of operation with retry

        """
        for attempt in range(max_attempts):
            try:
                result = operation()
                if result.is_success:
                    return result
                should_retry = (
                    should_retry_func(attempt, result.error)
                    if should_retry_func
                    else False
                )
                if not should_retry or attempt >= max_attempts - 1:
                    return result
                if cleanup_func:
                    cleanup_func()
            except (
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
                KeyError,
                OSError,
            ) as e:
                if attempt >= max_attempts - 1:
                    return r[TResult].fail(f"Operation failed: {e}")
                if cleanup_func:
                    cleanup_func()
        return r[TResult].fail("Max retries exceeded")

    @staticmethod
    def with_timeout[TTimeout](
        operation: Callable[[], r[TTimeout]],
        timeout_seconds: float,
    ) -> r[TTimeout]:
        """Execute an operation with a hard timeout using railway patterns."""
        if timeout_seconds <= c.INITIAL_TIME:
            return r[TTimeout].fail("Timeout must be positive")
        result_container: MutableSequence[r[TTimeout] | None] = [None]
        exception_container: MutableSequence[Exception | None] = [None]

        def run_operation() -> None:
            try:
                result_container[0] = operation()
            except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
                exception_container[0] = e

        context = contextvars.copy_context()
        thread = threading.Thread(target=context.run, args=(run_operation,))
        thread.daemon = True
        thread.start()
        thread.join(timeout_seconds)
        if thread.is_alive():
            return r[TTimeout].fail(
                f"Operation timed out after {timeout_seconds} seconds",
            )
        if exception_container[0]:
            return r[TTimeout].fail(
                f"Operation failed with exception: {exception_container[0]}",
            )
        if result_container[0] is None:
            return r[TTimeout].fail("Operation completed but returned no result")
        return result_container[0]

    mt = match


__all__ = ["FlextUtilitiesReliability"]
