"""Reliability helpers aligned with dispatcher-centric CQRS flows.

Utilities extracted from ``flext_core.utilities`` to keep retry and
timeout behaviors modular. All helpers return ``FlextRuntime.RuntimeResult`` so
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

from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core._utilities.mapper import FlextUtilitiesMapper
from flext_core.constants import c
from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import t


class FlextUtilitiesReliability:
    """Reliability patterns for resilient, dispatcher-safe operations."""

    @property
    def logger(self) -> p.Log.StructlogLogger:
        """Get logger instance using FlextRuntime (avoids circular imports).

        Returns structlog logger instance with all logging methods (debug, info, warning, error, etc).
        Uses same structure/config as FlextLogger but without circular import.
        """
        return FlextRuntime.get_logger(__name__)

    @staticmethod
    def with_timeout[TTimeout](
        operation: Callable[[], r[TTimeout]],
        timeout_seconds: float,
    ) -> r[TTimeout]:
        """Execute an operation with a hard timeout using railway patterns."""
        if timeout_seconds <= c.INITIAL_TIME:
            return r[TTimeout].fail("Timeout must be positive")

        # Use proper typing for containers
        result_container: list[r[TTimeout] | None] = [None]
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
            return r[TTimeout].fail(
                f"Operation timed out after {timeout_seconds} seconds",
            )

        if exception_container[0]:
            return r[TTimeout].fail(
                f"Operation failed with exception: {exception_container[0]}",
            )

        if result_container[0] is None:
            return r[TTimeout].fail(
                "Operation completed but returned no result",
            )

        return result_container[0]

    @staticmethod
    def _calculate_retry_delay(
        attempt: int,
        delay_seconds: float,
        backoff_multiplier: float,
    ) -> float:
        """Calculate delay for retry attempt with exponential backoff."""
        current_delay = delay_seconds * (backoff_multiplier**attempt)
        return min(current_delay, c.Reliability.RETRY_BACKOFF_MAX)

    @staticmethod
    def _normalize_operation_result[TResult](
        result_raw: r[TResult] | TResult,
    ) -> r[TResult]:
        """Convert operation result to RuntimeResult."""
        # Check if result_raw is a RuntimeResult by checking for is_success/is_failure attributes
        # Structural typing validates: if it has Result interface, treat as Result
        has_result_interface = (
            hasattr(result_raw, "is_success")
            and hasattr(result_raw, "is_failure")
            and callable(getattr(result_raw, "unwrap", None))
        )
        if has_result_interface:
            # Type narrowing: result_raw has RuntimeResult structure (is_success, is_failure, unwrap)
            # Structural typing validates this is a Result type - return directly
            # Python's structural typing allows this without explicit cast
            return result_raw

        # Type narrowing: result_raw is not a Result - must be TResult
        # Wrap the value in a successful Result
        return r[TResult].ok(result_raw)

    @staticmethod
    def retry[TResult](
        operation: Callable[[], r[TResult] | TResult],
        max_attempts: int | None = None,
        delay: float | None = None,
        delay_seconds: float | None = None,
        backoff_multiplier: float | None = None,
        retry_on: tuple[type[Exception], ...] | None = None,
    ) -> r[TResult]:
        """Execute an operation with retry logic using railway patterns.

        Args:
            operation: Operation to execute. Can return RuntimeResult or direct value.
            max_attempts: Maximum number of attempts
            delay: Delay between retries in seconds (alias for delay_seconds)
            delay_seconds: Delay between retries in seconds
            backoff_multiplier: Multiplier for exponential backoff
            retry_on: Tuple of exception types to retry on. If None, retries on all exceptions.

        Returns:
            r[TResult]: Result of operation with retry logic applied

        Fast fail: explicit default values instead of 'or' fallback.

        """
        # Use delay if provided, otherwise delay_seconds, otherwise default
        delay_value: float | None = delay if delay is not None else delay_seconds

        # Fast fail: explicit default values instead of 'or' fallback
        max_attempts_value: int = (
            max_attempts
            if max_attempts is not None
            else c.Reliability.MAX_RETRY_ATTEMPTS
        )
        delay_seconds_value: float = (
            delay_value
            if delay_value is not None
            else c.Reliability.DEFAULT_RETRY_DELAY_SECONDS
        )
        backoff_multiplier_value: float = (
            backoff_multiplier
            if backoff_multiplier is not None
            else c.Reliability.RETRY_BACKOFF_BASE
        )

        if max_attempts_value < c.Reliability.RETRY_COUNT_MIN:
            return r[TResult].fail(
                f"Max attempts must be at least {c.Reliability.RETRY_COUNT_MIN}",
            )

        last_error: str | None = None

        for attempt in range(max_attempts_value):
            try:
                result_raw = operation()
                result = FlextUtilitiesReliability._normalize_operation_result(
                    result_raw,
                )

                if result.is_success:
                    return result

                # When is_failure is True, error is never None (fail() converts None to "")
                last_error = result.error or "Unknown error"

                # Don't delay on the last attempt
                if attempt < max_attempts_value - 1:
                    current_delay = FlextUtilitiesReliability._calculate_retry_delay(
                        attempt,
                        delay_seconds_value,
                        backoff_multiplier_value,
                    )
                    if current_delay > 0:
                        time.sleep(current_delay)

            except Exception as e:
                # Check if this exception should be retried
                if retry_on is not None and not isinstance(e, retry_on):
                    # Exception not in retry_on list, propagate it
                    raise

                last_error = str(e)

                # Don't delay on the last attempt
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
    def calculate_delay(
        attempt: int,
        config: t.ConfigurationDict | None,
    ) -> float:
        """Calculate delay for retry attempt using configuration.

        Args:
            attempt: Current attempt number (0-based)
            config: Retry configuration object

        Returns:
            float: Delay in seconds for next attempt

        """
        # Extract configuration values safely with proper type conversion
        # config is t.ConfigurationDict | None, use dict.get() instead of getattr()
        if config is None:
            config = {}
        # Type narrowing: ensure values are numeric before conversion
        initial_delay_raw = config.get("initial_delay_seconds", 0.1)
        max_delay_raw = config.get("max_delay_seconds", 60.0)
        exponential_backoff_raw = config.get("exponential_backoff", False)
        backoff_multiplier_raw = config.get("backoff_multiplier")

        # Convert to float with type narrowing
        initial_delay = (
            float(initial_delay_raw)
            if isinstance(initial_delay_raw, (int, float))
            else 0.1
        )
        max_delay = (
            float(max_delay_raw) if isinstance(max_delay_raw, (int, float)) else 60.0
        )
        exponential_backoff = (
            bool(exponential_backoff_raw)
            if FlextUtilitiesGuards.is_type(exponential_backoff_raw, bool)
            else False
        )
        backoff_multiplier: float | None = None
        if backoff_multiplier_raw is not None and isinstance(
            backoff_multiplier_raw,
            (int, float),
        ):
            backoff_multiplier = float(backoff_multiplier_raw)

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
        operation: Callable[[], r[TResult]],
        max_attempts: int = c.Reliability.MAX_RETRY_ATTEMPTS,
        should_retry_func: Callable[[int, str | None], bool] | None = None,
        cleanup_func: Callable[[], None] | None = None,
    ) -> FlextRuntime.RuntimeResult[TResult]:
        """Execute operation with retry logic using railway patterns.

        Args:
            operation: Operation to execute (should return RuntimeResult)
            max_attempts: Maximum number of attempts
            should_retry_func: Function to determine if retry should happen
            cleanup_func: Function to call for cleanup after each attempt

        Returns:
            r[TResult]: Result of operation with retry

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
                    return r[TResult].fail(f"Operation failed: {e}")

                if cleanup_func:
                    cleanup_func()

        return r[TResult].fail("Max retries exceeded")

    # ========================================================================
    # Compose methods (pipe, chain, flow) - Functional composition patterns
    # ========================================================================

    @staticmethod
    def pipe(
        value: object,
        *operations: Callable[[object], object],
        on_error: str = "stop",
    ) -> r[object]:
        """Functional pipeline with railway-oriented error handling.

        Business Rule: Chains operations sequentially, unwrapping r
        values automatically. Error handling modes: "stop" (fail fast) or
        "skip" (continue with previous value). Railway pattern ensures errors
        propagate correctly through the pipeline.

        Args:
            value: Initial value to process
            *operations: Functions to apply in sequence
            on_error: Error handling ("stop" or "skip")

        Returns:
            r containing final value or error

        Example:
            result = FlextUtilitiesReliability.pipe(
                "  hello world  ",
                str.strip,
                str.upper,
                lambda s: s.replace(" ", "_"),
            )
            # → r.ok("HELLO_WORLD")

        """
        if not operations:
            return r[object].ok(value)

        # Type annotation: current starts as object (value parameter)
        # Will be updated through pipeline operations
        current: object = value
        for i, op in enumerate(operations):
            try:
                result = op(current)

                # Unwrap r if returned - use structural typing check
                has_result_interface = (
                    hasattr(result, "is_success")
                    and hasattr(result, "is_failure")
                    and hasattr(result, "value")
                )
                if has_result_interface:
                    # Structural typing: result has Result interface methods
                    # result implements Result protocol - access via protocol
                    result_typed: p.Result[object] = result
                    if result_typed.is_failure:
                        if on_error == "stop":
                            err_msg = result_typed.error or "Unknown error"
                            return r[object].fail(
                                f"Pipeline step {i} failed: {err_msg}",
                            )
                        # on_error == "skip": continue with previous value
                        # Type narrowing: current remains unchanged (previous value)
                        continue
                    # Type narrowing: result.is_success is True, so result.value is valid
                    # Extract value from Result - result_typed implements Result protocol
                    current = result_typed.value
                else:
                    # Type annotation: result is object (non-RuntimeResult return)
                    current = result

            except Exception as e:
                if on_error == "stop":
                    return r[object].fail(f"Pipeline step {i} failed: {e}")
                # on_error == "skip": continue with previous value

        return r[object].ok(current)

    @staticmethod
    def chain(
        value: object,
        *funcs: Callable[[object], object],
    ) -> object:
        """Chain operations (mnemonic: chain = pipeline).

        Business Rule: Execute a sequence of functions in order, passing each
        result to the next function. This is the functional pipeline pattern.

        Generic replacement for: func3(func2(func1(value))) patterns

        Args:
            value: Initial value (any type)
            *funcs: Functions to apply in sequence

        Returns:
            Final result after all operations

        Example:
            result = FlextUtilitiesReliability.chain(
                data,
                lambda x: x.get("items", []),
                lambda x: [i for i in x if i > 0],
                lambda x: [i * 2 for i in x],
            )

        """
        current: object = value
        for func in funcs:
            current = func(current)
        return current

    @staticmethod
    def flow(
        value: object,
        *ops: t.ConfigurationDict | Callable[[object], object],
    ) -> object:
        """Flow operations using DSL or functions (mnemonic: flow = fluent pipeline).

        Generic replacement for: build() + chain() combinations

        Args:
            value: Initial value
            *ops: Operations (dict DSL or callable functions)

        Returns:
            Final result

        Example:
            result = FlextUtilitiesReliability.flow(
                data,
                {"ensure": "dict"},
                {"get": "items", "default": []},
                lambda x: [i for i in x if i > 0],
                {"map": lambda i: i * 2},
            )

        """
        current: object = value
        for op in ops:
            if isinstance(op, dict):
                # Type narrowing: op is dict, isinstance provides type narrowing to ConfigurationDict
                # build() expects ops: t.ConfigurationDict | None
                op_dict: t.ConfigurationDict = op
                current = FlextUtilitiesMapper.build(current, ops=op_dict)
            elif callable(op):
                current = op(current)
        return current

    @staticmethod
    def compose(
        *funcs: Callable[[object], object],
        mode: str = "pipe",
    ) -> Callable[[object], object]:
        """Compose multiple functions into a single function.

        Unifies pipe/chain/flow patterns into a single super-method.

        Args:
            *funcs: Functions to compose
            mode: Composition mode ("pipe", "chain", "flow")

        Returns:
            Composed function

        Example:
            composed = FlextUtilitiesReliability.compose(
                str.strip,
                str.upper,
                mode="pipe",
            )
            result = composed("  hello  ")  # → "HELLO"

        """
        if mode == "pipe":

            def piped(value: object) -> object:
                result = FlextUtilitiesReliability.pipe(value, *funcs)

                if hasattr(result, "is_success") and hasattr(result, "is_failure"):
                    return result.value if result.is_success else result
                return result

            return piped

        if mode == "chain":
            return lambda value: FlextUtilitiesReliability.chain(value, *funcs)

        # mode == "flow"
        return lambda value: FlextUtilitiesReliability.flow(value, *funcs)


__all__ = ["FlextUtilitiesReliability"]
