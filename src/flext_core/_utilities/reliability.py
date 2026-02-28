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
from collections.abc import Callable, Mapping
from typing import TypeGuard

from pydantic import TypeAdapter, ValidationError

from flext_core import FlextRuntime, c, p, r, t
from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core._utilities.mapper import FlextUtilitiesMapper


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
            return r.fail("Timeout must be positive")

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
            return r.fail(
                f"Operation timed out after {timeout_seconds} seconds",
            )

        if exception_container[0]:
            return r.fail(
                f"Operation failed with exception: {exception_container[0]}",
            )

        if result_container[0] is None:
            return r.fail(
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
        if isinstance(result_raw, r):
            return result_raw

        # result_raw is a plain value - wrap in successful Result
        # Safe to call ok() because at this point result_raw is TResult
        return r.ok(result_raw)

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
            return r.fail(
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

            except (
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
                KeyError,
                OSError,
            ) as e:
                if retry_on is not None and not isinstance(e, retry_on):
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

        return r.fail(
            f"Operation failed after {max_attempts_value} attempts: {last_error}",
        )

    @staticmethod
    def calculate_delay(
        attempt: int,
        config: Mapping[str, t.ConfigMapValue] | None,
    ) -> float:
        """Calculate delay for retry attempt using configuration.

        Args:
            attempt: Current attempt number (0-based)
            config: Retry configuration object

        Returns:
            float: Delay in seconds for next attempt

        """
        # Extract configuration values safely with proper type conversion
        # config is Mapping[str, t.ConfigMapValue] | None, use dict.get() instead of getattr()
        if config is None:
            config = {}
        # Type narrowing: ensure values are numeric before conversion
        initial_delay_raw = config.get("initial_delay_seconds", 0.1)
        max_delay_raw = config.get("max_delay_seconds", 60.0)
        exponential_backoff_raw = config.get("exponential_backoff", False)
        backoff_multiplier_raw = config.get("backoff_multiplier")

        initial_delay = (
            float(initial_delay_raw)
            if isinstance(initial_delay_raw, int | float)
            and not isinstance(initial_delay_raw, bool)
            else 0.1
        )
        max_delay = (
            float(max_delay_raw)
            if isinstance(max_delay_raw, int | float)
            and not isinstance(max_delay_raw, bool)
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
            and isinstance(backoff_multiplier_raw, int | float)
            and not isinstance(backoff_multiplier_raw, bool)
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

            except (
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
                KeyError,
                OSError,
            ) as e:
                # If operation throws exception, consider it a failure
                if attempt >= max_attempts - 1:
                    return r.fail(f"Operation failed: {e}")

                if cleanup_func:
                    cleanup_func()

        return r.fail("Max retries exceeded")

    # ========================================================================
    # Compose methods (pipe, chain, flow) - Functional composition patterns
    # ========================================================================

    @staticmethod
    def pipe(
        value: t.ConfigMapValue,
        *operations: Callable[[t.ConfigMapValue], t.ConfigMapValue],
        on_error: str = "stop",
    ) -> r[t.ConfigMapValue]:
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
            return r.ok(value)

        # Type annotation: current starts as object (value parameter)
        # Will be updated through pipeline operations
        current: t.ConfigMapValue = value
        for i, op in enumerate(operations):
            try:
                op_result = op(current)

                if FlextUtilitiesGuards.is_result_like(op_result):
                    if op_result.is_failure:
                        if on_error == "stop":
                            err_msg = op_result.error or "Unknown error"
                            return r.fail(
                                f"Pipeline step {i} failed: {err_msg}",
                            )
                        # on_error == "skip": continue with previous value
                        continue
                    # Extract value from successful Result
                    current = op_result.value
                else:
                    current = op_result

            except (
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
                KeyError,
                OSError,
            ) as e:
                if on_error == "stop":
                    return r.fail(f"Pipeline step {i} failed: {e}")
                # on_error == "skip": continue with previous value

        return r.ok(current)

    @staticmethod
    def chain(
        value: t.ConfigMapValue,
        *funcs: Callable[[t.ConfigMapValue], t.ConfigMapValue],
    ) -> t.ConfigMapValue:
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
        current: t.ConfigMapValue = value
        for func in funcs:
            current = func(current)
        return current

    @staticmethod
    def flow(
        value: t.ConfigMapValue,
        *ops: Mapping[str, t.ConfigMapValue]
        | Callable[[t.ConfigMapValue], t.ConfigMapValue],
    ) -> t.ConfigMapValue:
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
        current: t.ConfigMapValue = value
        for op in ops:
            if isinstance(op, Mapping):
                op_dict = op
                # Check if current is PayloadValue compatible
                if FlextUtilitiesGuards.is_general_value_type(current):
                    current = FlextUtilitiesMapper.build(current, ops=op_dict)
                else:
                    # For non-PayloadValue, skip build and just pass through
                    # This maintains backward compatibility for generic object pipelines
                    pass
            elif callable(op):
                current = op(current)
        return current

    @staticmethod
    def _is_match_predicate(
        value: type | t.ConfigMapValue | Callable[[t.ConfigMapValue], bool],
    ) -> TypeGuard[Callable[[t.ConfigMapValue], bool]]:
        return callable(value) and not isinstance(value, type)

    @staticmethod
    def _is_match_mapper(
        value: t.ConfigMapValue | Callable[[t.ConfigMapValue], t.ConfigMapValue],
    ) -> TypeGuard[Callable[[t.ConfigMapValue], t.ConfigMapValue]]:
        return callable(value)

    @staticmethod
    def _resolve_match_output(
        candidate: t.ConfigMapValue | Callable[[t.ConfigMapValue], t.ConfigMapValue],
        value: t.ConfigMapValue,
    ) -> t.ConfigMapValue:
        resolved: t.ConfigMapValue | object
        if FlextUtilitiesReliability._is_match_mapper(candidate):
            resolved = candidate(value)
        else:
            resolved = candidate
        try:
            return TypeAdapter(t.ConfigMapValue).validate_python(resolved)
        except ValidationError:
            return str(resolved)

    @staticmethod
    def compose(
        *funcs: Callable[[t.ConfigMapValue], t.ConfigMapValue],
        mode: str = "pipe",
    ) -> Callable[[t.ConfigMapValue], t.ConfigMapValue | r[t.ConfigMapValue]]:
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

            def piped(
                value: t.ConfigMapValue,
            ) -> t.ConfigMapValue | r[t.ConfigMapValue]:
                result = FlextUtilitiesReliability.pipe(value, *funcs)
                return result.value if result.is_success else result

            return piped

        if mode == "chain":
            return lambda value: FlextUtilitiesReliability.chain(value, *funcs)

        # mode == "flow"
        return lambda value: FlextUtilitiesReliability.flow(value, *funcs)

    # ========================================================================
    # Result Pipeline Methods (Railway-Oriented Programming)
    # ========================================================================

    @staticmethod
    def flow_result[T](result: r[T], *funcs: Callable[[T], r[T]]) -> r[T]:
        """Chain multiple operations on FlextResult.

        Applies each function in sequence, short-circuiting on failure.
        Railway-oriented programming pattern for composing result-returning operations.

        Args:
            result: Initial FlextResult to chain
            *funcs: Functions that take a value and return r[T]

        Returns:
            Final FlextResult after all operations or first failure

        Example:
            result = u.flow_result(
                r.ok(user),
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
    def then[T, U](result: r[T], func: Callable[[T], r[U]]) -> r[U]:
        """Chain single operation on FlextResult (monadic bind).

        Also known as flatMap or bind in other languages.

        Args:
            result: FlextResult to chain
            func: Function that takes value and returns new FlextResult

        Returns:
            Result of applying func, or original failure

        Example:
            user_result = u.then(
                validate_input(data),
                lambda data: create_user(data)
            )

        """
        if result.is_failure:
            return r.fail(result.error or "Unknown error")
        return func(result.value)

    @staticmethod
    def fold_result[T, U](
        result: r[T],
        on_failure: Callable[[str], U],
        on_success: Callable[[T], U],
    ) -> U:
        """Fold FlextResult into single value (catamorphism).

        Allows handling both success and failure cases uniformly.

        Args:
            result: FlextResult to fold
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
        if result.is_failure:
            return on_failure(result.error or "Unknown error")
        return on_success(result.value)

    @staticmethod
    def tap_result[T](result: r[T], func: Callable[[T], None]) -> r[T]:
        """Execute side effect on success without changing result.

        Useful for logging, metrics, or other side effects.

        Args:
            result: FlextResult to tap
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
    def flow_through[T, U](
        result: r[T | U],
        *funcs: Callable[[T | U], r[T | U]],
    ) -> r[T | U]:
        """Chain multiple operations in a pipeline.

        Args:
            result: Initial result to start the pipeline
            *funcs: Functions to chain, each takes value and returns result

        Returns:
            Final result after all operations

        Example:
            result = u.flow_through(
                initial_result,
                validate_user,
                save_user,
                notify_user,
            )

        """
        current_result = result
        for func in funcs:
            if current_result.is_failure:
                break
            current_result = func(current_result.value)
        return current_result

    # ========================================================================
    # Pattern Matching Methods (Functional Programming)
    # ========================================================================

    @staticmethod
    def match(
        value: t.ConfigMapValue,
        *cases: tuple[
            type | t.ConfigMapValue | Callable[[t.ConfigMapValue], bool],
            t.ConfigMapValue | Callable[[t.ConfigMapValue], t.ConfigMapValue],
        ],
        default: t.ConfigMapValue
        | Callable[[t.ConfigMapValue], t.ConfigMapValue]
        | None = None,
    ) -> t.ConfigMapValue:
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
            >>> u.Reliability.match(
            ...     "REDACTED_LDAP_BIND_PASSWORD",
            ...     (str, lambda s: s.upper()),  # type match
            ...     (
            ...         "REDACTED_LDAP_BIND_PASSWORD",
            ...         "is_REDACTED_LDAP_BIND_PASSWORD",
            ...     ),  # value match
            ...     default="unknown",
            ... )
            'ADMIN'

            >>> u.Reliability.match(
            ...     15,
            ...     (lambda x: x > 10, "big"),  # predicate match
            ...     (lambda x: x > 5, "medium"),
            ...     default="small",
            ... )
            'big'

        """
        input_value: t.ConfigMapValue = value
        for pattern, result in cases:
            if isinstance(pattern, type) and isinstance(input_value, pattern):
                return FlextUtilitiesReliability._resolve_match_output(result, value)
            if pattern == input_value:
                return FlextUtilitiesReliability._resolve_match_output(
                    result,
                    input_value,
                )
            if FlextUtilitiesReliability._is_match_predicate(pattern):
                try:
                    pred_result = pattern(input_value)
                    if pred_result:
                        return FlextUtilitiesReliability._resolve_match_output(
                            result,
                            input_value,
                        )
                except (ValueError, TypeError, AttributeError):
                    pass
        # Default handling
        if default is not None:
            return FlextUtilitiesReliability._resolve_match_output(default, input_value)
        return None

    # Mnemonic alias
    mt = match


__all__ = ["FlextUtilitiesReliability"]
