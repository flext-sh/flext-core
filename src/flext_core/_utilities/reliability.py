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

_RELIABILITY_CONTAINER_DICT_ADAPTER = TypeAdapter(dict[str, t.ContainerValue])


class FlextUtilitiesReliability:
    """Reliability patterns for resilient, dispatcher-safe operations."""

    @property
    def logger(self) -> p.Log.StructlogLogger:
        """Get structlog logger via FlextRuntime (infrastructure-level, no FlextLogger)."""
        return FlextRuntime.get_logger(__name__)

    @staticmethod
    def _calculate_retry_delay(
        attempt: int, delay_seconds: float, backoff_multiplier: float
    ) -> float:
        """Calculate delay for retry attempt with exponential backoff."""
        current_delay = delay_seconds * backoff_multiplier**attempt
        return min(current_delay, c.Reliability.RETRY_BACKOFF_MAX)

    @staticmethod
    def _is_match_mapper(
        value: t.ContainerValue | Callable[[t.ContainerValue], t.ContainerValue],
    ) -> TypeGuard[Callable[[t.ContainerValue], t.ContainerValue]]:
        return callable(value)

    @staticmethod
    def _is_match_predicate(
        value: type | t.ContainerValue | Callable[[t.ContainerValue], bool],
    ) -> TypeGuard[Callable[[t.ContainerValue], bool]]:
        return callable(value) and (not isinstance(value, type))

    @staticmethod
    def _normalize_operation_mapping[TKey, TValue](
        value: Mapping[TKey, TValue],
    ) -> dict[str, t.ContainerValue]:
        normalized: dict[str, t.ContainerValue] = {}
        for raw_key, raw_value in value.items():
            normalized_value = (
                raw_value
                if FlextUtilitiesGuards.is_general_value_type(raw_value)
                else str(raw_value)
            )
            normalized[str(raw_key)] = normalized_value
        return normalized

    @staticmethod
    def _resolve_match_output(
        candidate: t.ContainerValue | Callable[[t.ContainerValue], t.ContainerValue],
        value: t.ContainerValue,
    ) -> t.ContainerValue:
        return candidate(value) if callable(candidate) else candidate

    @staticmethod
    def calculate_delay(
        attempt: int, config: Mapping[str, t.ContainerValue] | None
    ) -> float:
        """Calculate delay for retry attempt using configuration.

        Args:
            attempt: Current attempt number (0-based)
            config: Retry configuration object

        Returns:
            float: Delay in seconds for next attempt

        """
        if config is None:
            config = {}
        initial_delay_raw = config.get("initial_delay_seconds", 0.1)
        max_delay_raw = config.get("max_delay_seconds", 60.0)
        exponential_backoff_raw = config.get("exponential_backoff", False)
        backoff_multiplier_raw = config.get("backoff_multiplier")
        initial_delay = (
            float(initial_delay_raw)
            if isinstance(initial_delay_raw, int | float)
            and (not isinstance(initial_delay_raw, bool))
            else 0.1
        )
        max_delay = (
            float(max_delay_raw)
            if isinstance(max_delay_raw, int | float)
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
            and isinstance(backoff_multiplier_raw, int | float)
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
    def chain(
        value: t.ContainerValue, *funcs: Callable[[t.ContainerValue], t.ContainerValue]
    ) -> t.ContainerValue:
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
        current: t.ContainerValue = value
        for func in funcs:
            current = func(current)
        return current

    @staticmethod
    def compose(
        *funcs: Callable[[t.ContainerValue], t.ContainerValue], mode: str = "pipe"
    ) -> Callable[[t.ContainerValue], t.ContainerValue | r[t.ContainerValue]]:
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
                value: t.ContainerValue,
            ) -> t.ContainerValue | r[t.ContainerValue]:
                result = FlextUtilitiesReliability.pipe(value, *funcs)
                return result.value if result.is_success else result

            return piped
        if mode == "chain":
            return lambda value: FlextUtilitiesReliability.chain(value, *funcs)
        return lambda value: FlextUtilitiesReliability.flow(value, *funcs)

    @staticmethod
    def flow(
        value: t.ContainerValue,
        *ops: Mapping[str, t.ContainerValue]
        | Callable[[t.ContainerValue], t.ContainerValue],
    ) -> t.ContainerValue:
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
        current: t.ContainerValue = value
        for op in ops:
            if isinstance(op, Mapping):
                try:
                    op_dict = _RELIABILITY_CONTAINER_DICT_ADAPTER.validate_python(op)
                except ValidationError:
                    op_dict: dict[str, t.ContainerValue] = {}
                if FlextUtilitiesGuards.is_general_value_type(current):
                    current = FlextUtilitiesMapper.build(current, ops=op_dict)
            elif callable(op):
                current = op(current)
        return current

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
    def flow_through[T, U](
        result: r[T | U], *funcs: Callable[[T | U], r[T | U]]
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

    @staticmethod
    def fold_result[T, U](
        result: r[T], on_failure: Callable[[str], U], on_success: Callable[[T], U]
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
    def match(
        value: t.ContainerValue,
        *cases: tuple[
            type | t.ContainerValue | Callable[[t.ContainerValue], bool],
            t.ContainerValue | Callable[[t.ContainerValue], t.ContainerValue],
        ],
        default: t.ContainerValue
        | Callable[[t.ContainerValue], t.ContainerValue]
        | None = None,
    ) -> r[t.ContainerValue]:
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
        input_value: t.ContainerValue = value
        for pattern, result in cases:
            if isinstance(pattern, type) and isinstance(input_value, pattern):
                return r[t.ContainerValue].ok(
                    FlextUtilitiesReliability._resolve_match_output(result, value)
                )
            if pattern == input_value:
                return r[t.ContainerValue].ok(
                    FlextUtilitiesReliability._resolve_match_output(result, input_value)
                )
            if FlextUtilitiesReliability._is_match_predicate(pattern):
                try:
                    pred_result = pattern(input_value)
                    if pred_result:
                        return r[t.ContainerValue].ok(
                            FlextUtilitiesReliability._resolve_match_output(
                                result, input_value
                            )
                        )
                except (ValueError, TypeError, AttributeError):
                    pass
        if default is not None:
            return r[t.ContainerValue].ok(
                FlextUtilitiesReliability._resolve_match_output(default, input_value)
            )
        return r[t.ContainerValue].fail("No match found and no default provided")

    @staticmethod
    def pipe(
        value: t.ContainerValue,
        *operations: Callable[[t.ContainerValue], t.ContainerValue],
        on_error: str = "stop",
    ) -> r[t.ContainerValue]:
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
        current: t.ContainerValue = value
        for i, op in enumerate(operations):
            try:
                op_result = op(current)
                if FlextUtilitiesGuards.is_result_like(op_result):
                    if op_result.is_failure:
                        if on_error == "stop":
                            err_msg = op_result.error or "Unknown error"
                            return r[t.ContainerValue].fail(
                                f"Pipeline step {i} failed: {err_msg}"
                            )
                        continue
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
                    return r[t.ContainerValue].fail(f"Pipeline step {i} failed: {e}")
        return r.ok(current)

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
            r[TResult]: Result of operation with retry logic applied

        Fast fail: explicit default values instead of 'or' fallback.

        """
        delay_value: float | None = delay if delay is not None else delay_seconds
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
                f"Max attempts must be at least {c.Reliability.RETRY_COUNT_MIN}"
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
                        attempt, delay_seconds_value, backoff_multiplier_value
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
                        attempt, delay_seconds_value, backoff_multiplier_value
                    )
                    if current_delay > 0:
                        time.sleep(current_delay)
        return r[TResult].fail(
            f"Operation failed after {max_attempts_value} attempts: {last_error}"
        )

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
            return r[U].fail(result.error or "Unknown error")
        return func(result.value)

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
        operation: Callable[[], r[TTimeout]], timeout_seconds: float
    ) -> r[TTimeout]:
        """Execute an operation with a hard timeout using railway patterns."""
        if timeout_seconds <= c.INITIAL_TIME:
            return r[TTimeout].fail("Timeout must be positive")
        result_container: list[r[TTimeout] | None] = [None]
        exception_container: list[Exception | None] = [None]

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
                f"Operation timed out after {timeout_seconds} seconds"
            )
        if exception_container[0]:
            return r[TTimeout].fail(
                f"Operation failed with exception: {exception_container[0]}"
            )
        if result_container[0] is None:
            return r[TTimeout].fail("Operation completed but returned no result")
        return result_container[0]

    mt = match


__all__ = ["FlextUtilitiesReliability"]
