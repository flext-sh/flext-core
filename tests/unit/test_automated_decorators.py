"""Real API tests for flext_core.decorators using flext_tests."""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest
from flext_tests import tm, tt
from hypothesis import given, settings, strategies as st

from flext_core import FlextDecorators, e


class TestAutomatedFlextDecorators:
    @pytest.mark.parametrize(
        ("case", "values"),
        [("small", (1, 2, 3)), ("larger", (5, 7, 12))],
        ids=lambda case: case[0],
    )
    def test_railway_wraps_success(
        self, case: str, values: tuple[int, int, int]
    ) -> None:
        @FlextDecorators.railway()
        def add(a: int, b: int) -> int:
            return a + b

        left, right, expected = values
        result = add(left, right)
        tm.ok(result, eq=expected)
        tm.that(case, none=False)

    def test_railway_wraps_exception(self) -> None:
        @FlextDecorators.railway(error_code="CALC_ERR")
        def parse(raw: str) -> int:
            return int(raw)

        result = parse("nan")
        tm.fail(result, has="invalid literal", code="CALC_ERR")

    def test_retry_retries_on_failure(self) -> None:
        call_count = 0

        @FlextDecorators.retry(max_attempts=3, delay_seconds=0.001)
        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                msg = "not yet"
                raise ValueError(msg)
            return "ok"

        result = flaky()
        tm.that(call_count, eq=3)
        tm.that(result, eq="ok")

    def test_timeout_enforces_deadline(self) -> None:
        @FlextDecorators.timeout(timeout_seconds=0.001)
        def slow() -> str:
            time.sleep(0.01)
            return "done"

        with pytest.raises(e.TimeoutError):
            slow()

    def test_deprecated_emits_warning(self) -> None:
        @FlextDecorators.deprecated("Use new_api")
        def old_api() -> str:
            return "legacy"

        with pytest.warns(DeprecationWarning, match="Use new_api"):
            value = old_api()
        tm.that(value, eq="legacy")

    def test_log_operation_and_track_operation(self) -> None:
        @FlextDecorators.log_operation(operation_name="logged_op")
        def logged() -> str:
            return "logged"

        @FlextDecorators.track_operation(operation_name="tracked_op")
        def tracked() -> str:
            return "tracked"

        tm.that(logged(), eq="logged")
        tm.that(tracked(), eq="tracked")

    def test_combined_uses_railway(self) -> None:
        @FlextDecorators.combined(operation_name="combined", use_railway=True)
        def multiply(a: int, b: int) -> int:
            return a * b

        result = multiply(3, 4)
        tm.ok(result, eq=12)

    @given(a=st.integers(), b=st.integers(min_value=1, max_value=1000))
    @settings(max_examples=50)
    def test_hypothesis_railway_division_always_returns_result(
        self, a: int, b: int
    ) -> None:
        @FlextDecorators.railway(error_code="DIV")
        def divide(x: int, y: int) -> float:
            return x / y

        result = divide(a, b)
        tm.ok(result)

    @pytest.mark.performance
    @pytest.mark.parametrize(
        "mode",
        [("raw", "raw"), ("railway", "railway")],
        ids=lambda case: case[0],
    )
    def test_railway_benchmark_overhead(
        self, mode: str, benchmark: Callable[..., object]
    ) -> None:
        raw_add = tt.op("add")

        @FlextDecorators.railway()
        def wrapped_add(a: int, b: int) -> int:
            return a + b

        if mode == "raw":
            raw_value = raw_add(21, 21)
            if isinstance(raw_value, int):
                tm.that(raw_value, eq=42)
            else:
                tm.that(False, eq=True)
            _ = benchmark(lambda: raw_add(21, 21))
            return
        tm.ok(wrapped_add(21, 21), eq=42)
        _ = benchmark(lambda: wrapped_add(21, 21))
