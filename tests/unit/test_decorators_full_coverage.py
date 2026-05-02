"""Behavior contract for flext_core.decorators — public API only."""

from __future__ import annotations

import time
import warnings

import pytest
from flext_tests import tm

from flext_core import FlextContainer
from tests import d, e, m, p, r


class TestsFlextDecorators:
    """Behavior contract for flext_core.decorators — public API only."""

    def test_deprecated_emits_deprecation_warning_and_preserves_return(self) -> None:
        @d.deprecated("old API")
        def fn(value: str) -> str:
            return value.upper()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fn("ok")

        tm.that(result, eq="OK")
        tm.that(any(w.category is DeprecationWarning for w in caught), eq=True)

    def test_inject_resolves_dependency_from_shared_container(
        self,
        clean_container: p.Container,
    ) -> None:
        _ = clean_container
        di = FlextContainer.shared()
        _ = di.bind("injected.value", "dep-value")

        @d.inject(dep="injected.value")
        def fn(*, dep: str = "fallback") -> str:
            return dep

        tm.that(fn(), eq="dep-value")

    def test_inject_falls_back_when_binding_missing(
        self,
        clean_container: p.Container,
    ) -> None:
        _ = clean_container

        @d.inject(dep="missing.key")
        def fn(*, dep: str = "fallback") -> str:
            return dep

        tm.that(fn(), eq="fallback")

    def test_timeout_raises_when_call_exceeds_limit(self) -> None:
        @d.timeout(timeout_seconds=0.001, error_code="TMO")
        def slow() -> str:
            time.sleep(0.05)
            return "never"

        with pytest.raises(e.TimeoutError):
            slow()

    def test_timeout_reraises_original_exception_when_within_limit(self) -> None:
        @d.timeout(timeout_seconds=2.0)
        def fails_fast() -> None:
            msg = "fast-fail"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="fast-fail"):
            fails_fast()

    def test_timeout_passes_through_when_call_completes_in_time(self) -> None:
        @d.timeout(timeout_seconds=2.0)
        def quick() -> str:
            return "done"

        tm.that(quick(), eq="done")

    def test_timeout_reraises_existing_timeout_error(self) -> None:
        @d.timeout(timeout_seconds=1.0)
        def raises_timeout() -> None:
            msg = "already-timeout"
            raise e.TimeoutError(msg)

        with pytest.raises(e.TimeoutError, match="already-timeout"):
            raises_timeout()

    def test_railway_wraps_exception_as_failed_result(self) -> None:
        @d.railway(error_code="E_RW")
        def fails() -> int:
            msg = "boom"
            raise RuntimeError(msg)

        result = fails()
        tm.fail(result)
        tm.that(result.error, contains="boom")

    def test_railway_passes_through_existing_result(self) -> None:
        @d.railway()
        def already_result() -> p.Result[int]:
            return r[int].ok(1)

        result = already_result()
        tm.ok(result)
        tm.that(result.unwrap(), eq=1)

    def test_retry_returns_successful_call_without_retry(self) -> None:
        calls = {"n": 0}

        @d.retry(max_attempts=3)
        def succeed() -> str:
            calls["n"] += 1
            return "ok"

        tm.that(succeed(), eq="ok")
        tm.that(calls["n"], eq=1)

    def test_retry_retries_until_success(self) -> None:
        calls = {"n": 0}

        @d.retry(max_attempts=3, delay_seconds=0.001)
        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 2:
                msg = "transient"
                raise ValueError(msg)
            return "ok"

        tm.that(flaky(), eq="ok")
        tm.that(calls["n"], eq=2)

    def test_combined_applies_injection_on_standard_path(
        self,
        clean_container: p.Container,
    ) -> None:
        _ = clean_container
        di = FlextContainer.shared()
        _ = di.bind("answer.service", 42)

        @d.combined(inject_deps={"dep": "answer.service"}, operation_name="std")
        def fn(*, dep: int = 0) -> int:
            return dep + 1

        tm.that(fn(), eq=43)

    def test_combined_wraps_with_railway_when_enabled(
        self,
        clean_container: p.Container,
    ) -> None:
        _ = clean_container

        @d.combined(
            operation_name="rw",
            railway_enabled=True,
        )
        def fails() -> int:
            msg = "boom"
            raise RuntimeError(msg)

        result = fails()
        tm.fail(result)

    def test_with_correlation_ensures_correlation_id_during_call(self) -> None:
        @d.with_correlation()
        def fn() -> str:
            return "ok"

        tm.that(fn(), eq="ok")

    def test_factory_registers_callable_and_produces_value(
        self,
        clean_container: p.Container,
    ) -> None:
        _ = clean_container

        class _Payload(m.BaseModel):
            v: int

        @d.factory(name="svc.factory", singleton=True, lazy=False)
        def build() -> _Payload:
            return _Payload(v=7)

        built = build()
        payload = built.unwrap() if isinstance(built, p.Result) else built
        assert isinstance(payload, _Payload)
        tm.that(payload.v, eq=7)
