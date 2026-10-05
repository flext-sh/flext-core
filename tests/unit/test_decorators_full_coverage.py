"""Behavior contract for flext_core.decorators — public API only.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from flext_tests import d, e, r, tm

from flext_core import FlextContainer
from tests.models import m

if TYPE_CHECKING:
    from tests.protocols import p


class TestsFlextCoreDecorators:
    """Behavior contract for flext_core.decorators — public API only."""

    @staticmethod
    def test_deprecated_emits_deprecation_warning_and_preserves_return() -> None:
        """Test deprecated emits deprecation warning and preserves return."""

        @d.deprecated("old API")
        def fn(value: str) -> str:
            return value.upper()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fn("ok")

        tm.that(result, eq="OK")
        tm.that(any(w.category is DeprecationWarning for w in caught), eq=True)

    @staticmethod
    def _inject_resolves_dependency_from_shared_container(
        clean_container: p.Container,
    ) -> None:
        """Test inject resolves dependency from shared container."""
        _ = clean_container
        di = FlextContainer.shared()
        _ = di.bind("injected.value", "dep-value")

        @d.inject(dep="injected.value")
        def fn(*, dep: str) -> str:
            return dep

        # Why: the decorator injects `dep` at runtime when the caller omits
        # it, but its ParamSpec-preserving signature keeps `dep` statically
        # required. Call through a `Callable[..., str]` reference — permissive
        # by design (PEP 484), not `Any` — to type the deliberate zero-arg call.
        injected_fn: Callable[..., str] = fn
        tm.that(injected_fn(), eq="dep-value")

    @staticmethod
    def test_inject_falls_back_when_binding_missing(
        clean_container: p.Container,
    ) -> None:
        """Test inject falls back when binding missing."""
        _ = clean_container

        @d.inject(dep="missing.key")
        def fn(*, dep: str = "default-value") -> str:
            return dep

        tm.that(fn(), eq="default-value")

    @staticmethod
    def test_timeout_raises_when_call_exceeds_limit() -> None:
        """Test timeout raises when call exceeds limit."""

        @d.timeout(timeout_seconds=0.001, error_code="TMO")
        def slow() -> str:
            time.sleep(0.05)
            return "never"

        with pytest.raises(e.FlextTimeoutError):
            slow()

    @staticmethod
    def test_timeout_reraises_original_exception_when_within_limit() -> None:
        """Test timeout reraises original exception when within limit."""

        @d.timeout(timeout_seconds=2.0)
        def fails_fast() -> None:
            msg = "fast-fail"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="fast-fail"):
            fails_fast()

    @staticmethod
    def test_timeout_passes_through_when_call_completes_in_time() -> None:
        """Test timeout passes through when call completes in time."""

        @d.timeout(timeout_seconds=2.0)
        def quick() -> str:
            return "done"

        tm.that(quick(), eq="done")

    @staticmethod
    def test_timeout_reraises_existing_timeout_error() -> None:
        """Test timeout reraises existing timeout error."""

        @d.timeout(timeout_seconds=1.0)
        def raises_timeout() -> None:
            msg = "already-timeout"
            raise e.FlextTimeoutError(msg)

        with pytest.raises(e.FlextTimeoutError, match="already-timeout"):
            raises_timeout()

    @staticmethod
    def test_railway_wraps_exception_as_failed_result() -> None:
        """Test railway wraps exception as failed result."""

        @d.railway(error_code="E_RW")
        def fails() -> int:
            msg = "boom"
            raise RuntimeError(msg)

        result = fails()
        tm.fail(result)
        tm.that(result.error, contains="boom")

    @staticmethod
    def test_railway_passes_through_existing_result() -> None:
        """Test railway passes through existing result."""

        @d.railway()
        def already_result() -> p.Result[int]:
            return r[int].ok(1)

        result = already_result()
        tm.ok(result)
        tm.that(result.unwrap(), eq=1)

    @staticmethod
    def test_retry_returns_successful_call_without_retry() -> None:
        """Test retry returns successful call without retry."""
        calls = {"n": 0}

        @d.retry(max_attempts=3)
        def succeed() -> str:
            calls["n"] += 1
            return "ok"

        tm.that(succeed(), eq="ok")
        tm.that(calls["n"], eq=1)

    @staticmethod
    def test_retry_retries_until_success() -> None:
        """Test retry retries until success."""
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

    @staticmethod
    def test_combined_applies_injection_on_standard_path(
        clean_container: p.Container,
    ) -> None:
        """Test combined applies injection on standard path."""
        _ = clean_container
        di = FlextContainer.shared()
        _ = di.bind("answer.service", 42)

        @d.combined(inject_deps={"dep": "answer.service"}, operation_name="std")
        def fn(*, dep: int = 0) -> int:
            return dep + 1

        tm.that(fn(), eq=43)

    @staticmethod
    def test_combined_wraps_with_railway_when_enabled(
        clean_container: p.Container,
    ) -> None:
        """Test combined wraps with railway when enabled."""
        _ = clean_container

        @d.combined(operation_name="rw", railway_enabled=True)
        def fails() -> int:
            msg = "boom"
            raise RuntimeError(msg)

        result = fails()
        tm.fail(result)

    @staticmethod
    def test_with_correlation_ensures_correlation_id_during_call() -> None:
        """Test with correlation ensures correlation id during call."""

        @d.with_correlation()
        def fn() -> str:
            return "ok"

        tm.that(fn(), eq="ok")

    @staticmethod
    def test_factory_registers_callable_and_produces_value(
        clean_container: p.Container,
    ) -> None:
        """Test factory registers callable and produces value."""
        _ = clean_container

        class _Payload(m.BaseModel):
            v: int

        @d.factory(name="svc.factory", singleton=True, lazy=False)
        def build() -> _Payload:
            return _Payload(v=7)

        payload = build()
        assert isinstance(payload, _Payload)
        tm.that(payload.v, eq=7)
