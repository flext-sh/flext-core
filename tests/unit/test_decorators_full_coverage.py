"""Tests for Decorators full coverage."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Mapping, MutableSequence
from typing import Annotated, ClassVar

import pytest

import flext_core as core_decorators
from flext_core import (
    FlextContainer,
    FlextContext,
)
from flext_tests import tm
from tests import d, e, m, p, r, t, u


class TestDecoratorsFullCoverage:
    class _FakeLogger:
        def __init__(self) -> None:
            self.warning_calls: MutableSequence[tuple[str, t.ConfigurationMapping]] = []
            self.error_calls: MutableSequence[tuple[str, t.ConfigurationMapping]] = []
            self.exception_calls: MutableSequence[
                tuple[str, t.ConfigurationMapping]
            ] = []
            self.logger = self

        def warning(self, message: str, **kwargs: t.Scalar) -> None:
            self.warning_calls.append((message, kwargs))

        def error(self, message: str, **kwargs: t.Scalar) -> None:
            self.error_calls.append((message, kwargs))

        def info(self, _message: str, **_kwargs: t.Scalar) -> None:
            return None

        def debug(self, _message: str, *_args: t.Scalar, **_kwargs: t.Scalar) -> None:
            return None

        def exception(self, message: str, **kwargs: t.Scalar) -> None:
            self.exception_calls.append((message, kwargs))

    class _ObjWithLogger(m.BaseModel):
        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            arbitrary_types_allowed=True
        )
        logger: Annotated[p.Logger, m.Field(description="Logger instance holder")]

    def test_deprecated_wrapper_emits_warning_and_returns_value(self) -> None:
        @d.deprecated("old API")
        def fn(value: str) -> str:
            return value.upper()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tm.that(fn("ok"), eq="OK")
        tm.that(any(w.category is DeprecationWarning for w in caught), eq=True)

    def test_inject_sets_missing_dependency_from_container(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _ = self

        class _Container:
            @staticmethod
            def resolve(_name: str) -> p.Result[str]:
                return r[str].ok("dep")

        monkeypatch.setattr(
            core_decorators.FlextContainer,
            "shared",
            lambda: _Container(),
        )

        @d.inject(dep="service.dep")
        def fn(*, dep: str = "fallback") -> str:
            return dep

        tm.that(fn(), eq="dep")

    def test_log_operation_track_perf_exception_adds_duration(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_logger = self._FakeLogger()

        def _bind_operation_context(**_kwargs: t.Scalar) -> str:
            return "cid-1"

        def _clear_operation_scope(**_kwargs: t.Scalar) -> None:
            return None

        def _logger_factory(_module: str) -> TestDecoratorsFullCoverage._FakeLogger:
            return fake_logger

        monkeypatch.setattr(
            core_decorators.FlextDecorators,
            "_bind_operation_context",
            staticmethod(_bind_operation_context),
        )
        monkeypatch.setattr(
            core_decorators.FlextDecorators,
            "_clear_operation_scope",
            staticmethod(_clear_operation_scope),
        )
        monkeypatch.setattr(
            core_decorators.u,
            "fetch_logger",
            _logger_factory,
        )

        @d.log_operation("boom", track_perf=True)
        def fn() -> None:
            msg = "x"
            raise ValueError(msg)

        with pytest.raises(ValueError):
            fn()
        assert fake_logger.exception_calls, "No exception calls captured"
        _message, kwargs = fake_logger.exception_calls[-1]
        assert "duration_ms" in kwargs, (
            f"Missing duration_ms in kwargs. "
            f"All calls: {[(m, list(k.keys())) for m, k in fake_logger.exception_calls]}"
        )
        assert "duration_seconds" in kwargs

    def test_retry_unreachable_timeouterror_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _ = self

        def _execute_retry_loop(
            _call: Callable[[], str],
            _func_name: str,
            _logger: p.Logger,
            *,
            retry_settings: m.RetryConfiguration,
        ) -> Exception:
            _ = retry_settings
            return ValueError("failed")

        def _handle_retry_exhaustion(*_args: t.Scalar, **_kwargs: t.Scalar) -> None:
            return None

        def _logger_factory(_module: str) -> TestDecoratorsFullCoverage._FakeLogger:
            return TestDecoratorsFullCoverage._FakeLogger()

        monkeypatch.setattr(
            core_decorators.d,
            "_execute_retry_loop",
            _execute_retry_loop,
        )
        monkeypatch.setattr(
            core_decorators.d,
            "_handle_retry_exhaustion",
            _handle_retry_exhaustion,
        )
        monkeypatch.setattr(
            core_decorators.u,
            "fetch_logger",
            _logger_factory,
        )

        @d.retry(max_attempts=1, error_code="X")
        def fn() -> str:
            return "ok"

        result = fn()
        tm.that(result, is_=Exception)
        tm.that(str(result), has="failed")

    def test_resolve_logger_prefers_logger_attribute(self) -> None:
        logger = u.create_module_logger(__name__)
        owner = self._ObjWithLogger(logger=logger)

        def target() -> str:
            return "ok"

        tm.that(d._resolve_logger(owner, func=target) is logger, eq=True)

    def test_execute_retry_loop_covers_default_linear_and_never_ran(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_logger = self._FakeLogger()

        def _sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr(time, "sleep", _sleep)
        calls = {"n": 0}

        def flaky() -> str:
            calls["n"] += 1
            msg = "nope"
            raise ValueError(msg)

        cfg = m.RetryConfiguration.model_validate(
            {
                "max_retries": 2,
                "initial_delay_seconds": 0.01,
                "exponential_backoff": False,
                "retry_on_exceptions": [],
                "retry_on_status_codes": [],
            },
        )
        execute_retry_loop = getattr(d, "_execute_retry_loop")
        result_exc = execute_retry_loop(
            flaky,
            "flaky",
            fake_logger,
            retry_settings=cfg,
        )
        tm.that(result_exc, is_=Exception)
        tm.that(calls["n"], eq=2)

    def test_handle_retry_exhaustion_falsey_exception_reaches_timeout_error(
        self,
    ) -> None:
        class FalseyError(Exception):
            def __bool__(self) -> bool:
                return False

        fake_logger = self._FakeLogger()

        def fn(*_args: t.Scalar, **_kwargs: t.Scalar) -> None:
            return None

        handle_retry_exhaustion = getattr(d, "_handle_retry_exhaustion")
        with pytest.raises(e.TimeoutError):
            handle_retry_exhaustion(
                FalseyError("x"),
                fn,
                2,
                None,
                fake_logger,
            )

    def test_bind_operation_context_without_ensure_correlation_and_bind_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_logger = self._FakeLogger()
        _ = FlextContext.Variables.CorrelationId.set("cid-existing")

        def _bind_context(*_args: t.Scalar, **_kwargs: t.Scalar) -> p.Result[bool]:
            return r[bool].fail("bind-fail", error_code="E_BIND")

        monkeypatch.setattr(
            core_decorators.u,
            "bind_context",
            _bind_context,
        )
        bind_operation_context = getattr(d, "_bind_operation_context")
        cid = bind_operation_context(
            operation="op",
            logger=fake_logger,
            function_name="fn",
            ensure_correlation=False,
        )
        tm.that(cid, eq="cid-existing")
        tm.that(len(fake_logger.warning_calls), gt=0)

    def test_clear_operation_scope_logs_warning_on_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_logger = self._FakeLogger()

        def _clear_scope(_scope: str) -> p.Result[bool]:
            return r[bool].fail("clear-fail", error_code="E_CLR")

        monkeypatch.setattr(
            core_decorators.u,
            "clear_scope",
            _clear_scope,
        )
        clear_operation_scope = getattr(d, "_clear_operation_scope")
        clear_operation_scope(
            logger=fake_logger,
            function_name="fn",
            operation="op",
        )
        tm.that(len(fake_logger.warning_calls), gt=0)

    def test_timeout_covers_exception_timeout_branch(self) -> None:
        @d.timeout(timeout_seconds=0.001, error_code="TMO")
        def fn() -> None:
            time.sleep(0.01)
            msg = "slow error"
            raise ValueError(msg)

        with pytest.raises(e.TimeoutError):
            fn()

    def test_timeout_reraises_original_exception_when_within_limit(self) -> None:
        @d.timeout(timeout_seconds=2.0)
        def fn() -> None:
            msg = "fast-fail"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="fast-fail"):
            fn()

    def test_combined_with_and_without_railway_uses_injection(
        self,
        clean_container: p.Container,
    ) -> None:
        _ = self
        _ = clean_container
        di = FlextContainer.shared()
        _ = di.bind("answer.service", 42)

        @d.combined(inject_deps={"dep": "answer.service"}, operation_name="std")
        def fn_standard(*, dep: int = 0) -> int:
            return dep + 1

        @d.combined(
            inject_deps={"dep": "answer.service"},
            operation_name="rw",
            use_railway=True,
        )
        def fn_railway(*, dep: int = 0) -> int:
            return dep + 2

        tm.that(fn_standard(), eq=43)
        fn_railway()

    def test_with_correlation_with_context_log_operation_and_factory(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _ = self
        ensure_calls: MutableSequence[int] = []
        fake_logger = self._FakeLogger()

        def _ensure_correlation_id() -> str:
            ensure_calls.append(1)
            return "cid-1"

        monkeypatch.setattr(
            core_decorators.FlextContext.Utilities,
            "ensure_correlation_id",
            _ensure_correlation_id,
        )

        @d.with_correlation()
        def fn() -> str:
            return "ok"

        tm.that(fn(), eq="ok")
        tm.that(ensure_calls, eq=[1])

        def _bind_global_context(**_kwargs: t.Scalar) -> p.Result[bool]:
            return r[bool].fail("bind", error_code="B")

        def _unbind_global_context(*_keys: str) -> p.Result[bool]:
            return r[bool].fail("unbind", error_code="U")

        def _bind_operation_context(**_kwargs: t.Scalar) -> None:
            return None

        def _clear_operation_scope(**_kwargs: t.Scalar) -> None:
            return None

        def _logger_factory(_module: str) -> TestDecoratorsFullCoverage._FakeLogger:
            return fake_logger

        setattr(_logger_factory, "bind_global_context", _bind_global_context)
        setattr(_logger_factory, "unbind_global_context", _unbind_global_context)

        @d.with_context(service="svc")
        def with_ctx() -> str:
            return "ctx"

        tm.that(with_ctx(), eq="ctx")
        monkeypatch.setattr(
            core_decorators.d,
            "_bind_operation_context",
            _bind_operation_context,
        )
        monkeypatch.setattr(
            core_decorators.d,
            "_clear_operation_scope",
            _clear_operation_scope,
        )

        @d.log_operation("tracked", ensure_correlation=True)
        def tracked() -> str:
            return "done"

        tm.that(tracked(), eq="done")

        class _FactoryPayload(m.BaseModel):
            v: int

        @d.factory(name="svc.factory", singleton=True, lazy=False)
        def build(_value: m.BaseModel) -> m.BaseModel:
            return _FactoryPayload(v=7)

        built = build(m.ConfigMap(root={"v": 1}))
        built_value = built.value if hasattr(built, "value") else built
        if isinstance(built_value, _FactoryPayload):
            tm.that(built_value.v, eq=7)
        elif isinstance(built_value, Mapping):
            tm.that(built_value.get("v"), eq=7)
        else:
            pytest.fail(f"Unexpected build() result type: {type(built_value)!r}")

    def test_track_performance_success_and_failure_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_logger = self._FakeLogger()

        def _bind_operation_context(**_kwargs: t.Scalar) -> str:
            return "cid-perf"

        def _clear_operation_scope(**_kwargs: t.Scalar) -> None:
            return None

        def _logger_factory(_module: str) -> TestDecoratorsFullCoverage._FakeLogger:
            return fake_logger

        monkeypatch.setattr(
            core_decorators.u,
            "fetch_logger",
            _logger_factory,
        )
        monkeypatch.setattr(
            core_decorators.d,
            "_bind_operation_context",
            _bind_operation_context,
        )
        monkeypatch.setattr(
            core_decorators.d,
            "_clear_operation_scope",
            _clear_operation_scope,
        )

        @d.log_operation("perf-op")
        def ok_fn() -> str:
            return "ok"

        @d.log_operation("perf-op-fail")
        def fail_fn() -> str:
            msg = "boom"
            raise ValueError(msg)

        tm.that(ok_fn(), eq="ok")
        with pytest.raises(ValueError):
            fail_fn()

    def test_railway_and_retry_additional_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _ = self

        @d.railway()
        def already_result() -> p.Result[int]:
            return r[int].ok(1)

        tm.ok(already_result())

        railway_fn = getattr(d, "railway")

        @railway_fn(error_code="123")
        def fails() -> int:
            msg = "x"
            raise RuntimeError(msg)

        fail_result = fails()
        tm.fail(fail_result)
        fake_logger = self._FakeLogger()

        def _execute_retry_loop(
            _call: Callable[[], str],
            _func_name: str,
            _logger: p.Logger,
            *,
            retry_settings: m.RetryConfiguration,
        ) -> str:
            _ = retry_settings
            return "done"

        def _logger_factory(_module: str) -> TestDecoratorsFullCoverage._FakeLogger:
            return fake_logger

        monkeypatch.setattr(
            core_decorators.d,
            "_execute_retry_loop",
            _execute_retry_loop,
        )
        monkeypatch.setattr(
            core_decorators.u,
            "fetch_logger",
            _logger_factory,
        )

        @d.retry(max_attempts=1)
        def retry_fn() -> str:
            return "ok"

        tm.that(retry_fn(), eq="done")

    def test_execute_retry_exponential_and_handle_exhaustion_raise_last_exception(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_logger = self._FakeLogger()

        def _sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr(time, "sleep", _sleep)
        calls = {"n": 0}

        def always_fails() -> str:
            calls["n"] += 1
            msg = "fail"
            raise KeyError(msg)

        cfg = m.RetryConfiguration.model_validate(
            {
                "max_retries": 2,
                "initial_delay_seconds": 0.01,
                "exponential_backoff": True,
                "retry_on_exceptions": [],
                "retry_on_status_codes": [],
            },
        )
        execute_retry_loop2 = getattr(d, "_execute_retry_loop")
        result = execute_retry_loop2(
            always_fails,
            "always_fails",
            fake_logger,
            retry_settings=cfg,
        )
        tm.that(result, is_=Exception)
        tm.that(calls["n"], eq=2)

        def fn(*_args: t.Scalar, **_kwargs: t.Scalar) -> None:
            return None

        handle_retry_exhaustion2 = getattr(d, "_handle_retry_exhaustion")
        with pytest.raises(e.TimeoutError, match="failed after 2 attempts"):
            handle_retry_exhaustion2(
                ValueError("last"),
                fn,
                2,
                "ERR",
                fake_logger,
            )

    def test_timeout_additional_success_and_reraise_timeout_paths(self) -> None:
        @d.timeout(timeout_seconds=1.0)
        def quick() -> str:
            return "fast"

        tm.that(quick(), eq="fast")

        @d.timeout(timeout_seconds=1.0)
        def raises_timeout() -> None:
            msg = "already-timeout"
            raise e.TimeoutError(msg)

        with pytest.raises(e.TimeoutError, match="already-timeout"):
            raises_timeout()

    def test_timeout_raises_when_successful_call_exceeds_limit(self) -> None:
        @d.timeout(timeout_seconds=0.001, error_code="SLOW_OK")
        def slow_success() -> str:
            time.sleep(0.01)
            return "ok"

        with pytest.raises(e.TimeoutError):
            slow_success()
