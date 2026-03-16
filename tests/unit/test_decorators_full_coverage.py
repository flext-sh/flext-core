"""Tests for Decorators full coverage."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated, cast

import pytest
from flext_tests import t as test_t, tm
from pydantic import BaseModel, Field

from flext_core import FlextContainer, FlextContext, FlextLogger, d, e, r
from tests import c, m, t


class _FakeLogger:
    def __init__(self) -> None:
        self.warning_calls: list[tuple[str, dict[str, t.Scalar]]] = []
        self.error_calls: list[tuple[str, dict[str, t.Scalar]]] = []
        self.exception_calls: list[tuple[str, dict[str, t.Scalar]]] = []
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


class _ObjWithLogger(BaseModel):
    logger: Annotated[object, Field(description="Logger instance holder")]


def test_deprecated_wrapper_emits_warning_and_returns_value() -> None:

    @d.deprecated("old API")
    def fn(value: str) -> str:
        return value.upper()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tm.that(fn("ok"), eq="OK")
    tm.that(any(w.category is DeprecationWarning for w in caught), eq=True)


def test_inject_sets_missing_dependency_from_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class _Container:
        @staticmethod
        def get(_name: str) -> r[str]:
            return r[str].ok("dep")

    monkeypatch.setattr(
        "flext_core.decorators.FlextContainer.create",
        lambda: _Container(),
    )

    @d.inject(dep="service.dep")
    def fn(*, dep: str = "fallback") -> str:
        return dep

    tm.that(fn(), eq="dep")


def test_log_operation_track_perf_exception_adds_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_logger = _FakeLogger()

    def _resolve_logger(
        _args: tuple[object, ...],
        _func: Callable[..., object],
    ) -> _FakeLogger:
        return fake_logger

    def _bind_operation_context(**_kwargs: t.Scalar) -> str:
        return "cid-1"

    def _clear_operation_scope(**_kwargs: t.Scalar) -> None:
        return None

    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._resolve_logger",
        _resolve_logger,
    )
    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._bind_operation_context",
        _bind_operation_context,
    )
    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._clear_operation_scope",
        _clear_operation_scope,
    )

    @d.log_operation("boom", track_perf=True)
    def fn() -> None:
        msg = "x"
        raise ValueError(msg)

    with pytest.raises(ValueError):
        fn()
    _message, kwargs = fake_logger.exception_calls[-1]
    tm.that("duration_ms" in kwargs, eq=True)
    tm.that("duration_seconds" in kwargs, eq=True)


def test_retry_unreachable_timeouterror_path(monkeypatch: pytest.MonkeyPatch) -> None:

    def _execute_retry_loop(*_args: t.Scalar, **_kwargs: t.Scalar) -> Exception:
        return ValueError("failed")

    def _handle_retry_exhaustion(*_args: t.Scalar, **_kwargs: t.Scalar) -> None:
        return None

    def _resolve_logger(
        _args: tuple[object, ...],
        _func: Callable[..., object],
    ) -> _FakeLogger:
        return _FakeLogger()

    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._execute_retry_loop",
        _execute_retry_loop,
    )
    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._handle_retry_exhaustion",
        _handle_retry_exhaustion,
    )
    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._resolve_logger",
        _resolve_logger,
    )

    @d.retry(max_attempts=1, error_code="X")
    def fn() -> str:
        return "ok"

    with pytest.raises(e.TimeoutError):
        fn()


def test_resolve_logger_prefers_logger_attribute() -> None:
    logger = FlextLogger(__name__)
    owner = _ObjWithLogger(logger=logger)

    def target() -> None:
        return None

    tm.that(d._resolve_logger((owner,), target) is logger, eq=True)


def test_execute_retry_loop_covers_default_linear_and_never_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_logger = _FakeLogger()

    def _sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("flext_core.decorators.time.sleep", _sleep)
    calls = {"n": 0}

    def flaky(*_args: t.Scalar, **_kwargs: t.Scalar) -> str:
        calls["n"] += 1
        msg = "nope"
        raise ValueError(msg)

    cfg = m.RetryConfiguration.model_validate(
        {
            "max_attempts": 2,
            "initial_delay_seconds": 0.01,
            "exponential_backoff": False,
        },
    )
    result_exc = d._execute_retry_loop(
        flaky,
        (),
        {},
        cast("FlextLogger", fake_logger),
        retry_config=cfg,
    )
    tm.that(isinstance(result_exc, Exception), eq=True)
    tm.that(calls["n"], eq=2)

    def _fake_retry_config(**_kw: test_t.Tests.object) -> SimpleNamespace:
        return SimpleNamespace(
            max_retries=0,
            initial_delay_seconds=0.1,
            exponential_backoff=False,
        )

    monkeypatch.setattr(
        "flext_core.decorators.m.RetryConfiguration",
        _fake_retry_config,
    )
    result_none = d._execute_retry_loop(
        lambda *_args, **_kwargs: "x",
        (),
        {},
        cast("FlextLogger", fake_logger),
        retry_config=None,
    )
    tm.that(isinstance(result_none, RuntimeError), eq=True)


def test_handle_retry_exhaustion_falsey_exception_reaches_timeout_error() -> None:

    class FalseyError(Exception):
        def __bool__(self) -> bool:
            return False

    fake_logger = _FakeLogger()

    def fn(*_args: t.Scalar, **_kwargs: t.Scalar) -> None:
        return None

    with pytest.raises(e.TimeoutError):
        d._handle_retry_exhaustion(
            FalseyError("x"),
            fn,
            2,
            None,
            cast("FlextLogger", fake_logger),
        )


def test_bind_operation_context_without_ensure_correlation_and_bind_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_logger = _FakeLogger()
    _ = FlextContext.Variables.CorrelationId.set("cid-existing")

    def _bind_context(*_args: t.Scalar, **_kwargs: t.Scalar) -> r[bool]:
        return r[bool].fail("bind-fail", error_code="E_BIND")

    monkeypatch.setattr("flext_core.decorators.FlextLogger.bind_context", _bind_context)
    cid = d._bind_operation_context(
        operation="op",
        logger=cast("FlextLogger", fake_logger),
        function_name="fn",
        ensure_correlation=False,
    )
    tm.that(cid, eq="cid-existing")
    tm.that(len(fake_logger.warning_calls) > 0, eq=True)


def test_clear_operation_scope_and_handle_log_result_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_logger = _FakeLogger()

    def _clear_scope(_scope: str) -> r[bool]:
        return r[bool].fail("clear-fail", error_code="E_CLR")

    monkeypatch.setattr("flext_core.decorators.FlextLogger.clear_scope", _clear_scope)
    d._clear_operation_scope(
        logger=cast("FlextLogger", fake_logger),
        function_name="fn",
        operation="op",
    )
    d._handle_log_result(
        result=r[bool].fail("x", error_code="E1"),
        logger=cast("FlextLogger", fake_logger),
        fallback_message="fallback",
        kwargs=m.ConfigMap(root={"extra": {"k": "v"}}),
    )
    d._handle_log_result(
        result=r[bool].fail("y", error_code="E2"),
        logger=cast("FlextLogger", fake_logger),
        fallback_message="fallback2",
        kwargs=m.ConfigMap(root={"extra": "not-a-dict"}),
    )
    tm.that(len(fake_logger.warning_calls) > 0, eq=True)


def test_handle_log_result_without_fallback_logger_and_non_dict_like_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class _NoFallback:
        logger: None = None

    d._handle_log_result(
        result=r[bool].fail("x"),
        logger=cast("FlextLogger", _NoFallback()),
        fallback_message="m",
        kwargs=m.ConfigMap(root={"extra": {"k": "v"}}),
    )
    fake_logger = _FakeLogger()

    def _is_dict_like(_value: t.Scalar) -> bool:
        return False

    monkeypatch.setattr(
        "flext_core.decorators.FlextRuntime.is_dict_like",
        _is_dict_like,
    )
    d._handle_log_result(
        result=r[bool].fail("x", error_code="E"),
        logger=cast("FlextLogger", fake_logger),
        fallback_message="fallback",
        kwargs=m.ConfigMap(root={"extra": {"k": "v"}}),
    )
    tm.that(len(fake_logger.warning_calls) > 0, eq=True)


def test_timeout_covers_exception_timeout_branch() -> None:

    @d.timeout(timeout_seconds=0.001, error_code="TMO")
    def fn() -> None:
        time.sleep(0.01)
        msg = "slow error"
        raise ValueError(msg)

    with pytest.raises(e.TimeoutError):
        fn()


def test_timeout_reraises_original_exception_when_within_limit() -> None:

    @d.timeout(timeout_seconds=2.0)
    def fn() -> None:
        msg = "fast-fail"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="fast-fail"):
        fn()


def test_combined_with_and_without_railway_uses_injection(
    clean_container: FlextContainer,
) -> None:
    _ = clean_container
    di = FlextContainer.create()
    _ = di.register("answer.service", 42)

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
    result = fn_railway()
    tm.that(getattr(result, "is_success", False), eq=True)
    tm.that(getattr(result, "value", None), eq=44)


def test_with_correlation_with_context_track_operation_and_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_calls: list[int] = []
    fake_logger = _FakeLogger()

    def _ensure_correlation_id() -> str:
        ensure_calls.append(1)
        return "cid-1"

    monkeypatch.setattr(
        "flext_core.decorators.FlextContext.Utilities.ensure_correlation_id",
        _ensure_correlation_id,
    )

    @d.with_correlation()
    def fn() -> str:
        return "ok"

    tm.that(fn(), eq="ok")
    tm.that(ensure_calls, eq=[1])

    def _resolve_logger(
        _args: tuple[object, ...],
        _func: Callable[..., object],
    ) -> _FakeLogger:
        return fake_logger

    def _bind_global_context(**_kwargs: t.Scalar) -> r[bool]:
        return r[bool].fail("bind", error_code="B")

    def _unbind_global_context(*_keys: str) -> r[bool]:
        return r[bool].fail("unbind", error_code="U")

    def _bind_operation_context(**_kwargs: t.Scalar) -> None:
        return None

    def _clear_operation_scope(**_kwargs: t.Scalar) -> None:
        return None

    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._resolve_logger",
        _resolve_logger,
    )
    monkeypatch.setattr(
        "flext_core.decorators.FlextLogger.bind_global_context",
        _bind_global_context,
    )
    monkeypatch.setattr(
        "flext_core.decorators.FlextLogger.unbind_global_context",
        _unbind_global_context,
    )

    @d.with_context(service="svc")
    def with_ctx() -> str:
        return "ctx"

    tm.that(with_ctx(), eq="ctx")
    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._bind_operation_context",
        _bind_operation_context,
    )
    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._clear_operation_scope",
        _clear_operation_scope,
    )

    @d.track_operation("tracked", track_correlation=True)
    def tracked() -> str:
        return "done"

    tm.that(tracked(), eq="done")

    @d.factory(name="svc.factory", singleton=True, lazy=False)
    def build(_value: BaseModel) -> BaseModel:
        return m.ConfigMap(root={"v": 7})

    tm.that(hasattr(build, c.Discovery.FACTORY_ATTR), eq=True)


def test_track_performance_success_and_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_logger = _FakeLogger()

    def _resolve_logger(
        _args: tuple[object, ...],
        _func: Callable[..., object],
    ) -> _FakeLogger:
        return fake_logger

    def _bind_operation_context(**_kwargs: t.Scalar) -> str:
        return "cid-perf"

    def _clear_operation_scope(**_kwargs: t.Scalar) -> None:
        return None

    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._resolve_logger",
        _resolve_logger,
    )
    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._bind_operation_context",
        _bind_operation_context,
    )
    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._clear_operation_scope",
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


def test_railway_and_retry_additional_paths(monkeypatch: pytest.MonkeyPatch) -> None:

    @d.railway()
    def already_result() -> r[int]:
        return r[int].ok(1)

    tm.ok(already_result())

    @d.railway(error_code=cast("str", 123))
    def fails() -> int:
        msg = "x"
        raise RuntimeError(msg)

    fail_result: r[int] = fails()
    tm.fail(fail_result)
    fake_logger = _FakeLogger()

    def _resolve_logger(
        _args: tuple[object, ...],
        _func: Callable[..., object],
    ) -> _FakeLogger:
        return fake_logger

    def _execute_retry_loop(*_args: t.Scalar, **_kwargs: t.Scalar) -> str:
        return "done"

    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._resolve_logger",
        _resolve_logger,
    )
    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._execute_retry_loop",
        _execute_retry_loop,
    )

    @d.retry(max_attempts=1)
    def retry_fn() -> str:
        return "ok"

    tm.that(retry_fn(), eq="done")


def test_execute_retry_exponential_and_handle_exhaustion_raise_last_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_logger = _FakeLogger()

    def _sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("flext_core.decorators.time.sleep", _sleep)
    calls = {"n": 0}

    def always_fails(*_args: t.Scalar, **_kwargs: t.Scalar) -> str:
        calls["n"] += 1
        msg = "fail"
        raise KeyError(msg)

    cfg = m.RetryConfiguration.model_validate(
        {
            "max_attempts": 2,
            "initial_delay_seconds": 0.01,
            "exponential_backoff": True,
        },
    )
    result = d._execute_retry_loop(
        always_fails,
        (),
        {},
        cast("FlextLogger", fake_logger),
        retry_config=cfg,
    )
    tm.that(isinstance(result, Exception), eq=True)
    tm.that(calls["n"], eq=2)

    def fn(*_args: t.Scalar, **_kwargs: t.Scalar) -> None:
        return None

    with pytest.raises(ValueError, match="last"):
        d._handle_retry_exhaustion(
            ValueError("last"),
            fn,
            2,
            "ERR",
            cast("FlextLogger", fake_logger),
        )


def test_timeout_additional_success_and_reraise_timeout_paths() -> None:

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


def test_timeout_raises_when_successful_call_exceeds_limit() -> None:

    @d.timeout(timeout_seconds=0.001, error_code="SLOW_OK")
    def slow_success() -> str:
        time.sleep(0.01)
        return "ok"

    with pytest.raises(e.TimeoutError):
        slow_success()
