from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest

from flext_core import FlextContext, c, d, e, m, r, t, u


class _ResultLogger:
    def __init__(self) -> None:
        self.exceptions: list[tuple[str, dict[str, object]]] = []

    def debug(self, _msg: str, *_args: object, extra: dict[str, object]) -> None:
        _ = extra

    def exception(self, message: str, **kwargs: object) -> None:
        self.exceptions.append((message, kwargs))


class _FakeLogger:
    def __init__(self) -> None:
        self.warning_calls: list[tuple[str, dict[str, object]]] = []
        self.error_calls: list[tuple[str, dict[str, object]]] = []
        self.logger = self
        self.result_logger = _ResultLogger()

    def with_result(self) -> _ResultLogger:
        return self.result_logger

    def warning(self, message: str, **kwargs: object) -> None:
        self.warning_calls.append((message, kwargs))

    def error(self, message: str, **kwargs: object) -> None:
        self.error_calls.append((message, kwargs))

    def info(self, _message: str, **_kwargs: object) -> None:
        return None

    def debug(self, _message: str, **_kwargs: object) -> None:
        return None

    def exception(self, _message: str, **_kwargs: object) -> None:
        return None


@dataclass
class _ObjWithLogger:
    logger: object


def test_deprecated_wrapper_emits_warning_and_returns_value() -> None:
    @d.deprecated("old API")
    def fn(value: str) -> str:
        return value.upper()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert fn("ok") == "OK"

    assert any(w.category is DeprecationWarning for w in caught)


def test_inject_sets_missing_dependency_from_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Container:
        @staticmethod
        def get(_name: str) -> r[object]:
            return r[object].ok("dep")

    monkeypatch.setattr(
        "flext_core.decorators.FlextContainer.create", lambda: _Container()
    )

    @d.inject(dep="service.dep")
    def fn(*, dep: str = "fallback") -> str:
        return dep

    assert fn() == "dep"


def test_log_operation_track_perf_exception_adds_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_logger = _FakeLogger()

    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._resolve_logger",
        lambda _a, _f: fake_logger,
    )
    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._bind_operation_context",
        lambda **_kw: "cid-1",
    )
    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._clear_operation_scope",
        lambda **_kw: None,
    )

    @d.log_operation("boom", track_perf=True)
    def fn() -> None:
        raise ValueError("x")

    with pytest.raises(ValueError):
        fn()

    _message, kwargs = fake_logger.result_logger.exceptions[-1]
    assert "duration_ms" in kwargs
    assert "duration_seconds" in kwargs


def test_retry_unreachable_timeouterror_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._execute_retry_loop",
        lambda *_args, **_kwargs: ValueError("failed"),
    )
    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._handle_retry_exhaustion",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._resolve_logger",
        lambda _a, _f: _FakeLogger(),
    )

    @d.retry(max_attempts=1, error_code="X")
    def fn() -> str:
        return "ok"

    with pytest.raises(e.TimeoutError):
        fn()


def test_resolve_logger_prefers_logger_attribute() -> None:
    from flext_core.loggings import FlextLogger

    logger = FlextLogger(__name__)
    owner = _ObjWithLogger(logger=logger)

    def target() -> None:
        return None

    assert d._resolve_logger((owner,), target) is logger


def test_execute_retry_loop_covers_default_linear_and_never_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_logger = _FakeLogger()
    monkeypatch.setattr("flext_core.decorators.time.sleep", lambda _s: None)

    calls = {"n": 0}

    def flaky(*_args: object, **_kwargs: object) -> str:
        calls["n"] += 1
        raise ValueError("nope")

    cfg = m.RetryConfiguration(
        max_attempts=2,
        initial_delay_seconds=0.01,
        exponential_backoff=False,
    )
    result_exc = d._execute_retry_loop(
        flaky, tuple(), {}, cast(Any, fake_logger), retry_config=cfg
    )
    assert isinstance(result_exc, Exception)
    assert calls["n"] == 2

    monkeypatch.setattr(
        "flext_core.decorators.m.RetryConfiguration",
        lambda: SimpleNamespace(
            max_retries=0,
            initial_delay_seconds=0.1,
            exponential_backoff=False,
        ),
    )
    result_none = d._execute_retry_loop(
        lambda *_args, **_kwargs: "x",
        tuple(),
        {},
        cast(Any, fake_logger),
        retry_config=None,
    )
    assert isinstance(result_none, RuntimeError)


def test_handle_retry_exhaustion_falsey_exception_reaches_timeout_error() -> None:
    class FalseyError(Exception):
        def __bool__(self) -> bool:
            return False

    fake_logger = _FakeLogger()

    def fn(*_args: object, **_kwargs: object) -> None:
        return None

    with pytest.raises(e.TimeoutError):
        d._handle_retry_exhaustion(
            FalseyError("x"),
            fn,
            2,
            None,
            cast(Any, fake_logger),
        )


def test_bind_operation_context_without_ensure_correlation_and_bind_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_logger = _FakeLogger()
    _ = FlextContext.Variables.CorrelationId.set("cid-existing")

    monkeypatch.setattr(
        "flext_core.decorators.FlextLogger.bind_context",
        lambda *_args, **_kwargs: r[bool].fail("bind-fail", error_code="E_BIND"),
    )

    cid = d._bind_operation_context(
        operation="op",
        logger=cast(Any, fake_logger),
        function_name="fn",
        ensure_correlation=False,
    )
    assert cid == "cid-existing"
    assert fake_logger.warning_calls


def test_clear_operation_scope_and_handle_log_result_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_logger = _FakeLogger()

    monkeypatch.setattr(
        "flext_core.decorators.FlextLogger.clear_scope",
        lambda _scope: r[bool].fail("clear-fail", error_code="E_CLR"),
    )

    d._clear_operation_scope(
        logger=cast(Any, fake_logger), function_name="fn", operation="op"
    )

    d._handle_log_result(
        result=r[bool].fail("x", error_code="E1"),
        logger=cast(Any, fake_logger),
        fallback_message="fallback",
        kwargs=m.ConfigMap(root={"extra": {"k": "v"}}),
    )
    d._handle_log_result(
        result=r[bool].fail("y", error_code="E2"),
        logger=cast(Any, fake_logger),
        fallback_message="fallback2",
        kwargs=m.ConfigMap(root={"extra": "not-a-dict"}),
    )
    assert fake_logger.warning_calls


def test_handle_log_result_without_fallback_logger_and_non_dict_like_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoFallback:
        logger = None

    d._handle_log_result(
        result=r[bool].fail("x"),
        logger=cast(Any, _NoFallback()),
        fallback_message="m",
        kwargs=m.ConfigMap(root={"extra": {"k": "v"}}),
    )

    fake_logger = _FakeLogger()
    monkeypatch.setattr(
        "flext_core.decorators.FlextRuntime.is_dict_like", lambda _v: False
    )
    d._handle_log_result(
        result=r[bool].fail("x", error_code="E"),
        logger=cast(Any, fake_logger),
        fallback_message="fallback",
        kwargs=m.ConfigMap(root={"extra": {"k": "v"}}),
    )
    assert fake_logger.warning_calls


def test_timeout_covers_exception_timeout_branch() -> None:
    @d.timeout(timeout_seconds=0.001, error_code="TMO")
    def fn() -> None:
        time.sleep(0.01)
        raise ValueError("slow error")

    with pytest.raises(e.TimeoutError):
        fn()


def test_timeout_reraises_original_exception_when_within_limit() -> None:
    @d.timeout(timeout_seconds=2.0)
    def fn() -> None:
        raise ValueError("fast-fail")

    with pytest.raises(ValueError, match="fast-fail"):
        fn()


def test_combined_with_and_without_railway_uses_injection(
    clean_container: object,
) -> None:
    _ = clean_container

    from flext_core.container import FlextContainer

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

    assert fn_standard() == 43
    result = fn_railway()
    assert getattr(result, "is_success", False) is True
    assert getattr(result, "value", None) == 44


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

    assert fn() == "ok"
    assert ensure_calls == [1]

    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._resolve_logger",
        lambda _a, _f: fake_logger,
    )
    monkeypatch.setattr(
        "flext_core.decorators.FlextLogger.bind_global_context",
        lambda **_kw: r[bool].fail("bind", error_code="B"),
    )
    monkeypatch.setattr(
        "flext_core.decorators.FlextLogger.unbind_global_context",
        lambda *_ks: r[bool].fail("unbind", error_code="U"),
    )

    @d.with_context(service="svc")
    def with_ctx() -> str:
        return "ctx"

    assert with_ctx() == "ctx"

    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._bind_operation_context",
        lambda **_kw: None,
    )
    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._clear_operation_scope",
        lambda **_kw: None,
    )

    @d.track_operation("tracked", track_correlation=True)
    def tracked() -> str:
        return "done"

    assert tracked() == "done"

    @d.factory(name="svc.factory", singleton=True, lazy=False)
    def build(_value: t.ScalarValue) -> t.ScalarValue:
        return 7

    assert hasattr(build, c.Discovery.FACTORY_ATTR)


def test_track_performance_success_and_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_logger = _FakeLogger()
    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._resolve_logger",
        lambda _a, _f: fake_logger,
    )
    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._bind_operation_context",
        lambda **_kw: "cid-perf",
    )
    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._clear_operation_scope",
        lambda **_kw: None,
    )

    @d.track_performance("perf-op")
    def ok_fn() -> str:
        return "ok"

    @d.track_performance("perf-op-fail")
    def fail_fn() -> str:
        raise ValueError("boom")

    assert ok_fn() == "ok"
    with pytest.raises(ValueError):
        fail_fn()


def test_railway_and_retry_additional_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    @d.railway()
    def already_result() -> r[int]:
        return r[int].ok(1)

    assert already_result().is_success

    @d.railway(error_code=cast(Any, 123))
    def fails() -> int:
        raise RuntimeError("x")

    fail_result = cast(Any, fails())
    assert fail_result.is_failure

    fake_logger = _FakeLogger()
    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._resolve_logger",
        lambda _a, _f: fake_logger,
    )
    monkeypatch.setattr(
        "flext_core.decorators.FlextDecorators._execute_retry_loop",
        lambda *_a, **_k: "done",
    )

    @d.retry(max_attempts=1)
    def retry_fn() -> str:
        return "ok"

    assert retry_fn() == "done"


def test_execute_retry_exponential_and_handle_exhaustion_raise_last_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_logger = _FakeLogger()
    monkeypatch.setattr("flext_core.decorators.time.sleep", lambda _s: None)

    calls = {"n": 0}

    def always_fails(*_args: object, **_kwargs: object) -> str:
        calls["n"] += 1
        raise KeyError("fail")

    cfg = m.RetryConfiguration(
        max_attempts=2,
        initial_delay_seconds=0.01,
        exponential_backoff=True,
    )
    result = d._execute_retry_loop(
        always_fails,
        tuple(),
        {},
        cast(Any, fake_logger),
        retry_config=cfg,
    )
    assert isinstance(result, Exception)
    assert calls["n"] == 2

    def fn(*_args: object, **_kwargs: object) -> None:
        return None

    with pytest.raises(ValueError, match="last"):
        d._handle_retry_exhaustion(
            ValueError("last"),
            fn,
            2,
            "ERR",
            cast(Any, fake_logger),
        )


def test_timeout_additional_success_and_reraise_timeout_paths() -> None:
    @d.timeout(timeout_seconds=1.0)
    def quick() -> str:
        return "fast"

    assert quick() == "fast"

    @d.timeout(timeout_seconds=1.0)
    def raises_timeout() -> None:
        raise e.TimeoutError("already-timeout")

    with pytest.raises(e.TimeoutError, match="already-timeout"):
        raises_timeout()


def test_timeout_raises_when_successful_call_exceeds_limit() -> None:
    @d.timeout(timeout_seconds=0.001, error_code="SLOW_OK")
    def slow_success() -> str:
        time.sleep(0.01)
        return "ok"

    with pytest.raises(e.TimeoutError):
        slow_success()
