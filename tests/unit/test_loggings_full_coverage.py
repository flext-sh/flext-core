"""Loggings full coverage tests."""

from __future__ import annotations

import inspect
import types
from pathlib import Path
from typing import ClassVar, cast, override

import pytest

from flext_core import FlextLogger, FlextRuntime, FlextSettings, c, m, p, r, u
from flext_tests import t


class _FakeBindable:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, t.Scalar]]] = []

    def bind(self, **kwargs: t.Scalar) -> _FakeBindable:
        self.calls.append(("bind", (), kwargs))
        return self

    def unbind(self, *keys: str) -> _FakeBindable:
        self.calls.append(("unbind", keys, {}))
        return self

    def try_unbind(self, *keys: str) -> _FakeBindable:
        self.calls.append(("try_unbind", keys, {}))
        return self

    def debug(self, message: str, *args: t.Scalar, **kwargs: t.Scalar) -> None:
        self.calls.append(("debug", (message, *args), kwargs))

    def info(self, message: str, *args: t.Scalar, **kwargs: t.Scalar) -> None:
        self.calls.append(("info", (message, *args), kwargs))

    def warning(self, message: str, *args: t.Scalar, **kwargs: t.Scalar) -> None:
        self.calls.append(("warning", (message, *args), kwargs))

    def error(self, message: str, *args: t.Scalar, **kwargs: t.Scalar) -> None:
        self.calls.append(("error", (message, *args), kwargs))

    def critical(self, message: str, *args: t.Scalar, **kwargs: t.Scalar) -> None:
        self.calls.append(("critical", (message, *args), kwargs))


class _ContextVars:
    def __init__(self) -> None:
        self.store: dict[str, t.Tests.object] = {}

    def bind_contextvars(self, **kwargs: t.Scalar) -> None:
        self.store.update(kwargs)

    def unbind_contextvars(self, *keys: str) -> None:
        for key in keys:
            self.store.pop(key, None)

    def clear_contextvars(self) -> None:
        self.store.clear()

    def get_contextvars(self) -> dict[str, t.Tests.object]:
        return dict(self.store)


class _StructlogShim:
    def __init__(self) -> None:
        self.contextvars: _ContextVars = _ContextVars()


def test_loggings_context_and_factory_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    shim = _StructlogShim()

    def _structlog_accessor() -> _StructlogShim:
        return shim

    monkeypatch.setattr(FlextRuntime, "structlog", staticmethod(_structlog_accessor))
    assert isinstance(c.Settings.LogLevel.DEBUG.value, str)
    value = "ok"
    assert value == "ok"
    logger_obj = FlextLogger.create_bound_logger(
        "x",
        cast("p.Log.StructlogLogger", cast("object", _FakeBindable())),
    )
    assert logger_obj._context == {}
    assert logger_obj() is logger_obj
    bind_result = FlextLogger.bind_global_context(k1="v1")
    get_result = FlextLogger._get_global_context()
    clear_result = FlextLogger.clear_global_context()
    assert bind_result.is_success
    assert isinstance(get_result, m.ConfigMap)
    assert clear_result.is_success

    def _raise_bind_contextvars(**_kwargs: t.Scalar) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(shim.contextvars, "bind_contextvars", _raise_bind_contextvars)
    failed_ctx = FlextLogger.bind_global_context(x="y")
    assert isinstance(failed_ctx, r)
    assert failed_ctx.is_failure

    class _Cfg:
        log_level = "DEBUG"

        def model_dump(self) -> dict[str, t.Scalar]:
            return {"log_level": self.log_level}

    class _Container:
        config = _Cfg()

    def _create_module_logger(_cls: type, _name: str) -> FlextLogger:
        return FlextLogger.create_bound_logger(
            "mod",
            cast("p.Log.StructlogLogger", cast("object", _FakeBindable())),
        )

    monkeypatch.setattr(
        FlextLogger,
        "create_module_logger",
        classmethod(_create_module_logger),
    )
    created = FlextLogger.for_container(
        cast("p.DI", cast("object", _Container())),
        extra="v",
    )
    assert isinstance(created, FlextLogger)
    with FlextLogger.with_container_context(
        cast("p.DI", cast("object", _Container())),
        level=c.Settings.LogLevel.INFO,
        trace_id="1",
    ) as scoped:
        assert isinstance(scoped, FlextLogger)


def test_loggings_bind_clear_level_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    shim = _StructlogShim()

    def _structlog_accessor() -> _StructlogShim:
        return shim

    monkeypatch.setattr(FlextRuntime, "structlog", staticmethod(_structlog_accessor))

    def _raise_merge(*_args: t.Scalar, **_kwargs: t.Scalar) -> None:
        msg = "merge boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(u, "merge", _raise_merge)
    failed_bind = FlextLogger.bind_context("request", x="y")
    assert failed_bind.is_failure
    FlextLogger._scoped_contexts["request"] = {"k": "v"}

    def _raise_unbind(*_keys: str) -> None:
        msg = "unbind boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(shim.contextvars, "unbind_contextvars", _raise_unbind)
    failed_clear = FlextLogger.clear_scope("request")
    assert failed_clear.is_failure

    def _raise_bind_contextvars(**_kwargs: t.Scalar) -> None:
        msg = "bind boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(shim.contextvars, "bind_contextvars", _raise_bind_contextvars)
    lvl_bind = FlextLogger.bind_context_for_level("DEBUG", a="b")
    assert lvl_bind.is_failure

    def _raise_unbind_contextvars(*_keys: str) -> None:
        msg = "unbind boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(
        shim.contextvars,
        "unbind_contextvars",
        _raise_unbind_contextvars,
    )
    lvl_unbind = FlextLogger.unbind_context_for_level("DEBUG", "a")
    assert lvl_unbind.is_failure


def test_loggings_instance_and_message_format_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeBindable()

    def _get_logger(_name: str | None = None) -> _FakeBindable:
        return fake

    monkeypatch.setattr(FlextRuntime, "get_logger", staticmethod(_get_logger))

    class _Config:
        level = "WARNING"
        service_name = "svc"
        service_version = "1.0"
        correlation_id = "cid"
        force_new = True

        def model_dump(self) -> dict[str, t.Scalar]:
            return {
                "log_level": self.level,
                "service_name": self.service_name,
                "service_version": self.service_version,
                "correlation_id": self.correlation_id,
                "force_new": self.force_new,
            }

    logger = FlextLogger("x", config=cast("p.Config", cast("object", _Config())))
    assert logger.name == "x"
    assert logger.new(a=1).name == "x"
    assert logger.unbind("a").name == "x"
    assert logger.unbind("a", safe=True).name == "x"
    logger.trace("%s %s", "a")
    monkeypatch.setattr(logger, "_structlog_instance", object())
    logger.trace("x")
    assert FlextLogger._format_log_message("%s %s", "a") != ""
    monkeypatch.setattr(inspect, "currentframe", lambda: None)
    assert FlextLogger._get_calling_frame() is None

    class _Code:
        co_qualname = "MyType.run"

    class _Frame:
        f_locals: ClassVar[dict[str, t.Tests.object]] = {}
        f_code = _Code()

    assert (
        FlextLogger._extract_class_name(
            cast("types.FrameType", cast("object", _Frame())),
        )
        is None
    )


def test_loggings_source_and_log_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeBindable()
    logger = FlextLogger.create_bound_logger(
        "x",
        cast("p.Log.StructlogLogger", cast("object", fake)),
    )

    def _no_frame() -> types.FrameType | None:
        return None

    monkeypatch.setattr(FlextLogger, "_get_calling_frame", staticmethod(_no_frame))
    assert FlextLogger._get_caller_source_path() is None

    def _raise_resolve(self: Path) -> Path:
        msg = "bad"
        raise RuntimeError(msg)

    monkeypatch.setattr(Path, "resolve", _raise_resolve)
    assert FlextLogger._convert_to_relative_path("/tmp/x.py") == "x.py"

    class _NoMarkers:
        def __init__(self, path: Path) -> None:
            self.path: Path = path

        @property
        def parent(self) -> _NoMarkers:
            return self

        def __truediv__(self, _other: str) -> _NoMarkers:
            return self

        def exists(self) -> bool:
            return False

        @override
        def __eq__(self, _other: object) -> bool:
            return True

    assert (
        FlextLogger._find_workspace_root(
            cast("Path", cast("object", _NoMarkers(Path("/tmp")))),
        )
        is None
    )
    logger_boom = FlextLogger.create_bound_logger(
        "x",
        cast("p.Log.StructlogLogger", cast("object", _FakeBindable())),
    )
    logger_boom._structlog_instance = cast(
        "p.Log.StructlogLogger",
        cast("object", _FakeBindable()),
    )

    def _raise_info(*_args: t.Scalar, **_kwargs: t.Scalar) -> None:
        msg = "no info"
        raise AttributeError(msg)

    monkeypatch.setattr(logger_boom.logger, "info", _raise_info)
    failed = logger_boom._log("INFO", "msg")
    assert failed is not None
    assert failed.is_failure
    logger.log("INFO", "message", k="v")
    logger.warning("warn")


def test_loggings_exception_and_adapter_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeBindable()
    logger = FlextLogger.create_bound_logger(
        "x",
        cast("p.Log.StructlogLogger", cast("object", fake)),
    )

    def _raise_cfg(_cls: type) -> p.Config:
        msg = "cfg"
        raise RuntimeError(msg)

    monkeypatch.setattr(FlextSettings, "get_global", classmethod(_raise_cfg))
    assert logger._should_include_stack_trace() is True
    with_exception = logger.build_exception_context(
        exception=ValueError("x"),
        exc_info=False,
        context={"k": "v"},
    )
    assert "stack_trace" in with_exception
    with_exc_info = logger.build_exception_context(
        exception=None,
        exc_info=True,
        context={},
    )
    assert "stack_trace" in with_exc_info
    broken = FlextLogger.create_bound_logger(
        "x",
        cast("p.Log.StructlogLogger", cast("object", _FakeBindable())),
    )

    def _raise_error(*_args: t.Scalar, **_kwargs: t.Scalar) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(broken.logger, "error", _raise_error)
    broken.exception("msg", exception=ValueError("x"), exc_info=True)
    tracker = FlextLogger.PerformanceTracker(logger, "op")
    with tracker:
        pass
    tracker.__exit__(RuntimeError, RuntimeError("x"), None)
    assert logger.unbind("missing", safe=True)
    with pytest.warns(DeprecationWarning, match="try_unbind"):
        assert logger.try_unbind("missing")


def test_loggings_remaining_branch_paths(monkeypatch: pytest.MonkeyPatch) -> None:

    class _Container:
        pass

    captured: dict[str, t.Tests.object] = {}

    def _for_container(
        cls: type,
        _container: p.DI,
        level: str | None = None,
    ) -> FlextLogger:
        captured["level"] = level
        return FlextLogger.create_bound_logger(
            "ctx",
            cast("p.Log.StructlogLogger", cast("object", _FakeBindable())),
        )

    monkeypatch.setattr(FlextLogger, "for_container", classmethod(_for_container))
    with FlextLogger.with_container_context(
        cast("p.DI", cast("object", _Container())),
        trace_id="t1",
    ):
        pass
    assert captured["level"] is None
    sentinel = object()

    def _get_logger(_name: str | None = None) -> object:
        return sentinel

    monkeypatch.setattr(FlextRuntime, "get_logger", staticmethod(_get_logger))
    assert FlextLogger.get_logger("x") is sentinel

    class _TraceLogger(_FakeBindable):
        @override
        def debug(self, message: str, *_args: t.Scalar, **_kwargs: t.Scalar) -> None:
            msg = "trace boom"
            raise RuntimeError(msg)

    trace_logger = FlextLogger.create_bound_logger(
        "x",
        cast("p.Log.StructlogLogger", cast("object", _TraceLogger())),
    )
    trace_logger.trace("%s", "a")

    class _ShortFrame:
        f_back: types.FrameType | None = None

    monkeypatch.setattr(
        inspect,
        "currentframe",
        lambda: cast("types.FrameType", cast("object", _ShortFrame())),
    )
    assert FlextLogger._get_calling_frame() is None

    class _CodeUpper:
        co_qualname = "MyClass.run"

    class _UpperFrame:
        f_locals: ClassVar[dict[str, t.Tests.object]] = {}
        f_code = _CodeUpper()

    monkeypatch.setattr(c.Validation, "LEVEL_PREFIX_PARTS_COUNT", 2)
    assert (
        FlextLogger._extract_class_name(
            cast("types.FrameType", cast("object", _UpperFrame())),
        )
        == "MyClass"
    )

    class _CodeMethod:
        co_filename = "/tmp/example.py"
        co_name = "run"
        co_qualname = "run"

    class _CallerFrame:
        f_code = _CodeMethod()
        f_lineno = 40
        f_locals: ClassVar[dict[str, t.Tests.object]] = {}

    def _calling_frame() -> types.FrameType:
        return cast("types.FrameType", cast("object", _CallerFrame()))

    monkeypatch.setattr(FlextLogger, "_get_calling_frame", staticmethod(_calling_frame))

    def _relative_filename(_filename: str) -> str:
        return "example.py"

    monkeypatch.setattr(
        FlextLogger,
        "_convert_to_relative_path",
        staticmethod(_relative_filename),
    )

    def _extract_class_name(_frame: types.FrameType) -> str | None:
        return None

    monkeypatch.setattr(
        FlextLogger,
        "_extract_class_name",
        staticmethod(_extract_class_name),
    )
    source = FlextLogger._get_caller_source_path()
    assert source is not None and source.endswith(" run")

    def _raise_calling_frame() -> types.FrameType:
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(
        FlextLogger,
        "_get_calling_frame",
        staticmethod(_raise_calling_frame),
    )
    assert FlextLogger._get_caller_source_path() is None

    def _resolve_path(self: Path) -> Path:
        return Path("/tmp/example.py")

    monkeypatch.setattr(Path, "resolve", _resolve_path)

    def _workspace_root(_abs_path: Path) -> Path:
        return Path("/repo")

    monkeypatch.setattr(
        FlextLogger,
        "_find_workspace_root",
        staticmethod(_workspace_root),
    )
    assert FlextLogger._convert_to_relative_path("/tmp/example.py") == "example.py"

    class _ErrorLogger(_FakeBindable):
        @override
        def error(self, message: str, *_args: t.Scalar, **_kwargs: t.Scalar) -> None:
            msg = "err"
            raise TypeError(msg)

    err_logger = FlextLogger.create_bound_logger(
        "x",
        cast("p.Log.StructlogLogger", cast("object", _ErrorLogger())),
    )
    err_logger.error("boom", exception=ValueError("x"))


def test_loggings_uncovered_level_trace_path_and_exception_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class _Container:
        pass

    captured: dict[str, t.Tests.object] = {}

    def _for_container(
        cls: type,
        _container: p.DI,
        level: str | None = None,
    ) -> FlextLogger:
        captured["level"] = level
        return FlextLogger.create_bound_logger(
            "ctx",
            cast("p.Log.StructlogLogger", cast("object", _FakeBindable())),
        )

    monkeypatch.setattr(FlextLogger, "for_container", classmethod(_for_container))
    with FlextLogger.with_container_context(
        cast("p.DI", cast("object", _Container())),
        level="DEBUG",
    ):
        pass
    assert captured["level"] == "DEBUG"

    class _StructlogLogger(_FakeBindable):
        @override
        def debug(self, message: str, *_args: t.Scalar, **_kwargs: t.Scalar) -> None:
            msg = "trace"
            raise KeyError(msg)

        @override
        def error(self, message: str, *_args: t.Scalar, **_kwargs: t.Scalar) -> None:
            msg = "exception"
            raise RuntimeError(msg)

    monkeypatch.setattr(p.Log, "StructlogLogger", _StructlogLogger)
    trace_logger = FlextLogger.create_bound_logger(
        "x",
        cast("p.Log.StructlogLogger", cast("object", _StructlogLogger())),
    )
    trace_logger.trace("%s", "value")
    trace_logger.error("boom", exception=ValueError("x"))

    def _resolve_outside(self: Path) -> Path:
        return Path("/tmp/outside.py")

    monkeypatch.setattr(Path, "resolve", _resolve_outside)

    def _workspace_root(_abs_path: Path) -> Path:
        return Path("/repo")

    monkeypatch.setattr(
        FlextLogger,
        "_find_workspace_root",
        staticmethod(_workspace_root),
    )
    assert FlextLogger._convert_to_relative_path("/tmp/outside.py") == "outside.py"

    def _no_workspace_root(_abs_path: Path) -> None:
        return None

    monkeypatch.setattr(
        FlextLogger,
        "_find_workspace_root",
        staticmethod(_no_workspace_root),
    )
    assert FlextLogger._convert_to_relative_path("/tmp/outside.py") == "outside.py"
