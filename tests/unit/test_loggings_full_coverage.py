"""Loggings full coverage tests."""

from __future__ import annotations

import inspect
import types
from pathlib import Path
from typing import ClassVar, cast, override

import pytest
from flext_tests import t, tm

import flext_core.loggings as loggings_module
from flext_core import FlextLogger, FlextRuntime, FlextSettings, c, p, r


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
    tm.that(isinstance(c.Settings.LogLevel.DEBUG.value, str), eq=True)
    value = "ok"
    tm.that(value, eq="ok")
    logger_obj = FlextLogger.create_bound_logger(
        "x",
        cast("p.Logger", cast("object", _FakeBindable())),
    )
    tm.that(logger_obj._context, eq={})
    tm.that(logger_obj() is logger_obj, eq=True)
    bind_result = FlextLogger.bind_global_context(k1="v1")
    get_result = FlextLogger._get_global_context()
    clear_result = FlextLogger.clear_global_context()
    tm.ok(bind_result)
    tm.that(isinstance(get_result, t.ConfigMap), eq=True)
    tm.ok(clear_result)

    def _raise_bind_contextvars(**_kwargs: t.Scalar) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(shim.contextvars, "bind_contextvars", _raise_bind_contextvars)
    failed_ctx = FlextLogger.bind_global_context(x="y")
    tm.that(isinstance(failed_ctx, r), eq=True)
    tm.fail(failed_ctx)

    class _Cfg:
        log_level = "DEBUG"

        def model_dump(self) -> dict[str, t.Scalar]:
            return {"log_level": self.log_level}

    class _Container:
        config = _Cfg()

    def _create_module_logger(_cls: type, _name: str) -> FlextLogger:
        return FlextLogger.create_bound_logger(
            "mod",
            cast("p.Logger", cast("object", _FakeBindable())),
        )

    monkeypatch.setattr(
        FlextLogger,
        "create_module_logger",
        cast("t.Tests.object", classmethod(_create_module_logger)),
    )
    created = FlextLogger.for_container(
        cast("p.Container", cast("object", _Container())),
        extra="v",
    )
    tm.that(isinstance(created, p.Logger), eq=True)
    with FlextLogger.with_container_context(
        cast("p.Container", cast("object", _Container())),
        level=c.Settings.LogLevel.INFO,
        trace_id="1",
    ) as scoped:
        tm.that(isinstance(scoped, p.Logger), eq=True)


def test_loggings_bind_clear_level_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    shim = _StructlogShim()

    def _structlog_accessor() -> _StructlogShim:
        return shim

    monkeypatch.setattr(FlextRuntime, "structlog", staticmethod(_structlog_accessor))

    def _raise_merge(*_args: t.Scalar, **_kwargs: t.Scalar) -> None:
        msg = "merge boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(loggings_module.u, "merge", _raise_merge)
    failed_bind = FlextLogger.bind_context("request", x="y")
    tm.fail(failed_bind)
    FlextLogger._scoped_contexts["request"] = {"k": "v"}

    def _raise_unbind(*_keys: str) -> None:
        msg = "unbind boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(shim.contextvars, "unbind_contextvars", _raise_unbind)
    failed_clear = FlextLogger.clear_scope("request")
    tm.fail(failed_clear)

    def _raise_bind_contextvars(**_kwargs: t.Scalar) -> None:
        msg = "bind boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(shim.contextvars, "bind_contextvars", _raise_bind_contextvars)
    lvl_bind = FlextLogger.bind_context_for_level("DEBUG", a="b")
    tm.fail(lvl_bind)

    def _raise_unbind_contextvars(*_keys: str) -> None:
        msg = "unbind boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(
        shim.contextvars,
        "unbind_contextvars",
        _raise_unbind_contextvars,
    )
    lvl_unbind = FlextLogger.unbind_context_for_level("DEBUG", "a")
    tm.fail(lvl_unbind)


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

    logger = FlextLogger("x", config=cast("p.Settings", cast("object", _Config())))
    tm.that(logger.name, eq="x")
    tm.that(logger.new(a=1).name, eq="x")
    tm.that(logger.unbind("a").name, eq="x")
    tm.that(logger.unbind("a", safe=True).name, eq="x")
    logger.trace("%s %s", "a")
    monkeypatch.setattr(logger, "_structlog_instance", object())
    logger.trace("x")
    tm.that(FlextLogger._format_log_message("%s %s", "a") != "", eq=True)
    monkeypatch.setattr(inspect, "currentframe", lambda: None)
    tm.that(FlextLogger._get_calling_frame() is None, eq=True)

    class _Code:
        co_qualname = "MyType.run"

    class _Frame:
        f_locals: ClassVar[dict[str, t.Tests.object]] = {}
        f_code = _Code()

    tm.that(
        FlextLogger._extract_class_name(
            cast("types.FrameType", cast("object", _Frame())),
        )
        is None,
        eq=True,
    )


def test_loggings_source_and_log_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeBindable()
    logger = FlextLogger.create_bound_logger(
        "x",
        cast("p.Logger", cast("object", fake)),
    )

    def _no_frame() -> types.FrameType | None:
        return None

    monkeypatch.setattr(FlextLogger, "_get_calling_frame", staticmethod(_no_frame))
    tm.that(FlextLogger._get_caller_source_path() is None, eq=True)

    def _raise_resolve(self: Path) -> Path:
        msg = "bad"
        raise RuntimeError(msg)

    monkeypatch.setattr(Path, "resolve", _raise_resolve)
    tm.that(FlextLogger._convert_to_relative_path("/tmp/x.py"), eq="x.py")

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

    tm.that(
        FlextLogger._find_workspace_root(
            cast("Path", cast("object", _NoMarkers(Path("/tmp")))),
        )
        is None,
        eq=True,
    )
    logger_boom = FlextLogger.create_bound_logger(
        "x",
        cast("p.Logger", cast("object", _FakeBindable())),
    )
    logger_boom._structlog_instance = cast(
        "p.Logger",
        cast("object", _FakeBindable()),
    )

    def _raise_info(*_args: t.Scalar, **_kwargs: t.Scalar) -> None:
        msg = "no info"
        raise AttributeError(msg)

    monkeypatch.setattr(logger_boom.logger, "info", _raise_info)
    failed = logger_boom._log("INFO", "msg")
    tm.that(failed is not None, eq=True)
    tm.fail(failed)
    logger.log("INFO", "message", k="v")
    logger.warning("warn")


def test_loggings_exception_and_adapter_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeBindable()
    logger = FlextLogger.create_bound_logger(
        "x",
        cast("p.Logger", cast("object", fake)),
    )

    def _raise_cfg(_cls: type) -> p.Settings:
        msg = "cfg"
        raise RuntimeError(msg)

    monkeypatch.setattr(
        FlextSettings, "get_global", cast("t.Tests.object", classmethod(_raise_cfg))
    )
    tm.that(logger._should_include_stack_trace() is True, eq=True)
    with_exception = logger.build_exception_context(
        exception=ValueError("x"),
        exc_info=False,
        context={"k": "v"},
    )
    tm.that("stack_trace" in with_exception, eq=True)
    with_exc_info = logger.build_exception_context(
        exception=None,
        exc_info=True,
        context={},
    )
    tm.that("stack_trace" in with_exc_info, eq=True)
    broken = FlextLogger.create_bound_logger(
        "x",
        cast("p.Logger", cast("object", _FakeBindable())),
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
    tm.that(isinstance(logger.unbind("missing", safe=True), p.Logger), eq=True)
    with pytest.warns(DeprecationWarning, match="try_unbind"):
        tm.that(isinstance(logger.try_unbind("missing"), p.Logger), eq=True)


def test_loggings_remaining_branch_paths(monkeypatch: pytest.MonkeyPatch) -> None:

    class _Container:
        pass

    captured: dict[str, t.Tests.object] = {}

    def _for_container(
        cls: type,
        _container: p.Container,
        level: str | None = None,
    ) -> FlextLogger:
        captured["level"] = level
        return FlextLogger.create_bound_logger(
            "ctx",
            cast("p.Logger", cast("object", _FakeBindable())),
        )

    monkeypatch.setattr(
        FlextLogger,
        "for_container",
        cast("t.Tests.object", classmethod(_for_container)),
    )
    with FlextLogger.with_container_context(
        cast("p.Container", cast("object", _Container())),
        trace_id="t1",
    ):
        pass
    tm.that(captured["level"] is None, eq=True)
    sentinel = object()

    def _get_logger(_name: str | None = None) -> object:
        return sentinel

    monkeypatch.setattr(FlextRuntime, "get_logger", staticmethod(_get_logger))
    tm.that(FlextLogger.get_logger("x") is sentinel, eq=True)

    class _TraceLogger(_FakeBindable):
        @override
        def debug(self, message: str, *_args: t.Scalar, **_kwargs: t.Scalar) -> None:
            msg = "trace boom"
            raise RuntimeError(msg)

    trace_logger = FlextLogger.create_bound_logger(
        "x",
        cast("p.Logger", cast("object", _TraceLogger())),
    )
    trace_logger.trace("%s", "a")

    class _ShortFrame:
        f_back: types.FrameType | None = None

    monkeypatch.setattr(
        inspect,
        "currentframe",
        lambda: cast("types.FrameType", cast("object", _ShortFrame())),
    )
    tm.that(FlextLogger._get_calling_frame() is None, eq=True)

    class _CodeUpper:
        co_qualname = "MyClass.run"

    class _UpperFrame:
        f_locals: ClassVar[dict[str, t.Tests.object]] = {}
        f_code = _CodeUpper()

    monkeypatch.setattr(c.Validation, "LEVEL_PREFIX_PARTS_COUNT", 2)
    tm.that(
        FlextLogger._extract_class_name(
            cast("types.FrameType", cast("object", _UpperFrame())),
        ),
        eq="MyClass",
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
    tm.that(source is not None and source.endswith(" run"), eq=True)

    def _raise_calling_frame() -> types.FrameType:
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(
        FlextLogger,
        "_get_calling_frame",
        staticmethod(_raise_calling_frame),
    )
    tm.that(FlextLogger._get_caller_source_path() is None, eq=True)

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
    tm.that(FlextLogger._convert_to_relative_path("/tmp/example.py"), eq="example.py")

    class _ErrorLogger(_FakeBindable):
        @override
        def error(self, message: str, *_args: t.Scalar, **_kwargs: t.Scalar) -> None:
            msg = "err"
            raise TypeError(msg)

    err_logger = FlextLogger.create_bound_logger(
        "x",
        cast("p.Logger", cast("object", _ErrorLogger())),
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
        _container: p.Container,
        level: str | None = None,
    ) -> FlextLogger:
        captured["level"] = level
        return FlextLogger.create_bound_logger(
            "ctx",
            cast("p.Logger", cast("object", _FakeBindable())),
        )

    monkeypatch.setattr(
        FlextLogger,
        "for_container",
        cast("t.Tests.object", classmethod(_for_container)),
    )
    with FlextLogger.with_container_context(
        cast("p.Container", cast("object", _Container())),
        level="DEBUG",
    ):
        pass
    tm.that(captured["level"], eq="DEBUG")

    class _StructlogLogger(_FakeBindable):
        @override
        def debug(self, message: str, *_args: t.Scalar, **_kwargs: t.Scalar) -> None:
            msg = "trace"
            raise KeyError(msg)

        @override
        def error(self, message: str, *_args: t.Scalar, **_kwargs: t.Scalar) -> None:
            msg = "exception"
            raise RuntimeError(msg)

    from flext_core._protocols.logging import FlextProtocolsLogging

    monkeypatch.setattr(FlextProtocolsLogging, "Logger", _StructlogLogger)
    trace_logger = FlextLogger.create_bound_logger(
        "x",
        cast("p.Logger", cast("object", _StructlogLogger())),
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
    tm.that(FlextLogger._convert_to_relative_path("/tmp/outside.py"), eq="outside.py")

    def _no_workspace_root(_abs_path: Path) -> None:
        return None

    monkeypatch.setattr(
        FlextLogger,
        "_find_workspace_root",
        staticmethod(_no_workspace_root),
    )
    tm.that(FlextLogger._convert_to_relative_path("/tmp/outside.py"), eq="outside.py")
