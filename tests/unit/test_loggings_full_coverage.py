from __future__ import annotations

import inspect
from pathlib import Path
import types
from typing import cast

from structlog.typing import BindableLogger

from flext_core import c, m, p, r, t, u
from flext_core.loggings import FlextLogger
from flext_core.runtime import FlextRuntime
from flext_core.settings import FlextSettings


class _FakeBindable:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def bind(self, **kwargs: object) -> _FakeBindable:
        self.calls.append(("bind", (), kwargs))
        return self

    def unbind(self, *keys: str) -> _FakeBindable:
        self.calls.append(("unbind", keys, {}))
        return self

    def try_unbind(self, *keys: str) -> _FakeBindable:
        self.calls.append(("try_unbind", keys, {}))
        return self

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        self.calls.append(("debug", (message, *args), kwargs))

    def info(self, message: str, *args: object, **kwargs: object) -> None:
        self.calls.append(("info", (message, *args), kwargs))

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        self.calls.append(("warning", (message, *args), kwargs))

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        self.calls.append(("error", (message, *args), kwargs))

    def critical(self, message: str, *args: object, **kwargs: object) -> None:
        self.calls.append(("critical", (message, *args), kwargs))


class _ContextVars:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}

    def bind_contextvars(self, **kwargs: object) -> None:
        self.store.update(kwargs)

    def unbind_contextvars(self, *keys: str) -> None:
        for key in keys:
            self.store.pop(key, None)

    def clear_contextvars(self) -> None:
        self.store.clear()

    def get_contextvars(self) -> dict[str, object]:
        return dict(self.store)


class _StructlogShim:
    def __init__(self) -> None:
        self.contextvars = _ContextVars()


def test_loggings_context_and_factory_paths(monkeypatch) -> None:
    shim = _StructlogShim()
    monkeypatch.setattr(FlextRuntime, "structlog", staticmethod(lambda: shim))
    assert isinstance(c.Settings.LogLevel.DEBUG.value, str)
    value: t.GeneralValueType = "ok"
    assert value == "ok"

    logger_obj = FlextLogger.create_bound_logger(
        "x", cast("BindableLogger", cast("object", _FakeBindable()))
    )
    assert logger_obj._context == {}
    assert logger_obj() is logger_obj

    bind_result = FlextLogger.bind_global_context(k1="v1")
    get_result = FlextLogger._get_global_context()
    clear_result = FlextLogger.clear_global_context()
    assert bind_result.is_success
    assert isinstance(get_result, m.ConfigMap)
    assert clear_result.is_success

    monkeypatch.setattr(
        FlextLogger,
        "_execute_context_op",
        classmethod(
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
        ),
    )
    failed_ctx = FlextLogger._context_operation("bind", x="y")
    assert isinstance(failed_ctx, r)
    assert failed_ctx.is_failure

    class _Cfg:
        log_level = "DEBUG"

    class _Container:
        config = _Cfg()

    monkeypatch.setattr(
        FlextLogger,
        "create_module_logger",
        classmethod(
            lambda _cls, _name: FlextLogger.create_bound_logger(
                "mod", cast("BindableLogger", cast("object", _FakeBindable()))
            )
        ),
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


def test_loggings_bind_clear_level_error_paths(monkeypatch) -> None:
    shim = _StructlogShim()
    monkeypatch.setattr(FlextRuntime, "structlog", staticmethod(lambda: shim))

    monkeypatch.setattr(
        u,
        "merge",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("merge boom")),
    )
    failed_bind = FlextLogger.bind_context("request", x="y")
    assert failed_bind.is_failure

    FlextLogger._scoped_contexts["request"] = {"k": "v"}

    def _raise_unbind(*_keys: str) -> None:
        raise RuntimeError("unbind boom")

    monkeypatch.setattr(shim.contextvars, "unbind_contextvars", _raise_unbind)
    failed_clear = FlextLogger.clear_scope("request")
    assert failed_clear.is_failure

    monkeypatch.setattr(
        shim.contextvars,
        "bind_contextvars",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("bind boom")),
    )
    lvl_bind = FlextLogger.bind_context_for_level("DEBUG", a="b")
    assert lvl_bind.is_failure

    monkeypatch.setattr(
        shim.contextvars,
        "unbind_contextvars",
        lambda *_keys: (_ for _ in ()).throw(RuntimeError("unbind boom")),
    )
    lvl_unbind = FlextLogger.unbind_context_for_level("DEBUG", "a")
    assert lvl_unbind.is_failure


def test_loggings_instance_and_message_format_paths(monkeypatch) -> None:
    fake = _FakeBindable()
    monkeypatch.setattr(
        FlextRuntime, "get_logger", staticmethod(lambda _name=None: fake)
    )

    class _Config:
        level = "WARNING"
        service_name = "svc"
        service_version = "1.0"
        correlation_id = "cid"
        force_new = True

    logger = FlextLogger("x", config=cast("p.Config", cast("object", _Config())))
    assert logger.name == "x"
    assert logger.new(a=1).name == "x"
    assert logger.unbind("a").name == "x"
    assert logger.try_unbind("a").name == "x"

    logger.trace("%s %s", "a")

    monkeypatch.setattr(logger, "logger", object())
    logger.trace("x")

    assert FlextLogger._format_log_message("%s %s", "a") != ""

    monkeypatch.setattr(inspect, "currentframe", lambda: None)
    assert FlextLogger._get_calling_frame() is None

    class _Code:
        co_qualname = "MyType.run"

    class _Frame:
        f_locals: dict[str, object] = {}
        f_code = _Code()

    assert (
        FlextLogger._extract_class_name(
            cast("types.FrameType", cast("object", _Frame()))
        )
        is None
    )


def test_loggings_source_and_log_error_paths(monkeypatch) -> None:
    fake = _FakeBindable()
    logger = FlextLogger.create_bound_logger(
        "x", cast("BindableLogger", cast("object", fake))
    )

    monkeypatch.setattr(FlextLogger, "_get_calling_frame", staticmethod(lambda: None))
    assert FlextLogger._get_caller_source_path() is None

    monkeypatch.setattr(
        Path, "resolve", lambda self: (_ for _ in ()).throw(RuntimeError("bad"))
    )
    assert FlextLogger._convert_to_relative_path("/tmp/x.py") == "x.py"

    class _NoMarkers:
        def __init__(self, path: Path) -> None:
            self.path = path

        @property
        def parent(self) -> _NoMarkers:
            return self

        def __truediv__(self, _other: str) -> _NoMarkers:
            return self

        def exists(self) -> bool:
            return False

        def __eq__(self, _other: object) -> bool:
            return True

    assert (
        FlextLogger._find_workspace_root(
            cast("Path", cast("object", _NoMarkers(Path("/tmp"))))
        )
        is None
    )

    logger_boom = FlextLogger.create_bound_logger(
        "x", cast("BindableLogger", cast("object", _FakeBindable()))
    )
    logger_boom.logger = cast("BindableLogger", cast("object", _FakeBindable()))
    monkeypatch.setattr(
        logger_boom.logger,
        "info",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AttributeError("no info")),
    )
    failed = logger_boom._log("INFO", "msg")
    assert failed.is_failure

    logger.log("INFO", "message", {"k": "v"})
    logger.warn("warn")


def test_loggings_exception_and_adapter_paths(monkeypatch) -> None:
    fake = _FakeBindable()
    logger = FlextLogger.create_bound_logger(
        "x", cast("BindableLogger", cast("object", fake))
    )

    monkeypatch.setattr(
        FlextSettings,
        "get_global_instance",
        classmethod(lambda _cls: (_ for _ in ()).throw(RuntimeError("cfg"))),
    )
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
        "x", cast("BindableLogger", cast("object", _FakeBindable()))
    )
    monkeypatch.setattr(
        broken.logger,
        "error",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    broken.exception("msg", exception=ValueError("x"), exc_info=True)

    tracker = FlextLogger.PerformanceTracker(logger, "op")
    with tracker:
        pass
    tracker.__exit__(RuntimeError, RuntimeError("x"), None)

    adapter = logger.with_result()
    assert adapter.track_performance("op") is None
    assert adapter.log_result(r[str].ok("v")) is None
    bound_ctx = adapter.bind_context(a="b")
    assert bound_ctx is not None
    assert adapter.get_context() is None
    assert adapter.start_tracking() is None
    assert adapter.stop_tracking() is None

    assert adapter.trace("x").is_success
    assert adapter.debug("x").is_success
    assert adapter.info("x").is_success
    assert adapter.warning("x").is_success
    assert adapter.error("x").is_success
    assert adapter.critical("x").is_success

    class _NonException(BaseException):
        pass

    adapter.exception("boom", exception=_NonException("x"), x=1)


def test_loggings_remaining_branch_paths(monkeypatch) -> None:
    class _Container:
        pass

    captured: dict[str, object] = {}

    def _for_container(cls, _container, level=None):
        captured["level"] = level
        return FlextLogger.create_bound_logger(
            "ctx",
            cast("BindableLogger", cast("object", _FakeBindable())),
        )

    monkeypatch.setattr(FlextLogger, "for_container", classmethod(_for_container))
    with FlextLogger.with_container_context(
        cast("p.DI", cast("object", _Container())),
        trace_id="t1",
    ):
        pass
    assert captured["level"] is None

    sentinel = object()
    monkeypatch.setattr(
        FlextRuntime, "get_logger", staticmethod(lambda _name=None: sentinel)
    )
    assert FlextLogger.get_logger("x") is sentinel

    class _TraceLogger(_FakeBindable):
        def debug(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("trace boom")

    trace_logger = FlextLogger.create_bound_logger(
        "x",
        cast("BindableLogger", cast("object", _TraceLogger())),
    )
    trace_logger.trace("%s", "a")

    class _ShortFrame:
        f_back = None

    monkeypatch.setattr(
        inspect,
        "currentframe",
        lambda: cast("types.FrameType", cast("object", _ShortFrame())),
    )
    assert FlextLogger._get_calling_frame() is None

    class _CodeUpper:
        co_qualname = "MyClass.run"

    class _UpperFrame:
        f_locals: dict[str, object] = {}
        f_code = _CodeUpper()

    monkeypatch.setattr(c.Validation, "LEVEL_PREFIX_PARTS_COUNT", 2)
    assert (
        FlextLogger._extract_class_name(
            cast("types.FrameType", cast("object", _UpperFrame()))
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
        f_locals: dict[str, object] = {}

    monkeypatch.setattr(
        FlextLogger,
        "_get_calling_frame",
        staticmethod(lambda: cast("types.FrameType", cast("object", _CallerFrame()))),
    )
    monkeypatch.setattr(
        FlextLogger,
        "_convert_to_relative_path",
        staticmethod(lambda _filename: "example.py"),
    )
    monkeypatch.setattr(
        FlextLogger, "_extract_class_name", staticmethod(lambda _f: None)
    )
    source = FlextLogger._get_caller_source_path()
    assert source is not None and source.endswith(" run")

    monkeypatch.setattr(
        FlextLogger,
        "_get_calling_frame",
        staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("boom"))),
    )
    assert FlextLogger._get_caller_source_path() is None

    monkeypatch.setattr(Path, "resolve", lambda self: Path("/tmp/example.py"))
    monkeypatch.setattr(
        FlextLogger,
        "_find_workspace_root",
        staticmethod(lambda _abs_path: Path("/repo")),
    )
    assert FlextLogger._convert_to_relative_path("/tmp/example.py") == "example.py"

    class _ErrorLogger(_FakeBindable):
        def error(self, *_args: object, **_kwargs: object) -> None:
            raise TypeError("err")

    err_logger = FlextLogger.create_bound_logger(
        "x",
        cast("BindableLogger", cast("object", _ErrorLogger())),
    )
    err_logger.exception("boom", exception=ValueError("x"), exc_info=True)


def test_loggings_uncovered_level_trace_path_and_exception_guards(monkeypatch) -> None:
    class _Container:
        pass

    captured: dict[str, object] = {}

    def _for_container(cls, _container, level=None):
        captured["level"] = level
        return FlextLogger.create_bound_logger(
            "ctx",
            cast("BindableLogger", cast("object", _FakeBindable())),
        )

    monkeypatch.setattr(FlextLogger, "for_container", classmethod(_for_container))
    with FlextLogger.with_container_context(
        cast("p.DI", cast("object", _Container())),
        level="DEBUG",
    ):
        pass
    assert captured["level"] == "DEBUG"

    class _StructlogLogger(_FakeBindable):
        def debug(self, *_args: object, **_kwargs: object) -> None:
            raise KeyError("trace")

        def error(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("exception")

    monkeypatch.setattr(p.Log, "StructlogLogger", _StructlogLogger)
    trace_logger = FlextLogger.create_bound_logger(
        "x",
        cast("BindableLogger", cast("object", _StructlogLogger())),
    )
    trace_logger.trace("%s", "value")
    trace_logger.exception("boom", exception=ValueError("x"), exc_info=True)

    monkeypatch.setattr(Path, "resolve", lambda self: Path("/tmp/outside.py"))
    monkeypatch.setattr(
        FlextLogger,
        "_find_workspace_root",
        staticmethod(lambda _abs_path: Path("/repo")),
    )
    assert FlextLogger._convert_to_relative_path("/tmp/outside.py") == "outside.py"

    monkeypatch.setattr(
        FlextLogger,
        "_find_workspace_root",
        staticmethod(lambda _abs_path: None),
    )
    assert FlextLogger._convert_to_relative_path("/tmp/outside.py") == "outside.py"
