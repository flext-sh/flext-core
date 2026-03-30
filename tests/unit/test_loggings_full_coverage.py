"""Loggings full coverage tests."""

from __future__ import annotations

import inspect
import types
from collections.abc import MutableSequence
from pathlib import Path
from typing import ClassVar, cast

import pytest
from flext_tests import tm

from flext_core import (
    FlextLogger,
    FlextRuntime,
    FlextSettings,
)
from tests import p, t


class TestModule:
    class _FakeBindable:
        def __init__(self) -> None:
            self.calls: MutableSequence[
                tuple[str, tuple[t.NormalizedValue, ...], t.ScalarMapping]
            ] = []

        def bind(self, **kwargs: t.Scalar) -> TestModule._FakeBindable:
            self.calls.append(("bind", (), kwargs))
            return self

        def unbind(self, *keys: str) -> TestModule._FakeBindable:
            self.calls.append(("unbind", keys, {}))
            return self

        def try_unbind(self, *keys: str) -> TestModule._FakeBindable:
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
            self.store: t.MutableContainerMapping = {}

        def bind_contextvars(self, **kwargs: t.Scalar) -> None:
            self.store.update(kwargs)

        def unbind_contextvars(self, *keys: str) -> None:
            for key in keys:
                self.store.pop(key, None)

        def clear_contextvars(self) -> None:
            self.store.clear()

        def get_contextvars(self) -> t.ContainerMapping:
            return dict(self.store)

    class _StructlogShim:
        def __init__(self) -> None:
            self.contextvars: TestModule._ContextVars = TestModule._ContextVars()

    def test_loggings_instance_and_message_format_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = self._FakeBindable()

        def _get_logger(_name: str | None = None) -> TestModule._FakeBindable:
            return fake

        monkeypatch.setattr(FlextRuntime, "get_logger", staticmethod(_get_logger))

        class _Config:
            level = "WARNING"
            service_name = "svc"
            service_version = "1.0"
            correlation_id = "cid"
            force_new = True

            def model_dump(self) -> t.ScalarMapping:
                return {
                    "log_level": self.level,
                    "service_name": self.service_name,
                    "service_version": self.service_version,
                    "correlation_id": self.correlation_id,
                    "force_new": self.force_new,
                }

        logger = FlextLogger(
            "x",
            config=cast("p.Settings", cast("t.NormalizedValue", _Config())),
        )
        tm.that(logger.name, eq="x")
        tm.that(logger.new(a=1).name, eq="x")
        tm.that(logger.unbind("a").name, eq="x")
        tm.that(logger.unbind("a", safe=True).name, eq="x")
        logger.trace("%s %s", "a")
        monkeypatch.setattr(logger, "_structlog_instance", "normalized")
        logger.trace("x")
        tm.that(FlextLogger._format_log_message("%s %s", "a"), ne="")
        monkeypatch.setattr(inspect, "currentframe", lambda: None)
        tm.that(FlextLogger._get_calling_frame(), none=True)

        class _Code:
            co_qualname = "MyType.run"

        class _Frame:
            f_locals: ClassVar[t.ContainerMapping] = {}
            f_code = _Code()

        tm.that(
            FlextLogger._extract_class_name(
                cast("types.FrameType", cast("t.NormalizedValue", _Frame())),
            ),
            none=True,
        )

    def test_loggings_source_and_log_error_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = self._FakeBindable()
        logger = FlextLogger.create_bound_logger(
            "x",
            cast("p.Logger", cast("t.NormalizedValue", fake)),
        )

        def _no_frame() -> types.FrameType | None:
            return None

        monkeypatch.setattr(FlextLogger, "_get_calling_frame", staticmethod(_no_frame))
        tm.that(FlextLogger._get_caller_source_path(), none=True)

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

        tm.that(
            FlextLogger._find_workspace_root(
                cast("Path", cast("t.NormalizedValue", _NoMarkers(Path("/tmp")))),
            ),
            none=True,
        )
        logger_boom = FlextLogger.create_bound_logger(
            "x",
            cast("p.Logger", cast("t.NormalizedValue", self._FakeBindable())),
        )
        logger_boom._structlog_instance = cast(
            "p.Logger",
            cast("t.NormalizedValue", self._FakeBindable()),
        )

        def _raise_info(*_args: t.Scalar, **_kwargs: t.Scalar) -> None:
            msg = "no info"
            raise AttributeError(msg)

        monkeypatch.setattr(logger_boom.logger, "info", _raise_info)
        failed = logger_boom._log("INFO", "msg")
        assert failed is not None
        tm.fail(failed)
        logger.log("INFO", "message", k="v")
        logger.warning("warn")

    def test_loggings_exception_and_adapter_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = self._FakeBindable()
        logger = FlextLogger.create_bound_logger(
            "x",
            cast("p.Logger", cast("t.NormalizedValue", fake)),
        )

        def _raise_cfg(_cls: type) -> p.Settings:
            msg = "cfg"
            raise RuntimeError(msg)

        monkeypatch.setattr(
            FlextSettings,
            "get_global",
            cast("t.NormalizedValue", classmethod(_raise_cfg)),
        )
        tm.that(isinstance(logger._should_include_stack_trace(), bool), eq=True)
        with_exception = logger.build_exception_context(
            exception=ValueError("x"),
            exc_info=False,
            context={"k": "v"},
        )
        tm.that(with_exception, has="exception_type")
        with_exc_info = logger.build_exception_context(
            exception=None,
            exc_info=True,
            context={},
        )
        tm.that(with_exc_info, is_=t.ConfigMap)
        broken = FlextLogger.create_bound_logger(
            "x",
            cast("p.Logger", cast("t.NormalizedValue", self._FakeBindable())),
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
        tm.that(logger.unbind("missing", safe=True), is_=p.Logger)
        with pytest.warns(DeprecationWarning, match="try_unbind"):
            tm.that(logger.try_unbind("missing"), is_=p.Logger)


__all__ = ["TestModule"]
