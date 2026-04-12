"""Loggings full coverage tests."""

from __future__ import annotations

import types
from collections.abc import MutableSequence
from pathlib import Path
from typing import ClassVar, cast, override

import pytest

from flext_core import (
    FlextSettings,
)
from flext_tests import tm
from tests import p, t, u


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
            self.store: t.MutableContainerMapping = dict[str, t.NormalizedValue]()

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

    class _FailingInfoBindable(_FakeBindable):
        @override
        def info(self, message: str, *args: t.Scalar, **kwargs: t.Scalar) -> None:
            super().info(message, *args, **kwargs)
            msg = "no info"
            raise AttributeError(msg)

    def test_loggings_instance_and_message_format_paths(self) -> None:
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

        logger = u.create_module_logger("x")
        logger = logger.bind(
            service_name="svc",
            service_version="1.0",
            correlation_id="cid",
        )
        tm.that(logger, none=False)
        tm.that(logger.new(a=1), none=False)
        with pytest.raises(KeyError):
            logger.unbind("a")
        tm.that(logger.try_unbind("a"), none=False)
        logger.trace("%s %s", "a")
        logger.trace("x")
        tm.that(u._format_log_message("%s %s", "a"), ne="")
        tm.that(u._calling_frame(), is_=types.FrameType)

        class _Code:
            co_qualname = "MyType.run"

        class _Frame:
            f_locals: ClassVar[t.ContainerMapping] = {}
            f_code = _Code()

        tm.that(
            u._extract_class_name(
                cast("types.FrameType", cast("t.NormalizedValue", _Frame())),
            ),
            none=True,
        )

    def test_loggings_source_and_log_error_paths(self) -> None:
        fake = self._FakeBindable()
        logger = u.create_bound_logger(
            "x",
            cast("p.Logger", cast("t.NormalizedValue", fake)),
        )

        tm.that(u._caller_source_path(), none=True)

        tm.that(u._convert_to_relative_path("/tmp/x.py"), eq="x.py")

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
            u._find_workspace_root(
                cast("Path", cast("t.NormalizedValue", _NoMarkers(Path("/tmp")))),
            ),
            none=True,
        )
        logger_boom = u.create_bound_logger(
            "x",
            cast("p.Logger", cast("t.NormalizedValue", self._FakeBindable())),
        )
        logger_boom._structlog_instance = cast(
            "p.Logger",
            cast("t.NormalizedValue", self._FailingInfoBindable()),
        )

        failed = logger_boom._log("INFO", "msg")
        assert failed is not None
        tm.fail(failed)
        tm.ok(logger.log("INFO", "message", k="v"))
        tm.ok(logger.warning("warn"))

    def test_loggings_exception_and_adapter_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = self._FakeBindable()
        logger = u.create_bound_logger(
            "x",
            cast("p.Logger", cast("t.NormalizedValue", fake)),
        )

        def _raise_cfg(_cls: type) -> p.Settings:
            msg = "cfg"
            raise RuntimeError(msg)

        monkeypatch.setattr(
            FlextSettings,
            "fetch_global",
            cast("t.NormalizedValue", classmethod(_raise_cfg)),
        )
        tm.that(logger._should_include_stack_trace(), is_=bool)
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
        broken = u.create_bound_logger(
            "x",
            cast("p.Logger", cast("t.NormalizedValue", self._FakeBindable())),
        )

        broken.exception("msg", exception=ValueError("x"), exc_info=True)
        tracker = u.PerformanceTracker(logger, "op")
        with tracker:
            pass
        tracker.__exit__(RuntimeError, RuntimeError("x"), None)
        tm.that(logger.unbind("missing", safe=True), none=False)
        with pytest.warns(DeprecationWarning, match="try_unbind"):
            tm.that(logger.try_unbind("missing"), none=False)


__all__: list[str] = ["TestModule"]
