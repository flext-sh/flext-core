"""Loggings full coverage tests."""

from __future__ import annotations

import types
from collections.abc import (
    Mapping,
    MutableSequence,
)
from pathlib import Path
from typing import ClassVar, override

import pytest

from flext_core import (
    FlextSettings,
)
from flext_tests import tm
from tests import m, p, t, u


class TestModule:
    class _FakeBindable:
        name: str = "test"

        def __init__(self) -> None:
            self.calls: MutableSequence[
                tuple[str, tuple[t.Container, ...], t.ScalarMapping]
            ] = []

        def bind(self, **kwargs: t.Scalar) -> TestModule._FakeBindable:
            self.calls.append(("bind", (), kwargs))
            return self

        def new(self, **kwargs: t.Scalar) -> TestModule._FakeBindable:
            self.calls.append(("new", (), kwargs))
            return self

        def unbind(self, *keys: str, safe: bool = False) -> TestModule._FakeBindable:
            self.calls.append(("unbind", keys, {}))
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

        def exception(self, message: str, *args: t.Scalar, **kwargs: t.Scalar) -> None:
            self.calls.append(("exception", (message, *args), kwargs))

        def log(
            self, level: str, message: str, *args: t.Scalar, **kwargs: t.Scalar
        ) -> None:
            self.calls.append(("log", (level, message, *args), kwargs))

        def trace(self, message: str, *args: t.Scalar, **kwargs: t.Scalar) -> None:
            self.calls.append(("trace", (message, *args), kwargs))

        def build_exception_context(
            self,
            *,
            exception: Exception | None,
            exc_info: bool,
            context: t.ScalarMapping,
        ) -> m.ConfigMap:
            return m.ConfigMap(root={})

    class _ContextVars:
        def __init__(self) -> None:
            self.store: t.MutableFlatContainerMapping = dict[str, t.Container]()

        def bind_contextvars(self, **kwargs: t.Scalar) -> None:
            self.store.update(kwargs)

        def unbind_contextvars(self, *keys: str) -> None:
            for key in keys:
                self.store.pop(key, None)

        def clear_contextvars(self) -> None:
            self.store.clear()

        def get_contextvars(self) -> Mapping[str, t.Container]:
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
        tm.that(logger.unbind("a", safe=True), none=False)
        logger.trace("%s %s", "a")
        logger.trace("x")
        tm.that(u.to_str(("%s %s", "a")), ne="")
        tm.that(u._calling_frame(), is_=types.FrameType)

        class _Code:
            co_qualname = "MyType.run"

        class _Frame:
            f_locals: ClassVar[Mapping[str, t.Container]] = {}
            f_code = _Code()

        extract_class_name = getattr(u, "_extract_class_name")
        tm.that(
            extract_class_name(
                _Frame(),
            ),
            none=True,
        )

    def test_loggings_source_and_log_error_paths(self) -> None:
        create_bound_logger = getattr(u, "create_bound_logger")
        fake = self._FakeBindable()
        logger = create_bound_logger("x", fake)

        tm.that(u._caller_source_path(), none=True)

        tm.that(Path("/tmp/x.py").name, eq="x.py")

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

        find_workspace_root = getattr(u, "_find_workspace_root")
        tm.that(
            find_workspace_root(
                _NoMarkers(Path("/tmp")),
            ),
            none=True,
        )
        boom_bindable = self._FakeBindable()
        logger_boom = create_bound_logger("x", boom_bindable)
        failing_bindable = self._FailingInfoBindable()
        logger_boom._structlog_instance = failing_bindable

        failed = logger_boom._log("INFO", "msg")
        assert failed is not None
        tm.fail(failed)
        tm.ok(logger.log("INFO", "message", k="v"))
        tm.ok(logger.warning("warn"))

    def test_loggings_exception_and_adapter_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        create_bound_logger = getattr(u, "create_bound_logger")
        fake = self._FakeBindable()
        logger = create_bound_logger("x", fake)

        def _raise_cfg(_cls: type) -> p.Settings:
            msg = "cfg"
            raise RuntimeError(msg)

        monkeypatch.setattr(
            FlextSettings,
            "fetch_global",
            classmethod(_raise_cfg),
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
        tm.that(with_exc_info, is_=dict)
        broken_bindable = self._FakeBindable()
        broken = create_bound_logger("x", broken_bindable)

        broken.exception("msg", exception=ValueError("x"), exc_info=True)
        tracker = u.PerformanceTracker(logger, "op")
        with tracker:
            pass
        tracker.__exit__(RuntimeError, RuntimeError("x"), None)
        assert logger.unbind("missing", safe=True) is not None


__all__: list[str] = ["TestModule"]
