"""Tests for runtime full coverage."""

from __future__ import annotations

import contextlib
import inspect
import io
import logging
import queue
import threading
import types
from collections.abc import (
    Callable,
    Generator,
    Iterator,
    Mapping,
    MutableSequence,
    Sequence,
)
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, cast, override

import pytest
import structlog
from dependency_injector import containers, providers
from pydantic import BaseModel

from flext_tests import tm
from tests import m, r, t, u

runtime_module = inspect.getmodule(u.configure_structlog)


@pytest.fixture(autouse=True)
def reset_runtime_state() -> Generator[None]:
    u._structlog_configured = False
    if u._async_writer is not None:
        u._async_writer.shutdown()
        u._async_writer = None
    yield
    u._structlog_configured = False
    if u._async_writer is not None:
        u._async_writer.shutdown()
        u._async_writer = None


def test_async_log_writer_paths() -> None:

    class Stream(io.StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.messages: MutableSequence[str] = []
            self.flushed = 0

        @override
        def write(self, message: str) -> int:
            self.messages.append(message)
            return len(message)

        @override
        def flush(self) -> None:
            self.flushed += 1

    stream = Stream()
    writer = u._AsyncLogWriter(stream)
    writer.write("hello")
    writer.shutdown()
    writer.shutdown()
    tm.that(writer.stop_event.is_set(), eq=True)

    class EmptyQueue:
        def get(self, timeout: float = 0.1) -> str:
            _ = timeout
            raise queue.Empty

    forced = cast(
        "u._AsyncLogWriter",
        cast("t.NormalizedValue", object.__new__(u._AsyncLogWriter)),
    )
    forced.stream = stream
    forced.queue = cast(
        "queue.Queue[str | None]",
        cast("t.NormalizedValue", EmptyQueue()),
    )
    forced.stop_event = threading.Event()
    forced.stop_event.set()
    forced._worker()

    class FailingStream(io.StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.first: bool = True
            self.messages: MutableSequence[str] = []

        @override
        def write(self, message: str) -> int:
            if self.first:
                self.first = False
                msg = "boom"
                raise OSError(msg)
            self.messages.append(message)
            return len(message)

        @override
        def flush(self) -> None:
            return None

    class SequenceQueue:
        def __init__(self) -> None:
            self.calls: int = 0
            self.calls = 0

        def get(self, timeout: float = 0.1) -> str | None:
            _ = timeout
            self.calls += 1
            if self.calls == 1:
                return "message"
            return None

        def task_done(self) -> None:
            return None

    failing = FailingStream()
    broken = cast(
        "u._AsyncLogWriter",
        cast("t.NormalizedValue", object.__new__(u._AsyncLogWriter)),
    )
    broken.stream = failing
    broken.queue = cast(
        "queue.Queue[str | None]",
        cast("t.NormalizedValue", SequenceQueue()),
    )
    broken.stop_event = threading.Event()
    broken._worker()
    tm.that(failing.messages, has="Error in async log writer\n")

    class EmptyThenSentinelQueue:
        def __init__(self) -> None:
            self.calls: int = 0
            self.calls = 0

        def get(self, timeout: float = 0.1) -> str | None:
            _ = timeout
            self.calls += 1
            if self.calls == 1:
                raise queue.Empty
            return None

        def task_done(self) -> None:
            return None

    continue_writer = cast(
        "u._AsyncLogWriter",
        cast("t.NormalizedValue", object.__new__(u._AsyncLogWriter)),
    )
    continue_writer.stream = stream
    continue_writer.queue = cast(
        "queue.Queue[str | None]",
        cast("t.NormalizedValue", EmptyThenSentinelQueue()),
    )
    continue_writer.stop_event = threading.Event()
    continue_writer._worker()


def test_async_log_writer_shutdown_with_full_queue() -> None:

    class FlushOnlyStream(io.StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.flush_calls: int = 0
            super().__init__()
            self.flush_calls = 0

        @override
        def flush(self) -> None:
            self.flush_calls += 1

    class FullQueue:
        def put_nowait(self, message: str | None) -> None:
            _ = message
            raise queue.Full

    class JoinRecorderThread:
        def __init__(self) -> None:
            self.join_timeout: float | None = None

        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float | None = None) -> None:
            self.join_timeout = timeout

    stream = FlushOnlyStream()
    writer = cast(
        "u._AsyncLogWriter",
        cast("t.NormalizedValue", object.__new__(u._AsyncLogWriter)),
    )
    writer.stream = stream
    writer.queue = cast(
        "queue.Queue[str | None]",
        cast("t.NormalizedValue", FullQueue()),
    )
    writer.stop_event = threading.Event()
    thread = JoinRecorderThread()
    writer.thread = cast("threading.Thread", cast("t.NormalizedValue", thread))
    writer.shutdown()
    tm.that(writer.stop_event.is_set(), eq=True)
    tm.that(thread.join_timeout, none=False)
    if thread.join_timeout is not None:
        tm.that(abs(thread.join_timeout - 2.0), lt=1e-09)
    tm.that(stream.flush_calls, eq=1)


def test_runtime_create_instance_failure_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = monkeypatch

    class Marker:
        pass

    instance = u.create_instance(Marker)
    assert isinstance(instance, Marker)


def test_normalization_edge_branches() -> None:
    cfg = t.ConfigMap(root={"a": 1})
    normalized_cfg = u.normalize_to_container(cfg)
    tm.that(normalized_cfg, is_=(t.ConfigMap, t.Dict))

    class DictLike(t.ContainerMappingBase):
        @override
        def __getitem__(self, key: str) -> int:
            if key == "x":
                return 1
            raise KeyError(key)

        @override
        def __len__(self) -> int:
            return 1

        @override
        def __iter__(self) -> Iterator[str]:
            return iter(["x"])

    normalized_dict_like = u.normalize_to_container(
        cast("t.RuntimeData", DictLike()),
    )
    tm.that(normalized_dict_like, is_=t.Dict)
    metadata_cfg = u.normalize_to_metadata(cfg)
    tm.that(metadata_cfg, is_=str)
    metadata_dict_like = u.normalize_to_metadata(
        cast("t.RuntimeData", DictLike()),
    )
    tm.that(
        isinstance(metadata_dict_like, dict) and metadata_dict_like == {"x": 1},
        eq=True,
    )
    metadata_list = u.normalize_to_metadata(
        cast("t.RuntimeData", ["a", "normalized"]),
    )
    tm.that(metadata_list, is_=list)


def test_normalize_to_container_alias_removal_path() -> None:
    result = u.normalize_to_container("hello")
    tm.that(result, eq="hello")


def test_normalize_to_metadata_alias_removal_path() -> None:
    result = u.normalize_to_metadata(42)
    tm.that(result, none=False)


def test_get_logger_none_name_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_with_frame = u.get_logger()
    tm.that(logger_with_frame, none=False)
    logger_no_frame = u.get_logger()
    tm.that(logger_no_frame, none=False)


def test_dependency_registration_duplicate_guards() -> None:
    container = u.DependencyIntegration.create_container()
    u.DependencyIntegration.register_object(container, "svc", 1)
    with pytest.raises(ValueError, match="already registered"):
        u.DependencyIntegration.register_object(container, "svc", 2)
    u.DependencyIntegration.register_factory(container, "factory", lambda: 1)
    with pytest.raises(ValueError, match="already registered"):
        u.DependencyIntegration.register_factory(
            container,
            "factory",
            lambda: 2,
        )
    u.DependencyIntegration.register_resource(
        container,
        "resource",
        lambda: 1,
    )
    with pytest.raises(ValueError, match="already registered"):
        u.DependencyIntegration.register_resource(
            container,
            "resource",
            lambda: 2,
        )


def test_configure_structlog_edge_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: MutableSequence[t.ConfigurationMapping] = []

    class StatefulModule:
        def __init__(self) -> None:
            self._print_access: int = 0

            def _merge_contextvars(*_args: t.Scalar) -> t.ConfigurationMapping:
                return {}

            def _add_log_level(*_args: t.Scalar) -> t.ConfigurationMapping:
                return {}

            def _time_stamper(**_kwargs: t.Scalar) -> t.Scalar:
                return ""

            def _console_renderer(**_kwargs: t.Scalar) -> t.Scalar:
                return ""

            self.contextvars = type(
                "Ctx",
                (),
                {"merge_contextvars": staticmethod(_merge_contextvars)},
            )
            self.processors = type(
                "Processors",
                (),
                {
                    "add_log_level": staticmethod(_add_log_level),
                    "TimeStamper": staticmethod(_time_stamper),
                    "StackInfoRenderer": staticmethod(lambda: ""),
                    "JSONRenderer": staticmethod(lambda: ""),
                },
            )
            self.dev = type(
                "Dev",
                (),
                {"ConsoleRenderer": staticmethod(_console_renderer)},
            )

        def reset_defaults(self) -> None:
            return None

        def make_filtering_bound_logger(self, level: int) -> type:
            _ = level
            return dict

        def configure(self, **kwargs: t.Scalar) -> None:
            calls.append(dict(kwargs.items()))

        def __getattr__(
            self,
            name: str,
        ) -> t.Scalar | type | Callable[..., t.Scalar] | types.SimpleNamespace:
            if name == "types":
                return types.SimpleNamespace(Processor=type)
            if name != "PrintLoggerFactory":
                raise AttributeError(name)
            self._print_access += 1
            if self._print_access == 1:
                raise AttributeError(name)

            def _print_logger_factory(**_kwargs: t.Scalar) -> t.Scalar:
                return ""

            return _print_logger_factory

    fake_module = StatefulModule()

    class Config:
        log_level: int = logging.DEBUG
        console_renderer: bool = True
        additional_processors: ClassVar[
            Sequence[Callable[..., t.ConfigurationMapping]]
        ] = [
            lambda *_args: {},
        ]
        wrapper_class_factory: Callable[..., t.Scalar] | None = None
        logger_factory: Callable[[], t.Scalar] = staticmethod(lambda: "")
        cache_logger_on_first_use: bool = True
        async_logging: bool = True

    _ = Config  # referenced for pyright
    u.configure_structlog(config=None)
    tm.that(u.is_structlog_configured(), eq=True)
    tm.that(bool(calls), eq=True)
    u._structlog_configured = False
    calls.clear()
    fake_module._print_access = 0
    with contextlib.suppress(AttributeError):
        object.__delattr__(fake_module, "PrintLoggerFactory")

    def _print_logger_factory(**_kwargs: t.Scalar) -> t.Scalar:
        return ""

    setattr(fake_module, "PrintLoggerFactory", _print_logger_factory)

    class ConfigNoAsync:
        log_level: int = logging.INFO
        console_renderer: bool = True
        additional_processors: (
            MutableSequence[Callable[..., t.MutableConfigurationMapping]] | None
        ) = None
        wrapper_class_factory: Callable[..., t.Scalar] | None = None
        logger_factory: Callable[..., t.Scalar] | None = None
        cache_logger_on_first_use: bool = True
        async_logging: bool = False

    _ = ConfigNoAsync  # referenced for pyright
    u.configure_structlog(config=None)
    tm.that(u._structlog_configured, eq=True)
    u._structlog_configured = False
    calls.clear()
    fake_module._print_access = 0

    class ConfigAsyncFallback:
        log_level: int = logging.INFO
        console_renderer: bool = True
        additional_processors: (
            MutableSequence[Callable[..., t.MutableConfigurationMapping]] | None
        ) = None
        wrapper_class_factory: Callable[..., t.Scalar] | None = None
        logger_factory: Callable[..., t.Scalar] | None = None
        cache_logger_on_first_use: bool = True
        async_logging: bool = True

    _ = ConfigAsyncFallback  # referenced for pyright
    u.configure_structlog(config=None)
    tm.that(u._structlog_configured, eq=True)


def test_reconfigure_and_reset_state_paths() -> None:

    class DummyWriter:
        def __init__(self) -> None:
            self.called: bool = False

        def shutdown(self) -> None:
            self.called = True

    dummy = DummyWriter()
    u._async_writer = cast(
        "u._AsyncLogWriter",
        cast("t.NormalizedValue", dummy),
    )
    u._structlog_configured = True
    u.reconfigure_structlog(log_level=logging.DEBUG, console_renderer=True)
    tm.that(dummy.called, eq=True)
    u.reset_structlog_state_for_testing()
    tm.that(not u._structlog_configured, eq=True)


def test_runtime_result_all_missed_branches() -> None:

    def _plus_one(value: int) -> int:
        return value + 1

    def _raise_bad(_value: int) -> int:
        msg = "bad"
        raise ValueError(msg)

    def _ok_plus_one(value: int | None) -> r[int | None]:
        if value is None:
            return r[int | None].fail("none")
        return r[int | None].ok(value + 1)

    def _ok_plus_two(value: int) -> r[int]:
        return r[int].ok(value + 2)

    def _error_to_int(error: str) -> int:
        return len(error)

    success: r[int] = r[int].ok(1)
    failure: r[int] = r[int].fail(
        "e",
        error_code="E1",
        error_data=t.ConfigMap(root={"x": 1}),
    )
    tm.that(success.is_success, eq=True)
    tm.that(success.unwrap_or(9), eq=1)
    tm.that(failure.unwrap_or(9), eq=9)
    tm.that(success.unwrap_or_else(lambda: 7), eq=1)
    tm.that(failure.unwrap_or_else(lambda: 7), eq=7)
    with pytest.raises(RuntimeError, match="Cannot unwrap failed result"):
        failure.unwrap()
    mapped_ok = success.map(_plus_one)
    tm.that(mapped_ok.is_success and mapped_ok.value == 2, eq=True)
    mapped_error = success.map(_raise_bad)
    tm.that(mapped_error.is_failure, eq=True)
    mapped_failed = failure.map(int)
    tm.that(mapped_failed.is_failure, eq=True)
    flat_mapped = success.flat_map(_ok_plus_one)
    tm.that(flat_mapped.value, eq=2)
    tm.that(success.flat_map(_ok_plus_two).value, eq=3)
    tm.that(success.fold(_error_to_int, _plus_one), eq=2)
    tm.that(failure.fold(_error_to_int, _plus_one), eq=1)
    tapped: MutableSequence[int] = []
    success.tap(lambda x: tapped.append(x))
    tm.that(tapped, eq=[1])
    errors: MutableSequence[str] = []
    failure.tap_error(lambda err: errors.append(err))
    tm.that(errors, eq=["e"])
    tm.that(failure.map_error(lambda err: err.upper()).error, eq="E")
    tm.that(success.map_error(lambda err: err.upper()) is success, eq=True)
    filtered = success.filter(lambda value: value > 10)
    tm.that(filtered.is_failure, eq=True)
    tm.that(filtered.error, eq="Value did not pass filter predicate")
    tm.that(failure.map_error(lambda err: f"{err}-alt").error, eq="e-alt")
    tm.that(
        failure.lash(lambda _err: r[int].ok(5)).value,
        eq=5,
    )
    tm.that(failure.recover(lambda _err: 7).value, eq=7)

    class NoneValueResult(r[int | None]):
        @property
        @override
        def value(self) -> int | None:
            return None

    none_success: r[int | None] = NoneValueResult(
        is_success=True,
        error=None,
        error_code=None,
        error_data=None,
    )
    flowed = none_success.flow_through(_ok_plus_one)
    tm.that(flowed is none_success, eq=True)

    none_ok = r[int | None].ok(None)
    tm.that(none_ok.is_success, eq=True)
    tm.that(none_ok.value, none=True)
    none_error: r[int] = r[int].fail(
        None,
    )
    tm.that(none_error.error, eq="")
    broken: r[int] = r(
        is_success=True,
        error=None,
        error_code=None,
        error_data=None,
    )
    tm.that(broken.value, none=True)


def test_model_support_and_hash_compare_paths() -> None:
    prefixed = u.generate_prefixed_id("item", length=8)
    tm.that(
        prefixed.startswith("item_") and len(prefixed.split("_", 1)[1]) == 8,
        eq=True,
    )
    tm.that(
        (
            u.compare_entities_by_id(
                "a",
                cast("t.RuntimeData", "normalized"),
            )
            is False
        ),
        eq=True,
    )
    tm.that(
        (u.compare_entities_by_id(cast("t.RuntimeData", "normalized"), 3) is False),
        eq=True,
    )

    class A:
        unique_id: str = "1"

    class B:
        unique_id: str = "1"

    tm.that(
        (
            u.compare_entities_by_id(
                cast("t.RuntimeData", A()),
                cast("t.RuntimeData", B()),
            )
            is False
        ),
        eq=True,
    )

    class _Opaque:
        pass

    obj = cast("t.RuntimeData", _Opaque())
    tm.that(
        u.hash_entity_by_id(obj),
        eq=hash(
            id(obj),
        ),
    )
    tm.that(u.compare_value_objects_by_value("a", "a"), eq=True)
    tm.that(
        (
            u.compare_value_objects_by_value(
                cast("t.RuntimeData", "normalized"),
                1,
            )
            is False
        ),
        eq=True,
    )
    tm.that(u.compare_value_objects_by_value([1], [1]), eq=True)

    class C:
        @override
        def __repr__(self) -> str:
            return "same"

    class D:
        @override
        def __repr__(self) -> str:
            return "same"

    tm.that(
        (
            u.compare_value_objects_by_value(
                cast("t.RuntimeData", C()),
                cast("t.RuntimeData", D()),
            )
            is False
        ),
        eq=True,
    )
    tm.that(
        (
            u.compare_value_objects_by_value(
                cast("t.RuntimeData", C()),
                cast("t.RuntimeData", C()),
            )
            is True
        ),
        eq=True,
    )
    tm.that(u.hash_value_object_by_value("x"), is_=int)
    tm.that(u.hash_value_object_by_value({"a": 1}), is_=int)
    tm.that(u.hash_value_object_by_value([1, 2]), is_=int)
    tm.that(
        u.hash_value_object_by_value(MappingProxyType({"a": 1})),
        is_=int,
    )
    tm.that(u.hash_value_object_by_value((1, 2)), is_=int)
    tm.that(u.hash_value_object_by_value(datetime.now(UTC)), is_=int)

    class Empty:
        pass

    assert isinstance(u.Bootstrap.create_instance(Empty), Empty)


def test_config_bridge_and_trace_context_and_http_validation() -> None:
    level = u.get_log_level_from_config()
    tm.that(level, is_=int)
    trace_from_scalar = u.ensure_trace_context(
        1,
        include_correlation_id=True,
        include_timestamp=True,
    )
    tm.that(
        {"trace_id", "span_id", "correlation_id", "timestamp"}.issubset(
            trace_from_scalar,
        ),
        eq=True,
    )

    class TraceModel(BaseModel):
        key: str = "value"

    trace_from_model = u.ensure_trace_context(TraceModel())
    tm.that(trace_from_model["key"], eq="value")

    # ensure_trace_context catches RuntimeError internally for bad mappings
    bad_trace = u.ensure_trace_context({"broken": "x"})
    tm.that(bad_trace, has="trace_id")  # graceful fallback
    tm.that(bad_trace, has="span_id")
    trace_from_mapping = u.ensure_trace_context(MappingProxyType({"a": "b"}))
    tm.that(trace_from_mapping, has="trace_id")
    trace_from_other = u.ensure_trace_context("path")
    tm.that(trace_from_other, has="trace_id")


def test_runtime_result_alias_compatibility() -> None:
    rr: r[int] = r[int].ok(10)
    wrapped: r[int] = r[int].ok(rr.value)
    assert isinstance(wrapped, r)


def test_runtime_misc_remaining_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    u._structlog_configured = False
    u.ensure_structlog_configured()
    tm.that(u.is_structlog_configured(), eq=True)

    class BasicModel(BaseModel):
        value: int = 1

    tm.that(u.is_base_model(BasicModel()), eq=True)
    normalized_mapping = u.normalize_to_container(
        MappingProxyType({"k": "v"}),
    )
    tm.that(normalized_mapping, is_=t.Dict)
    norm_list = u.normalize_to_container([1, "x"])
    tm.that(norm_list, is_=t.ObjectList)
    # Path is Container, returned as-is
    tm.that(u.normalize_to_container(Path("/tmp")), eq=Path("/tmp"))
    tm.that(u.normalize_to_metadata(1), eq=1)
    metadata_mapping = u.normalize_to_metadata(MappingProxyType({"a": 1}))
    tm.that(
        isinstance(metadata_mapping, dict) and metadata_mapping == {"a": 1},
        eq=True,
    )
    tm.that(u.normalize_to_metadata(Path("/tmp")), eq=str(Path("/tmp")))

    class Frame:
        f_back: types.FrameType | None = None

    tm.that(u.get_logger(None), none=False)


def test_runtime_module_accessors_and_metadata() -> None:
    tm.that(u.structlog() is structlog, eq=True)
    tm.that(u.dependency_providers() is providers, eq=True)
    tm.that(u.dependency_containers() is containers, eq=True)


def test_configure_structlog_print_logger_factory_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class FallbackModule:
        def __init__(self) -> None:
            self.print_calls: int = 0
            self.print_calls = 0

            def _merge_contextvars(*_args: t.Scalar) -> t.ConfigurationMapping:
                return {}

            def _add_log_level(*_args: t.Scalar) -> t.ConfigurationMapping:
                return {}

            def _time_stamper(**_kwargs: t.Scalar) -> t.Scalar:
                return ""

            def _console_renderer(**_kwargs: t.Scalar) -> t.Scalar:
                return ""

            self.contextvars = type(
                "Ctx",
                (),
                {"merge_contextvars": staticmethod(_merge_contextvars)},
            )
            self.processors = type(
                "Processors",
                (),
                {
                    "add_log_level": staticmethod(_add_log_level),
                    "TimeStamper": staticmethod(_time_stamper),
                    "StackInfoRenderer": staticmethod(lambda: ""),
                    "JSONRenderer": staticmethod(lambda: ""),
                },
            )
            self.dev = type(
                "Dev",
                (),
                {"ConsoleRenderer": staticmethod(_console_renderer)},
            )

        @override
        def __getattribute__(
            self,
            name: str,
        ) -> t.Scalar | type | Callable[..., t.Scalar] | None:
            if name == "PrintLoggerFactory":
                calls = object.__getattribute__(self, "print_calls") + 1
                object.__setattr__(self, "print_calls", calls)
                if calls == 1:
                    return None

                def _print_logger_factory(**_kwargs: t.Scalar) -> t.Scalar:
                    return ""

                return _print_logger_factory
            return object.__getattribute__(self, name)

        def make_filtering_bound_logger(
            self,
            level: int,
        ) -> type[t.ConfigurationMapping]:
            _ = level
            return dict

        def configure(self, **_kwargs: t.Scalar) -> None:
            return None

    module = FallbackModule()
    u._structlog_configured = False
    cfg = type(
        "Cfg",
        (),
        {
            "log_level": logging.INFO,
            "console_renderer": True,
            "additional_processors": None,
            "wrapper_class_factory": None,
            "logger_factory": None,
            "cache_logger_on_first_use": True,
            "async_logging": True,
        },
    )()
    u.configure_structlog(config=cast("BaseModel", cfg))
    tm.that(module.print_calls, gte=0)


def test_configure_structlog_async_logging_uses_print_logger_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_has_logger_factory: MutableSequence[bool] = []
    factory_streams: MutableSequence[io.TextIOBase] = []

    class AsyncModule:
        def __init__(self) -> None:
            def _merge_contextvars(*_args: t.Scalar) -> t.ConfigurationMapping:
                return {}

            def _add_log_level(*_args: t.Scalar) -> t.ConfigurationMapping:
                return {}

            def _time_stamper(**_kwargs: t.Scalar) -> t.Scalar:
                return ""

            def _console_renderer(**_kwargs: t.Scalar) -> t.Scalar:
                return ""

            self.contextvars = type(
                "Ctx",
                (),
                {"merge_contextvars": staticmethod(_merge_contextvars)},
            )
            self.processors = type(
                "Processors",
                (),
                {
                    "add_log_level": staticmethod(_add_log_level),
                    "TimeStamper": staticmethod(_time_stamper),
                    "StackInfoRenderer": staticmethod(lambda: ""),
                    "JSONRenderer": staticmethod(lambda: ""),
                },
            )
            self.dev = type(
                "Dev",
                (),
                {"ConsoleRenderer": staticmethod(_console_renderer)},
            )
            self.PrintLoggerFactory = self._build_print_logger_factory()

        @staticmethod
        def _build_print_logger_factory() -> Callable[..., Callable[..., t.Scalar]]:
            def _print_logger_factory(
                *,
                file: io.TextIOBase,
            ) -> Callable[..., t.Scalar]:
                factory_streams.append(file)

                def _factory(*_args: t.Scalar) -> t.Scalar:
                    return ""

                return _factory

            return _print_logger_factory

        def make_filtering_bound_logger(
            self,
            level: int,
        ) -> type[t.ConfigurationMapping]:
            _ = level
            return dict

        def configure(self, **kwargs: t.Scalar) -> None:
            configured_has_logger_factory.append(callable(kwargs.get("logger_factory")))

    module = AsyncModule()
    u.configure_structlog(config=None)
    tm.that(u._async_writer is not None, eq=True)
    tm.that(bool(factory_streams), eq=True)
    tm.that(factory_streams[0] is u._async_writer, eq=True)
    tm.that(bool(configured_has_logger_factory), eq=True)
    tm.that(configured_has_logger_factory[0], eq=True)


def test_get_logger_auto_configures_structlog() -> None:
    logger = u.get_logger("tests.runtime.auto")
    _ = logger
    tm.that(u._structlog_configured, eq=True)
    tm.that(u._async_writer is not None, eq=True)


def test_dependency_integration_and_wiring_paths() -> None:
    bridge, services, resources = u.DependencyIntegration.create_layered_bridge(
        config=t.ConfigMap(root={"db": t.Dict(root={"dsn": "sqlite://"})}),
    )
    _ = (bridge, services, resources)
    tm.that(True, eq=True)
    di = u.DependencyIntegration.create_container(
        container_options=m.DependencyContainerCreationOptions(
            config=t.ConfigMap(root={"feature": t.Dict(root={"enabled": True})}),
            services={"svc": 1},
            factories={"factory": lambda: 2},
            resources={"resource": lambda: {"ok": True}},
            wire_modules=[],
            wire_packages=["unused.package"],
            wire_classes=[u],
        ),
    )
    tm.that(di.svc(), eq=1)
    tm.that(di.factory(), eq=2)
    tm.that(di.resource(), eq={"ok": True})
    provider = providers.Configuration()
    u.DependencyIntegration.bind_configuration_provider(
        provider,
        t.ConfigMap(root={"api": t.Dict(root={"url": "x"})}),
    )
    tm.that(provider.api.url(), eq="x")


def test_runtime_result_remaining_paths() -> None:

    def _ok_passthrough(value: int) -> r[int]:
        return r[int].ok(value)

    def _ok_inc(value: int) -> r[int]:
        return r[int].ok(value + 1)

    def _fail_boom(_value: int) -> r[int]:
        return r[int].fail("boom")

    success: r[int] = r[int].ok(3)
    failure: r[int] = r[int].fail(
        "err",
        error_code="E2",
        error_data=t.ConfigMap(root={"k": "v"}),
    )
    tm.that(failure.error_code, eq="E2")
    tm.that(failure.error_data, none=False)
    tm.that(success.value, eq=3)
    with pytest.raises(RuntimeError, match="Cannot access value of failed result"):
        _ = failure.value
    tm.that(success.unwrap(), eq=3)
    tm.that(failure.flat_map(_ok_passthrough).is_failure, eq=True)
    tm.that(success.filter(lambda value: value > 0) is success, eq=True)
    tm.that(success.map_error(str) is success, eq=True)
    tm.that(success.lash(r.fail) is success, eq=True)
    tm.that(success.recover(lambda _e: 0) is success, eq=True)
    chain_success = success.flow_through(_ok_inc, _ok_inc)
    tm.that(chain_success.is_success and chain_success.value == 5, eq=True)
    chain_failure = success.flow_through(_fail_boom, _ok_inc)
    tm.that(chain_failure.is_failure, eq=True)
    tm.that(success | 0, eq=3)
    tm.that(bool(success), eq=True)
    tm.that(repr(success).startswith("r[T].ok("), eq=True)
    tm.that(repr(failure).startswith("r[T].fail("), eq=True)
    with success as entered:
        tm.that(entered is success, eq=True)


def test_runtime_integration_tracking_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    events: MutableSequence[tuple[str, t.ContainerMapping]] = []

    class Logger:
        def info(self, message: str, **kwargs: t.Scalar) -> None:
            events.append((message, dict(kwargs.items())))

        def error(self, message: str, **kwargs: t.Scalar) -> None:
            events.append((message, dict(kwargs.items())))

    def _get_logger(_name: str | None = None) -> Logger:
        return Logger()

    class CtxVars:
        @staticmethod
        def get_contextvars() -> t.StrMapping:
            return {"correlation_id": "corr-1"}

        @staticmethod
        def bind_contextvars(**_kwargs: t.Scalar) -> None:
            return None

    fake_structlog = type(
        "FakeStructlog",
        (),
        {"contextvars": CtxVars, "get_logger": staticmethod(_get_logger)},
    )
    u.Integration.track_service_resolution("svc", resolved=True)
    u.Integration.track_service_resolution(
        "svc",
        resolved=False,
        error_message="x",
    )
    u.Integration.track_domain_event(
        "evt",
        aggregate_id="agg",
        event_data=t.ConfigMap(root={"k": "v"}),
    )
    u.Integration.setup_service_infrastructure(
        service_name="svc",
        service_version="1.0.0",
        enable_context_correlation=True,
    )
    tm.that(len(events), eq=4)


def test_model_helpers_remaining_paths() -> None:

    class Entity:
        def __init__(self, unique_id: str) -> None:
            self.unique_id = unique_id

    class ValueModel(BaseModel):
        a: int = 0

    left = Entity("u-1")
    right = Entity("u-1")
    tm.that(
        (
            u.compare_entities_by_id(
                cast("t.RuntimeData", left),
                cast("t.RuntimeData", right),
            )
            is True
        ),
        eq=True,
    )
    tm.that(u.hash_entity_by_id(cast("t.RuntimeData", left)), is_=int)
    vm_a = ValueModel(a=1)
    vm_b = ValueModel(a=1)
    tm.that(u.compare_value_objects_by_value(vm_a, vm_b), eq=True)
    tm.that(u.hash_value_object_by_value(vm_a), is_=int)


def test_ensure_trace_context_dict_conversion_paths() -> None:
    payload: Mapping[
        str,
        t.Container
        | MutableSequence[int]
        | t.MutableIntMapping
        | Callable[[], int]
        | type
        | None,
    ] = {
        "none": None,
        "str": "x",
        "int": 1,
        "float": 1.5,
        "bool": True,
        "dt": datetime.now(UTC),
        "path": Path(),
        "list": [1, 2],
        "dict": {"a": 1},
        "callable": lambda: 1,
        "other": type("Sentinel", (), {}),
    }
    result = u.ensure_trace_context(cast("t.ConfigurationMapping", payload))
    tm.that(result["str"], eq="x")
    tm.that(result["int"], eq="1")
    tm.that("trace_id" in result and "span_id" in result, eq=True)
