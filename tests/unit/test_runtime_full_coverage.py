"""Tests for runtime full coverage."""

from __future__ import annotations

import contextlib
import io
import logging
import queue
import types
from collections.abc import Callable, Generator, Iterator, Mapping
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import ClassVar, Self, cast, override

import pytest
from flext_tests import t, tm
from pydantic import BaseModel

import flext_core.runtime as runtime_module
from flext_core import FlextRuntime, r
from tests import c, m, u

runtime_tests: ModuleType = import_module("tests.unit.test_runtime")
runtime_cov_tests: ModuleType = import_module("tests.unit.test_runtime_coverage_100")


@pytest.fixture(autouse=True)
def reset_runtime_state() -> Generator[None]:
    FlextRuntime._structlog_configured = False
    if FlextRuntime._async_writer is not None:
        FlextRuntime._async_writer.shutdown()
        FlextRuntime._async_writer = None
    yield
    FlextRuntime._structlog_configured = False
    if FlextRuntime._async_writer is not None:
        FlextRuntime._async_writer.shutdown()
        FlextRuntime._async_writer = None


def test_reuse_existing_runtime_scenarios() -> None:
    suite = runtime_tests.TestFlextRuntime()
    _ = c.Logging.DEFAULT_LEVEL
    _ = u.ensure_str("x")
    for case in runtime_tests.RuntimeScenarios.DICT_LIKE_SCENARIOS:
        suite.test_dict_like_validation(case)
    for case in runtime_tests.RuntimeScenarios.LIST_LIKE_SCENARIOS:
        suite.test_list_like_validation(case)
    for case in runtime_tests.RuntimeScenarios.JSON_SCENARIOS:
        suite.test_json_validation(case)
    for case in runtime_tests.RuntimeScenarios.IDENTIFIER_SCENARIOS:
        suite.test_identifier_validation(case)
    for case in runtime_tests.RuntimeScenarios.SERIALIZATION_SCENARIOS:
        suite.test_safe_get_attribute(case)
    for case in runtime_tests.RuntimeScenarios.GENERIC_ARGS_SCENARIOS:
        suite.test_extract_generic_args(case)
    for case in runtime_tests.RuntimeScenarios.SEQUENCE_TYPE_SCENARIOS:
        suite.test_sequence_type_detection(case)
    for case in runtime_tests.RuntimeScenarios.STRUCTLOG_CONFIG_SCENARIOS:
        suite.test_structlog_configuration(case)


def test_reuse_existing_runtime_coverage_branches() -> None:
    dict_like = runtime_cov_tests.TestRuntimeDictLike()
    dict_like.test_is_dict_like_with_exception_on_items()
    dict_like.test_is_dict_like_with_exception_on_items_typeerror()
    dict_like.test_is_dict_like_with_userdict()
    dict_like.test_is_dict_like_with_missing_attributes()
    dict_like.test_is_dict_like_with_missing_keys()
    dict_like.test_is_dict_like_with_missing_items()
    dict_like.test_is_dict_like_with_missing_get()
    type_check = runtime_cov_tests.TestRuntimeTypeChecking()
    type_check.test_extract_generic_args_with_type_mapping()
    type_check.test_is_sequence_type_with_type_mapping()
    type_check.test_level_based_context_filter_malformed_prefix()
    type_check.test_configure_structlog_with_config_object()
    type_check.test_enable_runtime_checking()
    type_check.test_is_valid_json_exception_path()
    type_check.test_is_valid_identifier_non_string()
    type_check.test_extract_generic_args_with_typing_get_args()
    type_check.test_extract_generic_args_exception_path()
    type_check.test_is_sequence_type_with_origin()
    type_check.test_is_sequence_type_with_sequence_subclass()
    type_check.test_is_sequence_type_exception_path()
    type_check.test_level_based_context_filter_with_level_prefixed()
    type_check.test_configure_structlog_with_config_additional_processors()


def test_async_log_writer_paths() -> None:

    class Stream(io.StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.messages: list[str] = []
            self.flushed = 0

        @override
        def write(self, message: str) -> int:
            self.messages.append(message)
            return len(message)

        @override
        def flush(self) -> None:
            self.flushed += 1

    stream = Stream()
    writer = FlextRuntime._AsyncLogWriter(stream)
    writer.write("hello")
    writer.shutdown()
    writer.shutdown()
    tm.that(writer.stop_event.is_set(), eq=True)

    class EmptyQueue:
        def get(self, timeout: float = 0.1) -> str:
            _ = timeout
            raise queue.Empty

    forced = cast(
        "FlextRuntime._AsyncLogWriter",
        cast("object", object.__new__(FlextRuntime._AsyncLogWriter)),
    )
    forced.stream = stream
    forced.queue = cast("queue.Queue[str | None]", cast("object", EmptyQueue()))
    forced.stop_event = runtime_module.threading.Event()
    forced.stop_event.set()
    forced._worker()

    class FailingStream(io.StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.first: bool = True
            self.messages: list[str] = []

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
        "FlextRuntime._AsyncLogWriter",
        cast("object", object.__new__(FlextRuntime._AsyncLogWriter)),
    )
    broken.stream = failing
    broken.queue = cast("queue.Queue[str | None]", cast("object", SequenceQueue()))
    broken.stop_event = runtime_module.threading.Event()
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
        "FlextRuntime._AsyncLogWriter",
        cast("object", object.__new__(FlextRuntime._AsyncLogWriter)),
    )
    continue_writer.stream = stream
    continue_writer.queue = cast(
        "queue.Queue[str | None]",
        cast("object", EmptyThenSentinelQueue()),
    )
    continue_writer.stop_event = runtime_module.threading.Event()
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
        "FlextRuntime._AsyncLogWriter",
        cast("object", object.__new__(FlextRuntime._AsyncLogWriter)),
    )
    writer.stream = stream
    writer.queue = cast("queue.Queue[str | None]", cast("object", FullQueue()))
    writer.stop_event = runtime_module.threading.Event()
    thread = JoinRecorderThread()
    writer.thread = cast("runtime_module.threading.Thread", cast("object", thread))
    writer.shutdown()
    tm.that(writer.stop_event.is_set(), eq=True)
    tm.that(thread.join_timeout is not None, eq=True)
    if thread.join_timeout is not None:
        tm.that(abs(thread.join_timeout - 2.0), lt=1e-09)
    tm.that(stream.flush_calls, eq=1)


def test_runtime_create_instance_failure_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class FakeObject:
        def __new__(cls) -> Self:
            _ = cls
            return cast("Self", object())

    monkeypatch.setattr(runtime_module, "object", FakeObject, raising=False)

    class Marker:
        pass

    with pytest.raises(TypeError, match="did not return instance"):
        FlextRuntime.create_instance(Marker)


def test_normalization_edge_branches() -> None:
    cfg = m.ConfigMap(root={"a": 1})
    normalized_cfg = FlextRuntime.normalize_to_container(cfg)
    tm.that(isinstance(normalized_cfg, (m.ConfigMap, m.Dict)), eq=True)
    tm.that(getattr(normalized_cfg, "root", None), eq={"a": 1})

    class DictLike(Mapping[str, object]):
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

    normalized_dict_like = FlextRuntime.normalize_to_container(
        cast("FlextRuntime.RuntimeData", DictLike()),
    )
    tm.that(isinstance(normalized_dict_like, m.Dict), eq=True)
    tm.that(getattr(normalized_dict_like, "root", None), eq={"x": 1})
    metadata_cfg = FlextRuntime.normalize_to_metadata(cfg)
    tm.that(isinstance(metadata_cfg, str), eq=True)
    metadata_dict_like = FlextRuntime.normalize_to_metadata(
        cast("FlextRuntime.RuntimeData", DictLike()),
    )
    tm.that(
        isinstance(metadata_dict_like, dict) and metadata_dict_like == {"x": 1}, eq=True
    )
    metadata_list = FlextRuntime.normalize_to_metadata(
        cast("FlextRuntime.RuntimeData", ["a", object()]),
    )
    tm.that(isinstance(metadata_list, list), eq=True)


def test_deprecated_normalize_to_general_value_warns() -> None:
    """Verify deprecated normalize_to_general_value emits DeprecationWarning."""
    # COMPATIBILITY: normalize_to_general_value deprecated → use normalize_to_container
    # Planned removal: v0.12
    with pytest.warns(
        DeprecationWarning, match="normalize_to_general_value is deprecated"
    ):
        result = FlextRuntime.normalize_to_general_value("hello")
    tm.that(result, eq="hello")


def test_deprecated_normalize_to_metadata_value_warns() -> None:
    """Verify deprecated normalize_to_metadata_value emits DeprecationWarning."""
    # COMPATIBILITY: normalize_to_metadata_value deprecated → use normalize_to_metadata
    # Planned removal: v0.12
    with pytest.warns(
        DeprecationWarning, match="normalize_to_metadata_value is deprecated"
    ):
        result = FlextRuntime.normalize_to_metadata_value(42)
    tm.that(result is not None, eq=True)


def test_get_logger_none_name_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_with_frame = FlextRuntime.get_logger()
    tm.that(logger_with_frame is not None, eq=True)
    monkeypatch.setattr(runtime_module.inspect, "currentframe", lambda: None)
    logger_no_frame = FlextRuntime.get_logger()
    tm.that(logger_no_frame is not None, eq=True)


def test_dependency_registration_duplicate_guards() -> None:
    container = FlextRuntime.DependencyIntegration.create_container()
    FlextRuntime.DependencyIntegration.register_object(container, "svc", 1)
    with pytest.raises(ValueError, match="already registered"):
        FlextRuntime.DependencyIntegration.register_object(container, "svc", 2)
    FlextRuntime.DependencyIntegration.register_factory(container, "factory", lambda: 1)
    with pytest.raises(ValueError, match="already registered"):
        FlextRuntime.DependencyIntegration.register_factory(
            container,
            "factory",
            lambda: 2,
        )
    FlextRuntime.DependencyIntegration.register_resource(
        container,
        "resource",
        lambda: 1,
    )
    with pytest.raises(ValueError, match="already registered"):
        FlextRuntime.DependencyIntegration.register_resource(
            container,
            "resource",
            lambda: 2,
        )


def test_configure_structlog_edge_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, t.Scalar]] = []

    class StatefulModule:
        def __init__(self) -> None:
            self._print_access: int = 0

            def _merge_contextvars(*_args: t.Scalar) -> dict[str, t.Scalar]:
                return {}

            def _add_log_level(*_args: t.Scalar) -> dict[str, t.Scalar]:
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
            self, name: str
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
    monkeypatch.setattr(runtime_module, "structlog", fake_module)

    class Config:
        log_level: int = logging.DEBUG
        console_renderer: bool = True
        additional_processors: ClassVar[list[Callable[..., dict[str, t.Scalar]]]] = [
            lambda *_args: {},
        ]
        wrapper_class_factory: Callable[..., t.Scalar] | None = None
        logger_factory: Callable[[], t.Scalar] = staticmethod(lambda: "")
        cache_logger_on_first_use: bool = True
        async_logging: bool = True

    _ = Config  # referenced for pyright
    FlextRuntime.configure_structlog(config=None)
    tm.that(FlextRuntime.is_structlog_configured(), eq=True)
    tm.that(bool(calls), eq=True)
    FlextRuntime._structlog_configured = False
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
        additional_processors: list[Callable[..., dict[str, t.Scalar]]] | None = None
        wrapper_class_factory: Callable[..., t.Scalar] | None = None
        logger_factory: Callable[..., t.Scalar] | None = None
        cache_logger_on_first_use: bool = True
        async_logging: bool = False

    _ = ConfigNoAsync  # referenced for pyright
    FlextRuntime.configure_structlog(config=None)
    tm.that(FlextRuntime._structlog_configured, eq=True)
    FlextRuntime._structlog_configured = False
    calls.clear()
    fake_module._print_access = 0

    class ConfigAsyncFallback:
        log_level: int = logging.INFO
        console_renderer: bool = True
        additional_processors: list[Callable[..., dict[str, t.Scalar]]] | None = None
        wrapper_class_factory: Callable[..., t.Scalar] | None = None
        logger_factory: Callable[..., t.Scalar] | None = None
        cache_logger_on_first_use: bool = True
        async_logging: bool = True

    _ = ConfigAsyncFallback  # referenced for pyright
    FlextRuntime.configure_structlog(config=None)
    tm.that(FlextRuntime._structlog_configured, eq=True)


def test_reconfigure_and_reset_state_paths() -> None:

    class DummyWriter:
        def __init__(self) -> None:
            self.called: bool = False

        def shutdown(self) -> None:
            self.called = True

    dummy = DummyWriter()
    FlextRuntime._async_writer = cast(
        "FlextRuntime._AsyncLogWriter",
        cast("object", dummy),
    )
    FlextRuntime._structlog_configured = True
    FlextRuntime.reconfigure_structlog(log_level=logging.DEBUG, console_renderer=True)
    tm.that(dummy.called, eq=True)
    FlextRuntime.reset_structlog_state_for_testing()
    tm.that(FlextRuntime._structlog_configured, eq=False)


def test_runtime_result_all_missed_branches() -> None:

    def _plus_one(value: int) -> int:
        return value + 1

    def _raise_bad(_value: int) -> int:
        msg = "bad"
        raise ValueError(msg)

    def _ok_plus_one(value: int | None) -> FlextRuntime.RuntimeResult[int | None]:
        if value is None:
            return FlextRuntime.RuntimeResult[int | None].fail("none")
        return FlextRuntime.RuntimeResult[int | None].ok(value + 1)

    def _ok_plus_two(value: int) -> FlextRuntime.RuntimeResult[int]:
        return FlextRuntime.RuntimeResult[int].ok(value + 2)

    def _error_to_int(error: str) -> int:
        return len(error)

    success: FlextRuntime.RuntimeResult[int] = FlextRuntime.RuntimeResult[int].ok(1)
    failure: FlextRuntime.RuntimeResult[int] = FlextRuntime.RuntimeResult[int].fail(
        "e",
        error_code="E1",
        error_data=m.ConfigMap(root={"x": 1}),
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
    tapped: list[int] = []
    success.tap(lambda x: tapped.append(x))
    tm.that(tapped, eq=[1])
    errors: list[str] = []
    failure.tap_error(lambda err: errors.append(err))
    tm.that(errors, eq=["e"])
    tm.that(failure.map_error(lambda err: err.upper()).error, eq="E")
    tm.that(success.map_error(lambda err: err.upper()) is success, eq=True)
    filtered = success.filter(lambda value: value > 10)
    tm.that(filtered.is_failure, eq=True)
    tm.that(filtered.error, eq="Filter predicate failed")
    tm.that(failure.map_error(lambda err: f"{err}-alt").error, eq="e-alt")
    tm.that(
        failure.lash(lambda _err: FlextRuntime.RuntimeResult[int].ok(5)).value, eq=5
    )
    tm.that(failure.recover(lambda _err: 7).value, eq=7)

    class NoneValueResult(FlextRuntime.RuntimeResult[int | None]):
        @property
        @override
        def value(self) -> int | None:
            return None

    none_success: FlextRuntime.RuntimeResult[int | None] = NoneValueResult(
        is_success=True, error=None, error_code=None, error_data=None
    )
    flowed = none_success.flow_through(_ok_plus_one)
    tm.that(flowed is none_success, eq=True)

    none_ok = FlextRuntime.RuntimeResult[int | None].ok(None)
    tm.that(none_ok.is_success, eq=True)
    tm.that(none_ok.value, none=True)
    none_error: FlextRuntime.RuntimeResult[int] = FlextRuntime.RuntimeResult[int].fail(
        None,
    )
    tm.that(none_error.error, eq="")
    broken = FlextRuntime.RuntimeResult[int](
        is_success=True, error=None, error_code=None, error_data=None
    )
    tm.that(broken.value, none=True)


def test_model_support_and_hash_compare_paths() -> None:
    prefixed = FlextRuntime.generate_prefixed_id("item", length=8)
    tm.that(
        prefixed.startswith("item_") and len(prefixed.split("_", 1)[1]) == 8, eq=True
    )
    tm.that(
        (
            FlextRuntime.compare_entities_by_id(
                "a", cast("FlextRuntime.RuntimeData", object())
            )
            is False
        ),
        eq=True,
    )
    tm.that(
        (
            FlextRuntime.compare_entities_by_id(
                cast("FlextRuntime.RuntimeData", object()), 3
            )
            is False
        ),
        eq=True,
    )

    class A:
        unique_id: str = "1"

    class B:
        unique_id: str = "1"

    tm.that(
        (
            FlextRuntime.compare_entities_by_id(
                cast("FlextRuntime.RuntimeData", A()),
                cast("FlextRuntime.RuntimeData", B()),
            )
            is False
        ),
        eq=True,
    )
    obj = cast("FlextRuntime.RuntimeData", object())
    tm.that(
        FlextRuntime.hash_entity_by_id(obj),
        eq=hash(
            id(obj),
        ),
    )
    tm.that(FlextRuntime.compare_value_objects_by_value("a", "a"), eq=True)
    tm.that(
        (
            FlextRuntime.compare_value_objects_by_value(
                cast("FlextRuntime.RuntimeData", object()),
                1,
            )
            is False
        ),
        eq=True,
    )
    tm.that(FlextRuntime.compare_value_objects_by_value([1], [1]), eq=True)

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
            FlextRuntime.compare_value_objects_by_value(
                cast("FlextRuntime.RuntimeData", C()),
                cast("FlextRuntime.RuntimeData", D()),
            )
            is False
        ),
        eq=True,
    )
    tm.that(
        (
            FlextRuntime.compare_value_objects_by_value(
                cast("FlextRuntime.RuntimeData", C()),
                cast("FlextRuntime.RuntimeData", C()),
            )
            is True
        ),
        eq=True,
    )
    tm.that(isinstance(FlextRuntime.hash_value_object_by_value("x"), int), eq=True)
    tm.that(isinstance(FlextRuntime.hash_value_object_by_value({"a": 1}), int), eq=True)
    tm.that(isinstance(FlextRuntime.hash_value_object_by_value([1, 2]), int), eq=True)
    tm.that(
        isinstance(
            FlextRuntime.hash_value_object_by_value(MappingProxyType({"a": 1})), int
        ),
        eq=True,
    )
    tm.that(isinstance(FlextRuntime.hash_value_object_by_value((1, 2)), int), eq=True)
    tm.that(
        isinstance(FlextRuntime.hash_value_object_by_value(datetime.now(UTC)), int),
        eq=True,
    )

    class Empty:
        pass

    tm.that(isinstance(FlextRuntime.Bootstrap.create_instance(Empty), Empty), eq=True)


def test_config_bridge_and_trace_context_and_http_validation() -> None:
    level = FlextRuntime.get_log_level_from_config()
    tm.that(isinstance(level, int), eq=True)
    trace_from_scalar = FlextRuntime.ensure_trace_context(
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

    trace_from_model = FlextRuntime.ensure_trace_context(TraceModel())
    tm.that(trace_from_model["key"], eq="value")

    # ensure_trace_context catches RuntimeError internally for bad mappings
    bad_trace = FlextRuntime.ensure_trace_context({"broken": "x"})
    tm.that(bad_trace, has="trace_id")  # graceful fallback
    tm.that(bad_trace, has="span_id")
    trace_from_mapping = FlextRuntime.ensure_trace_context(MappingProxyType({"a": "b"}))
    tm.that(trace_from_mapping, has="trace_id")
    trace_from_other = FlextRuntime.ensure_trace_context("path")
    tm.that(trace_from_other, has="trace_id")
    ok_statuses: list[int | str] = [200, "201"]
    ok_result = FlextRuntime.validate_http_status_codes(ok_statuses)
    tm.that(ok_result.is_success and ok_result.value == [200, 201], eq=True)
    bad_range = FlextRuntime.validate_http_status_codes([99])
    tm.that(
        bad_range.is_failure and "Invalid HTTP status code" in (bad_range.error or ""),
        eq=True,
    )
    invalid_statuses: list[int | str] = cast("list[int | str]", [object()])
    bad_type = FlextRuntime.validate_http_status_codes(invalid_statuses)
    tm.that(
        bad_type.is_failure and "Cannot convert to integer" in (bad_type.error or ""),
        eq=True,
    )
    bad_value = FlextRuntime.validate_http_status_codes(["abc"])
    tm.that(
        bad_value.is_failure and "Cannot convert to integer" in (bad_value.error or ""),
        eq=True,
    )


def test_runtime_result_alias_compatibility() -> None:
    rr: FlextRuntime.RuntimeResult[int] = FlextRuntime.RuntimeResult[int].ok(10)
    wrapped: r[int] = r[int].ok(rr.value)
    tm.that(isinstance(wrapped, r), eq=True)


def test_runtime_misc_remaining_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    FlextRuntime._structlog_configured = False
    FlextRuntime.ensure_structlog_configured()
    tm.that(FlextRuntime.is_structlog_configured(), eq=True)

    class BasicModel(BaseModel):
        value: int = 1

    tm.that(FlextRuntime.is_base_model(BasicModel()), eq=True)
    normalized_mapping = FlextRuntime.normalize_to_container(
        MappingProxyType({"k": "v"}),
    )
    tm.that(isinstance(normalized_mapping, m.Dict), eq=True)
    tm.that(getattr(normalized_mapping, "root", None), eq={"k": "v"})
    norm_list = FlextRuntime.normalize_to_container([1, "x"])
    tm.that(isinstance(norm_list, m.ObjectList), eq=True)
    tm.that(list(getattr(norm_list, "root", [])), eq=[1, "x"])
    # Path is Container, returned as-is
    tm.that(FlextRuntime.normalize_to_container(Path("/tmp")), eq=Path("/tmp"))
    tm.that(FlextRuntime.normalize_to_metadata(1), eq=1)
    metadata_mapping = FlextRuntime.normalize_to_metadata(MappingProxyType({"a": 1}))
    tm.that(
        isinstance(metadata_mapping, dict) and metadata_mapping == {"a": 1}, eq=True
    )
    tm.that(FlextRuntime.normalize_to_metadata(Path("/tmp")), eq=str(Path("/tmp")))

    class Frame:
        f_back: types.FrameType | None = None

    monkeypatch.setattr(runtime_module.inspect, "currentframe", lambda: Frame())
    tm.that(FlextRuntime.get_logger(None) is not None, eq=True)


def test_runtime_module_accessors_and_metadata() -> None:
    metadata_ref = FlextRuntime.Metadata
    tm.that(isinstance(metadata_ref, type), eq=True)
    metadata = metadata_ref()
    tm.that(metadata.version, eq="1.0.0")
    tm.that(FlextRuntime.structlog() is runtime_module.structlog, eq=True)
    tm.that(FlextRuntime.dependency_providers() is runtime_module.providers, eq=True)
    tm.that(FlextRuntime.dependency_containers() is runtime_module.containers, eq=True)


def test_configure_structlog_print_logger_factory_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class FallbackModule:
        def __init__(self) -> None:
            self.print_calls: int = 0
            self.print_calls = 0

            def _merge_contextvars(*_args: t.Scalar) -> dict[str, t.Scalar]:
                return {}

            def _add_log_level(*_args: t.Scalar) -> dict[str, t.Scalar]:
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
            self, name: str
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

        def make_filtering_bound_logger(self, level: int) -> type[dict[str, t.Scalar]]:
            _ = level
            return dict

        def configure(self, **_kwargs: t.Scalar) -> None:
            return None

    module = FallbackModule()
    monkeypatch.setattr(runtime_module, "structlog", module)
    FlextRuntime._structlog_configured = False
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
    FlextRuntime.configure_structlog(config=cast("BaseModel", cfg))
    tm.that(module.print_calls >= 2, eq=True)


def test_dependency_integration_and_wiring_paths() -> None:
    bridge, services, resources = (
        FlextRuntime.DependencyIntegration.create_layered_bridge(
            config=m.ConfigMap(root={"db": m.Dict(root={"dsn": "sqlite://"})}),
        )
    )
    tm.that(
        bridge is not None and services is not None and (resources is not None), eq=True
    )
    di = FlextRuntime.DependencyIntegration.create_container(
        config=m.ConfigMap(root={"feature": m.Dict(root={"enabled": True})}),
        services={"svc": 1},
        factories={"factory": lambda: 2},
        resources={"resource": lambda: {"ok": True}},
        wire_modules=[],
        wire_packages=["unused.package"],
        wire_classes=[FlextRuntime],
    )
    tm.that(getattr(getattr(di.config, "feature"), "enabled")(), eq=True)
    tm.that(di.svc(), eq=1)
    tm.that(di.factory(), eq=2)
    tm.that(di.resource(), eq={"ok": True})
    provider = runtime_module.providers.Configuration()
    FlextRuntime.DependencyIntegration.bind_configuration_provider(
        provider,
        m.ConfigMap(root={"api": m.Dict(root={"url": "x"})}),
    )
    tm.that(provider.api.url(), eq="x")


def test_runtime_result_remaining_paths() -> None:

    def _ok_passthrough(value: int) -> FlextRuntime.RuntimeResult[int]:
        return FlextRuntime.RuntimeResult[int].ok(value)

    def _ok_inc(value: int) -> FlextRuntime.RuntimeResult[int]:
        return FlextRuntime.RuntimeResult[int].ok(value + 1)

    def _fail_boom(_value: int) -> FlextRuntime.RuntimeResult[int]:
        return FlextRuntime.RuntimeResult[int].fail("boom")

    success: FlextRuntime.RuntimeResult[int] = FlextRuntime.RuntimeResult[int].ok(3)
    failure: FlextRuntime.RuntimeResult[int] = FlextRuntime.RuntimeResult[int].fail(
        "err",
        error_code="E2",
        error_data=m.ConfigMap(root={"k": "v"}),
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
    tm.that(success.lash(FlextRuntime.RuntimeResult.fail) is success, eq=True)
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
    events: list[tuple[str, dict[str, t.Tests.object]]] = []

    class Logger:
        def info(self, message: str, **kwargs: t.Scalar) -> None:
            events.append((message, dict(kwargs.items())))

        def error(self, message: str, **kwargs: t.Scalar) -> None:
            events.append((message, dict(kwargs.items())))

    def _get_logger(_name: str | None = None) -> Logger:
        return Logger()

    class CtxVars:
        @staticmethod
        def get_contextvars() -> dict[str, str]:
            return {"correlation_id": "corr-1"}

        @staticmethod
        def bind_contextvars(**_kwargs: t.Scalar) -> None:
            return None

    fake_structlog = type(
        "FakeStructlog",
        (),
        {"contextvars": CtxVars, "get_logger": staticmethod(_get_logger)},
    )
    monkeypatch.setattr(runtime_module, "structlog", fake_structlog)
    FlextRuntime.Integration.track_service_resolution("svc", resolved=True)
    FlextRuntime.Integration.track_service_resolution(
        "svc",
        resolved=False,
        error_message="x",
    )
    FlextRuntime.Integration.track_domain_event(
        "evt",
        aggregate_id="agg",
        event_data=m.ConfigMap(root={"k": "v"}),
    )
    FlextRuntime.Integration.setup_service_infrastructure(
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
            FlextRuntime.compare_entities_by_id(
                cast("FlextRuntime.RuntimeData", left),
                cast("FlextRuntime.RuntimeData", right),
            )
            is True
        ),
        eq=True,
    )
    tm.that(
        isinstance(
            FlextRuntime.hash_entity_by_id(cast("FlextRuntime.RuntimeData", left)), int
        ),
        eq=True,
    )
    vm_a = ValueModel(a=1)
    vm_b = ValueModel(a=1)
    tm.that(FlextRuntime.compare_value_objects_by_value(vm_a, vm_b), eq=True)
    tm.that(isinstance(FlextRuntime.hash_value_object_by_value(vm_a), int), eq=True)


def test_ensure_trace_context_dict_conversion_paths() -> None:
    payload: dict[
        str,
        t.Scalar | Path | list[int] | dict[str, int] | Callable[[], int] | type | None,
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
    result = FlextRuntime.ensure_trace_context(cast("Mapping[str, t.Scalar]", payload))
    tm.that(result["str"], eq="x")
    tm.that(result["int"], eq="1")
    tm.that("trace_id" in result and "span_id" in result, eq=True)
