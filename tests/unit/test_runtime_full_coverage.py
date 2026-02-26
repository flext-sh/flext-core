from __future__ import annotations

import logging
import queue
import contextlib
import io
from importlib import import_module
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Callable, Generator, cast

import pytest
from pydantic import BaseModel

import flext_core.runtime as runtime_module
from flext_core import c, m, r, t, u
from flext_core.runtime import FlextRuntime

runtime_tests: ModuleType = import_module("tests.unit.test_runtime")
runtime_cov_tests: ModuleType = import_module(
    "tests.unit.test_runtime_coverage_100",
)


@pytest.fixture(autouse=True)
def reset_runtime_state() -> Generator[None, None, None]:
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
    _ = u.Mapper.ensure_str("x")

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

        def write(self, message: str) -> int:
            self.messages.append(message)
            return len(message)

        def flush(self) -> None:
            self.flushed += 1

    stream = Stream()
    writer = FlextRuntime._AsyncLogWriter(stream)
    writer.write("hello")
    writer.shutdown()
    writer.shutdown()
    assert writer.stop_event.is_set()

    class EmptyQueue:
        def get(self, timeout: float = 0.1) -> str:
            _ = timeout
            raise queue.Empty

    forced = cast(
        "FlextRuntime._AsyncLogWriter",
        cast(object, object.__new__(FlextRuntime._AsyncLogWriter)),
    )
    forced.stream = stream
    forced.queue = cast("queue.Queue[str | None]", cast(object, EmptyQueue()))
    forced.stop_event = runtime_module.threading.Event()
    forced.stop_event.set()
    forced._worker()

    class FailingStream(io.StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.first = True
            self.messages: list[str] = []

        def write(self, message: str) -> int:
            if self.first:
                self.first = False
                raise OSError("boom")
            self.messages.append(message)
            return len(message)

        def flush(self) -> None:
            return None

    class SequenceQueue:
        def __init__(self) -> None:
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
        cast(object, object.__new__(FlextRuntime._AsyncLogWriter)),
    )
    broken.stream = failing
    broken.queue = cast("queue.Queue[str | None]", cast(object, SequenceQueue()))
    broken.stop_event = runtime_module.threading.Event()
    broken._worker()
    assert "Error in async log writer\n" in failing.messages

    class EmptyThenSentinelQueue:
        def __init__(self) -> None:
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
        cast(object, object.__new__(FlextRuntime._AsyncLogWriter)),
    )
    continue_writer.stream = stream
    continue_writer.queue = cast(
        "queue.Queue[str | None]",
        cast(object, EmptyThenSentinelQueue()),
    )
    continue_writer.stop_event = runtime_module.threading.Event()
    continue_writer._worker()


def test_async_log_writer_shutdown_with_full_queue() -> None:
    class FlushOnlyStream(io.StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.flush_calls = 0

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
        cast(object, object.__new__(FlextRuntime._AsyncLogWriter)),
    )
    writer.stream = stream
    writer.queue = cast("queue.Queue[str | None]", cast(object, FullQueue()))
    writer.stop_event = runtime_module.threading.Event()
    thread = JoinRecorderThread()
    writer.thread = cast("runtime_module.threading.Thread", cast(object, thread))

    writer.shutdown()

    assert writer.stop_event.is_set()
    assert thread.join_timeout == 2.0
    assert stream.flush_calls == 1


def test_runtime_create_instance_failure_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeObject:
        def __new__(cls) -> FakeObject:
            _ = cls
            return cast("FakeObject", object())

    monkeypatch.setattr(runtime_module, "object", FakeObject, raising=False)

    class Marker:
        pass

    with pytest.raises(TypeError, match="did not return instance"):
        FlextRuntime.create_instance(Marker)


def test_normalization_edge_branches() -> None:
    cfg = t.ConfigMap(root={"a": 1})
    normalized_cfg = FlextRuntime.normalize_to_general_value(cfg)
    assert normalized_cfg == {"a": 1}

    class DictLike:
        def __getitem__(self, key: str) -> object:
            if key == "x":
                return 1
            raise KeyError(key)

        def keys(self) -> list[str]:
            return ["x"]

        def items(self) -> list[tuple[str, object]]:
            return [("x", 1)]

        def get(self, key: str, default: object = None) -> object:
            _ = key
            return default

        def __iter__(self):
            return iter([("x", 1)])

    normalized_dict_like = FlextRuntime.normalize_to_general_value(
        cast("t.GeneralValueType", cast(object, DictLike())),
    )
    assert normalized_dict_like == {"x": 1}

    metadata_cfg = FlextRuntime.normalize_to_metadata_value(cfg)
    assert metadata_cfg == '{"a": 1}'

    metadata_dict_like = FlextRuntime.normalize_to_metadata_value(
        cast("t.GeneralValueType", cast(object, DictLike())),
    )
    assert metadata_dict_like == '{"x": 1}'

    metadata_list = FlextRuntime.normalize_to_metadata_value(
        cast("t.GeneralValueType", ["a", object()]),
    )
    assert isinstance(metadata_list, list)
    assert metadata_list[0] == "a"
    assert isinstance(metadata_list[1], str)


def test_get_logger_none_name_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_with_frame = FlextRuntime.get_logger()
    assert logger_with_frame is not None

    monkeypatch.setattr(runtime_module.inspect, "currentframe", lambda: None)
    logger_no_frame = FlextRuntime.get_logger()
    assert logger_no_frame is not None


def test_dependency_registration_duplicate_guards() -> None:
    container = FlextRuntime.DependencyIntegration.create_container()
    FlextRuntime.DependencyIntegration.register_object(container, "svc", 1)
    with pytest.raises(ValueError, match="already registered"):
        FlextRuntime.DependencyIntegration.register_object(container, "svc", 2)

    FlextRuntime.DependencyIntegration.register_factory(container, "factory", lambda: 1)
    with pytest.raises(ValueError, match="already registered"):
        FlextRuntime.DependencyIntegration.register_factory(
            container, "factory", lambda: 2
        )

    FlextRuntime.DependencyIntegration.register_resource(
        container, "resource", lambda: 1
    )
    with pytest.raises(ValueError, match="already registered"):
        FlextRuntime.DependencyIntegration.register_resource(
            container, "resource", lambda: 2
        )


def test_configure_structlog_edge_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class StatefulModule:
        def __init__(self) -> None:
            self._print_access = 0
            self.contextvars = type(
                "Ctx",
                (),
                {"merge_contextvars": staticmethod(lambda *_args: {})},
            )
            self.processors = type(
                "Processors",
                (),
                {
                    "add_log_level": staticmethod(lambda *_args: {}),
                    "TimeStamper": staticmethod(lambda **_kw: object()),
                    "StackInfoRenderer": staticmethod(lambda: object()),
                    "JSONRenderer": staticmethod(lambda: object()),
                },
            )
            self.dev = type(
                "Dev",
                (),
                {"ConsoleRenderer": staticmethod(lambda **_kw: object())},
            )

        def reset_defaults(self) -> None:
            return None

        def make_filtering_bound_logger(self, level: int) -> type[object]:
            _ = level
            return dict

        def configure(self, **kwargs: object) -> None:
            calls.append(kwargs)

        def __getattr__(self, name: str) -> object:
            if name != "PrintLoggerFactory":
                raise AttributeError(name)
            self._print_access += 1
            if self._print_access == 1:
                raise AttributeError(name)
            return lambda **_kwargs: object()

    fake_module = StatefulModule()
    monkeypatch.setattr(runtime_module, "structlog", fake_module)

    class Config:
        log_level = logging.DEBUG
        console_renderer = True
        additional_processors: list[Callable[..., object]] = [lambda *_args: {}]
        wrapper_class_factory = None
        logger_factory = staticmethod(lambda: object())
        cache_logger_on_first_use = True
        async_logging = True

    FlextRuntime.configure_structlog(
        config=cast("t.GeneralValueType", cast(object, Config())),
    )
    assert FlextRuntime.is_structlog_configured() is True
    assert calls

    FlextRuntime._structlog_configured = False
    calls.clear()
    fake_module._print_access = 0
    with contextlib.suppress(AttributeError):
        delattr(fake_module, "PrintLoggerFactory")
    setattr(fake_module, "PrintLoggerFactory", lambda **_kwargs: object())

    class ConfigNoAsync:
        log_level = logging.INFO
        console_renderer = True
        additional_processors = None
        wrapper_class_factory = None
        logger_factory = None
        cache_logger_on_first_use = True
        async_logging = False

    FlextRuntime.configure_structlog(
        config=cast("t.GeneralValueType", cast(object, ConfigNoAsync())),
    )
    assert FlextRuntime._structlog_configured

    FlextRuntime._structlog_configured = False
    calls.clear()
    fake_module._print_access = 0

    class ConfigAsyncFallback:
        log_level = logging.INFO
        console_renderer = True
        additional_processors = None
        wrapper_class_factory = None
        logger_factory = None
        cache_logger_on_first_use = True
        async_logging = True

    FlextRuntime.configure_structlog(
        config=cast("t.GeneralValueType", cast(object, ConfigAsyncFallback())),
    )
    assert FlextRuntime._structlog_configured


def test_reconfigure_and_reset_state_paths() -> None:
    class DummyWriter:
        def __init__(self) -> None:
            self.called = False

        def shutdown(self) -> None:
            self.called = True

    dummy = DummyWriter()
    FlextRuntime._async_writer = cast(
        "FlextRuntime._AsyncLogWriter",
        cast(object, dummy),
    )
    FlextRuntime._structlog_configured = True
    FlextRuntime.reconfigure_structlog(log_level=logging.DEBUG, console_renderer=True)
    assert dummy.called is True
    FlextRuntime.reset_structlog_state_for_testing()
    assert not FlextRuntime._structlog_configured


def test_runtime_result_all_missed_branches() -> None:
    success = FlextRuntime.RuntimeResult.ok(1)
    failure: FlextRuntime.RuntimeResult[int] = FlextRuntime.RuntimeResult.fail(
        "e", error_code="E1", error_data=m.ConfigMap(root={"x": 1})
    )

    assert success.result is success
    assert success.unwrap_or(9) == 1
    assert failure.unwrap_or(9) == 9
    assert success.unwrap_or_else(lambda: 7) == 1
    assert failure.unwrap_or_else(lambda: 7) == 7

    with pytest.raises(RuntimeError, match="Cannot unwrap failed result"):
        failure.unwrap()

    mapped_ok = success.map(lambda x: x + 1)
    assert mapped_ok.is_success and mapped_ok.value == 2
    mapped_error = success.map(lambda _x: (_ for _ in ()).throw(ValueError("bad")))
    assert mapped_error.is_failure
    mapped_failed = failure.map(lambda x: x)
    assert mapped_failed.is_failure

    flat_mapped = success.flat_map(lambda x: FlextRuntime.RuntimeResult.ok(x + 1))
    assert flat_mapped.value == 2
    assert success.and_then(lambda x: FlextRuntime.RuntimeResult.ok(x + 2)).value == 3

    assert success.fold(lambda err: err, lambda x: x + 1) == 2
    assert failure.fold(lambda err: f"{err}!", lambda x: x) == "e!"

    tapped: list[int] = []
    success.tap(lambda x: tapped.append(x))
    assert tapped == [1]
    errors: list[str] = []
    failure.tap_error(lambda err: errors.append(err))
    assert errors == ["e"]

    assert failure.map_error(lambda err: err.upper()).error == "E"
    assert success.map_error(lambda err: err.upper()) is success

    filtered = success.filter(lambda x: x > 10)
    assert filtered.is_failure
    assert filtered.error == "Filter predicate failed"

    assert failure.alt(lambda err: f"{err}-alt").error == "e-alt"
    assert failure.lash(lambda _err: FlextRuntime.RuntimeResult.ok(5)).value == 5
    assert failure.recover(lambda _err: 7).value == 7

    class NoneValueResult(FlextRuntime.RuntimeResult[int | None]):
        @property
        def value(self) -> int | None:
            return None

    none_success = NoneValueResult(value=1, is_success=True)
    flowed = none_success.flow_through(
        lambda x: FlextRuntime.RuntimeResult.ok(cast("int", x) + 1)
    )
    assert flowed is none_success

    assert success._protocol_name() == "RuntimeResult"
    with pytest.raises(
        ValueError, match="Cannot create success result with None value"
    ):
        FlextRuntime.RuntimeResult.ok(None)

    none_error: FlextRuntime.RuntimeResult[int] = FlextRuntime.RuntimeResult.fail(None)
    assert none_error.error == ""

    broken = FlextRuntime.RuntimeResult[int](value=None, is_success=True)
    with pytest.raises(RuntimeError, match="Invariant violation"):
        _ = broken.value


def test_model_support_and_hash_compare_paths() -> None:
    prefixed = FlextRuntime.generate_prefixed_id("item", length=8)
    assert prefixed.startswith("item_") and len(prefixed.split("_", 1)[1]) == 8

    assert (
        FlextRuntime.compare_entities_by_id(
            "a",
            cast("t.GeneralValueType", object()),
        )
        is False
    )
    assert (
        FlextRuntime.compare_entities_by_id(
            cast("t.GeneralValueType", object()),
            3,
        )
        is False
    )

    class A:
        unique_id = "1"

    class B:
        unique_id = "1"

    assert (
        FlextRuntime.compare_entities_by_id(
            cast("t.GeneralValueType", cast(object, A())),
            cast("t.GeneralValueType", cast(object, B())),
        )
        is False
    )
    # _is_scalar only matches datetime/None; strings fall to hash(id(entity))
    obj = object()
    assert FlextRuntime.hash_entity_by_id(
        cast("t.GeneralValueType", obj),
    ) == hash(id(obj))

    assert FlextRuntime.compare_value_objects_by_value("a", "a") is True
    assert (
        FlextRuntime.compare_value_objects_by_value(
            cast("t.GeneralValueType", object()),
            1,
        )
        is False
    )
    assert FlextRuntime.compare_value_objects_by_value([1], [1]) is True

    class C:
        def __repr__(self) -> str:
            return "same"

    class D:
        def __repr__(self) -> str:
            return "same"

    assert (
        FlextRuntime.compare_value_objects_by_value(
            cast("t.GeneralValueType", cast(object, C())),
            cast("t.GeneralValueType", cast(object, D())),
        )
        is False
    )
    assert (
        FlextRuntime.compare_value_objects_by_value(
            cast("t.GeneralValueType", cast(object, C())),
            cast("t.GeneralValueType", cast(object, C())),
        )
        is True
    )

    assert isinstance(FlextRuntime.hash_value_object_by_value("x"), int)
    assert isinstance(FlextRuntime.hash_value_object_by_value({"a": 1}), int)
    assert isinstance(FlextRuntime.hash_value_object_by_value([1, 2]), int)
    assert isinstance(
        FlextRuntime.hash_value_object_by_value(MappingProxyType({"a": 1})),
        int,
    )
    assert isinstance(FlextRuntime.hash_value_object_by_value((1, 2)), int)
    assert isinstance(FlextRuntime.hash_value_object_by_value(datetime.now(UTC)), int)

    class Empty:
        pass

    assert isinstance(FlextRuntime.Bootstrap.create_instance(Empty), Empty)


def test_config_bridge_and_trace_context_and_http_validation() -> None:
    level = FlextRuntime.get_log_level_from_config()
    assert isinstance(level, int)

    trace_from_scalar = FlextRuntime.ensure_trace_context(
        1,
        include_correlation_id=True,
        include_timestamp=True,
    )
    assert {"trace_id", "span_id", "correlation_id", "timestamp"}.issubset(
        trace_from_scalar
    )

    class TraceModel(BaseModel):
        key: str = "value"

    trace_from_model = FlextRuntime.ensure_trace_context(TraceModel())
    assert trace_from_model["key"] == "value"

    class BadDict(dict[object, object]):
        def items(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        FlextRuntime.ensure_trace_context(
            cast("t.GeneralValueType", BadDict())
        )

    trace_from_mapping = FlextRuntime.ensure_trace_context(MappingProxyType({"a": "b"}))
    assert "trace_id" in trace_from_mapping

    trace_from_other = FlextRuntime.ensure_trace_context(Path("."))
    assert "trace_id" in trace_from_other

    ok_statuses: list[t.GeneralValueType] = [200, "201"]
    ok_result = FlextRuntime.validate_http_status_codes(ok_statuses)
    assert ok_result.is_success and ok_result.value == [200, 201]

    bad_range = FlextRuntime.validate_http_status_codes([99])
    assert bad_range.is_failure and "Invalid HTTP status code" in (
        bad_range.error or ""
    )

    invalid_statuses: list[t.GeneralValueType] = [cast("t.GeneralValueType", object())]
    bad_type = FlextRuntime.validate_http_status_codes(invalid_statuses)
    assert bad_type.is_failure and "Invalid HTTP status code type" in (
        bad_type.error or ""
    )

    bad_value = FlextRuntime.validate_http_status_codes(["abc"])
    assert bad_value.is_failure and "Cannot convert to integer" in (
        bad_value.error or ""
    )


def test_runtime_result_alias_compatibility() -> None:
    rr: FlextRuntime.RuntimeResult[int] = FlextRuntime.RuntimeResult.ok(10)
    wrapped: r[int] = r[int].ok(rr.value)
    assert isinstance(wrapped, r)


def test_runtime_misc_remaining_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    FlextRuntime._structlog_configured = False
    FlextRuntime.ensure_structlog_configured()
    assert FlextRuntime.is_structlog_configured() is True

    class BasicModel(BaseModel):
        value: int = 1

    assert FlextRuntime.is_base_model(BasicModel()) is True

    normalized_mapping = FlextRuntime.normalize_to_general_value(
        MappingProxyType({"k": "v"}),
    )
    assert normalized_mapping == {"k": "v"}
    assert FlextRuntime.normalize_to_general_value([1, "x"]) == [1, "x"]
    assert FlextRuntime.normalize_to_general_value(Path("/tmp")) == "/tmp"

    assert FlextRuntime.normalize_to_metadata_value(1) == "1"
    assert (
        FlextRuntime.normalize_to_metadata_value(MappingProxyType({"a": 1}))
        == '{"a": 1}'
    )
    assert FlextRuntime.normalize_to_metadata_value(Path("/tmp")) == "/tmp"

    class Frame:
        f_back = None

    monkeypatch.setattr(runtime_module.inspect, "currentframe", lambda: Frame())
    assert FlextRuntime.get_logger(None) is not None


def test_runtime_module_accessors_and_metadata() -> None:
    metadata = FlextRuntime.Metadata()
    assert metadata.version == "1.0.0"
    assert FlextRuntime.structlog() is runtime_module.structlog
    assert FlextRuntime.dependency_providers() is runtime_module.providers
    assert FlextRuntime.dependency_containers() is runtime_module.containers


def test_configure_structlog_print_logger_factory_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FallbackModule:
        def __init__(self) -> None:
            self.print_calls = 0
            self.contextvars = type(
                "Ctx",
                (),
                {"merge_contextvars": staticmethod(lambda *_args: {})},
            )
            self.processors = type(
                "Processors",
                (),
                {
                    "add_log_level": staticmethod(lambda *_args: {}),
                    "TimeStamper": staticmethod(lambda **_kw: object()),
                    "StackInfoRenderer": staticmethod(lambda: object()),
                    "JSONRenderer": staticmethod(lambda: object()),
                },
            )
            self.dev = type(
                "Dev",
                (),
                {"ConsoleRenderer": staticmethod(lambda **_kw: object())},
            )

        def __getattribute__(self, name: str) -> object:
            if name == "PrintLoggerFactory":
                calls = object.__getattribute__(self, "print_calls") + 1
                object.__setattr__(self, "print_calls", calls)
                if calls == 1:
                    return None
                return lambda **_kwargs: object()
            return object.__getattribute__(self, name)

        def make_filtering_bound_logger(self, level: int) -> type[object]:
            _ = level
            return dict

        def configure(self, **_kwargs: object) -> None:
            return None

    module = FallbackModule()
    monkeypatch.setattr(runtime_module, "structlog", module)
    FlextRuntime._structlog_configured = False
    FlextRuntime.configure_structlog(
        config=cast(
            "t.GeneralValueType",
            cast(
                object,
                type(
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
                )(),
            ),
        ),
    )
    assert module.print_calls >= 2


def test_dependency_integration_and_wiring_paths() -> None:
    bridge, services, resources = (
        FlextRuntime.DependencyIntegration.create_layered_bridge(
            config=t.ConfigMap(root={"db": {"dsn": "sqlite://"}}),
        )
    )
    assert bridge is not None and services is not None and resources is not None

    di = FlextRuntime.DependencyIntegration.create_container(
        config=t.ConfigMap(root={"feature": {"enabled": True}}),
        services={"svc": 1},
        factories={"factory": lambda: 2},
        resources={"resource": lambda: {"ok": True}},
        wire_modules=[],
        wire_packages=["unused.package"],
        wire_classes=[FlextRuntime],
    )
    assert getattr(getattr(di.config, "feature"), "enabled")() is True
    assert di.svc() == 1
    assert di.factory() == 2
    assert di.resource() == {"ok": True}

    provider = runtime_module.providers.Configuration()
    FlextRuntime.DependencyIntegration.bind_configuration_provider(
        provider,
        t.ConfigMap(root={"api": {"url": "x"}}),
    )
    assert provider.api.url() == "x"


def test_runtime_result_remaining_paths() -> None:
    success = FlextRuntime.RuntimeResult.ok(3)
    failure: FlextRuntime.RuntimeResult[int] = FlextRuntime.RuntimeResult.fail(
        "err",
        error_code="E2",
        error_data=m.ConfigMap(root={"k": "v"}),
    )

    assert failure.error_code == "E2"
    assert failure.error_data is not None
    assert success.data == 3
    with pytest.raises(RuntimeError, match="Cannot access value of failed result"):
        _ = failure.value
    assert success.unwrap() == 3

    assert failure.flat_map(lambda x: FlextRuntime.RuntimeResult.ok(x)).is_failure
    assert success.filter(lambda x: x > 0) is success
    assert success.alt(lambda e: e) is success
    assert success.lash(lambda e: FlextRuntime.RuntimeResult.fail(e)) is success
    assert success.recover(lambda _e: 0) is success

    chain_success = success.flow_through(
        lambda x: FlextRuntime.RuntimeResult.ok(x + 1),
        lambda x: FlextRuntime.RuntimeResult.ok(x + 1),
    )
    assert chain_success.is_success and chain_success.value == 5

    chain_failure = success.flow_through(
        lambda _x: FlextRuntime.RuntimeResult.fail("boom"),
        lambda x: FlextRuntime.RuntimeResult.ok(x + 1),
    )
    assert chain_failure.is_failure

    assert (success | 0) == 3
    assert bool(success) is True
    assert repr(success).startswith("r.ok(")
    assert repr(failure).startswith("r.fail(")
    with success as entered:
        assert entered is success


def test_runtime_integration_tracking_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict[str, object]]] = []

    class Logger:
        def info(self, message: str, **kwargs: object) -> None:
            events.append((message, kwargs))

        def error(self, message: str, **kwargs: object) -> None:
            events.append((message, kwargs))

    class CtxVars:
        @staticmethod
        def get_contextvars() -> dict[str, str]:
            return {"correlation_id": "corr-1"}

        @staticmethod
        def bind_contextvars(**_kwargs: object) -> None:
            return None

    fake_structlog = type(
        "FakeStructlog",
        (),
        {
            "contextvars": CtxVars,
            "get_logger": staticmethod(lambda _name=None: Logger()),
        },
    )
    monkeypatch.setattr(runtime_module, "structlog", fake_structlog)

    FlextRuntime.Integration.track_service_resolution("svc", resolved=True)
    FlextRuntime.Integration.track_service_resolution(
        "svc", resolved=False, error_message="x"
    )
    FlextRuntime.Integration.track_domain_event(
        "evt",
        aggregate_id="agg",
        event_data=t.ConfigMap(root={"k": "v"}),
    )
    FlextRuntime.Integration.setup_service_infrastructure(
        service_name="svc",
        service_version="1.0.0",
        enable_context_correlation=True,
    )
    assert len(events) == 4


def test_model_helpers_remaining_paths() -> None:
    class Entity:
        def __init__(self, unique_id: str) -> None:
            self.unique_id = unique_id

    class ValueModel(BaseModel):
        a: int

    left = Entity("u-1")
    right = Entity("u-1")
    assert (
        FlextRuntime.compare_entities_by_id(
            cast("t.GeneralValueType", cast(object, left)),
            cast("t.GeneralValueType", cast(object, right)),
        )
        is True
    )
    assert isinstance(
        FlextRuntime.hash_entity_by_id(cast("t.GeneralValueType", cast(object, left))),
        int,
    )

    vm_a = ValueModel(a=1)
    vm_b = ValueModel(a=1)
    assert FlextRuntime.compare_value_objects_by_value(vm_a, vm_b) is True
    assert isinstance(FlextRuntime.hash_value_object_by_value(vm_a), int)


def test_ensure_trace_context_dict_conversion_paths() -> None:
    payload = {
        "none": None,
        "str": "x",
        "int": 1,
        "float": 1.5,
        "bool": True,
        "dt": datetime.now(UTC),
        "path": Path("."),
        "list": [1, 2],
        "dict": {"a": 1},
        "callable": lambda: 1,
        "other": object(),
    }
    result = FlextRuntime.ensure_trace_context(cast("t.GeneralValueType", payload))
    assert result["str"] == "x"
    assert result["int"] == "1"
    assert "trace_id" in result and "span_id" in result
