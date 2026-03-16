"""Tests for Mixins."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from types import SimpleNamespace
from typing import cast, override

import pytest
from flext_tests import t, tm
from pydantic import BaseModel

from flext_core import FlextLogger, FlextMixins, r, x
from tests import c, p, u

from ._models import _SvcModel


def _noop(*_a: t.Scalar, **_k: t.Scalar) -> None:
    """Typed no-op for protocol stubs."""
    return


def _return_true(*_a: t.Scalar, **_k: t.Scalar) -> bool:
    """Typed return-True for protocol stubs."""
    return True


def _return_true_no_args() -> bool:
    """Typed callable for process/validate stubs."""
    return True


def _mock_register_fail(_name: str) -> r[bool]:
    """Mock _register_in_container that returns failure."""
    return cast(
        "r[bool]",
        cast("object", SimpleNamespace(is_failure=True, error=None)),
    )


def _validation_ok_true(v: t.NormalizedValue) -> r[bool]:
    """Validator that always returns ok(True)."""
    return r[bool].ok(True)


def _validation_ok_false(v: t.NormalizedValue) -> r[bool]:
    """Validator that always returns ok(False)."""
    return r[bool].ok(False)


def _validation_fail_no(v: t.NormalizedValue) -> r[bool]:
    """Validator that always returns fail('no')."""
    return r[bool].fail("no")


class _RuntimeContainer:
    def __init__(self) -> None:
        super().__init__()
        self.configured: dict[str, t.Tests.object] | None = None
        self.wired: dict[str, t.Tests.object] | None = None

    def scoped(self, **_kwargs: t.Scalar) -> _RuntimeContainer:
        return self

    def configure(self, overrides: dict[str, t.Tests.object]) -> None:
        self.configured = overrides

    def wire_modules(self, **kwargs: t.Scalar) -> None:
        self.wired = dict(kwargs)

    def list_services(self) -> Sequence[str]:
        return []

    def has_service(self, _name: str) -> bool:
        return False

    def register(
        self, _name: str, _value: t.Tests.object, *, kind: str = "service"
    ) -> _RuntimeContainer:
        return self

    def clear_all(self) -> None:
        pass

    def get_config(self) -> t.ConfigMap:
        return t.ConfigMap(root={})

    def get(self, _key: str, **_kwargs: t.Scalar) -> r[t.Tests.object]:
        return r[t.Tests.object].fail("not implemented")

    @property
    def context(self) -> None:
        return None

    @property
    def config(self) -> None:
        return None


class _ContainerForLogger:
    def __init__(
        self,
        success: bool,
        logger: t.Container | BaseModel | None = None,
    ) -> None:
        super().__init__()
        self.success: bool = success
        self.logger: t.Container | BaseModel | None = logger
        self.factories: dict[str, t.Tests.object] = {}
        self.register_calls: list[tuple[str, str]] = []

    def get_typed(
        self, _key: str, _tp: type[t.Container | BaseModel]
    ) -> r[t.Container | BaseModel]:
        if self.success:
            return r[t.Container | BaseModel].ok(self.logger or "logger")
        return r[t.Container | BaseModel].fail("missing")

    def get(
        self,
        _key: str,
        *,
        type_cls: type[t.Container | BaseModel] | None = None,
    ) -> r[t.Container | BaseModel]:
        _ = type_cls
        if self.success:
            return r[t.Container | BaseModel].ok(self.logger or "logger")
        return r[t.Container | BaseModel].fail("missing")

    def register_factory(self, key: str, factory: t.FactoryCallable) -> r[bool]:
        _ = key
        _ = factory
        msg = "register_factory path should not be used"
        raise AssertionError(msg)

    def register(
        self,
        name: str,
        value: t.Tests.object,
        *,
        kind: str = "service",
    ) -> _ContainerForLogger:
        self.register_calls.append((name, kind))
        if kind == "factory":
            self.factories[name] = value
        return self


def test_mixins_result_and_model_conversion_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tm.that(isinstance(c.Context.SCOPE_OPERATION, str), eq=True)
    tm.that(hasattr(u, "merge"), eq=True)
    tm.fail(x.fail("error"))
    conf = t.ConfigMap(root={"a": "b"})
    tm.that(x.normalize_to_container(conf) is conf, eq=True)
    model = _SvcModel(value="ok")
    tm.that(x.normalize_to_container(model) is model, eq=True)

    class _BadMap(Mapping[str, t.NormalizedValue]):
        @override
        def __iter__(self) -> Iterator[str]:
            return iter(["k"])

        @override
        def __len__(self) -> int:
            return 1

        @override
        def __getitem__(self, _key: str) -> t.NormalizedValue:
            msg = "boom"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="boom"):
        _ = x.normalize_to_container(_BadMap())


def test_mixins_runtime_bootstrap_and_track_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_container = _RuntimeContainer()
    monkeypatch.setattr(
        "flext_core.mixins.FlextContainer.create",
        staticmethod(lambda: runtime_container),
    )

    class _Service(x):
        @classmethod
        def _runtime_bootstrap_options(cls) -> p.RuntimeBootstrapOptions:
            return cast(
                "p.RuntimeBootstrapOptions",
                cast(
                    "t.Tests.object",
                    SimpleNamespace(
                        config_type=None,
                        config_overrides=None,
                        context=None,
                        services=None,
                        wire_modules=["pkg"],
                        resources={"res": lambda: "x"},
                        factories={"fac": lambda: "y"},
                    ),
                ),
            )

    service = _Service(
        config_type=None,
        config_overrides=None,
        initial_context=None,
    )
    runtime = service._get_runtime()
    tm.that(runtime, none=False)
    tm.that(runtime_container.wired, none=False)
    with service.track("op") as metrics:
        cast("dict[str, t.Tests.object]", metrics)["duration_ms"] = 2.0
    tm.that(hasattr(service, "_operation_stats"), eq=True)
    tm.that(service._operation_stats, has="op")
    try:
        with service.track("op_fail"):
            msg = "boom"
            raise RuntimeError(msg)
    except RuntimeError:
        pass


def test_mixins_container_registration_and_logger_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class _Service(x):
        pass

    service = _Service(
        config_type=None,
        config_overrides=None,
        initial_context=None,
    )
    ok_register = service._register_in_container("svc")
    tm.ok(ok_register)

    class _AlreadyContainer:
        def has_service(self, _name: str) -> bool:
            return True

        def register(self, _name: str, _value: t.Tests.object) -> _AlreadyContainer:
            return self

    monkeypatch.setattr(
        _Service,
        "container",
        property(lambda _self: _AlreadyContainer()),
    )
    tm.ok(service._register_in_container("svc"))

    class _FailContainer:
        def has_service(self, _name: str) -> bool:
            return False

        def register(self, _name: str, _value: t.Tests.object) -> _FailContainer:
            return self

    monkeypatch.setattr(_Service, "container", property(lambda _self: _FailContainer()))
    service._init_service("svc")
    FlextMixins._logger_cache.clear()
    monkeypatch.setattr(
        "flext_core.mixins.FlextContainer.create",
        staticmethod(
            lambda: _ContainerForLogger(True, logger="l"),
        ),
    )
    logger_from_di = _Service._get_or_create_logger()
    tm.that(logger_from_di, none=False)
    monkeypatch.setattr(
        "flext_core.mixins.FlextContainer.create",
        staticmethod(lambda: _ContainerForLogger(False)),
    )
    logger_created = _Service._get_or_create_logger()
    tm.that(logger_created, none=False)
    monkeypatch.setattr(
        "flext_core.mixins.FlextContainer.create",
        staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("no container"))),
    )
    logger_fallback = _Service._get_or_create_logger()
    tm.that(logger_fallback, none=False)


def test_mixins_context_logging_and_cqrs_paths(monkeypatch: pytest.MonkeyPatch) -> None:

    class _LocalLogger:
        def info(self, *_args: t.Scalar, **_kwargs: t.Scalar) -> None:
            return None

        def warning(self, *_args: t.Scalar, **_kwargs: t.Scalar) -> None:
            return None

    class _Service(x):
        @override
        @classmethod
        def _get_or_create_logger(cls) -> FlextLogger:
            return cast("FlextLogger", cast("object", _LocalLogger()))

    service = _Service(
        config_type=None,
        config_overrides=None,
        initial_context=None,
    )
    config = t.ConfigMap(root={"k": "v"})
    service._log_config_once(config, message="cfg")
    service._with_operation_context(
        "run",
        params="k=v",
        stack_trace="s",
        normal="n",
    )
    service._clear_operation_context()
    monkeypatch.delattr(x.CQRS.MetricsTracker, "_metrics", raising=False)
    mt = x.CQRS.MetricsTracker()
    result_record = mt.record_metric("k", 1)
    tm.ok(result_record)
    result_metrics = mt.get_metrics()
    tm.ok(result_metrics)
    monkeypatch.delattr(x.CQRS.ContextStack, "_stack", raising=False)
    cs = x.CQRS.ContextStack()
    result_push1 = cs.push_context({"handler_name": "h", "handler_mode": "query"})
    tm.ok(result_push1)
    result_push2 = cs.push_context({"x": "y"})
    tm.ok(result_push2)
    result_pop = cs.pop_context()
    tm.ok(result_pop)
    result_pop2 = cs.pop_context()
    tm.ok(result_pop2)
    tm.that(cs.current_context(), none=True)
    result_push3 = cs.push_context({"handler_name": "h2", "handler_mode": "command"})
    tm.ok(result_push3)
    tm.that(cs.current_context(), none=False)


def test_mixins_validation_and_protocol_paths() -> None:
    validators: list[Callable[..., r[bool]]] = [
        _validation_ok_false,
    ]
    bad_true = x.Validation.validate_with_result("v", validators)
    tm.fail(bad_true)
    fail_validators: list[Callable[..., r[bool]]] = [
        _validation_fail_no,
    ]
    fail_result = x.Validation.validate_with_result("v", fail_validators)
    tm.fail(fail_result)
    tm.that(
        x.ProtocolValidation.is_handler(
            cast("t.Tests.object", SimpleNamespace(handle=_noop)),
        ),
        eq=False,
    )
    tm.that(
        x.ProtocolValidation.is_service(
            cast("p.Service[bool]", cast("t.Tests.object", SimpleNamespace())),
        ),
        eq=False,
    )
    cmd_bus = SimpleNamespace(
        dispatch=_noop,
        publish=_noop,
        register_handler=_noop,
    )
    tm.that(x.ProtocolValidation.is_command_bus(cmd_bus), eq=True)
    unknown = x.ProtocolValidation.validate_protocol_compliance(
        t.ConfigMap(root={}),
        "Nope",
    )
    service_like = SimpleNamespace(
        execute=_noop,
        get_service_info=_noop,
        is_valid=_return_true,
    )
    known = x.ProtocolValidation.validate_protocol_compliance(
        cast("p.Base", service_like),
        "Service",
    )
    tm.fail(unknown)
    tm.ok(known)

    class _ModelDumpOnly:
        pass

    missing = x.ProtocolValidation.validate_processor_protocol(
        cast("p.HasModelDump", cast("object", _ModelDumpOnly())),
    )
    bad_callable = x.ProtocolValidation.validate_processor_protocol(
        cast(
            "p.HasModelDump",
            cast(
                "object",
                SimpleNamespace(model_dump=dict, process=1, validate=lambda: True),
            ),
        ),
    )
    good = x.ProtocolValidation.validate_processor_protocol(
        cast(
            "p.HasModelDump",
            cast(
                "object",
                SimpleNamespace(
                    model_dump=dict,
                    process=_return_true_no_args,
                    validate=_return_true_no_args,
                ),
            ),
        ),
    )
    tm.fail(missing)
    tm.fail(bad_callable)
    tm.ok(good)


def test_mixins_remaining_branch_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_container = _RuntimeContainer()

    def _create_runtime_container() -> _RuntimeContainer:
        return runtime_container

    monkeypatch.setattr(
        "flext_core.mixins.FlextContainer.create",
        staticmethod(_create_runtime_container),
    )

    class _WireService(x):
        @classmethod
        def _runtime_bootstrap_options(cls) -> p.RuntimeBootstrapOptions:
            return cast(
                "p.RuntimeBootstrapOptions",
                cast("t.Tests.object", {"wire_packages": ["pkg", 1]}),
            )

    _ = _WireService(
        config_type=None,
        config_overrides=None,
        initial_context=None,
    )._get_runtime()
    tm.that(runtime_container.wired, none=True)

    class _ModelMarker:
        pass

    monkeypatch.setattr("flext_core.mixins.BaseModel", _ModelMarker)

    class _ModelService(_ModelMarker, x):
        pass

    model_service = _ModelService(
        config_type=None,
        config_overrides=None,
        initial_context=None,
    )
    captured: dict[str, t.Tests.object] = {}

    class _RegContainer:
        def __init__(self) -> None:
            super().__init__()
            self._services: dict[str, t.Tests.object] = {}

        def has_service(self, name: str) -> bool:
            return name in self._services

        def register(self, name: str, value: t.Tests.object) -> _RegContainer:
            self._services[name] = value
            captured["name"] = name
            captured["value"] = value
            return self

    def _container_getter(_self: _ModelService) -> _RegContainer:
        return _RegContainer()

    monkeypatch.setattr(
        _ModelService,
        "container",
        property(_container_getter),
    )
    tm.ok(model_service._register_in_container("svc_model"))
    tm.that(isinstance(captured["value"], _ModelMarker), eq=True)

    class _WarnLogger:
        def warning(self, *_args: t.Scalar, **_kwargs: t.Scalar) -> None:
            return None

    class _WarnService(x):
        @override
        @classmethod
        def _get_or_create_logger(cls) -> FlextLogger:
            return cast("FlextLogger", cast("object", _WarnLogger()))

    class _LoggerService(x):
        pass

    warn_service = _WarnService(
        config_type=None,
        config_overrides=None,
        initial_context=None,
    )
    monkeypatch.setattr(
        warn_service,
        "_register_in_container",
        _mock_register_fail,
    )
    warn_service._init_service("svc_warn")
    monkeypatch.delattr(x.CQRS.MetricsTracker, "_metrics", raising=False)
    tracker = x.CQRS.MetricsTracker()
    del tracker._metrics
    tm.ok(tracker.record_metric("a", 1))
    del tracker._metrics
    tm.ok(tracker.get_metrics())
    monkeypatch.delattr(x.CQRS.ContextStack, "_stack", raising=False)
    stack = x.CQRS.ContextStack()
    del stack._stack
    tm.ok(stack.push_context({"handler_name": "h", "handler_mode": "event"}))
    object.__setattr__(stack, "_stack", [{"k": "v"}])
    popped_dict = stack.pop_context()
    tm.ok(popped_dict)
    tm.that(popped_dict.value, eq={"k": "v"})
    del stack._stack
    tm.that(stack.current_context(), none=True)
    object.__setattr__(stack, "_stack", [{"k": "v"}])
    tm.that(stack.current_context(), none=True)
    valid = x.Validation.validate_with_result(
        "v",
        [_validation_ok_true],
    )
    tm.ok(valid)
    FlextMixins._logger_cache.clear()
    logger_obj = _LoggerService._get_or_create_logger()
    tm.that(isinstance(logger_obj, FlextLogger), eq=True)

    class _BrokenContainer:
        def get_typed(
            self, _key: str, _tp: type[t.Container | BaseModel]
        ) -> r[t.Container | BaseModel]:
            msg = "boom"
            raise RuntimeError(msg)

    def _create_broken_container() -> _BrokenContainer:
        return _BrokenContainer()

    FlextMixins._logger_cache.clear()
    monkeypatch.setattr(
        "flext_core.mixins.FlextContainer.create",
        staticmethod(_create_broken_container),
    )
    fallback_logger = _LoggerService._get_or_create_logger()
    tm.that(fallback_logger, none=False)


def test_mixins_context_stack_pop_initializes_missing_stack_attr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(x.CQRS.ContextStack, "_stack", raising=False)
    stack = x.CQRS.ContextStack()
    del stack._stack
    popped = stack.pop_context()
    tm.ok(popped)
    tm.that(popped.value, eq={})
