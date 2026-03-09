"""Tests for Mixins."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from types import SimpleNamespace
from typing import cast, override

import pytest

from flext_core import FlextLogger, FlextMixins, FlextRuntime, c, m, p, r, t, u, x

from ._models import _SvcModel


def _normalize_to_one(_v: t.ContainerValue) -> int:
    """Staticmethod helper for monkeypatch; type known for pyright."""
    return 1


def _noop(*_a: object, **_k: object) -> None:
    """Typed no-op for protocol stubs."""
    return


def _return_true(*_a: object, **_k: object) -> bool:
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


def _validation_ok_true(v: t.ContainerValue) -> r[bool]:
    """Validator that always returns ok(True)."""
    return r[bool].ok(True)


def _validation_ok_false(v: t.ContainerValue) -> r[bool]:
    """Validator that always returns ok(False)."""
    return r[bool].ok(False)


def _validation_fail_no(v: t.ContainerValue) -> r[bool]:
    """Validator that always returns fail('no')."""
    return r[bool].fail("no")


class _RuntimeContainer:
    def __init__(self) -> None:
        super().__init__()
        self.configured: dict[str, t.ContainerValue] | None = None
        self.wired: dict[str, object] | None = None

    def scoped(self, **_kwargs: object) -> _RuntimeContainer:
        return self

    def configure(self, overrides: dict[str, t.ContainerValue]) -> None:
        self.configured = overrides

    def wire_modules(self, **kwargs: object) -> None:
        self.wired = kwargs


class _ContainerForLogger:
    def __init__(self, success: bool, logger: object | None = None) -> None:
        super().__init__()
        self.success: bool = success
        self.logger: object | None = logger
        self.factories: dict[str, object] = {}
        self.register_calls: list[tuple[str, str]] = []

    def get_typed(self, _key: str, _tp: object) -> r[object]:
        if self.success:
            return r[object].ok(self.logger or object())
        return r[object].fail("missing")

    def get(self, _key: str, *, type_cls: object | None = None) -> r[object]:
        _ = type_cls
        if self.success:
            return r[object].ok(self.logger or object())
        return r[object].fail("missing")

    def register_factory(self, key: str, factory: object) -> r[bool]:
        _ = key
        _ = factory
        msg = "register_factory path should not be used"
        raise AssertionError(msg)

    def register(
        self,
        name: str,
        value: object,
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
    assert isinstance(c.Context.SCOPE_OPERATION, str)
    assert hasattr(u, "merge")
    assert x.fail("error").is_failure
    conf = m.ConfigMap(root={"a": "b"})
    assert x.to_dict(conf) is conf
    monkeypatch.setattr(
        FlextRuntime,
        "normalize_to_general_value",
        staticmethod(_normalize_to_one),
    )
    scalar_wrapped = x.to_dict(_SvcModel(value="ok"))
    assert scalar_wrapped.root == {"value": 1}

    class _BadMap(t.ConfigurationMapping):
        @override
        def __iter__(self) -> Iterator[str]:
            return iter(["k"])

        @override
        def __len__(self) -> int:
            return 1

        @override
        def __getitem__(self, _key: str) -> t.ContainerValue:
            msg = "boom"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="boom") as exc_info:
        _ = x.to_dict(cast("Mapping[str, t.ContainerValue]", _BadMap()))
    assert exc_info.value is not None


def test_mixins_runtime_bootstrap_and_track_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_container = _RuntimeContainer()
    monkeypatch.setattr(
        "flext_core.mixins.FlextContainer.create",
        staticmethod(lambda: runtime_container),
    )

    class _Service(x):
        @override
        @classmethod
        def _runtime_bootstrap_options(cls) -> p.RuntimeBootstrapOptions:
            return cast(
                "p.RuntimeBootstrapOptions",
                cast(
                    "object",
                    {
                        "container_overrides": {"debug": True},
                        "wire_packages": ["pkg"],
                        "resources": {"res": lambda: "x"},
                        "factories": {"fac": lambda: "y"},
                    },
                ),
            )

    service = _Service()
    runtime = service._get_runtime()
    assert runtime is not None
    assert runtime_container.configured == {"debug": True}
    assert runtime_container.wired is not None
    with service.track("op") as metrics:
        cast("dict[str, t.ContainerValue]", metrics)["duration_ms"] = 2.0
    assert hasattr(service, "_stats_op")
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

    service = _Service()
    ok_register = service._register_in_container("svc")
    assert ok_register.is_success

    class _AlreadyContainer:
        def has_service(self, _name: str) -> bool:
            return True

        def register(self, _name: str, _value: object) -> _AlreadyContainer:
            return self

    monkeypatch.setattr(
        _Service,
        "container",
        property(lambda _self: _AlreadyContainer()),
    )
    assert service._register_in_container("svc").is_success

    class _FailContainer:
        def has_service(self, _name: str) -> bool:
            return False

        def register(self, _name: str, _value: object) -> _FailContainer:
            return self

    monkeypatch.setattr(_Service, "container", property(lambda _self: _FailContainer()))
    service._init_service("svc")
    FlextMixins._logger_cache.clear()
    monkeypatch.setattr(
        "flext_core.mixins.FlextContainer.create",
        staticmethod(
            lambda: _ContainerForLogger(True, logger=SimpleNamespace(name="l")),
        ),
    )
    logger_from_di = _Service._get_or_create_logger()
    assert logger_from_di is not None
    monkeypatch.setattr(
        "flext_core.mixins.FlextContainer.create",
        staticmethod(lambda: _ContainerForLogger(False)),
    )
    logger_created = _Service._get_or_create_logger()
    assert logger_created is not None
    monkeypatch.setattr(
        "flext_core.mixins.FlextContainer.create",
        staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("no container"))),
    )
    logger_fallback = _Service._get_or_create_logger()
    assert logger_fallback is not None


def test_mixins_context_logging_and_cqrs_paths(monkeypatch: pytest.MonkeyPatch) -> None:

    class _LocalLogger:
        def info(self, *_args: object, **_kwargs: object) -> None:
            return None

        def warning(self, *_args: object, **_kwargs: object) -> None:
            return None

    class _Service(x):
        @override
        @classmethod
        def _get_or_create_logger(cls) -> FlextLogger:
            return cast("FlextLogger", cast("object", _LocalLogger()))

    service = _Service()
    config = m.ConfigMap(root={"k": "v"})
    service._log_config_once(config, message="cfg")
    service._with_operation_context(
        "run",
        params={"k": "v"},
        stack_trace="s",
        normal="n",
    )
    service._clear_operation_context()
    monkeypatch.delattr(x.CQRS.MetricsTracker, "_metrics", raising=False)
    mt = x.CQRS.MetricsTracker()
    result_record = mt.record_metric("k", 1)
    assert result_record.is_success
    result_metrics = mt.get_metrics()
    assert result_metrics.is_success
    monkeypatch.delattr(x.CQRS.ContextStack, "_stack", raising=False)
    cs = x.CQRS.ContextStack()
    result_push1 = cs.push_context({"handler_name": "h", "handler_mode": "query"})
    assert result_push1.is_success
    result_push2 = cs.push_context({"x": "y"})
    assert result_push2.is_success
    result_pop = cs.pop_context()
    assert result_pop.is_success
    result_pop2 = cs.pop_context()
    assert result_pop2.is_success
    assert cs.current_context() is None
    result_push3 = cs.push_context({"handler_name": "h2", "handler_mode": "command"})
    assert result_push3.is_success
    assert cs.current_context() is not None


def test_mixins_validation_and_protocol_paths() -> None:
    validators: list[Callable[[t.ContainerValue], r[bool]]] = [
        _validation_ok_false,
    ]
    bad_true = x.Validation.validate_with_result("v", validators)
    assert bad_true.is_failure
    fail_validators: list[Callable[[t.ContainerValue], r[bool]]] = [
        _validation_fail_no,
    ]
    fail_result = x.Validation.validate_with_result("v", fail_validators)
    assert fail_result.is_failure
    assert (
        x.ProtocolValidation.is_handler(
            cast(
                "t.ContainerValue",
                cast("object", SimpleNamespace(handle=_noop)),
            ),
        )
        is False
    )
    assert (
        x.ProtocolValidation.is_service(
            cast("p.Service[t.ContainerValue]", cast("object", SimpleNamespace())),
        )
        is False
    )
    cmd_bus = SimpleNamespace(
        dispatch=_noop,
        publish=_noop,
        register_handler=_noop,
    )
    assert x.ProtocolValidation.is_command_bus(cmd_bus) is True
    unknown = x.ProtocolValidation.validate_protocol_compliance(
        cast(
            "p.Handler[t.ContainerValue, t.ContainerValue]",
            cast("object", SimpleNamespace()),
        ),
        "Nope",
    )
    service_like = SimpleNamespace(
        execute=_noop,
        get_service_info=_noop,
        is_valid=_return_true,
    )
    known = x.ProtocolValidation.validate_protocol_compliance(
        cast(
            "p.Handler[t.ContainerValue, t.ContainerValue]",
            cast("object", service_like),
        ),
        "Service",
    )
    assert unknown.is_failure
    assert known.is_success

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
    assert missing.is_failure
    assert bad_callable.is_failure
    assert good.is_success


def test_mixins_remaining_branch_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_container = _RuntimeContainer()

    def _create_runtime_container() -> _RuntimeContainer:
        return runtime_container

    monkeypatch.setattr(
        "flext_core.mixins.FlextContainer.create",
        staticmethod(_create_runtime_container),
    )

    class _WireService(x):
        @override
        @classmethod
        def _runtime_bootstrap_options(cls) -> p.RuntimeBootstrapOptions:
            return cast(
                "p.RuntimeBootstrapOptions",
                cast("object", {"wire_packages": ["pkg", 1]}),
            )

    _ = _WireService()._get_runtime()
    assert runtime_container.wired is None

    class _ModelMarker:
        pass

    monkeypatch.setattr("flext_core.mixins.BaseModel", _ModelMarker)

    class _ModelService(_ModelMarker, x):
        pass

    model_service = _ModelService()
    captured: dict[str, object] = {}

    class _RegContainer:
        def __init__(self) -> None:
            super().__init__()
            self._services: dict[str, object] = {}

        def has_service(self, name: str) -> bool:
            return name in self._services

        def register(self, name: str, value: object) -> _RegContainer:
            self._services[name] = value
            captured["name"] = name
            captured["value"] = value
            return self

    def _container_getter(_self: object) -> _RegContainer:
        return _RegContainer()

    monkeypatch.setattr(
        _ModelService,
        "container",
        property(_container_getter),
    )
    assert model_service._register_in_container("svc_model").is_success
    assert isinstance(captured["value"], _ModelMarker)

    class _WarnLogger:
        def warning(self, *_args: object, **_kwargs: object) -> None:
            return None

    class _WarnService(x):
        @override
        @classmethod
        def _get_or_create_logger(cls) -> FlextLogger:
            return cast("FlextLogger", cast("object", _WarnLogger()))

    class _LoggerService(x):
        pass

    warn_service = _WarnService()
    monkeypatch.setattr(
        warn_service,
        "_register_in_container",
        _mock_register_fail,
    )
    warn_service._init_service("svc_warn")
    monkeypatch.delattr(x.CQRS.MetricsTracker, "_metrics", raising=False)
    tracker = x.CQRS.MetricsTracker()
    delattr(tracker, "_metrics")
    assert tracker.record_metric("a", 1).is_success
    delattr(tracker, "_metrics")
    assert tracker.get_metrics().is_success
    monkeypatch.delattr(x.CQRS.ContextStack, "_stack", raising=False)
    stack = x.CQRS.ContextStack()
    delattr(stack, "_stack")
    assert stack.push_context({"handler_name": "h", "handler_mode": "event"}).is_success
    object.__setattr__(stack, "_stack", [{"k": "v"}])
    popped_dict = stack.pop_context()
    assert popped_dict.is_success
    assert popped_dict.value == {"k": "v"}
    delattr(stack, "_stack")
    assert stack.current_context() is None
    object.__setattr__(stack, "_stack", [{"k": "v"}])
    assert stack.current_context() is None
    valid = x.Validation.validate_with_result(
        "v",
        [_validation_ok_true],
    )
    assert valid.is_success
    FlextMixins._logger_cache.clear()

    class _FactoryContainer(_ContainerForLogger):
        def __init__(self) -> None:
            super().__init__(False)

    factory_container = _FactoryContainer()

    def _create_factory_container() -> _ContainerForLogger:
        return factory_container

    monkeypatch.setattr(
        "flext_core.mixins.FlextContainer.create",
        staticmethod(_create_factory_container),
    )
    _ = _LoggerService._get_or_create_logger()
    assert any((kind == "factory" for _name, kind in factory_container.register_calls))
    factory = next(iter(factory_container.factories.values()))
    assert callable(factory)
    factory_value = factory()
    assert isinstance(factory_value, str)
    assert "_LoggerService" in factory_value

    class _BrokenContainer:
        def get_typed(self, _key: str, _tp: object) -> r[object]:
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
    assert fallback_logger is not None


def test_mixins_context_stack_pop_initializes_missing_stack_attr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(x.CQRS.ContextStack, "_stack", raising=False)
    stack = x.CQRS.ContextStack()
    delattr(stack, "_stack")
    popped = stack.pop_context()
    assert popped.is_success
    assert popped.value == {}
