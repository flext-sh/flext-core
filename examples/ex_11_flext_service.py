"""Golden-file example for FlextService (s) public APIs."""

from __future__ import annotations

import sys
from collections import UserDict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import ClassVar, override

from pydantic import BaseModel, PrivateAttr

from flext_core import (
    FlextContext,
    FlextExceptions,
    FlextService,
    FlextSettings,
    m,
    p,
    r,
    s,
    t,
    u,
)

_RESULTS: list[str] = []


def _check(label: str, value: object) -> None:
    _RESULTS.append(f"{label}: {_ser(value)}")


def _section(name: str) -> None:
    if _RESULTS:
        _RESULTS.append("")
    _RESULTS.append(f"[{name}]")


def _ser(v: object) -> str:
    if v is None:
        return "None"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, type):
        return v.__name__

    if isinstance(v, Mapping):
        map_value = m.ConfigMap(root={})
        if u.is_dict_like(map_value):
            return "mapping"

    if isinstance(v, Sequence) and not isinstance(v, str):
        seq_value: list[str] = []
        if u.is_list(seq_value):
            return "sequence"

    if isinstance(v, datetime | Path | BaseModel):
        return repr(v)
    return "object"


def _verify() -> None:
    actual = "\n".join(_RESULTS).strip() + "\n"
    me = Path(__file__)
    expected_path = me.with_suffix(".expected")
    n = sum(1 for line in _RESULTS if ": " in line and not line.startswith("["))
    if expected_path.exists():
        expected = expected_path.read_text(encoding="utf-8")
        if actual == expected:
            sys.stdout.write(f"PASS: {me.stem} ({n} checks)\n")
        else:
            actual_path = me.with_suffix(".actual")
            actual_path.write_text(actual, encoding="utf-8")
            sys.stdout.write(
                f"FAIL: {me.stem} — diff {expected_path.name} {actual_path.name}\n"
            )
            sys.exit(1)
    else:
        expected_path.write_text(actual, encoding="utf-8")
        sys.stdout.write(f"GENERATED: {expected_path.name} ({n} checks)\n")


class _Payload(BaseModel):
    text: str
    count: int


class _EchoService(s[str]):
    """Simple typed service implementation for execute()."""

    @override
    def execute(self) -> r[str]:
        return r[str].ok("echo:ok")


class _FailingService(s[str]):
    """Service returning failed result to exercise result property failure path."""

    @override
    def execute(self) -> r[str]:
        return r[str].fail("boom-service")


class _RuleService(s[str]):
    """Service overriding business-rule validation."""

    @override
    def execute(self) -> r[str]:
        return r[str].ok("rules")

    @override
    def validate_business_rules(self) -> r[bool]:
        return r[bool].fail("invalid-rule", error_code="E_RULE")


class _ValidationCrashService(s[str]):
    """Service raising from validate_business_rules() to test is_valid() guard."""

    @override
    def execute(self) -> r[str]:
        return r[str].ok("no-op")

    @override
    def validate_business_rules(self) -> r[bool]:
        msg = "rule-crash"
        raise RuntimeError(msg)


class _DeclarativeService(s[str]):
    """Declarative auto_execute service pattern."""

    auto_execute: ClassVar[bool] = True
    _execute_count: int = PrivateAttr(default=0)

    def __init__(self) -> None:
        super().__init__()
        if self.auto_execute:
            self._execution_result = self.execute()

    @override
    def execute(self) -> r[str]:
        self._execute_count += 1
        return r[str].ok(f"auto:{self._execute_count}")

    @property
    def execution_count(self) -> int:
        return self._execute_count


class _RuntimeFactoryService(s[str]):
    @override
    def execute(self) -> r[str]:
        return r[str].ok("factory")

    @classmethod
    def create_runtime_default(cls) -> m.ServiceRuntime:
        return cls._create_runtime()

    @classmethod
    def create_runtime_full(cls) -> m.ServiceRuntime:
        return cls._create_runtime(
            config_type=FlextSettings,
            config_overrides={},
            context=FlextContext.create(),
            subproject="examples",
            services={"svc_name": "service-value"},
            factories={"factory_name": lambda: "factory-value"},
            resources={"resource_name": lambda: "resource-value"},
            container_overrides={"feature_flag": True},
            wire_modules=[sys.modules[__name__]],
            wire_packages=["flext_core"],
            wire_classes=[_EchoService],
        )


class _HandlerLike(UserDict[str, str]):
    """Minimal handler-like object for protocol checks."""

    def handle(self) -> str:
        return "ok"

    def validate(self) -> bool:
        return True


class _TinyType:
    """Small type for Bootstrap.create_instance testing."""

    def __init__(self) -> None:
        self.initialized = True


class _EntityStub(BaseModel):
    unique_id: str


class _ServiceLike:
    def __init__(self) -> None:
        self._inner = _EchoService()

    @property
    def context(self) -> p.Context:
        return self._inner.context

    @property
    def config(self) -> p.Config:
        return self._inner.config

    @property
    def container(self) -> p.DI:
        return self._inner.container

    def execute(self) -> s.RuntimeResult[t.ContainerValue]:
        return s.RuntimeResult[t.ContainerValue].ok("service-like")

    def validate_business_rules(self) -> s.RuntimeResult[bool]:
        return s.RuntimeResult[bool].ok(True)

    def _protocol_name(self) -> str:
        return "ServiceLike"


class _ProcessorProtocolGood:
    def process(self) -> str:
        return "ok"

    def validate(self) -> bool:
        return True

    def model_dump(self) -> dict[str, str]:
        return {"status": "ok"}

    def _protocol_name(self) -> str:
        return "ProcessorProtocolGood"


class _ProcessorProtocolBad:
    def validate(self) -> bool:
        return False

    def model_dump(self) -> dict[str, str]:
        return {"status": "bad"}

    def _protocol_name(self) -> str:
        return "ProcessorProtocolBad"


def demo_service_core_api() -> None:
    """Exercise constructor, execute, properties, validation, and metadata APIs."""
    _section("service_core_api")

    _check("alias.FlextService_is_s", FlextService is s)

    service = _EchoService()
    _check("execute.unwrap", service.execute().unwrap_or("none"))
    _check("result.property", service.result)
    runtime_view = service.runtime
    _check("runtime.type", type(runtime_view).__name__)
    _check("runtime.has_config_attr", hasattr(runtime_view, "config"))
    _check("runtime.has_context_attr", hasattr(runtime_view, "context"))
    _check("runtime.has_container_attr", hasattr(runtime_view, "container"))
    _check("context.type", type(service.context).__name__)
    _check("config.type", type(service.config).__name__)
    _check("container.type", type(service.container).__name__)

    _check(
        "validate_business_rules.default", service.validate_business_rules().is_success
    )
    _check("is_valid.default", service.is_valid())
    _check("service_info.type", service.get_service_info().get("service_type"))

    rule_service = _RuleService()
    _check(
        "validate_business_rules.override.success",
        rule_service.validate_business_rules().is_success,
    )
    _check(
        "validate_business_rules.override.error",
        rule_service.validate_business_rules().error,
    )
    _check("is_valid.override", rule_service.is_valid())

    crashing = _ValidationCrashService()
    _check("is_valid.exception_guard", crashing.is_valid())

    failing = _FailingService()
    try:
        _ = failing.result
        _check("result.failure.raises", False)
    except FlextExceptions.BaseError as exc:
        _check("result.failure.raises", True)
        _check("result.failure.type", type(exc).__name__)

    declarative = _DeclarativeService()
    _check("auto_execute.declared", declarative.auto_execute)
    _check("auto_execute.execute_count_after_init", declarative.execution_count)
    _check("auto_execute.result", declarative.result)
    _check("auto_execute.execute_count_after_result", declarative.execution_count)


def demo_runtime_creation_and_serialization() -> None:
    """Exercise runtime factory params and serialization helpers."""
    _section("runtime_creation_and_serialization")

    base = _EchoService()

    runtime_default = _RuntimeFactoryService.create_runtime_default()
    _check("create_runtime.default.context", type(runtime_default.context).__name__)

    runtime_full = _RuntimeFactoryService.create_runtime_full()
    _check("create_runtime.full.container", type(runtime_full.container).__name__)
    _check("create_runtime.full.config", type(runtime_full.config).__name__)

    payload = _Payload(text="hello", count=2)
    _check("to_dict.none", s.to_dict(None))
    _check("to_dict.mapping", s.to_dict({"a": 1, "b": "x"}))
    _check("to_dict.model", s.to_dict(payload))

    model_dump_value = base.model_dump(exclude={"runtime"})
    _check("model_dump.type", type(model_dump_value).__name__)
    _check("model_dump.has_result", "result" in model_dump_value)
    model_dump_json_value = base.model_dump_json(exclude={"runtime"})
    _check("model_dump_json.type", type(model_dump_json_value).__name__)


def demo_mixins_and_runtime_methods() -> None:
    """Exercise inherited mixin/runtime APIs on FlextService."""
    _section("mixins_and_runtime_methods")

    service = _EchoService()

    _check("ok.unwrap", service.ok(123).unwrap_or(-1))
    _check(
        "fail.error",
        service.fail(
            "broken",
            error_code="E1",
            error_data=m.ConfigMap(root={"k": "v"}),
        ).error,
    )

    _check("ensure_result.raw", service.ensure_result("wrapped").unwrap_or("none"))
    _check(
        "ensure_result.result",
        service.ensure_result(r[str].ok("already")).unwrap_or("none"),
    )

    generated_id = service.generate_id()
    prefixed = service.generate_prefixed_id("svc", length=8)
    _check("generate_id.length", len(generated_id))
    _check("generate_prefixed_id.prefix", prefixed.startswith("svc_"))
    _check("generate_prefixed_id.length", len(prefixed))
    _check(
        "generate_datetime_utc.tz", service.generate_datetime_utc().tzinfo is not None
    )

    with service.track("sample_op") as metrics:
        _check("track.context.type", type(metrics).__name__)
    _check("track.has_operation_count", "operation_count" in metrics)
    _check("track.has_success_rate", "success_rate" in metrics)

    def _to_even(num: int) -> r[int]:
        if num % 2 == 0:
            return r[int].ok(num)
        return r[int].fail(f"odd:{num}")

    _check(
        "traverse.success",
        service.traverse([2, 4], _to_even, fail_fast=True).unwrap_or([]),
    )
    _check(
        "traverse.fail_fast", service.traverse([2, 3], _to_even, fail_fast=True).error
    )
    _check(
        "accumulate_errors.ok",
        service.accumulate_errors(r[int].ok(1), r[int].ok(2)).is_success,
    )
    fail_one = service.fail("e1")
    fail_two = service.fail("e2")
    _check(
        "accumulate_errors.fail", service.accumulate_errors(fail_one, fail_two).error
    )


def demo_namespace_bootstrap_cqrs_validation() -> None:
    """Exercise Bootstrap/CQRS/ProtocolValidation/Validation namespace APIs."""
    _section("namespace_bootstrap_cqrs_validation")

    tiny = s.Bootstrap.create_instance(_TinyType)
    _check("Bootstrap.create_instance.type", type(tiny).__name__)
    _check("Bootstrap.create_instance.init_called", hasattr(tiny, "initialized"))

    metrics = s.CQRS.MetricsTracker()
    _check("CQRS.MetricsTracker.record", metrics.record_metric("hits", 5).is_success)
    _check(
        "CQRS.MetricsTracker.get",
        metrics.get_metrics().unwrap_or(m.ConfigMap(root={})).get("hits"),
    )

    stack = s.CQRS.ContextStack()
    _check(
        "CQRS.ContextStack.push.dict",
        stack.push_context({
            "handler_name": "demo",
            "handler_mode": "query",
        }).is_success,
    )
    current = stack.current_context()
    _check(
        "CQRS.ContextStack.current.type",
        type(current).__name__ if current is not None else "None",
    )
    popped = stack.pop_context().unwrap_or({})
    _check("CQRS.ContextStack.pop.handler_name", popped.get("handler_name"))
    _check("CQRS.ContextStack.pop.empty", stack.pop_context().unwrap_or({}))

    proto_handler = s.ProtocolValidation.is_handler(_HandlerLike())
    service_like = _ServiceLike()
    is_service_fn = getattr(s.ProtocolValidation, "is_service")
    proto_service = bool(is_service_fn(service_like))
    proto_bus = s.ProtocolValidation.is_command_bus()
    validate_protocol_fn = getattr(s.ProtocolValidation, "validate_protocol_compliance")
    proto_ok = validate_protocol_fn(service_like, "Service")
    proto_unknown = validate_protocol_fn(service_like, "UnknownProto")
    processor_ok = s.ProtocolValidation.validate_processor_protocol(
        _ProcessorProtocolGood(),
    )
    processor_bad = s.ProtocolValidation.validate_processor_protocol(
        _ProcessorProtocolBad(),
    )

    _check("ProtocolValidation.is_handler", proto_handler)
    _check("ProtocolValidation.is_service", proto_service)
    _check("ProtocolValidation.is_command_bus", proto_bus)
    _check("ProtocolValidation.protocol_ok", proto_ok.is_success)
    _check("ProtocolValidation.protocol_unknown", proto_unknown.error)
    _check("ProtocolValidation.processor_ok", processor_ok.is_success)
    _check("ProtocolValidation.processor_bad", processor_bad.error)

    def _validator_len(data: object) -> r[bool]:
        if isinstance(data, str) and len(data) >= 3:
            return r[bool].ok(True)
        return r[bool].fail("too-short")

    def _validator_upper(data: object) -> r[bool]:
        if isinstance(data, str) and data.isupper():
            return r[bool].ok(True)
        return r[bool].fail("not-upper")

    valid = s.Validation.validate_with_result("ABC", [_validator_len, _validator_upper])
    invalid = s.Validation.validate_with_result(
        "ab", [_validator_len, _validator_upper]
    )
    _check("Validation.validate_with_result.ok", valid.unwrap_or(""))
    _check("Validation.validate_with_result.fail", invalid.error)


def demo_namespace_runtime_and_integration() -> None:
    """Exercise RuntimeResult, Integration, DependencyIntegration, and Metadata."""
    _section("namespace_runtime_and_integration")

    # Metadata namespace class (lazy-loaded from runtime)
    metadata = s.Metadata(attributes={"service": "ex11"})
    _check("Metadata.version", metadata.version)
    _check("Metadata.attributes", metadata.attributes)

    # Integration namespace methods (return None)
    s.Integration.setup_service_infrastructure(
        service_name="ex11", service_version="1.0.0"
    )
    s.Integration.track_service_resolution("demo_service", resolved=True)
    s.Integration.track_service_resolution(
        "demo_service_fail",
        resolved=False,
        error_message="not-found",
    )
    s.Integration.track_domain_event(
        event_name="demo_event",
        aggregate_id="agg-1",
        event_data=m.ConfigMap(root={"k": "v"}),
    )
    _check("Integration.calls", "ok")

    # DependencyIntegration namespace
    di = s.DependencyIntegration.create_container(
        config=m.ConfigMap(root={"env": "test"}),
        services={"object_item": "obj"},
        factories={"factory_item": lambda: "factory"},
        resources={"resource_item": lambda: m.ConfigMap(root={"value": 9})},
        wire_modules=[sys.modules[__name__]],
        wire_packages=["flext_core"],
        wire_classes=[_EchoService],
        factory_cache=False,
    )
    _check("DependencyIntegration.service", di.object_item())
    _check("DependencyIntegration.factory", di.factory_item())
    _check("DependencyIntegration.resource.type", type(di.resource_item()).__name__)

    bridge, service_mod, resource_mod = s.DependencyIntegration.create_layered_bridge(
        config=m.ConfigMap(root={"region": "us"}),
    )
    _check("DependencyIntegration.bridge.type", type(bridge).__name__)
    _check("DependencyIntegration.service_module.type", type(service_mod).__name__)
    _check("DependencyIntegration.resource_module.type", type(resource_mod).__name__)

    # RuntimeResult namespace methods
    rr_ok = s.RuntimeResult[int].ok(5)
    rr_fail = s.RuntimeResult[int].fail("bad", error_code="E_BAD")

    _check("RuntimeResult.is_success", rr_ok.is_success)
    _check("RuntimeResult.is_failure", rr_fail.is_failure)
    _check("RuntimeResult.value", rr_ok.value)
    _check("RuntimeResult.error", rr_fail.error)
    _check("RuntimeResult.error_code", rr_fail.error_code)
    _check("RuntimeResult.unwrap_or", rr_fail.unwrap_or(99))
    _check("RuntimeResult.unwrap_or_else", rr_fail.unwrap_or_else(lambda: 11))
    _check("RuntimeResult.map", rr_ok.map(lambda num: num + 2).unwrap_or(0))
    _check(
        "RuntimeResult.flat_map",
        rr_ok.flat_map(lambda num: s.RuntimeResult[int].ok(num * 3)).unwrap_or(0),
    )
    _check(
        "RuntimeResult.and_then",
        rr_ok.and_then(lambda num: s.RuntimeResult[int].ok(num + 1)).unwrap_or(0),
    )
    _check(
        "RuntimeResult.flow_through",
        rr_ok.flow_through(
            lambda num: s.RuntimeResult[int].ok(num + 1),
            lambda num: s.RuntimeResult[int].ok(num * 2),
        ).unwrap_or(0),
    )
    _check(
        "RuntimeResult.fold.success",
        rr_ok.fold(
            on_failure=lambda err: f"f:{err}", on_success=lambda num: f"s:{num}"
        ),
    )
    _check(
        "RuntimeResult.fold.failure",
        rr_fail.fold(
            on_failure=lambda err: f"f:{err}", on_success=lambda num: f"s:{num}"
        ),
    )

    taps: list[int] = []
    tap_errors: list[str] = []
    _check("RuntimeResult.tap", rr_ok.tap(lambda num: taps.append(num)).is_success)
    _check(
        "RuntimeResult.tap_error",
        rr_fail.tap_error(lambda err: tap_errors.append(err)).is_failure,
    )
    _check("RuntimeResult.tap.values", taps)
    _check("RuntimeResult.tap_error.values", tap_errors)

    _check("RuntimeResult.map_error", rr_fail.map_error(lambda err: f"x:{err}").error)
    _check("RuntimeResult.filter.pass", rr_ok.filter(lambda num: num > 0).is_success)
    _check("RuntimeResult.filter.fail", rr_ok.filter(lambda num: num < 0).error)
    _check("RuntimeResult.alt", rr_fail.alt(lambda err: f"alt:{err}").error)
    _check(
        "RuntimeResult.lash",
        rr_fail.lash(lambda err: s.RuntimeResult[int].ok(len(err))).unwrap_or(0),
    )
    _check("RuntimeResult.recover", rr_fail.recover(lambda err: len(err)).unwrap_or(0))
    _check("RuntimeResult.operator_or", rr_fail | 77)
    _check("RuntimeResult.bool.success", bool(rr_ok))
    _check("RuntimeResult.bool.failure", bool(rr_fail))
    _check("RuntimeResult.repr.success", repr(rr_ok))
    _check("RuntimeResult.repr.failure", repr(rr_fail))

    with rr_ok as ctx_ok:
        _check("RuntimeResult.context_manager", ctx_ok.unwrap_or(0))

    try:
        _ = s.RuntimeResult[object].ok(None)
        _check("RuntimeResult.ok.none_raises", False)
    except ValueError as exc:
        _check("RuntimeResult.ok.none_raises", True)
        _check("RuntimeResult.ok.none_type", type(exc).__name__)

    # Runtime model helper methods inherited by FlextService
    e1 = _EntityStub(unique_id="u1")
    e2 = _EntityStub(unique_id="u1")
    e3 = _EntityStub(unique_id="u2")
    _check("compare_entities_by_id.true", s.compare_entities_by_id(e1, e2))
    _check("compare_entities_by_id.false", s.compare_entities_by_id(e1, e3))
    _check("hash_entity_by_id.type", type(s.hash_entity_by_id(e1)).__name__)
    _check("compare_value_objects_by_value", s.compare_value_objects_by_value(e1, e2))
    _check(
        "hash_value_object_by_value.type",
        type(s.hash_value_object_by_value(e1)).__name__,
    )

    trace = s.ensure_trace_context(
        {"source": "demo"},
        include_correlation_id=True,
        include_timestamp=True,
    )
    _check("ensure_trace_context.has_trace_id", "trace_id" in trace)
    _check("ensure_trace_context.has_span_id", "span_id" in trace)
    _check("ensure_trace_context.has_correlation_id", "correlation_id" in trace)
    _check(
        "validate_http_status_codes.ok",
        s.validate_http_status_codes([200, "201"]).unwrap_or([]),
    )
    _check("validate_http_status_codes.fail", s.validate_http_status_codes([99]).error)


def main() -> None:
    """Run all sections and verify against the golden file."""
    demo_service_core_api()
    demo_runtime_creation_and_serialization()
    demo_mixins_and_runtime_methods()
    demo_namespace_bootstrap_cqrs_validation()
    demo_namespace_runtime_and_integration()
    _verify()


if __name__ == "__main__":
    main()
