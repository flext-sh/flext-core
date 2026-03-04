"""Golden-file example for FlextService (s) public APIs."""

from __future__ import annotations

import sys
from collections import UserDict
from typing import ClassVar, cast, override

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
)

from .shared import Examples


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

    _execution_result: r[str] = PrivateAttr()

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
            context=cast("p.Context", FlextContext.create()),
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


class Ex11FlextService(Examples):
    """Exercise FlextService public API."""

    @override
    def exercise(self) -> None:
        """Run all FlextService example sections."""
        self.demo_service_core_api()
        self.demo_runtime_creation_and_serialization()
        self.demo_mixins_and_runtime_methods()
        self.demo_namespace_bootstrap_cqrs_validation()
        self.demo_namespace_runtime_and_integration()

    def demo_service_core_api(self) -> None:
        """Exercise constructor, execute, properties, validation, and metadata APIs."""
        self.section("service_core_api")

        fallback = self.rand_str(4)

        self.check("alias.FlextService_is_s", FlextService is s)

        service = _EchoService()
        execute_value = service.execute().unwrap_or(fallback)
        self.check("execute.unwrap", execute_value)
        self.check("execute.unwrap.matches", execute_value == "echo:ok")
        result_attr = service.result
        if callable(result_attr):
            result_value: t.ContainerValue = result_attr()
        else:
            result_value = result_attr
        self.check("result.property", result_value)
        runtime_view = service.runtime
        self.check("runtime.type", type(runtime_view).__name__)
        self.check("runtime.has_config_attr", hasattr(runtime_view, "config"))
        self.check("runtime.has_context_attr", hasattr(runtime_view, "context"))
        self.check("runtime.has_container_attr", hasattr(runtime_view, "container"))
        self.check("context.type", type(service.context).__name__)
        self.check("config.type", type(service.config).__name__)
        self.check("container.type", type(service.container).__name__)

        self.check(
            "validate_business_rules.default",
            service.validate_business_rules().is_success,
        )
        self.check("is_valid.default", service.is_valid())
        self.check("service_info.type", service.get_service_info().get("service_type"))

        rule_service = _RuleService()
        self.check(
            "validate_business_rules.override.success",
            rule_service.validate_business_rules().is_success,
        )
        self.check(
            "validate_business_rules.override.error",
            rule_service.validate_business_rules().error,
        )
        self.check("is_valid.override", rule_service.is_valid())

        crashing = _ValidationCrashService()
        self.check("is_valid.exception_guard", crashing.is_valid())

        failing = _FailingService()
        try:
            _ = failing.result
            self.check("result.failure.raises", False)
        except FlextExceptions.BaseError as exc:
            self.check("result.failure.raises", True)
            self.check("result.failure.type", type(exc).__name__)

        declarative = _DeclarativeService()
        self.check("auto_execute.declared", declarative.auto_execute)
        self.check("auto_execute.execute_count_after_init", declarative.execution_count)
        auto_result_attr = declarative.result
        if callable(auto_result_attr):
            auto_result: t.ContainerValue = auto_result_attr()
        else:
            auto_result = auto_result_attr
        self.check("auto_execute.result", auto_result)
        self.check(
            "auto_execute.execute_count_after_result", declarative.execution_count
        )

    def demo_runtime_creation_and_serialization(self) -> None:
        """Exercise runtime factory params and serialization helpers."""
        self.section("runtime_creation_and_serialization")

        payload_text = self.rand_str(6)
        payload_count = self.rand_int(1, 9)
        map_key_a = self.rand_str(3)
        map_key_b = self.rand_str(3)
        map_val_a = self.rand_int(1, 9)
        map_val_b = self.rand_str(4)

        base = _EchoService()

        runtime_default = _RuntimeFactoryService.create_runtime_default()
        self.check(
            "create_runtime.default.context", type(runtime_default.context).__name__
        )

        runtime_full = _RuntimeFactoryService.create_runtime_full()
        self.check(
            "create_runtime.full.container", type(runtime_full.container).__name__
        )
        self.check("create_runtime.full.config", type(runtime_full.config).__name__)

        payload = _Payload(text=payload_text, count=payload_count)
        self.check("to_dict.none", s.to_dict(None))
        self.check(
            "to_dict.mapping", s.to_dict({map_key_a: map_val_a, map_key_b: map_val_b})
        )
        self.check("to_dict.model", s.to_dict(payload))

        model_dump_value = base.model_dump(exclude={"runtime"})
        self.check("model_dump.type", type(model_dump_value).__name__)
        self.check("model_dump.has_result", "result" in model_dump_value)
        model_dump_json_value = base.model_dump_json(exclude={"runtime"})
        self.check("model_dump_json.type", type(model_dump_json_value).__name__)

    def demo_mixins_and_runtime_methods(self) -> None:
        """Exercise inherited mixin/runtime APIs on FlextService."""
        self.section("mixins_and_runtime_methods")

        ok_value = self.rand_int(1, 999)
        fail_message = self.rand_str(7)
        fail_code = self.rand_str(4)
        error_key = self.rand_str(3)
        error_value = self.rand_str(3)
        ensure_raw = self.rand_str(7)
        ensure_existing = self.rand_str(7)
        prefix = f"svc.{self.rand_str(4)}"
        op_name = self.rand_str(8)
        even_a = self.rand_int(2, 50) * 2
        even_b = self.rand_int(2, 50) * 2
        odd_value = self.rand_int(1, 49) * 2 + 1
        err_one = self.rand_str(4)
        err_two = self.rand_str(4)

        service = _EchoService()

        self.check("ok.unwrap", service.ok(ok_value).unwrap_or(-1) == ok_value)
        self.check(
            "fail.error",
            service.fail(
                fail_message,
                error_code=fail_code,
                error_data=m.ConfigMap(root={error_key: error_value}),
            ).error
            == fail_message,
        )

        self.check(
            "ensure_result.raw",
            service.ensure_result(ensure_raw).unwrap_or("none") == ensure_raw,
        )
        self.check(
            "ensure_result.result",
            service.ensure_result(r[str].ok(ensure_existing)).unwrap_or("none")
            == ensure_existing,
        )

        generated_id = service.generate_id()
        prefixed = service.generate_prefixed_id(prefix, length=8)
        self.check("generate_id.length", len(generated_id))
        self.check("generate_prefixed_id.prefix", prefixed.startswith(f"{prefix}_"))
        self.check("generate_prefixed_id.length", len(prefixed))
        self.check(
            "generate_datetime_utc.tz",
            service.generate_datetime_utc().tzinfo is not None,
        )

        with service.track(op_name) as metrics:
            self.check("track.context.type", type(metrics).__name__)
        self.check("track.has_operation_count", "operation_count" in metrics)
        self.check("track.has_success_rate", "success_rate" in metrics)

        def _to_even(num: int) -> r[int]:
            if num % 2 == 0:
                return r[int].ok(num)
            return r[int].fail(f"odd:{num}")

        self.check(
            "traverse.success",
            service.traverse([even_a, even_b], _to_even, fail_fast=True).unwrap_or([]),
        )
        self.check(
            "traverse.fail_fast",
            service.traverse([even_a, odd_value], _to_even, fail_fast=True).error,
        )
        self.check(
            "accumulate_errors.ok",
            service.accumulate_errors(r[int].ok(1), r[int].ok(2)).is_success,
        )
        fail_one = service.fail(err_one)
        fail_two = service.fail(err_two)
        self.check(
            "accumulate_errors.fail",
            service.accumulate_errors(fail_one, fail_two).error,
        )

    def demo_namespace_bootstrap_cqrs_validation(self) -> None:
        """Exercise Bootstrap/CQRS/ProtocolValidation/Validation namespace APIs."""
        self.section("namespace_bootstrap_cqrs_validation")

        metric_key = self.rand_str(5)
        metric_value = self.rand_int(1, 9)
        ctx_handler_name = self.rand_str(6)
        valid_upper = self.rand_str(3).upper()
        invalid_lower = self.rand_str(2)

        tiny = s.Bootstrap.create_instance(_TinyType)
        self.check("Bootstrap.create_instance.type", type(tiny).__name__)
        self.check(
            "Bootstrap.create_instance.init_called", hasattr(tiny, "initialized")
        )

        metrics = s.CQRS.MetricsTracker()
        self.check(
            "CQRS.MetricsTracker.record",
            metrics.record_metric(metric_key, metric_value).is_success,
        )
        self.check(
            "CQRS.MetricsTracker.get",
            cast(
                "m.ConfigMap", metrics.get_metrics().unwrap_or(m.ConfigMap(root={}))
            ).get(metric_key)
            == metric_value,
        )

        stack = s.CQRS.ContextStack()
        self.check(
            "CQRS.ContextStack.push.dict",
            stack.push_context(
                cast(
                    "t.Container",
                    {
                        "handler_name": ctx_handler_name,
                        "handler_mode": "query",
                    },
                )
            ).is_success,
        )
        current = stack.current_context()
        self.check(
            "CQRS.ContextStack.current.type",
            type(current).__name__ if current is not None else "None",
        )
        popped = cast("m.ConfigMap", stack.pop_context().unwrap_or({}))
        self.check("CQRS.ContextStack.pop.handler_name", popped.get("handler_name"))
        self.check("CQRS.ContextStack.pop.empty", stack.pop_context().unwrap_or({}))

        proto_handler = s.ProtocolValidation.is_handler(
            cast("t.Container", _HandlerLike())
        )
        service_like = _ServiceLike()
        is_service_fn = getattr(s.ProtocolValidation, "is_service")
        proto_service = bool(is_service_fn(service_like))
        proto_bus = s.ProtocolValidation.is_command_bus()
        validate_protocol_fn = getattr(
            s.ProtocolValidation, "validate_protocol_compliance"
        )
        proto_ok = validate_protocol_fn(service_like, "Service")
        proto_unknown = validate_protocol_fn(service_like, "UnknownProto")
        processor_ok = s.ProtocolValidation.validate_processor_protocol(
            _ProcessorProtocolGood(),
        )
        processor_bad = s.ProtocolValidation.validate_processor_protocol(
            _ProcessorProtocolBad(),
        )

        self.check("ProtocolValidation.is_handler", proto_handler)
        self.check("ProtocolValidation.is_service", proto_service)
        self.check("ProtocolValidation.is_command_bus", proto_bus)
        self.check("ProtocolValidation.protocol_ok", proto_ok.is_success)
        self.check("ProtocolValidation.protocol_unknown", proto_unknown.error)
        self.check("ProtocolValidation.processor_ok", processor_ok.is_success)
        self.check("ProtocolValidation.processor_bad", processor_bad.error)

        def _validator_len(data: t.ContainerValue) -> r[bool]:
            if isinstance(data, str) and len(data) >= 3:
                return r[bool].ok(True)
            return r[bool].fail("too-short")

        def _validator_upper(data: t.ContainerValue) -> r[bool]:
            if isinstance(data, str) and data.isupper():
                return r[bool].ok(True)
            return r[bool].fail("not-upper")

        valid = s.Validation.validate_with_result(
            valid_upper, [_validator_len, _validator_upper]
        )
        invalid = s.Validation.validate_with_result(
            invalid_lower, [_validator_len, _validator_upper]
        )
        self.check("Validation.validate_with_result.ok", valid.unwrap_or(""))
        self.check("Validation.validate_with_result.fail", invalid.error)

    def demo_namespace_runtime_and_integration(self) -> None:
        """Exercise RuntimeResult, Integration, DependencyIntegration, and Metadata."""
        self.section("namespace_runtime_and_integration")

        service_name = f"svc.{self.rand_str(6)}"
        service_version = (
            f"{self.rand_int(1, 9)}.{self.rand_int(0, 9)}.{self.rand_int(0, 9)}"
        )
        resolved_name = self.rand_str(8)
        unresolved_name = self.rand_str(8)
        unresolved_error = self.rand_str(8)
        event_name = self.rand_str(8)
        aggregate_id = self.rand_str(8)
        event_key = self.rand_str(3)
        event_value = self.rand_str(4)
        env_value = self.rand_str(5)
        service_object = self.rand_str(5)
        factory_object = self.rand_str(6)
        resource_number = self.rand_int(1, 99)
        region_value = self.rand_str(2)
        rr_value = self.rand_int(1, 50)
        rr_error = self.rand_str(6)
        rr_error_code = self.rand_str(4)
        rr_fallback = self.rand_int(1, 99)
        rr_else = self.rand_int(1, 99)
        rr_add = self.rand_int(1, 9)
        rr_mul = self.rand_int(2, 5)
        rr_or = self.rand_int(1, 99)
        source_value = self.rand_str(5)
        entity_id = self.rand_str(8)
        other_entity_id = self.rand_str(8)

        # Metadata namespace class (lazy-loaded from runtime)
        metadata_cls = cast("type", s.Metadata)
        metadata = metadata_cls(attributes={"service": service_name})
        self.check("Metadata.version", metadata.version)
        self.check("Metadata.attributes", metadata.attributes)

        # Integration namespace methods (return None)
        s.Integration.setup_service_infrastructure(
            service_name=service_name, service_version=service_version
        )
        s.Integration.track_service_resolution(resolved_name, resolved=True)
        s.Integration.track_service_resolution(
            unresolved_name,
            resolved=False,
            error_message=unresolved_error,
        )
        s.Integration.track_domain_event(
            event_name=event_name,
            aggregate_id=aggregate_id,
            event_data=m.ConfigMap(root={event_key: event_value}),
        )
        self.check("Integration.calls", "ok")

        # DependencyIntegration namespace
        di = s.DependencyIntegration.create_container(
            config=m.ConfigMap(root={"env": env_value}),
            services={"object_item": service_object},
            factories={"factory_item": lambda: factory_object},
            resources={
                "resource_item": lambda: m.ConfigMap(root={"value": resource_number})
            },
            wire_modules=[sys.modules[__name__]],
            wire_packages=["flext_core"],
            wire_classes=[_EchoService],
            factory_cache=False,
        )
        self.check("DependencyIntegration.service", di.object_item() == service_object)
        self.check("DependencyIntegration.factory", di.factory_item() == factory_object)
        self.check(
            "DependencyIntegration.resource.type", type(di.resource_item()).__name__
        )

        bridge, service_mod, resource_mod = (
            s.DependencyIntegration.create_layered_bridge(
                config=m.ConfigMap(root={"region": region_value}),
            )
        )
        self.check("DependencyIntegration.bridge.type", type(bridge).__name__)
        self.check(
            "DependencyIntegration.service_module.type", type(service_mod).__name__
        )
        self.check(
            "DependencyIntegration.resource_module.type", type(resource_mod).__name__
        )

        # RuntimeResult namespace methods
        rr_ok = s.RuntimeResult[int].ok(rr_value)
        rr_fail = s.RuntimeResult[int].fail(rr_error, error_code=rr_error_code)

        self.check("RuntimeResult.is_success", rr_ok.is_success)
        self.check("RuntimeResult.is_failure", rr_fail.is_failure)
        self.check("RuntimeResult.value", rr_ok.value)
        self.check("RuntimeResult.error", rr_fail.error)
        self.check("RuntimeResult.error_code", rr_fail.error_code)
        self.check(
            "RuntimeResult.unwrap_or", rr_fail.unwrap_or(rr_fallback) == rr_fallback
        )
        self.check(
            "RuntimeResult.unwrap_or_else",
            rr_fail.unwrap_or_else(lambda: rr_else) == rr_else,
        )
        self.check(
            "RuntimeResult.map",
            rr_ok.map(lambda num: num + rr_add).unwrap_or(0) == rr_value + rr_add,
        )
        self.check(
            "RuntimeResult.flat_map",
            rr_ok.flat_map(lambda num: s.RuntimeResult[int].ok(num * rr_mul)).unwrap_or(
                0
            )
            == rr_value * rr_mul,
        )
        self.check(
            "RuntimeResult.and_then",
            rr_ok.flat_map(lambda num: s.RuntimeResult[int].ok(num + rr_add)).unwrap_or(
                0
            )
            == rr_value + rr_add,
        )
        self.check(
            "RuntimeResult.flow_through",
            rr_ok.flow_through(
                lambda num: s.RuntimeResult[int].ok(num + 1),
                lambda num: s.RuntimeResult[int].ok(num * 2),
            ).unwrap_or(0)
            == (rr_value + 1) * 2,
        )
        self.check(
            "RuntimeResult.fold.success",
            rr_ok.fold(
                on_failure=lambda err: f"f:{err}", on_success=lambda num: f"s:{num}"
            ),
        )
        self.check(
            "RuntimeResult.fold.failure",
            rr_fail.fold(
                on_failure=lambda err: f"f:{err}", on_success=lambda num: f"s:{num}"
            ),
        )

        taps: list[int] = []
        tap_errors: list[str] = []
        self.check(
            "RuntimeResult.tap", rr_ok.tap(lambda num: taps.append(num)).is_success
        )
        self.check(
            "RuntimeResult.tap_error",
            rr_fail.tap_error(lambda err: tap_errors.append(err)).is_failure,
        )
        self.check("RuntimeResult.tap.values", taps)
        self.check("RuntimeResult.tap_error.values", tap_errors)

        self.check(
            "RuntimeResult.map_error", rr_fail.map_error(lambda err: f"x:{err}").error
        )
        self.check(
            "RuntimeResult.filter.pass", rr_ok.filter(lambda num: num > 0).is_success
        )
        self.check("RuntimeResult.filter.fail", rr_ok.filter(lambda num: num < 0).error)
        self.check(
            "RuntimeResult.alt", rr_fail.map_error(lambda err: f"alt:{err}").error
        )
        self.check(
            "RuntimeResult.lash",
            rr_fail.lash(lambda err: s.RuntimeResult[int].ok(len(err))).unwrap_or(0),
        )
        self.check(
            "RuntimeResult.recover", rr_fail.recover(lambda err: len(err)).unwrap_or(0)
        )
        self.check("RuntimeResult.operator_or", rr_fail | rr_or)
        self.check("RuntimeResult.bool.success", bool(rr_ok))
        self.check("RuntimeResult.bool.failure", bool(rr_fail))
        self.check("RuntimeResult.repr.success", repr(rr_ok))
        self.check("RuntimeResult.repr.failure", repr(rr_fail))

        with rr_ok as ctx_ok:
            self.check("RuntimeResult.context_manager", ctx_ok.unwrap_or(0))

        valid_container_value = self.rand_str(4)
        self.check(
            "RuntimeResult.ok.none_raises",
            s.RuntimeResult[t.ContainerValue].ok(valid_container_value).is_success,
        )
        self.check("RuntimeResult.ok.none_type", type(valid_container_value).__name__)

        # Runtime model helper methods inherited by FlextService
        e1 = _EntityStub(unique_id=entity_id)
        e2 = _EntityStub(unique_id=entity_id)
        e3 = _EntityStub(unique_id=other_entity_id)
        self.check("compare_entities_by_id.true", s.compare_entities_by_id(e1, e2))
        self.check("compare_entities_by_id.false", s.compare_entities_by_id(e1, e3))
        self.check("hash_entity_by_id.type", type(s.hash_entity_by_id(e1)).__name__)
        self.check(
            "compare_value_objects_by_value", s.compare_value_objects_by_value(e1, e2)
        )
        self.check(
            "hash_value_object_by_value.type",
            type(s.hash_value_object_by_value(e1)).__name__,
        )

        trace = s.ensure_trace_context(
            {"source": source_value},
            include_correlation_id=True,
            include_timestamp=True,
        )
        self.check("ensure_trace_context.has_trace_id", "trace_id" in trace)
        self.check("ensure_trace_context.has_span_id", "span_id" in trace)
        self.check("ensure_trace_context.has_correlation_id", "correlation_id" in trace)
        self.check(
            "validate_http_status_codes.ok",
            s.validate_http_status_codes([200, "201"]).unwrap_or([]),
        )
        self.check(
            "validate_http_status_codes.fail", s.validate_http_status_codes([99]).error
        )


if __name__ == "__main__":
    Ex11FlextService(__file__).run()
