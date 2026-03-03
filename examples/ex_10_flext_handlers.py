# pyright: reportGeneralTypeIssues=false, reportArgumentType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportIncompatibleMethodOverride=false, reportUnusedImport=false, reportUnnecessaryComparison=false, reportUnreachable=false, reportIncompatibleVariableOverride=false, reportAttributeAccessIssue=false
"""Golden-file example for FlextHandlers (h) public APIs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import ClassVar, override

from flext_core import FlextHandlers, c, e, h, m, r, t, u

_RESULTS: list[str] = []


def _check(label: str, value: t.ContainerValue | type) -> None:
    _RESULTS.append(f"{label}: {_ser(value)}")


def _section(name: str) -> None:
    if _RESULTS:
        _RESULTS.append("")
    _RESULTS.append(f"[{name}]")


def _ser(v: t.ContainerValue | type) -> str:
    if v is None:
        return "None"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return repr(v)
    if u.is_list(v):
        return "[" + ", ".join(_ser(item) for item in v) + "]"
    if u.is_dict_like(v):
        pairs = ", ".join(
            f"{_ser(k)}: {_ser(val)}"
            for k, val in sorted(v.items(), key=lambda kv: str(kv[0]))
        )
        return "{" + pairs + "}"
    if isinstance(v, type):
        return v.__name__
    return type(v).__name__


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


class _Message(m.Command):
    text: str


class _DerivedMessage(_Message):
    pass


class _Entity(m.Value):
    unique_id: str


class _NoArgs:
    def __init__(self) -> None:
        self.marker = "created"


class _ProtocolHandler:
    def handle(self, message: t.ContainerValue) -> r[t.ContainerValue]:
        return r[t.ContainerValue].ok(message)

    def validate(self, data: t.ContainerValue) -> r[bool]:
        return r[bool].ok(data is not None)

    @staticmethod
    def _protocol_name() -> str:
        return "ProtocolHandler"


class _ServiceStub:
    def execute(self) -> r[t.ContainerValue]:
        return r[t.ContainerValue].ok(m.ConfigMap(root={"ok": True}))

    def validate_business_rules(self) -> r[bool]:
        return r[bool].ok(True)

    @property
    def is_valid(self) -> bool:
        return True

    def get_service_info(self) -> m.ConfigMap:
        return m.ConfigMap(root={"service": "stub"})

    @staticmethod
    def _protocol_name() -> str:
        return "ServiceStub"


class _ProcessorGood(m.Value):
    marker: str = "good"

    def process(self) -> str:
        return "ok"

    def validate(self) -> bool:
        return True

    @staticmethod
    def _protocol_name() -> str:
        return "ProcessorGood"


class _ProcessorBad(m.Value):
    marker: str = "bad"

    def process(self) -> str:
        return "ok"

    @staticmethod
    def _protocol_name() -> str:
        return "ProcessorBad"


class _NotImplementedPatternHandler(h[t.ContainerValue, t.ContainerValue]):
    @override
    def handle(self, message: t.ContainerValue) -> r[t.ContainerValue]:
        return super().handle(message)


class _DemoHandler(h[t.ContainerValue, str]):
    _expected_message_type: ClassVar[type | None] = _Message

    @override
    def handle(self, message: t.ContainerValue) -> r[str]:
        if message == "explode":
            error_message = "forced boom"
            raise RuntimeError(error_message)
        if u.is_dict_like(message):
            return r[str].ok(f"dict:{len(message)}")
        return r[str].ok(f"msg:{message}")

    @override
    def validate(self, data: t.ContainerValue) -> r[bool]:
        base = super().validate(data)
        if base.is_failure:
            return base
        if data == "bad":
            return r[bool].fail("blocked")
        return r[bool].ok(True)


def demo_handler_core() -> None:
    _section("handler_core")

    pattern_handler = _NotImplementedPatternHandler()
    try:
        pattern_handler.handle("x")
        pattern_value: t.ContainerValue = "no-error"
    except NotImplementedError as exc:
        pattern_value = f"{type(exc).__name__}:{exc}"
    _check("handle.not_implemented_pattern", pattern_value)

    handler = _DemoHandler(
        config=m.Handler(handler_id="demo", handler_name="DemoHandler")
    )
    _check("handler.handler_name", handler.handler_name)
    _check("handler.mode", handler.mode.value)

    _check("validate.none.failure", handler.validate(None).is_failure)
    _check("validate.ok.success", handler.validate("ok").is_success)
    _check("validate_command.blocked", handler.validate_command("bad").error)
    _check("validate_query.blocked", handler.validate_query("bad").error)
    _check(
        "validate_alias.same_behavior",
        handler.validate("ok").unwrap_or(False)
        and handler.validate_command("ok").unwrap_or(False)
        and handler.validate_query("ok").unwrap_or(False),
    )

    _check("can_handle.expected", handler.can_handle(_Message))
    _check("can_handle.derived", handler.can_handle(_DerivedMessage))
    _check("can_handle.other", handler.can_handle(str))

    execute_value = handler.execute(_Message(text="ping")).unwrap_or("-")
    _check(
        "execute.success.value",
        str(execute_value).startswith("msg:message_type='command'"),
    )
    _check("execute.validation_failure", handler.execute("bad").error)

    dispatch_value = handler.dispatch_message(_Message(text="go")).unwrap_or("-")
    _check(
        "dispatch.success",
        str(dispatch_value).startswith("msg:message_type='command'"),
    )
    _check(
        "dispatch.mode_mismatch",
        handler.dispatch_message(
            _Message(text="go"), operation=c.Dispatcher.HANDLER_MODE_QUERY
        ).error,
    )
    _check(
        "dispatch.pipeline_exception",
        handler.dispatch_message(
            "explode", operation=c.Dispatcher.HANDLER_MODE_COMMAND
        ).error,
    )

    _check("record_metric.ok", handler.record_metric("k", 1).is_success)
    _check(
        "push_context.mapping",
        handler.push_context({
            "handler_name": "h1",
            "handler_mode": "query",
        }).is_success,
    )
    _check(
        "push_context.execution",
        handler.push_context(
            m.Handler.ExecutionContext.create_for_handler(
                handler_name="h2",
                handler_mode="event",
            )
        ).is_success,
    )
    _check(
        "pop_context.1",
        handler.pop_context().unwrap_or(m.ConfigMap(root={})).get("handler_name", "-"),
    )
    _check(
        "pop_context.2",
        handler.pop_context().unwrap_or(m.ConfigMap(root={})).get("handler_name", "-"),
    )


def demo_create_from_callable() -> None:
    _section("create_from_callable")

    default_h = FlextHandlers.create_from_callable(lambda message: f"default:{message}")
    _check("callable.default", default_h.handle("x").unwrap_or("-"))

    named_h = FlextHandlers.create_from_callable(
        lambda message: f"named:{message}",
        handler_name="NamedCallable",
        handler_type=c.Cqrs.HandlerType.QUERY,
    )
    _check("callable.named.handler_name", named_h.handler_name)
    _check("callable.named.mode", named_h.mode.value)

    mode_enum_h = FlextHandlers.create_from_callable(
        lambda message: f"enum:{message}",
        mode=c.Cqrs.HandlerType.EVENT,
    )
    _check("callable.mode_enum", mode_enum_h.mode.value)

    mode_str_h = FlextHandlers.create_from_callable(
        lambda message: f"str:{message}",
        mode="query",
    )
    _check("callable.mode_str", mode_str_h.mode.value)

    config_h = FlextHandlers.create_from_callable(
        lambda message: f"cfg:{message}",
        handler_config=m.Handler(
            handler_id="cfg_id",
            handler_name="ConfiguredCallable",
            handler_mode=c.Cqrs.HandlerType.SAGA,
            handler_type=c.Cqrs.HandlerType.SAGA,
        ),
    )
    _check("callable.handler_config.name", config_h.handler_name)
    _check("callable.handler_config.mode", config_h.mode.value)

    try:
        FlextHandlers.create_from_callable(lambda message: message, mode="invalid")
        invalid_mode: t.ContainerValue = "no-error"
    except e.ValidationError as exc:
        invalid_mode = f"{type(exc).__name__}:{exc}"
    _check("callable.invalid_mode", invalid_mode)


def demo_discovery() -> None:
    _section("discovery")

    class _Service:
        @staticmethod
        @h.handler(_Message, priority=2)
        def high(_message: t.ContainerValue) -> t.ContainerValue:
            return "high"

        @staticmethod
        @h.handler(_Message, priority=1, timeout=3.0, middleware=[])
        def low(_message: t.ContainerValue) -> t.ContainerValue:
            return "low"

    class_scan = h.Discovery.scan_class(_Service)
    _check("scan_class.count", len(class_scan))
    _check("scan_class.first", class_scan[0][0] if class_scan else "none")
    _check("has_handlers.class", h.Discovery.has_handlers(_Service))
    _check("has_handlers.class_none", h.Discovery.has_handlers(_NoArgs))

    module = ModuleType("ex10_handlers_module")

    @h.handler(_Message, priority=9)
    def mod_handler(message: t.ContainerValue) -> t.ContainerValue:
        return f"module:{message}"

    def plain_function(_message: t.ContainerValue) -> t.ContainerValue:
        return "plain"

    setattr(module, "mod_handler", mod_handler)
    setattr(module, "plain_function", plain_function)

    module_scan = h.Discovery.scan_module(module)
    _check("scan_module.count", len(module_scan))
    _check("scan_module.name", module_scan[0][0] if module_scan else "none")
    wrapped_result = module_scan[0][1](_Message(text="m")) if module_scan else "none"
    _check(
        "scan_module.wrapped_result",
        str(wrapped_result).startswith("module:message_type='command'"),
    )
    _check("has_handlers_module.true", h.Discovery.has_handlers_module(module))
    _check(
        "has_handlers_module.false",
        h.Discovery.has_handlers_module(ModuleType("empty")),
    )


def demo_namespaces_and_mixins() -> None:
    _section("namespaces_and_mixins")

    created = h.Bootstrap.create_instance(_NoArgs)
    _check("bootstrap.create_instance", created.__class__.__name__)

    tracker = h.CQRS.MetricsTracker()
    _check("cqrs.record_metric", tracker.record_metric("hits", 2).is_success)
    _check(
        "cqrs.get_metrics",
        tracker.get_metrics().unwrap_or(m.ConfigMap(root={})).get("hits", -1),
    )

    stack = h.CQRS.ContextStack()
    _check(
        "cqrs.push_context.mapping",
        stack.push_context({
            "handler_name": "ctx",
            "handler_mode": "command",
        }).is_success,
    )
    current_context = stack.current_context()
    _check("cqrs.current_context", getattr(current_context, "handler_name", "-"))
    _check(
        "cqrs.pop_context",
        stack.pop_context().unwrap_or(m.ConfigMap(root={})).get("handler_name", "-"),
    )

    di = h.DependencyIntegration
    di_container = di.create_container(config=m.ConfigMap(root={"env": "test"}))
    _check("di.bind_configuration_exists", hasattr(di_container, "config"))
    _check("di.register_object", di.register_object(di_container, "obj", 3)() == 3)
    _check(
        "di.register_factory.cached",
        di.register_factory(di_container, "factory", lambda: "v", cache=True)() == "v",
    )
    resource_provider = di.register_resource(
        di_container,
        "res",
        lambda: m.ConfigMap(root={"k": "v"}),
    )
    _check("di.register_resource", resource_provider().get("k", "-") == "v")
    try:
        di.register_object(di_container, "obj", 4)
        duplicate_error: t.ContainerValue = "no-error"
    except ValueError as exc:
        duplicate_error = f"{type(exc).__name__}:{exc}"
    _check("di.duplicate_error", duplicate_error)

    bridge, services_mod, resources_mod = di.create_layered_bridge(
        m.ConfigMap(root={"x": "1"})
    )
    _check("di.layered.bridge", bridge.__class__.__name__)
    _check("di.layered.services", services_mod.__class__.__name__)
    _check("di.layered.resources", resources_mod.__class__.__name__)
    di.wire(di_container, modules=[])
    _check("di.wire.noop", True)

    h.Integration.track_service_resolution("svc", resolved=True)
    h.Integration.track_service_resolution(
        "svc", resolved=False, error_message="not-found"
    )
    h.Integration.track_domain_event(
        "evt",
        aggregate_id="agg-1",
        event_data=m.ConfigMap(root={"n": 1}),
    )
    h.Integration.setup_service_infrastructure(
        service_name="svc",
        service_version="1.2.3",
        enable_context_correlation=True,
    )
    _check("integration.calls", True)

    meta = h.Metadata(version="2.0.0", attributes={"tag": "v2"})
    _check("metadata.version", meta.version)
    _check("metadata.attributes", meta.attributes)

    protocol_handler = _ProtocolHandler()
    _check(
        "protocol.is_handler.true", h.ProtocolValidation.is_handler(protocol_handler)
    )
    _check(
        "protocol.is_handler.false",
        h.ProtocolValidation.is_handler(m.ConfigMap(root={})),
    )
    _check("protocol.is_service", h.ProtocolValidation.is_service(_ServiceStub()))
    _check("protocol.is_command_bus", h.ProtocolValidation.is_command_bus())
    _check(
        "protocol.validate_known",
        h.ProtocolValidation.validate_protocol_compliance(
            protocol_handler, "Handler"
        ).is_success,
    )
    _check(
        "protocol.validate_unknown",
        h.ProtocolValidation.validate_protocol_compliance(
            protocol_handler, "Unknown"
        ).error,
    )
    _check(
        "protocol.validate_processor.good",
        h.ProtocolValidation.validate_processor_protocol(_ProcessorGood()).is_success,
    )
    _check(
        "protocol.validate_processor.bad",
        h.ProtocolValidation.validate_processor_protocol(_ProcessorBad()).error,
    )

    def positive_validator(item: t.ContainerValue) -> r[bool]:
        return r[bool].ok(item == "ok")

    def strict_validator(item: t.ContainerValue) -> r[bool]:
        return r[bool].ok(bool(item))

    def fail_validator(_item: t.ContainerValue) -> r[bool]:
        return r[bool].fail("rule-failed")

    _check(
        "validation.chain.ok",
        h.Validation.validate_with_result(
            "ok",
            [positive_validator, strict_validator],
        ).is_success,
    )
    _check(
        "validation.chain.fail",
        h.Validation.validate_with_result(
            "x",
            [positive_validator, fail_validator],
        ).error,
    )


def demo_runtime_result_and_utilities() -> None:
    _section("runtime_result_and_utilities")

    rr_ok = h.RuntimeResult[int].ok(5)
    rr_fail = h.RuntimeResult[int].fail("boom", error_code="E1")
    _check("rr.is_success", rr_ok.is_success)
    _check("rr.is_failure", rr_fail.is_failure)
    _check("rr.unwrap_or", rr_fail.unwrap_or(9))
    _check("rr.map", rr_ok.map(lambda n: n + 2).unwrap_or(-1))
    _check(
        "rr.flat_map",
        rr_ok.flat_map(lambda n: h.RuntimeResult[int].ok(n * 3)).unwrap_or(-1),
    )
    _check(
        "rr.and_then",
        rr_ok.and_then(lambda n: h.RuntimeResult[int].ok(n - 1)).unwrap_or(-1),
    )
    _check("rr.alt", rr_fail.alt(lambda err: f"x:{err}").error)
    _check(
        "rr.lash",
        rr_fail.lash(lambda _err: h.RuntimeResult[int].ok(42)).unwrap_or(-1),
    )
    _check("rr.recover", rr_fail.recover(lambda _err: 17).unwrap_or(-1))
    _check("rr.fold", rr_ok.fold(lambda err: err, lambda n: f"ok:{n}"))

    _check("mixin.ok", h.ok("v").unwrap_or("-"))
    _check("mixin.fail", h.fail("err", error_code="E2").error_code)
    _check("mixin.ensure_result.value", h.ensure_result(8).unwrap_or(-1))
    _check("mixin.ensure_result.result", h.ensure_result(r[int].ok(9)).unwrap_or(-1))
    _check("mixin.to_dict", h.to_dict(m.ConfigMap(root={"a": 1})))

    generated_a = h.generate_id()
    generated_b = h.generate_id()
    _check("runtime.generate_id.length", len(generated_a) == len(generated_b) == 36)
    _check("runtime.generate_id.unique", generated_a != generated_b)
    _check(
        "runtime.generate_prefixed_id.default",
        h.generate_prefixed_id("cmd").startswith("cmd_"),
    )
    _check(
        "runtime.generate_prefixed_id.length",
        len(h.generate_prefixed_id("q", length=8).split("_")[1]) == 8,
    )
    _check(
        "runtime.generate_datetime_utc", h.generate_datetime_utc().tzinfo is not None
    )

    e1 = _Entity(unique_id="abc")
    e2 = _Entity(unique_id="abc")
    e3 = _Entity(unique_id="xyz")
    _check("runtime.compare_entities.true", h.compare_entities_by_id(e1, e2))
    _check("runtime.compare_entities.false", h.compare_entities_by_id(e1, e3))
    _check("runtime.hash_entity", h.hash_entity_by_id(e1) != 0)

    _check(
        "runtime.compare_value_objects.scalar",
        h.compare_value_objects_by_value("a", "a"),
    )
    _check(
        "runtime.compare_value_objects.model",
        h.compare_value_objects_by_value(_Message(text="x"), _Message(text="x")),
    )
    _check(
        "runtime.hash_value_object",
        h.hash_value_object_by_value(_Message(text="x")) != 0,
    )

    trace_context = h.ensure_trace_context(
        {"source": "test"},
        include_correlation_id=True,
        include_timestamp=True,
    )
    _check("runtime.ensure_trace_context.source", trace_context.get("source", "-"))
    _check(
        "runtime.ensure_trace_context.keys",
        [
            "trace_id" in trace_context,
            "span_id" in trace_context,
            "correlation_id" in trace_context,
            "timestamp" in trace_context,
        ],
    )
    _check("runtime.get_log_level", h.get_log_level_from_config() >= 0)

    _check(
        "runtime.validate_http.success",
        h.validate_http_status_codes([200, "404"]).unwrap_or([]),
    )
    _check("runtime.validate_http.range_fail", h.validate_http_status_codes([99]).error)
    _check(
        "runtime.validate_http.type_fail",
        h.validate_http_status_codes([Path("x")]).error,
    )

    _check("runtime.is_dict_like.true", h.is_dict_like({"a": 1}))
    _check("runtime.is_dict_like.false", h.is_dict_like([1, 2]))
    _check("runtime.is_list_like.true", h.is_list_like([1, 2]))
    _check("runtime.is_list_like.false", h.is_list_like("ab"))
    _check("runtime.is_valid_json.true", h.is_valid_json(json.dumps({"a": 1})))
    _check("runtime.is_valid_json.false", h.is_valid_json("{bad"))
    _check("runtime.is_valid_identifier.true", h.is_valid_identifier("valid_name"))
    _check("runtime.is_valid_identifier.false", h.is_valid_identifier("1bad"))
    _check("runtime.safe_get_attribute", h.safe_get_attribute(e1, "unique_id", "-"))
    _check(
        "runtime.extract_generic_args", len(h.extract_generic_args(dict[str, int])) >= 1
    )
    _check("runtime.is_sequence_type.true", h.is_sequence_type(list[int]))
    _check("runtime.is_sequence_type.false", h.is_sequence_type(dict[str, int]))
    _check("runtime.normalize_general", h.normalize_to_general_value({"n": 1}))
    _check("runtime.normalize_metadata", h.normalize_to_metadata_value({"k": [1, 2]}))


def main() -> None:
    demo_handler_core()
    demo_create_from_callable()
    demo_discovery()
    demo_namespaces_and_mixins()
    demo_runtime_result_and_utilities()
    _verify()


if __name__ == "__main__":
    main()
