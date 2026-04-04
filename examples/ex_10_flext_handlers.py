"""Golden-file example for FlextHandlers (h) public APIs."""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import ClassVar, override

from pydantic import BaseModel

from flext_core import FlextHandlers, c, e, h, m, r, t, u

from ._models.ex10 import Ex10ProtocolHandler
from .shared import Examples


class _Message(m.Command):
    text: str


class _DerivedMessage(_Message):
    pass


class _PayloadModel(m.Value):
    text: str


class _Entity(m.Value):
    unique_id: str


class _NoArgs:
    def __init__(self) -> None:
        self.marker = "created"


class _NotImplementedPatternHandler(FlextHandlers[_Message, str]):
    @override
    def handle(self, message: _Message) -> r[str]:
        raise NotImplementedError


class _DemoHandler(FlextHandlers[_Message, str]):
    _expected_message_type: ClassVar[type | None] = _Message

    @override
    def handle(self, message: _Message) -> r[str]:
        if message.text == "explode":
            error_message = "forced boom"
            raise RuntimeError(error_message)
        return r[str].ok(f"msg:{message.text}")

    @override
    def validate_message(self, data: _Message) -> r[bool]:
        if data.text == "bad":
            return r[bool].fail("blocked")
        return r[bool].ok(True)


class Ex10FlextHandlers(Examples):
    """Exercise FlextHandlers public API."""

    def demo_create_from_callable(self) -> None:
        """Exercise FlextHandlers.create_from_callable variants."""
        self.section("create_from_callable")
        probe_default = self.rand_str(6)
        probe_named = self.rand_str(6)
        probe_enum = self.rand_str(6)
        probe_str = self.rand_str(6)
        probe_cfg = self.rand_str(6)
        named_handler_name = self.rand_str(10)
        cfg_handler_id = self.rand_str(8)
        cfg_handler_name = self.rand_str(10)
        default_h = FlextHandlers.create_from_callable(
            lambda message: f"default:{message}",
        )
        default_value = default_h.handle(probe_default).unwrap_or("-")
        self.check("callable.default", default_value)
        self.check(
            "callable.default.matches",
            default_value == f"default:{probe_default}",
        )
        named_h = FlextHandlers.create_from_callable(
            lambda message: f"named:{message}",
            handler_name=named_handler_name,
            handler_type=c.HandlerType.QUERY,
        )
        self.check("callable.named.handler_name", named_h.handler_name)
        self.check(
            "callable.named.name_matches",
            named_h.handler_name == named_handler_name,
        )
        self.check("callable.named.mode", named_h.mode.value)
        self.check(
            "callable.named.value_matches",
            named_h.handle(probe_named).unwrap_or("-") == f"named:{probe_named}",
        )
        mode_enum_h = FlextHandlers.create_from_callable(
            lambda message: f"enum:{message}",
            mode=c.HandlerType.EVENT,
        )
        self.check("callable.mode_enum", mode_enum_h.mode.value)
        self.check(
            "callable.mode_enum.value_matches",
            mode_enum_h.handle(probe_enum).unwrap_or("-") == f"enum:{probe_enum}",
        )
        mode_str_h = FlextHandlers.create_from_callable(
            lambda message: f"str:{message}",
            mode="query",
        )
        self.check("callable.mode_str", mode_str_h.mode.value)
        self.check(
            "callable.mode_str.value_matches",
            mode_str_h.handle(probe_str).unwrap_or("-") == f"str:{probe_str}",
        )
        config_h = FlextHandlers.create_from_callable(
            lambda message: f"cfg:{message}",
            handler_config=m.Handler(
                handler_id=cfg_handler_id,
                handler_name=cfg_handler_name,
                handler_mode=c.HandlerType.SAGA,
                handler_type=c.HandlerType.SAGA,
            ),
        )
        self.check("callable.handler_config.name", config_h.handler_name)
        self.check("callable.handler_config.mode", config_h.mode.value)
        self.check(
            "callable.handler_config.name_matches",
            config_h.handler_name == cfg_handler_name,
        )
        self.check(
            "callable.handler_config.value_matches",
            config_h.handle(probe_cfg).unwrap_or("-") == f"cfg:{probe_cfg}",
        )
        try:
            FlextHandlers.create_from_callable(lambda message: message, mode="invalid")
            invalid_mode: str = "no-error"
        except e.ValidationError as exc:
            invalid_mode = f"{type(exc).__name__}:{exc}"
        self.check("callable.invalid_mode", invalid_mode)

    def demo_discovery(self) -> None:
        """Exercise class/module discovery and handler scans."""
        self.section("discovery")
        mod_priority = self.rand_int(1, 20)
        module_text = self.rand_str(5)

        class _Service:
            @staticmethod
            @h.handler(_Message, priority=2)
            def high(_message: BaseModel) -> BaseModel:
                return _PayloadModel(text="high")

            @staticmethod
            @h.handler(_Message, priority=1, timeout=3.0, middleware=[])
            def low(_message: BaseModel) -> BaseModel:
                return _PayloadModel(text="low")

        class_scan = h.Discovery.scan_class(_Service)
        self.check("scan_class.count", len(class_scan))
        self.check("scan_class.first", class_scan[0][0] if class_scan else "none")
        self.check("has_handlers.class", h.Discovery.has_handlers(_Service))
        self.check("has_handlers.class_none", h.Discovery.has_handlers(_NoArgs))
        module = ModuleType("ex10_handlers_module")

        @h.handler(_Message, priority=mod_priority)
        def mod_handler(message: BaseModel) -> BaseModel:
            text = message.model_dump().get("text", "")
            return _PayloadModel(text=f"module:{text}")

        def plain_function(_message: BaseModel) -> BaseModel:
            return _PayloadModel(text="plain")

        setattr(module, "mod_handler", mod_handler)
        setattr(module, "plain_function", plain_function)
        module_scan = h.Discovery.scan_module(module)
        self.check("scan_module.count", len(module_scan))
        self.check("scan_module.name", module_scan[0][0] if module_scan else "none")
        wrapped_result = (
            module_scan[0][1](_Message(text=module_text)) if module_scan else "none"
        )
        self.check("scan_module.wrapped_result", module_text in str(wrapped_result))
        self.check("has_handlers.module.true", h.Discovery.has_handlers(type(module)))
        self.check(
            "has_handlers.module.false",
            not h.Discovery.has_handlers(type(ModuleType("empty"))),
        )

    def demo_handler_core(self) -> None:
        """Exercise base handler operations and validation paths."""
        self.section("handler_core")
        pattern_probe = self.rand_str(4)
        message_ok = self.rand_str(6)
        payload_text = self.rand_str(6)
        dispatch_text = self.rand_str(6)
        metric_key = self.rand_str(6)
        metric_value = self.rand_int(1, 9)
        context_name_1 = self.rand_str(6)
        context_name_2 = self.rand_str(6)
        pattern_handler = _NotImplementedPatternHandler()
        try:
            pattern_handler.handle(_Message(text=pattern_probe))
            pattern_value: str = "no-error"
        except NotImplementedError as exc:
            pattern_value = f"{type(exc).__name__}:{exc}"
        self.check("handle.not_implemented_pattern", pattern_value)
        handler: FlextHandlers[_Message, str] = _DemoHandler()
        self.check("handler.handler_name", handler.handler_name)
        self.check("handler.name_matches", bool(handler.handler_name))
        self.check("handler.mode", handler.mode.value)
        self.check(
            "validate.none.failure",
            handler.validate_message(_Message(text="")).is_failure is False,
        )
        self.check(
            "validate.ok.success",
            handler.validate_message(_Message(text=message_ok)).is_success,
        )
        self.check(
            "validate.blocked_cmd", handler.validate_message(_Message(text="bad")).error
        )
        self.check(
            "validate.blocked_qry", handler.validate_message(_Message(text="bad")).error
        )
        self.check(
            "validate.consistent",
            handler.validate_message(_Message(text=message_ok)).unwrap_or(False)
            and handler.validate_message(_Message(text=message_ok)).unwrap_or(False)
            and handler.validate_message(_Message(text=message_ok)).unwrap_or(False),
        )
        self.check("can_handle.expected", handler.can_handle(_Message))
        self.check("can_handle.derived", handler.can_handle(_DerivedMessage))
        self.check("can_handle.other", handler.can_handle(str))
        execute_result: r[str] = handler.execute(_Message(text=payload_text))
        execute_value = execute_result.unwrap_or("-")
        self.check("execute.success.value", payload_text in str(execute_value))
        self.check(
            "execute.validation_failure",
            handler.execute(_Message(text="bad")).error,
        )
        dispatch_result: r[str] = handler.dispatch_message(
            _Message(text=dispatch_text),
        )
        dispatch_value = dispatch_result.unwrap_or("-")
        self.check("dispatch.success", dispatch_text in str(dispatch_value))
        self.check(
            "dispatch.mode_mismatch",
            handler.dispatch_message(
                _Message(text="go"),
                operation=c.HANDLER_MODE_QUERY,
            ).error,
        )
        self.check(
            "dispatch.pipeline_exception",
            handler.dispatch_message(
                _Message(text="explode"),
                operation=c.DEFAULT_HANDLER_MODE,
            ).error,
        )
        self.check(
            "record_metric.ok",
            handler.record_metric(metric_key, metric_value).is_success,
        )
        context_payload_query: t.StrMapping = {
            "handler_name": context_name_1,
            "handler_mode": "query",
        }
        self.check(
            "push_context.mapping",
            handler.push_context(context_payload_query).is_success,
        )
        context_payload_event: t.StrMapping = {
            "handler_name": context_name_2,
            "handler_mode": "event",
        }
        self.check(
            "push_context.execution",
            handler.push_context(context_payload_event).is_success,
        )
        pop_ctx_1 = handler.pop_context().unwrap_or(t.ConfigMap(root={}))
        pop_ctx_1_val = pop_ctx_1.get("handler_name", "-")
        self.check("pop_context.1", pop_ctx_1_val)
        pop_ctx_2 = handler.pop_context().unwrap_or(t.ConfigMap(root={}))
        pop_ctx_2_val = pop_ctx_2.get("handler_name", "-")
        self.check("pop_context.2", pop_ctx_2_val)

    def demo_namespaces_and_mixins(self) -> None:
        """Exercise namespaces, protocol validation, and mixins."""
        self.section("namespaces_and_mixins")
        hit_key = self.rand_str(5)
        hit_value = self.rand_int(1, 9)
        ctx_name = self.rand_str(6)
        env_value = self.rand_str(6)
        obj_key = self.rand_str(5)
        obj_value = self.rand_int(1, 99)
        factory_key = self.rand_str(5)
        factory_value = self.rand_str(5)
        resource_key = self.rand_str(5)
        resource_value = self.rand_str(5)
        service_name = f"svc.{self.rand_str(6)}"
        error_message = self.rand_str(8)
        event_name = self.rand_str(6)
        aggregate_id = self.rand_str(8)
        metadata_version = (
            f"{self.rand_int(1, 9)}.{self.rand_int(0, 9)}.{self.rand_int(0, 9)}"
        )
        metadata_tag = self.rand_str(4)
        created = h.Bootstrap.create_instance(_NoArgs)
        self.check("bootstrap.create_instance", created.__class__.__name__)
        tracker = m.MetricsTracker()
        self.check(
            "cqrs.record_metric",
            tracker.record_metric(hit_key, hit_value).is_success,
        )
        metrics_map = tracker.get_metrics().unwrap_or(t.ConfigMap(root={}))
        metrics_val = metrics_map.get(hit_key, -1)
        self.check("cqrs.get_metrics", metrics_val)
        stack = m.ContextStack()
        self.check(
            "cqrs.push_context.mapping",
            stack.push_context({
                "handler_name": ctx_name,
                "handler_mode": "command",
            }).is_success,
        )
        current_context = stack.current_context()
        self.check(
            "cqrs.current_context",
            getattr(current_context, "handler_name", "-"),
        )
        self.check(
            "cqrs.pop_context",
            stack
            .pop_context()
            .unwrap_or({"handler_name": "-"})
            .get("handler_name", "-"),
        )
        di = h.DependencyIntegration
        di_container = di.create_container(config=t.ConfigMap(root={"env": env_value}))
        self.check("di.bind_configuration_exists", hasattr(di_container, "config"))
        self.check(
            "di.register_object",
            di.register_object(di_container, obj_key, obj_value)() == obj_value,
        )
        self.check(
            "di.register_factory.cached",
            di.register_factory(
                di_container,
                factory_key,
                lambda: factory_value,
                cache=True,
            )()
            == factory_value,
        )
        resource_provider = di.register_resource(
            di_container,
            resource_key,
            lambda: t.ConfigMap(root={resource_key: resource_value}),
        )
        self.check(
            "di.register_resource",
            resource_provider().get(resource_key, "-") == resource_value,
        )
        try:
            di.register_object(di_container, obj_key, self.rand_int(100, 200))
            duplicate_error: str = "no-error"
        except ValueError as exc:
            duplicate_error = f"{type(exc).__name__}:{exc}"
        self.check("di.duplicate_error", duplicate_error)
        bridge, services_mod, resources_mod = di.create_layered_bridge(
            t.ConfigMap(root={self.rand_str(2): self.rand_str(2)}),
        )
        self.check("di.layered.bridge", bridge.__class__.__name__)
        self.check("di.layered.services", services_mod.__class__.__name__)
        self.check("di.layered.resources", resources_mod.__class__.__name__)
        di.wire(di_container, modules=[])
        self.check("di.wire.noop", True)
        h.Integration.track_service_resolution(service_name, resolved=True)
        h.Integration.track_service_resolution(
            service_name,
            resolved=False,
            error_message=error_message,
        )
        h.Integration.track_domain_event(
            event_name,
            aggregate_id=aggregate_id,
            event_data=t.ConfigMap(root={self.rand_str(3): self.rand_int(1, 9)}),
        )
        h.Integration.setup_service_infrastructure(
            service_name=service_name,
            service_version=f"{self.rand_int(1, 9)}.{self.rand_int(0, 9)}.{self.rand_int(0, 9)}",
            enable_context_correlation=True,
        )
        self.check("integration.calls", True)
        meta = m.Metadata(version=metadata_version, attributes={"tag": metadata_tag})
        self.check("metadata.version", meta.version)
        self.check("metadata.attributes", meta.attributes)
        protocol_handler = Ex10ProtocolHandler()
        self.check(
            "protocol.is_handler.true",
            u.is_handler(protocol_handler),
        )
        self.check(
            "protocol.is_handler.false",
            not u.is_handler(t.ConfigMap(root={})),
        )

    def demo_runtime_result_and_utilities(self) -> None:
        """Exercise RuntimeResult API and utility helpers."""
        self.section("runtime_result_and_utilities")
        rr_value = self.rand_int(1, 20)
        rr_delta = self.rand_int(1, 10)
        rr_multiplier = self.rand_int(2, 5)
        rr_error = self.rand_str(6)
        rr_error_code = self.rand_str(4)
        mixin_value = self.rand_str(4)
        mixin_error = self.rand_str(5)
        mixin_error_code = self.rand_str(4)
        ensured_raw = self.rand_int(1, 20)
        ensured_result = self.rand_int(1, 20)
        entity_id = self.rand_str(8)
        other_entity_id = self.rand_str(8)
        scalar_value = self.rand_str(3)
        model_text = self.rand_str(5)
        source_value = self.rand_str(5)
        valid_identifier = f"{self.rand_str(1)}_{self.rand_str(4)}"
        invalid_identifier = f"{self.rand_int(1, 9)}{self.rand_str(4)}"
        attr_fallback = self.rand_str(3)
        dict_key = self.rand_str(2)
        dict_value = self.rand_int(1, 9)
        rr_ok = r[int].ok(rr_value)
        rr_fail = r[int].fail(rr_error, error_code=rr_error_code)
        self.check("rr.is_success", rr_ok.is_success)
        self.check("rr.is_failure", rr_fail.is_failure)
        self.check("rr.unwrap_or", rr_fail.unwrap_or(rr_delta) == rr_delta)
        self.check(
            "rr.map",
            rr_ok.map(lambda n: n + rr_delta).unwrap_or(-1) == rr_value + rr_delta,
        )
        self.check(
            "rr.flat_map",
            rr_ok.flat_map(
                lambda n: r[int].ok(n * rr_multiplier),
            ).unwrap_or(-1)
            == rr_value * rr_multiplier,
        )
        self.check(
            "rr.and_then",
            rr_ok.flat_map(lambda n: r[int].ok(n - rr_delta)).unwrap_or(
                -1,
            )
            == rr_value - rr_delta,
        )
        self.check(
            "rr.alt",
            rr_fail.map_error(lambda err: f"x:{err}").error == f"x:{rr_error}",
        )
        self.check(
            "rr.lash",
            rr_fail.lash(lambda _err: r[int].ok(rr_value)).unwrap_or(-1) == rr_value,
        )
        self.check(
            "rr.recover",
            rr_fail.recover(lambda _err: rr_delta).unwrap_or(-1) == rr_delta,
        )
        self.check(
            "rr.fold",
            rr_ok.fold(lambda err: err, lambda n: f"ok:{n}") == f"ok:{rr_value}",
        )
        self.check("mixin.ok", r[str].ok(mixin_value).unwrap_or("-") == mixin_value)
        self.check(
            "mixin.fail",
            r[int].fail(mixin_error, error_code=mixin_error_code).error_code
            == mixin_error_code,
        )
        self.check(
            "mixin.ensure_result.value",
            r[int].ok(ensured_raw).unwrap_or(-1) == ensured_raw,
        )
        self.check(
            "mixin.ensure_result.result",
            r[int].ok(ensured_result).unwrap_or(-1) == ensured_result,
        )
        self.check("mixin.to_dict", t.ConfigMap(root={dict_key: dict_value}).root)
        generated_a = h.generate_id()
        generated_b = h.generate_id()
        self.check(
            "runtime.generate_id.length",
            len(generated_a) == len(generated_b) == 36,
        )
        self.check("runtime.generate_id.unique", generated_a != generated_b)
        self.check(
            "runtime.generate_prefixed_id.default",
            h.generate_prefixed_id("cmd").startswith("cmd_"),
        )
        self.check(
            "runtime.generate_prefixed_id.length",
            len(h.generate_prefixed_id("q", length=8).split("_")[1]) == 8,
        )
        self.check(
            "runtime.generate_datetime_utc",
            h.generate_datetime_utc().tzinfo is not None,
        )
        e1 = _Entity(unique_id=entity_id)
        e2 = _Entity(unique_id=entity_id)
        e3 = _Entity(unique_id=other_entity_id)
        self.check("runtime.compare_entities.true", h.compare_entities_by_id(e1, e2))
        self.check("runtime.compare_entities.false", h.compare_entities_by_id(e1, e3))
        self.check("runtime.hash_entity", h.hash_entity_by_id(e1) != 0)
        self.check(
            "runtime.compare_value_objects.scalar",
            h.compare_value_objects_by_value(scalar_value, scalar_value),
        )
        self.check(
            "runtime.compare_value_objects.model",
            h.compare_value_objects_by_value(
                _Message(text=model_text),
                _Message(text=model_text),
            ),
        )
        self.check(
            "runtime.hash_value_object",
            h.hash_value_object_by_value(_Message(text=model_text)) != 0,
        )
        trace_context = h.ensure_trace_context(
            {"source": source_value},
            include_correlation_id=True,
            include_timestamp=True,
        )
        self.check(
            "runtime.ensure_trace_context.source",
            trace_context.get("source", "-"),
        )
        self.check(
            "runtime.ensure_trace_context.keys",
            [
                "trace_id" in trace_context,
                "span_id" in trace_context,
                "correlation_id" in trace_context,
                "timestamp" in trace_context,
            ],
        )
        self.check("runtime.get_log_level", u.get_log_level_from_config() >= 0)
        self.check("runtime.is_dict_like.true", u.is_dict_like({"a": 1}))
        self.check("runtime.is_dict_like.false", u.is_dict_like([1, 2]))
        self.check("runtime.is_list_like.true", u.is_list_like([1, 2]))
        self.check("runtime.is_list_like.false", u.is_list_like("ab"))
        self.check(
            "runtime.is_valid_identifier.true",
            u.is_valid_identifier(valid_identifier),
        )
        self.check(
            "runtime.is_valid_identifier.false",
            u.is_valid_identifier(invalid_identifier),
        )
        self.check(
            "runtime.safe_get_attribute",
            u.safe_get_attribute(e1, "unique_id", attr_fallback) == entity_id,
        )
        self.check(
            "runtime.extract_generic_args",
            len(u.extract_generic_args(t.IntMapping)) >= 1,
        )
        self.check("runtime.is_sequence_type.true", h.is_sequence_type(Sequence[int]))
        self.check(
            "runtime.is_sequence_type.false",
            h.is_sequence_type(t.IntMapping),
        )
        self.check(
            "runtime.normalize_general",
            h.normalize_to_container({dict_key: dict_value}),
        )
        self.check("runtime.normalize_metadata", h.normalize_to_metadata({"k": [1, 2]}))

    @override
    def exercise(self) -> None:
        """Run all FlextHandlers example sections."""
        self.demo_handler_core()
        self.demo_create_from_callable()
        self.demo_discovery()
        self.demo_namespaces_and_mixins()
        self.demo_runtime_result_and_utilities()


if __name__ == "__main__":
    Ex10FlextHandlers(__file__).run()
