"""Golden-file example for FlextRegistry public APIs."""

from __future__ import annotations

from typing import override

from flext_core import FlextDispatcher, FlextHandlers, FlextRegistry, c, m, r, t

from .shared import Examples


class _CommandA(m.Command):
    value: str


class _CommandB(m.Command):
    amount: int


class _ProtocolHandler:
    def __init__(self, label: str, message_type: type[t.ContainerValue]) -> None:
        self._label = label
        self.message_type = message_type

    def can_handle(self, message_type: type) -> bool:
        return message_type is self.message_type

    def handle(self, message: t.ContainerValue) -> r[t.ContainerValue]:
        value = ""
        if hasattr(message, "value"):
            value = str(getattr(message, "value"))
        if hasattr(message, "amount"):
            value = str(getattr(message, "amount"))
        return r[t.ContainerValue].ok(f"{self._label}:{value}")

    def _protocol_name(self) -> str:
        return f"example-protocol-handler::{self._label}"


@FlextHandlers.handler(_CommandA, priority=3)
def _discovered_handler(_message: t.ContainerValue) -> t.ContainerValue:
    return "decorated"


class Ex12FlextRegistry(Examples):
    """Exercise FlextRegistry public API."""

    @override
    def exercise(self) -> None:
        """Run all FlextRegistry example sections."""
        registry, dispatcher = self._exercise_create_and_service_methods()
        self._exercise_summary_and_mixins(registry)
        handler_a, handler_b = self._exercise_registration_and_dispatch(
            registry, dispatcher
        )
        self._exercise_bindings_and_plugin_apis(registry, handler_a, handler_b)
        self._exercise_register_method_and_tracking(registry)

    def _exercise_bindings_and_plugin_apis(
        self,
        registry: FlextRegistry,
        handler_a: _ProtocolHandler,
        handler_b: _ProtocolHandler,
    ) -> None:
        self.section("bindings_and_plugins")
        custom_binding_name = self.rand_str(8)
        plugin_ns = f"svc.{self.rand_str(6)}"
        plugin_name_a = self.rand_str(6)
        plugin_value_a = self.rand_str(8)
        plugin_name_b = self.rand_str(6)
        plugin_value_b = self.rand_str(8)
        plugin_bad_name = self.rand_str(6)
        plugin_bad_value = self.rand_str(3)
        plugin_missing_name = self.rand_str(6)
        plugin_unreg_missing_name = self.rand_str(6)
        class_ns = f"cls.{self.rand_str(6)}"
        class_plugin_name = self.rand_str(6)
        class_plugin_value = self.rand_str(8)
        class_missing_name = self.rand_str(6)
        class_unreg_missing_name = self.rand_str(6)
        invalid_error = self.rand_str(7)
        boom_message = self.rand_str(7)
        bindings_result = registry.register_bindings({
            _CommandA: handler_a,
            custom_binding_name: handler_b,
        })
        self.check("register_bindings.success", bindings_result.is_success)
        self.check(
            "register_bindings.registered_len",
            len(bindings_result.value.registered) if bindings_result.is_success else -1,
        )
        plugin_ok = registry.register_plugin(plugin_ns, plugin_name_a, plugin_value_a)
        plugin_dup = registry.register_plugin(plugin_ns, plugin_name_a, plugin_value_a)
        plugin_empty = registry.register_plugin(plugin_ns, "", plugin_value_a)
        plugin_validated = registry.register_plugin(
            plugin_ns,
            plugin_name_b,
            plugin_value_b,
            validate=lambda pval: r[bool].ok(bool(pval)),
        )
        plugin_validate_fail = registry.register_plugin(
            plugin_ns,
            plugin_bad_name,
            plugin_bad_value,
            validate=lambda _pval: r[bool].fail(invalid_error),
        )
        plugin_validate_exc = registry.register_plugin(
            plugin_ns,
            self.rand_str(6),
            self.rand_str(3),
            validate=lambda _pval: (_ for _ in ()).throw(RuntimeError(boom_message)),
        )
        self.check("register_plugin.ok", plugin_ok.is_success)
        self.check("register_plugin.dup", plugin_dup.is_success)
        self.check("register_plugin.empty_name", plugin_empty.is_failure)
        self.check("register_plugin.validated", plugin_validated.is_success)
        self.check("register_plugin.validate_fail", plugin_validate_fail.is_failure)
        self.check("register_plugin.validate_exc", plugin_validate_exc.is_failure)
        plugin_get_ok = registry.get_plugin(plugin_ns, plugin_name_a)
        plugin_get_missing = registry.get_plugin(plugin_ns, plugin_missing_name)
        plugin_list = registry.list_plugins(plugin_ns)
        plugin_unreg_ok = registry.unregister_plugin(plugin_ns, plugin_name_a)
        plugin_unreg_missing = registry.unregister_plugin(
            plugin_ns, plugin_unreg_missing_name
        )
        self.check("get_plugin.ok", plugin_get_ok.unwrap_or("") == plugin_value_a)
        self.check("get_plugin.missing", plugin_get_missing.is_failure)
        self.check("list_plugins.transports", sorted(plugin_list.unwrap_or([])))
        self.check("unregister_plugin.ok", plugin_unreg_ok.is_success)
        self.check("unregister_plugin.missing", plugin_unreg_missing.is_failure)
        class_ok = registry.register_plugin(
            class_ns, class_plugin_name, class_plugin_value, scope="class"
        )
        class_dup = registry.register_plugin(
            class_ns, class_plugin_name, class_plugin_value, scope="class"
        )
        class_empty = registry.register_plugin(
            class_ns, "", class_plugin_value, scope="class"
        )
        class_get_ok = registry.get_plugin(class_ns, class_plugin_name, scope="class")
        class_get_missing = registry.get_plugin(
            class_ns, class_missing_name, scope="class"
        )
        class_list = registry.list_plugins(class_ns, scope="class")
        class_unreg_ok = registry.unregister_plugin(
            class_ns, class_plugin_name, scope="class"
        )
        class_unreg_missing = registry.unregister_plugin(
            class_ns, class_unreg_missing_name, scope="class"
        )
        self.check("register_class_plugin.ok", class_ok.is_success)
        self.check("register_class_plugin.dup", class_dup.is_success)
        self.check("register_class_plugin.empty_name", class_empty.is_failure)
        self.check(
            "get_class_plugin.ok", class_get_ok.unwrap_or("") == class_plugin_value
        )
        self.check("get_class_plugin.missing", class_get_missing.is_failure)
        self.check("list_class_plugins.auth", class_list.unwrap_or([]))
        self.check("unregister_class_plugin.ok", class_unreg_ok.is_success)
        self.check("unregister_class_plugin.missing", class_unreg_missing.is_failure)

    def _exercise_create_and_service_methods(
        self,
    ) -> tuple[FlextRegistry, FlextDispatcher]:
        self.section("create_and_service_methods")
        discovered_value = self.rand_str(4)
        dispatcher = FlextDispatcher()
        reg_default = FlextRegistry.create()
        reg_explicit = FlextRegistry.create(dispatcher=None)
        reg_auto_false = FlextRegistry.create(auto_discover_handlers=False)
        reg_auto_true = FlextRegistry.create(auto_discover_handlers=True)
        self.check("create.default.type", type(reg_default).__name__)
        self.check("create.explicit.type", type(reg_explicit).__name__)
        self.check("create.auto_false.type", type(reg_auto_false).__name__)
        self.check("create.auto_true.type", type(reg_auto_true).__name__)
        self.check(
            "decorated_handler.type",
            type(_discovered_handler(_CommandA(value=discovered_value))).__name__,
        )
        self.check("execute.success", reg_explicit.execute().is_success)
        self.check(
            "validate_business_rules.success",
            reg_explicit.validate_business_rules().is_success,
        )
        self.check("is_valid", reg_explicit.is_valid())
        self.check("service_info", reg_explicit.get_service_info())
        self.check("result_property", reg_explicit.result is not None)
        self.check("runtime.type", type(reg_explicit.runtime).__name__)
        self.check("context.type", type(reg_explicit.context).__name__)
        self.check("config.type", type(reg_explicit.config).__name__)
        self.check("container.type", type(reg_explicit.container).__name__)
        return (reg_explicit, dispatcher)

    def _exercise_register_method_and_tracking(self, registry: FlextRegistry) -> None:
        self.section("register_method_and_tracking")
        team_value = self.rand_str(5)
        version_value = str(self.rand_int(1, 9))
        owner_value = self.rand_str(7)
        svc_plain_name = f"svc.{self.rand_str(6)}"
        svc_plain_value = self.rand_str(6)
        svc_dict_name = f"svc.{self.rand_str(6)}"
        svc_dict_value = self.rand_str(7)
        svc_meta_name = f"svc.{self.rand_str(6)}"
        callable_value = self.rand_str(10)
        bad_value = self.rand_str(4)
        track_name = self.rand_str(8)
        meta_dict = m.ConfigMap(root={"team": team_value, "version": version_value})
        meta_model = m.Metadata(attributes={"owner": owner_value, "enabled": True})
        reg_plain = registry.register(svc_plain_name, svc_plain_value)
        reg_meta_dict = registry.register(
            svc_dict_name, svc_dict_value, metadata=meta_dict
        )
        reg_meta_model = registry.register(
            svc_meta_name, lambda: callable_value, metadata=meta_model
        )
        reg_bad = registry.register("", bad_value)
        self.check("register.service.plain", reg_plain.is_success)
        self.check("register.service.meta_dict", reg_meta_dict.is_success)
        self.check("register.service.meta_model", reg_meta_model.is_success)
        self.check("register.service.bad", reg_bad.is_failure)
        with registry.track(track_name) as metrics:
            self.check("track.has_operation_count", "operation_count" in metrics)
            self.check("track.operation_count", metrics.get("operation_count", -1))

    def _exercise_registration_and_dispatch(
        self, registry: FlextRegistry, dispatcher: FlextDispatcher
    ) -> tuple[_ProtocolHandler, _ProtocolHandler]:
        self.section("registration_and_dispatch")
        label_a = self.rand_str(3)
        label_b = self.rand_str(3)
        callable_prefix = self.rand_str(3)
        callable_name = self.rand_str(10)
        cmd_a_value = self.rand_str(6)
        cmd_b_value = self.rand_int(1, 100)
        handler_a = _ProtocolHandler(label_a, _CommandA)
        handler_b = _ProtocolHandler(label_b, _CommandB)
        handler_mode = FlextHandlers.create_from_callable(
            lambda msg: f"{callable_prefix}:{msg}",
            handler_name=callable_name,
            mode=c.Cqrs.HandlerType.COMMAND,
        )
        reg_one = registry.register_handler(handler_a)
        reg_dup = registry.register_handler(handler_a)
        reg_two = registry.register_handler(handler_b)
        reg_mode = registry.register_handler(handler_a)
        self.check("register_handler.a.success", reg_one.is_success)
        self.check(
            "register_handler.a.id",
            reg_one.value.registration_id if reg_one.is_success else "",
        )
        self.check("register_handler.duplicate.success", reg_dup.is_success)
        self.check("register_handler.b.success", reg_two.is_success)
        self.check("register_handler.mode.success", reg_mode.is_success)
        self.check("create_from_callable.type", type(handler_mode).__name__)
        batch = registry.register_handlers([handler_a, handler_b, handler_a])
        self.check("register_handlers.success", batch.is_success)
        self.check(
            "register_handlers.registered_len",
            len(batch.value.registered) if batch.is_success else -1,
        )
        self.check(
            "register_handlers.errors_len",
            len(batch.value.errors) if batch.is_success else -1,
        )
        cmd_a = _CommandA(value=cmd_a_value)
        dispatch_a = dispatcher.dispatch(cmd_a)
        self.check("dispatch.a.success", dispatch_a.is_success)
        self.check(
            "dispatch.a.value", dispatch_a.unwrap_or("") == f"{label_a}:{cmd_a_value}"
        )
        cmd_b = _CommandB(amount=cmd_b_value)
        dispatch_b = dispatcher.dispatch(cmd_b)
        self.check("dispatch.b.success", dispatch_b.is_success)
        self.check(
            "dispatch.b.value", dispatch_b.unwrap_or("") == f"{label_b}:{cmd_b_value}"
        )
        return (handler_a, handler_b)

    def _exercise_summary_and_mixins(self, registry: FlextRegistry) -> None:
        self.section("summary_and_mixins")
        summary_error = self.rand_str(5)
        ok_value = self.rand_str(6)
        fail_message = self.rand_str(7)
        fail_code = self.rand_str(5)
        fail_data_key = self.rand_str(3)
        fail_data_value = self.rand_str(4)
        ensured_raw = self.rand_int(1, 200)
        ensured_existing = self.rand_int(1, 200)
        map_key_a = self.rand_str(3)
        map_key_b = self.rand_str(3)
        map_val_a = self.rand_int(1, 9)
        map_val_b = self.rand_str(3)
        handler_name = self.rand_str(6)
        handler_id = self.rand_str(8)
        prefix = f"reg.{self.rand_str(4)}"
        summary_ok = FlextRegistry.Summary()
        summary_fail = FlextRegistry.Summary(errors=[summary_error])
        ok_success_attr = summary_ok.is_success
        summary_ok_success = (
            ok_success_attr() if callable(ok_success_attr) else ok_success_attr
        )
        ok_failure_attr = summary_ok.is_failure
        summary_ok_failure = (
            ok_failure_attr() if callable(ok_failure_attr) else ok_failure_attr
        )
        fail_success_attr = summary_fail.is_success
        summary_fail_success = (
            fail_success_attr() if callable(fail_success_attr) else fail_success_attr
        )
        fail_failure_attr = summary_fail.is_failure
        summary_fail_failure = (
            fail_failure_attr() if callable(fail_failure_attr) else fail_failure_attr
        )
        self.check("summary.ok.success", summary_ok_success)
        self.check("summary.ok.failure", summary_ok_failure)
        self.check("summary.fail.success", summary_fail_success)
        self.check("summary.fail.failure", summary_fail_failure)
        ok_result = registry.ok(ok_value)
        fail_result = registry.fail(
            fail_message,
            error_code=fail_code,
            error_data=m.ConfigMap(root={fail_data_key: fail_data_value}),
        )
        self.check("mixin.ok.unwrap_or", ok_result.unwrap_or("") == ok_value)
        self.check("mixin.fail.error", fail_result.error == fail_message)
        self.check("mixin.fail.error_code", fail_result.error_code == fail_code)
        self.check(
            "ensure_result.raw",
            registry.ensure_result(ensured_raw).unwrap_or(0) == ensured_raw,
        )
        self.check(
            "ensure_result.existing",
            registry.ensure_result(r[int].ok(ensured_existing)).unwrap_or(0)
            == ensured_existing,
        )
        self.check("to_dict.none", registry.to_dict(None))
        self.check(
            "to_dict.mapping",
            registry.to_dict({map_key_a: map_val_a, map_key_b: map_val_b}),
        )
        self.check(
            "to_dict.basemodel",
            registry.to_dict(
                m.Handler(handler_name=handler_name, handler_id=handler_id)
            ),
        )
        self.check("generate_id.len", len(registry.generate_id()))
        self.check(
            "generate_prefixed_id",
            registry.generate_prefixed_id(prefix, 6).startswith(f"{prefix}_"),
        )
        self.check(
            "generate_datetime_utc.type",
            type(registry.generate_datetime_utc()).__name__,
        )


if __name__ == "__main__":
    Ex12FlextRegistry(__file__).run()
