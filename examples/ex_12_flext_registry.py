"""Golden-file example for the registry DSL public APIs."""

from __future__ import annotations

from typing import override

from examples import (
    ExamplesFlextCoreShared,
    c,
    m,
    p,
    r,
    t,
    u,
)
from examples._models.ex12 import Ex12CommandA, Ex12CommandB
from flext_core import h


class _ProtocolHandler:
    message_type: type[m.Command]

    def __init__(self, label: str, message_type: type[m.Command]) -> None:
        self._label = label
        self.message_type = message_type

    def can_handle(self, message_type: type) -> bool:
        return message_type is self.message_type

    def handle(self, message: p.Routable) -> p.Result[t.Scalar]:
        value = ""
        if isinstance(message, Ex12CommandA):
            value = str(message.value)
        elif isinstance(message, Ex12CommandB):
            value = str(message.amount)
        return r[t.Scalar].ok(f"{self._label}:{value}")

    def __call__(self, message: p.Routable) -> p.Result[t.Scalar]:
        """Callable adapter for registry handler protocols expecting callables."""
        return self.handle(message)


def _as_registry_handler(
    handler: _ProtocolHandler,
) -> t.DispatchableHandler:
    """Adapt protocol handlers to the registry callable contract."""

    def call(message: p.Routable) -> t.JsonPayload:
        return u.normalize_to_container(handler.handle(message).unwrap_or(""))

    handler_name = handler.message_type.__name__
    setattr(call, "__name__", handler_name)
    setattr(call, "handler_id", handler_name)
    setattr(call, "message_type", handler.message_type)
    return call


@h.handler(Ex12CommandA, priority=3)
def _discovered_handler(message: m.Command) -> m.Command:
    if isinstance(message, Ex12CommandA):
        return Ex12CommandA(value=message.value)
    return Ex12CommandA(value="decorated")


class Ex12RegistryDsl(ExamplesFlextCoreShared):
    """Exercise the canonical registry DSL public API."""

    @override
    def exercise(self) -> None:
        """Run all registry DSL example sections."""
        registry, dispatcher = self._exercise_create_and_service_methods()
        self._exercise_summary_and_mixins()
        handler_a, handler_b = self._exercise_registration_and_dispatch(
            registry,
            dispatcher,
        )
        self._exercise_bindings_and_plugin_apis(registry, handler_a, handler_b)
        self._exercise_register_method_and_tracking(registry)

    def _exercise_bindings_and_plugin_apis(
        self,
        registry: p.Registry,
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
            Ex12CommandA: _as_registry_handler(handler_a),
            custom_binding_name: _as_registry_handler(handler_b),
        })
        self.check("register_bindings.success", bindings_result.success)
        self.check(
            "register_bindings.registered_len",
            len(bindings_result.value.registered) if bindings_result.success else -1,
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
        self.check("register_plugin.ok", plugin_ok.success)
        self.check("register_plugin.dup", plugin_dup.success)
        self.check("register_plugin.empty_name", plugin_empty.failure)
        self.check("register_plugin.validated", plugin_validated.success)
        self.check("register_plugin.validate_fail", plugin_validate_fail.failure)
        self.check("register_plugin.validate_exc", plugin_validate_exc.failure)
        plugin_fetch_ok = registry.fetch_plugin(plugin_ns, plugin_name_a)
        plugin_fetch_missing = registry.fetch_plugin(plugin_ns, plugin_missing_name)
        plugin_list = registry.list_plugins(plugin_ns)
        plugin_unreg_ok = registry.unregister_plugin(plugin_ns, plugin_name_a)
        plugin_unreg_missing = registry.unregister_plugin(
            plugin_ns,
            plugin_unreg_missing_name,
        )
        self.check("fetch_plugin.ok", plugin_fetch_ok.value == plugin_value_a)
        self.check("fetch_plugin.missing", plugin_fetch_missing.failure)
        self.check(
            "list_plugins.transports",
            sorted(plugin_list.value) if plugin_list.success else [],
        )
        self.check("unregister_plugin.ok", plugin_unreg_ok.success)
        self.check("unregister_plugin.missing", plugin_unreg_missing.failure)
        class_ok = registry.register_plugin(
            class_ns,
            class_plugin_name,
            class_plugin_value,
            scope=c.RegistrationScope.CLASS,
        )
        class_dup = registry.register_plugin(
            class_ns,
            class_plugin_name,
            class_plugin_value,
            scope=c.RegistrationScope.CLASS,
        )
        class_empty = registry.register_plugin(
            class_ns,
            "",
            class_plugin_value,
            scope=c.RegistrationScope.CLASS,
        )
        class_fetch_ok = registry.fetch_plugin(
            class_ns,
            class_plugin_name,
            scope=c.RegistrationScope.CLASS,
        )
        class_fetch_missing = registry.fetch_plugin(
            class_ns,
            class_missing_name,
            scope=c.RegistrationScope.CLASS,
        )
        class_list = registry.list_plugins(class_ns, scope=c.RegistrationScope.CLASS)
        class_unreg_ok = registry.unregister_plugin(
            class_ns,
            class_plugin_name,
            scope=c.RegistrationScope.CLASS,
        )
        class_unreg_missing = registry.unregister_plugin(
            class_ns,
            class_unreg_missing_name,
            scope=c.RegistrationScope.CLASS,
        )
        self.check("register_class_plugin.ok", class_ok.success)
        self.check("register_class_plugin.dup", class_dup.success)
        self.check("register_class_plugin.empty_name", class_empty.failure)
        self.check(
            "fetch_class_plugin.ok",
            class_fetch_ok.value == class_plugin_value,
        )
        self.check("fetch_class_plugin.missing", class_fetch_missing.failure)
        self.check(
            "list_class_plugins.auth",
            class_list.value if class_list.success else [],
        )
        self.check("unregister_class_plugin.ok", class_unreg_ok.success)
        self.check("unregister_class_plugin.missing", class_unreg_missing.failure)

    def _exercise_create_and_service_methods(
        self,
    ) -> t.Pair[p.Registry, p.Dispatcher]:
        self.section("create_and_service_methods")
        discovered_value = self.rand_str(4)
        dispatcher = u.build_dispatcher()
        registry = u.build_registry(dispatcher=dispatcher)
        reg_default = u.build_registry()
        reg_explicit = u.build_registry(dispatcher=None)
        reg_auto_false = u.build_registry(auto_discover_handlers=False)
        reg_auto_true = u.build_registry(auto_discover_handlers=True)
        self.check("dispatcher.protocol", isinstance(dispatcher, p.Dispatcher))
        self.check("create.shared.protocol", isinstance(registry, p.Registry))
        self.check("create.default.protocol", isinstance(reg_default, p.Registry))
        self.check("create.explicit.protocol", isinstance(reg_explicit, p.Registry))
        self.check("create.auto_false.protocol", isinstance(reg_auto_false, p.Registry))
        self.check("create.auto_true.protocol", isinstance(reg_auto_true, p.Registry))
        self.check(
            "decorated_handler.type",
            type(_discovered_handler(Ex12CommandA(value=discovered_value))).__name__,
        )
        self.check("execute.success", registry.execute().success)
        return (registry, dispatcher)

    def _exercise_register_method_and_tracking(self, registry: p.Registry) -> None:
        self.section("register_method_and_tracking")
        team_value = self.rand_str(5)
        version_value = str(self.rand_int(1, 9))
        owner_value = self.rand_str(7)
        svc_plain_name = f"svc.{self.rand_str(6)}"
        svc_plain_value = self.rand_str(6)
        svc_dict_name = f"svc.{self.rand_str(6)}"
        svc_meta_name = f"svc.{self.rand_str(6)}"
        bad_value = self.rand_str(4)
        meta_model = m.Metadata(attributes={"owner": owner_value, "enabled": True})
        reg_plain = registry.register(svc_plain_name, svc_plain_value)
        reg_meta_dict = registry.register(
            svc_dict_name,
            m.Metadata(attributes={"team": team_value, "version": version_value}),
        )
        reg_meta_model = registry.register(
            svc_meta_name,
            meta_model,
        )
        reg_bad = registry.register("", bad_value)
        self.check("register.service.plain", reg_plain.success)
        self.check("register.service.meta_dict", reg_meta_dict.success)
        self.check("register.service.meta_model", reg_meta_model.success)
        self.check("register.service.bad", reg_bad.failure)

    def _exercise_registration_and_dispatch(
        self,
        registry: p.Registry,
        dispatcher: p.Dispatcher,
    ) -> t.Pair[_ProtocolHandler, _ProtocolHandler]:
        self.section("registration_and_dispatch")
        label_a = self.rand_str(3)
        label_b = self.rand_str(3)
        callable_prefix = self.rand_str(3)
        callable_name = self.rand_str(10)
        cmd_a_value = self.rand_str(6)
        cmd_b_value = self.rand_int(1, 100)
        handler_a = _ProtocolHandler(label_a, Ex12CommandA)
        handler_b = _ProtocolHandler(label_b, Ex12CommandB)
        handler_mode = h.create_from_callable(
            lambda msg: f"{callable_prefix}:{msg}",
            handler_name=callable_name,
            mode=c.HandlerType.COMMAND,
        )
        reg_one = registry.register_handler(_as_registry_handler(handler_a))
        reg_dup = registry.register_handler(_as_registry_handler(handler_a))
        reg_two = registry.register_handler(_as_registry_handler(handler_b))
        reg_mode = registry.register_handler(_as_registry_handler(handler_a))
        self.check("register_handler.a.success", reg_one.success)
        self.check(
            "register_handler.a.id_present",
            bool(reg_one.value.registration_id) if reg_one.success else False,
        )
        self.check("register_handler.duplicate.success", reg_dup.success)
        self.check("register_handler.b.success", reg_two.success)
        self.check("register_handler.mode.success", reg_mode.success)
        handler_mode_probe = handler_mode.handle(callable_prefix)
        self.check("create_from_callable.handle_success", handler_mode_probe.success)
        batch = registry.register_handlers([
            _as_registry_handler(handler_a),
            _as_registry_handler(handler_b),
            _as_registry_handler(handler_a),
        ])
        self.check("register_handlers.success", batch.success)
        self.check(
            "register_handlers.registered_len",
            len(batch.value.registered) if batch.success else -1,
        )
        self.check(
            "register_handlers.errors_len",
            len(batch.value.errors) if batch.success else -1,
        )
        cmd_a = Ex12CommandA(value=cmd_a_value)
        dispatch_a = dispatcher.dispatch(cmd_a)
        self.check("dispatch.a.success", dispatch_a.success)
        self.check(
            "dispatch.a.value",
            dispatch_a.value == f"{label_a}:{cmd_a_value}",
        )
        cmd_b = Ex12CommandB(amount=cmd_b_value)
        dispatch_b = dispatcher.dispatch(cmd_b)
        self.check("dispatch.b.success", dispatch_b.success)
        self.check(
            "dispatch.b.value",
            dispatch_b.value == f"{label_b}:{cmd_b_value}",
        )
        return (handler_a, handler_b)

    def _exercise_summary_and_mixins(self) -> None:
        self.section("summary_and_mixins")
        summary_error = self.rand_str(5)
        ok_value = self.rand_str(6)
        fail_message = self.rand_str(7)
        fail_code = self.rand_str(5)
        ensured_raw = self.rand_int(1, 200)
        ensured_existing = self.rand_int(1, 200)
        map_key_a = self.rand_str(3)
        map_key_b = self.rand_str(3)
        map_val_a = self.rand_int(1, 9)
        map_val_b = self.rand_str(3)
        handler_name = self.rand_str(6)
        handler_id = self.rand_str(8)
        prefix = f"reg.{self.rand_str(4)}"
        summary_ok = m.RegistrySummary()
        summary_fail = m.RegistrySummary(errors=[summary_error])
        ok_success_attr = summary_ok.success
        summary_ok_success = (
            ok_success_attr() if callable(ok_success_attr) else ok_success_attr
        )
        ok_failure_attr = summary_ok.failure
        summary_ok_failure = (
            ok_failure_attr() if callable(ok_failure_attr) else ok_failure_attr
        )
        fail_success_attr = summary_fail.success
        summary_fail_success = (
            fail_success_attr() if callable(fail_success_attr) else fail_success_attr
        )
        fail_failure_attr = summary_fail.failure
        summary_fail_failure = (
            fail_failure_attr() if callable(fail_failure_attr) else fail_failure_attr
        )
        self.check("summary.ok.success", summary_ok_success)
        self.check("summary.ok.failure", summary_ok_failure)
        self.check("summary.fail.success", summary_fail_success)
        self.check("summary.fail.failure", summary_fail_failure)
        ok_result = r[str].ok(ok_value)
        fail_result = r[str].fail(fail_message, error_code=fail_code)
        self.check("mixin.ok.unwrap_or", ok_result.unwrap_or("") == ok_value)
        self.check("mixin.fail.error", fail_result.error == fail_message)
        self.check("mixin.fail.error_code", fail_result.error_code == fail_code)
        self.check(
            "ensure_result.raw",
            r[int].ok(ensured_raw).unwrap_or(0) == ensured_raw,
        )
        self.check(
            "ensure_result.existing",
            r[int].ok(ensured_existing).unwrap_or(0) == ensured_existing,
        )
        self.check("to_dict.none", m.ConfigMap(root={}))
        self.check(
            "to_dict.mapping",
            m.ConfigMap(root={map_key_a: map_val_a, map_key_b: map_val_b}),
        )
        self.check(
            "to_dict.basemodel",
            m.Handler(handler_name=handler_name, handler_id=handler_id).model_dump(),
        )
        self.check("generate_id.len", len(u.generate_id()))
        self.check(
            "generate_prefixed_id",
            u.generate_prefixed_id(prefix, 6).startswith(f"{prefix}_"),
        )
        self.check(
            "generate_datetime_utc.type",
            type(u.generate_datetime_utc()).__name__,
        )


if __name__ == "__main__":
    Ex12RegistryDsl(caller_file=__file__).run()
