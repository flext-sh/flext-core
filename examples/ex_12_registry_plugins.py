"""Registry plugin and service registration example sections."""

from __future__ import annotations

from typing import TYPE_CHECKING

from examples.constants import c
from examples.models import m
from flext_core import r

from .ex_12_registry_flow import Ex12RegistryFlow
from .ex_12_registry_support import ProtocolHandler, as_registry_handler

from examples.protocols import p



class Ex12RegistryPlugins(Ex12RegistryFlow):
    """Plugin and service registration checks for the registry example."""

    def _exercise_bindings_and_plugin_apis(
        self,
        registry: p.Registry,
        handler_a: ProtocolHandler,
        handler_b: ProtocolHandler,
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
            m.Examples.CommandA: as_registry_handler(handler_a),
            custom_binding_name: as_registry_handler(handler_b),
        })
        self.audit_check("register_bindings.success", bindings_result.success)
        self.audit_check(
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
        self.audit_check("register_plugin.ok", plugin_ok.success)
        self.audit_check("register_plugin.dup", plugin_dup.success)
        self.audit_check("register_plugin.empty_name", plugin_empty.failure)
        self.audit_check("register_plugin.validated", plugin_validated.success)
        self.audit_check("register_plugin.validate_fail", plugin_validate_fail.failure)
        self.audit_check("register_plugin.validate_exc", plugin_validate_exc.failure)
        plugin_fetch_ok = registry.fetch_plugin(plugin_ns, plugin_name_a)
        plugin_fetch_missing = registry.fetch_plugin(plugin_ns, plugin_missing_name)
        plugin_list = registry.list_plugins(plugin_ns)
        plugin_unreg_ok = registry.unregister_plugin(plugin_ns, plugin_name_a)
        plugin_unreg_missing = registry.unregister_plugin(
            plugin_ns, plugin_unreg_missing_name
        )
        self.audit_check("fetch_plugin.ok", plugin_fetch_ok.value == plugin_value_a)
        self.audit_check("fetch_plugin.missing", plugin_fetch_missing.failure)
        self.audit_check(
            "list_plugins.transports",
            sorted(plugin_list.value) if plugin_list.success else [],
        )
        self.audit_check("unregister_plugin.ok", plugin_unreg_ok.success)
        self.audit_check("unregister_plugin.missing", plugin_unreg_missing.failure)
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
            class_ns, "", class_plugin_value, scope=c.RegistrationScope.CLASS
        )
        class_fetch_ok = registry.fetch_plugin(
            class_ns, class_plugin_name, scope=c.RegistrationScope.CLASS
        )
        class_fetch_missing = registry.fetch_plugin(
            class_ns, class_missing_name, scope=c.RegistrationScope.CLASS
        )
        class_list = registry.list_plugins(class_ns, scope=c.RegistrationScope.CLASS)
        class_unreg_ok = registry.unregister_plugin(
            class_ns, class_plugin_name, scope=c.RegistrationScope.CLASS
        )
        class_unreg_missing = registry.unregister_plugin(
            class_ns, class_unreg_missing_name, scope=c.RegistrationScope.CLASS
        )
        self.audit_check("register_class_plugin.ok", class_ok.success)
        self.audit_check("register_class_plugin.dup", class_dup.success)
        self.audit_check("register_class_plugin.empty_name", class_empty.failure)
        self.audit_check(
            "fetch_class_plugin.ok", class_fetch_ok.value == class_plugin_value
        )
        self.audit_check("fetch_class_plugin.missing", class_fetch_missing.failure)
        self.audit_check(
            "list_class_plugins.auth", class_list.value if class_list.success else []
        )
        self.audit_check("unregister_class_plugin.ok", class_unreg_ok.success)
        self.audit_check("unregister_class_plugin.missing", class_unreg_missing.failure)

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
        reg_meta_model = registry.register(svc_meta_name, meta_model)
        reg_bad = registry.register("", bad_value)
        self.audit_check("register.service.plain", reg_plain.success)
        self.audit_check("register.service.meta_dict", reg_meta_dict.success)
        self.audit_check("register.service.meta_model", reg_meta_model.success)
        self.audit_check("register.service.bad", reg_bad.failure)
