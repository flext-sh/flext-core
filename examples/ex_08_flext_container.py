"""Golden-file example for FlextContainer public APIs."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from types import ModuleType
from typing import override

from flext_core import FlextContainer, FlextContext, FlextRuntime, c, m, r, u

from .shared import Examples


class _WireProbe:
    """Probe class used to exercise wire_modules(classes=...)."""


class Ex08FlextContainer(Examples):
    """Exercise FlextContainer public APIs."""

    _registered_service_name: str
    _registered_service_value: int

    @override
    def exercise(self) -> None:
        """Run all sections and record deterministic golden output."""
        root = self._exercise_singleton_and_creation()
        self._exercise_registration_and_resolution(root)
        self._exercise_fluent_and_config(root)
        scoped_full = self._exercise_wiring_and_scoped(root)
        self._exercise_internal_and_cleanup(scoped_full, root)

    def _exercise_fluent_and_config(self, container: FlextContainer) -> None:
        """Exercise fluent registration and configuration APIs."""
        self.section("fluent_and_config")
        fluent_service_name = f"svc.{self.rand_str(6)}"
        fluent_factory_name = f"svc.{self.rand_str(6)}"
        fluent_resource_name = f"svc.{self.rand_str(6)}"
        fluent_service_value = self.rand_str(10)
        fluent_factory_value = self.rand_str(10)
        fluent_resource_value = self.rand_str(10)
        max_factories = self.rand_int(1, 1000)
        with_service_result = container.register(
            fluent_service_name, fluent_service_value
        )
        with_factory_result = container.register(
            fluent_factory_name, lambda: fluent_factory_value, kind="factory"
        )
        with_resource_result = container.register(
            fluent_resource_name, lambda: fluent_resource_value, kind="resource"
        )
        with_config_result = container.configure({"max_factories": max_factories})
        self.check("with_service.returns_self", with_service_result is container)
        self.check("with_factory.returns_self", with_factory_result is container)
        self.check("with_resource.returns_self", with_resource_result is container)
        self.check("with_config.returns_self", with_config_result is container)
        configured_max_services = self.rand_int(1, 1000)
        configured_factory_caching = self.rand_bool()
        container.configure({
            "max_services": configured_max_services,
            "enable_factory_caching": configured_factory_caching,
        })
        config_map = container.get_config()
        max_services = config_map["max_services"]
        enable_factory_caching = config_map["enable_factory_caching"]
        max_services_num = max_services if u.Guards.is_type(max_services, int) else -1
        factory_cache_flag = (
            enable_factory_caching
            if u.Guards.is_type(enable_factory_caching, bool)
            else False
        )
        self.check(
            "configure.get_config.max_services_matches",
            max_services_num == configured_max_services,
        )
        self.check(
            "configure.get_config.enable_factory_caching_matches",
            factory_cache_flag == configured_factory_caching,
        )
        self.check(
            "with_service.get.value_matches",
            container.get(fluent_service_name, type_cls=str).unwrap_or("")
            == fluent_service_value,
        )
        self.check(
            "with_factory.get.value_matches",
            container.get(fluent_factory_name, type_cls=str).unwrap_or("")
            == fluent_factory_value,
        )
        self.check(
            "with_resource.get.value_matches",
            container.get(fluent_resource_name, type_cls=str).unwrap_or("")
            == fluent_resource_value,
        )

    def _exercise_internal_and_cleanup(
        self, container: FlextContainer, root: FlextContainer
    ) -> None:
        """Exercise lifecycle helpers and cleanup APIs."""
        self.section("internal_and_cleanup")
        container.initialize_di_components()
        self.check(
            "initialize_di_components.bridge_exists", hasattr(container, "_di_bridge")
        )
        self.check(
            "initialize_di_components.container_exists",
            hasattr(container, "_di_container"),
        )
        container.initialize_registrations(
            config=root.config.model_copy(deep=True), context=FlextContext()
        )
        self.check(
            "initialize_registrations.list_services_empty",
            len(container.list_services()),
        )
        container.sync_config_to_di()
        container.register_existing_providers()
        container.register_core_services()
        self.check(
            "sync_config_to_di.service_config_present", container.has_service("config")
        )
        self.check(
            "register_core_services.logger_present", container.has_service("logger")
        )
        self.check(
            "register_core_services.command_bus_present",
            container.has_service("command_bus"),
        )
        logger_default = container.create_module_logger()
        logger_custom = container.create_module_logger(f"examples.{self.rand_str(6)}")
        self.check("create_module_logger.default.type", type(logger_default).__name__)
        self.check("create_module_logger.custom.type", type(logger_custom).__name__)
        removable_name = f"svc.{self.rand_str(6)}"
        missing_remove_name = f"svc.{self.rand_str(6)}"
        _ = container.register(removable_name, self.rand_int(1, 1000))
        unregister_ok = container.unregister(removable_name)
        unregister_missing = container.unregister(missing_remove_name)
        self.check("unregister.existing.success", unregister_ok.is_success)
        self.check("unregister.missing.failure", unregister_missing.is_failure)
        container.clear_all()
        self.check("clear_all.count", len(container.list_services()))
        before_reset = root
        FlextContainer.reset_for_testing()
        after_reset = FlextContainer.get_global(context=FlextContext())
        self.check("reset_singleton.new_instance", before_reset is not after_reset)
        self.check(
            "reset_singleton.get_global.same_after_reset",
            after_reset is FlextContainer.get_global(),
        )

    def _exercise_registration_and_resolution(self, container: FlextContainer) -> None:
        """Exercise register APIs plus get/get_typed/list/has checks."""
        self.section("registration_and_resolution")
        service_name = f"svc.{self.rand_str(6)}"
        service_value = self.rand_int(1, 1000)
        self._registered_service_name = service_name
        self._registered_service_value = service_value
        factory_name = f"svc.{self.rand_str(6)}"
        resource_name = f"svc.{self.rand_str(6)}"
        missing_name = f"svc.{self.rand_str(6)}"
        bad_factory_name = f"svc.{self.rand_str(6)}"
        empty_name_value = self.rand_int(1, 1000)
        register_ok = container.register(service_name, service_value)
        service_before_dup = container.has_service(service_name)
        register_dup = container.register(service_name, self.rand_int(1, 1000))
        service_after_dup = container.has_service(service_name)
        empty_before_register = container.has_service("")
        register_empty = container.register("", empty_name_value)
        empty_after_register = container.has_service("")
        self.check("register.service.returns_self", register_ok is container)
        self.check("register.service.success", container.get(service_name).is_success)
        self.check(
            "register.service.stored_value_matches",
            container.get(service_name, type_cls=int).unwrap_or(-1) == service_value,
        )
        self.check("register.service.duplicate_returns_self", register_dup is container)
        self.check(
            "register.service.duplicate_failure",
            service_before_dup
            and service_after_dup
            and (
                container.get(service_name, type_cls=int).unwrap_or(-1) == service_value
            ),
        )
        self.check(
            "register.service.empty_name_returns_self", register_empty is container
        )
        self.check(
            "register.service.empty_name_failure",
            not empty_before_register and (not empty_after_register),
        )
        factory_calls = {"count": 0}

        def _factory_counter() -> int:
            factory_calls["count"] = int(factory_calls["count"]) + 1
            return int(factory_calls["count"])

        register_factory_ok = container.register(
            factory_name, _factory_counter, kind="factory"
        )
        register_factory_dup = container.register(
            factory_name, _factory_counter, kind="factory"
        )

        def _factory_raises() -> int:
            error_message = self.rand_str(10)
            raise RuntimeError(error_message)

        register_factory_bad = container.register(
            bad_factory_name, _factory_raises, kind="factory"
        )
        self.check("register.factory.returns_self", register_factory_ok is container)
        self.check("register.factory.success", container.get(factory_name).is_success)
        self.check(
            "register.factory.duplicate_failure",
            register_factory_dup is container
            and container.get(factory_name).is_success,
        )
        self.check(
            "register.factory.bad_registration_success",
            register_factory_bad is container
            and container.get(bad_factory_name).is_success,
        )
        resource_calls = {"count": 0}

        def _resource_data() -> Mapping[str, int]:
            resource_calls["count"] = int(resource_calls["count"]) + 1
            return {self.rand_str(4): int(resource_calls["count"])}

        register_resource_ok = container.register(
            resource_name, _resource_data, kind="resource"
        )
        register_resource_dup = container.register(
            resource_name, _resource_data, kind="resource"
        )
        self.check("register.resource.returns_self", register_resource_ok is container)
        self.check("register.resource.success", container.get(resource_name).is_success)
        self.check(
            "register.resource.duplicate_failure",
            register_resource_dup is container
            and container.get(resource_name).is_success,
        )
        get_service = container.get(service_name)
        get_factory = container.get(factory_name)
        get_resource = container.get(resource_name)
        get_missing = container.get(missing_name)
        get_bad_factory = container.get(bad_factory_name)
        self.check("get.service.success", get_service.is_success)
        self.check(
            "get.service.value_matches",
            container.get(service_name, type_cls=int).unwrap_or(-1) == service_value,
        )
        self.check("get.factory.success", get_factory.is_success)
        self.check(
            "get.factory.value_first_call",
            container.get(factory_name, type_cls=int).unwrap_or(-1) == 1,
        )
        self.check("get.resource.success", get_resource.is_success)
        self.check("get.resource.call_count_is_one", resource_calls["count"] == 1)
        self.check("get.missing.failure", get_missing.is_failure)
        self.check("get.bad_factory.failure", get_bad_factory.is_failure)
        get_typed_service = container.get(service_name, type_cls=int)
        get_typed_service_bad = container.get(service_name, type_cls=str)
        get_typed_factory = container.get(factory_name, type_cls=int)
        get_typed_missing = container.get(missing_name, type_cls=int)
        self.check("get_typed.service.success", get_typed_service.is_success)
        self.check(
            "get_typed.service.value_matches",
            get_typed_service.unwrap_or(-1) == service_value,
        )
        self.check(
            "get_typed.service.type_mismatch_failure", get_typed_service_bad.is_failure
        )
        self.check("get_typed.factory.success", get_typed_factory.is_success)
        self.check(
            "get_typed.resource_via_get.success",
            container.get(resource_name).is_success,
        )
        self.check("get_typed.missing.failure", get_typed_missing.is_failure)
        self.check("has_service.service.true", container.has_service(service_name))
        self.check("has_service.factory.true", container.has_service(factory_name))
        self.check("has_service.resource.true", container.has_service(resource_name))
        self.check("has_service.missing.false", container.has_service(missing_name))
        service_list = list(container.list_services())
        self.check("list_services.contains.service", service_name in service_list)
        self.check("list_services.contains.factory", factory_name in service_list)
        self.check("list_services.contains.resource", resource_name in service_list)

    def _exercise_singleton_and_creation(self) -> FlextContainer:
        """Exercise get_global/create/builder creation and singleton semantics."""
        self.section("singleton_and_creation")
        FlextContainer.reset_for_testing()
        root_context = FlextContext()
        root = FlextContainer.get_global(context=root_context)
        self.check("get_global.type", type(root).__name__)
        self.check("get_global.context.type", type(root.context).__name__)
        self.check("get_global.config.type", type(root.config).__name__)
        self.check("get_global.same_instance", root is FlextContainer.get_global())
        created_false = FlextContainer.create(auto_register_factories=False)
        created_true = FlextContainer.create(auto_register_factories=True)
        builder_false = FlextContainer.Builder.create(auto_register_factories=False)
        builder_true = FlextContainer.Builder.create(auto_register_factories=True)
        self.check("create.false.same_instance", created_false is root)
        self.check("create.true.same_instance", created_true is root)
        self.check("builder.create.false.same_instance", builder_false is root)
        self.check("builder.create.true.same_instance", builder_true is root)
        random_ok_val = self.rand_int(1, 1000)
        self.check(
            "result.ok.roundtrip", r[int].ok(random_ok_val).value == random_ok_val
        )
        self.check(
            "runtime.normalize.bool", FlextRuntime.normalize_to_general_value(True)
        )
        self.check("constants.default_max_services", c.Container.DEFAULT_MAX_SERVICES)
        return root

    def _exercise_wiring_and_scoped(self, container: FlextContainer) -> FlextContainer:
        """Exercise wire_modules and scoped with all supported parameter styles."""
        self.section("wiring_and_scoped")
        this_module: ModuleType = sys.modules[__name__]
        container.wire_modules(modules=[this_module])
        container.wire_modules(packages=[])
        container.wire_modules(classes=[_WireProbe])
        self.check("wire_modules.calls_completed", True)
        scoped_default = container.scoped()
        subproject_alpha = self.rand_str(6)
        subproject_beta = self.rand_str(6)
        scoped_subproject = container.scoped(subproject=subproject_alpha)
        explicit_context = FlextContext()
        explicit_config = container.config.model_copy(
            update={"app_name": f"scoped.{self.rand_str(8)}"}
        )
        scoped_service_name = f"svc.{self.rand_str(6)}"
        scoped_factory_name = f"svc.{self.rand_str(6)}"
        scoped_resource_name = f"svc.{self.rand_str(6)}"
        scoped_service_value = self.rand_str(8)
        scoped_factory_value = self.rand_int(1, 1000)
        scoped_resource_value = self.rand_str(8)
        scoped_full = container.scoped(
            config=explicit_config,
            context=explicit_context,
            subproject=subproject_beta,
            services={scoped_service_name: scoped_service_value},
            factories={scoped_factory_name: lambda: scoped_factory_value},
            resources={
                scoped_resource_name: lambda: m.ConfigMap(
                    root={"res": scoped_resource_value}
                )
            },
        )
        self.check("scoped.default.new_instance", scoped_default is not container)
        self.check(
            "scoped.default.inherits_service",
            scoped_default.has_service(self._registered_service_name),
        )
        self.check(
            "scoped.default.get_typed_service_matches",
            scoped_default.get(self._registered_service_name, type_cls=int).unwrap_or(
                -1
            )
            == self._registered_service_value,
        )
        self.check(
            "scoped.subproject.app_name_suffix",
            scoped_subproject.config.app_name.endswith(f".{subproject_alpha}"),
        )
        self.check("scoped.full.new_instance", scoped_full is not container)
        self.check("scoped.full.config_app_name", scoped_full.config.app_name)
        self.check(
            "scoped.full.uses_explicit_context", scoped_full.context is explicit_context
        )
        self.check(
            "scoped.full.has_service", scoped_full.has_service(scoped_service_name)
        )
        self.check(
            "scoped.full.has_factory", scoped_full.has_service(scoped_factory_name)
        )
        self.check(
            "scoped.full.has_resource", scoped_full.has_service(scoped_resource_name)
        )
        self.check(
            "scoped.full.get_service_matches",
            scoped_full.get(scoped_service_name, type_cls=str).unwrap_or("")
            == scoped_service_value,
        )
        self.check(
            "scoped.full.get_factory_matches",
            scoped_full.get(scoped_factory_name, type_cls=int).unwrap_or(-1)
            == scoped_factory_value,
        )
        self.check(
            "scoped.full.get_resource.success",
            scoped_full.get(scoped_resource_name).is_success,
        )
        return scoped_full


if __name__ == "__main__":
    Ex08FlextContainer(__file__).run()
