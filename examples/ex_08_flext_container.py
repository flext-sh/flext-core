"""Golden-file example for FlextContainer public APIs."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import override

from examples import c, m, p, t, u
from examples.shared import ExamplesFlextCoreShared
from flext_core import FlextContainer, r


class _WireProbe:
    """Probe class used to exercise wire_modules(classes=...)."""


class Ex08FlextContainer(ExamplesFlextCoreShared):
    """Exercise FlextContainer public APIs."""

    _registered_service_name: str = u.PrivateAttr(default_factory=str)
    _registered_service_value: int = u.PrivateAttr(default_factory=int)

    @override
    def exercise(self) -> None:
        """Run all sections and record deterministic golden output."""
        root = self._exercise_singleton_and_creation()
        self._exercise_registration_and_resolution(root)
        self._exercise_fluent_and_settings(root)
        scoped_full = self._exercise_wiring_and_scoped(root)
        self._exercise_internal_and_cleanup(scoped_full, root)

    def _exercise_fluent_and_settings(self, container: p.Container) -> None:
        """Exercise fluent registration and configuration APIs."""
        self.section("fluent_and_settings")
        fluent_service_name = f"svc.{self.rand_str(6)}"
        fluent_factory_name = f"svc.{self.rand_str(6)}"
        fluent_resource_name = f"svc.{self.rand_str(6)}"
        fluent_service_value = self.rand_str(10)
        fluent_factory_value = self.rand_str(10)
        fluent_resource_value = self.rand_str(10)
        max_factories = self.rand_int(1, 1000)
        with_service_result = container.bind(fluent_service_name, fluent_service_value)
        with_factory_result = container.factory(
            fluent_factory_name, lambda: fluent_factory_value
        )
        with_resource_result = container.resource(
            fluent_resource_name, lambda: fluent_resource_value
        )
        with_settings_result = container.apply({"max_factories": max_factories})
        self.check("with_service.returns_self", with_service_result is container)
        self.check("with_factory.returns_self", with_factory_result is container)
        self.check("with_resource.returns_self", with_resource_result is container)
        self.check("with_settings.returns_self", with_settings_result is container)
        configured_max_services = self.rand_int(1, 1000)
        configured_factory_caching = self.rand_bool()
        container.apply({
            "max_services": configured_max_services,
            "enable_factory_caching": configured_factory_caching,
        })
        settings_map = container.snapshot()
        max_services = settings_map["max_services"]
        enable_factory_caching = settings_map["enable_factory_caching"]
        max_services_num = max_services if isinstance(max_services, int) else -1
        factory_cache_flag = (
            enable_factory_caching
            if u.matches_type(enable_factory_caching, bool)
            else False
        )
        self.check(
            "configure.resolve_settings.max_services_matches",
            max_services_num == configured_max_services,
        )
        self.check(
            "configure.resolve_settings.enable_factory_caching_matches",
            factory_cache_flag == configured_factory_caching,
        )
        self.check(
            "with_service.get.value_matches",
            (
                container.resolve(fluent_service_name, type_cls=str).value
                if container.resolve(fluent_service_name, type_cls=str).success
                else ""
            )
            == fluent_service_value,
        )
        self.check(
            "with_factory.get.value_matches",
            (
                container.resolve(fluent_factory_name, type_cls=str).value
                if container.resolve(fluent_factory_name, type_cls=str).success
                else ""
            )
            == fluent_factory_value,
        )
        self.check(
            "with_resource.get.value_matches",
            (
                container.resolve(fluent_resource_name, type_cls=str).value
                if container.resolve(fluent_resource_name, type_cls=str).success
                else ""
            )
            == fluent_resource_value,
        )

    def _exercise_internal_and_cleanup(
        self,
        container: p.ContainerLifecycle,
        root: p.ContainerLifecycle,
    ) -> None:
        """Exercise lifecycle helpers and cleanup APIs."""
        self.section("internal_and_cleanup")
        container.initialize_di_components()
        self.check(
            "initialize_di_components.bridge_exists",
            hasattr(container, "_di_bridge"),
        )
        self.check(
            "initialize_di_components.container_exists",
            hasattr(container, "_di_container"),
        )
        container.initialize_registrations(
            registration=m.ServiceRegistrationSpec(
                settings=root.settings.model_copy(deep=True),
                context=root.context,
            ),
        )
        self.check(
            "initialize_registrations.list_services_empty",
            len(container.names()),
        )
        container.sync_config_to_di()
        container.register_existing_providers()
        container.register_core_services()
        self.check(
            "sync_settings_to_di.service_settings_present",
            container.has("settings"),
        )
        self.check(
            "register_core_services.logger_present",
            container.has("logger"),
        )
        self.check(
            "register_core_services.command_bus_present",
            container.has("command_bus"),
        )
        logger_default = container.logger()
        logger_custom = container.logger(f"examples.{self.rand_str(6)}")
        self.check("create_module_logger.default.type", type(logger_default).__name__)
        self.check("create_module_logger.custom.type", type(logger_custom).__name__)
        removable_name = f"svc.{self.rand_str(6)}"
        missing_remove_name = f"svc.{self.rand_str(6)}"
        _ = container.bind(removable_name, self.rand_int(1, 1000))
        unregister_ok = container.drop(removable_name)
        unregister_missing = container.drop(missing_remove_name)
        self.check("unregister.existing.success", unregister_ok.success)
        self.check("unregister.missing.failure", unregister_missing.failure)
        container.clear()
        self.check("clear_all.count", len(container.names()))
        before_reset = root
        FlextContainer.reset_for_testing()
        after_reset = FlextContainer.shared()
        self.check("reset_singleton.new_instance", before_reset is not after_reset)
        self.check(
            "reset_singleton.fetch_global.same_after_reset",
            after_reset is FlextContainer.shared(),
        )

    def _exercise_registration_and_resolution(self, container: p.Container) -> None:
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
        register_ok = container.bind(service_name, service_value)
        service_before_dup = container.has(service_name)
        register_dup = container.bind(service_name, self.rand_int(1, 1000))
        service_after_dup = container.has(service_name)
        empty_before_register = container.has("")
        register_empty = container.bind("", empty_name_value)
        empty_after_register = container.has("")
        self.check("register.service.returns_self", register_ok is container)
        self.check("register.service.success", container.resolve(service_name).success)
        self.check(
            "register.service.stored_value_matches",
            (
                container.resolve(service_name, type_cls=int).value
                if container.resolve(service_name, type_cls=int).success
                else -1
            )
            == service_value,
        )
        self.check("register.service.duplicate_returns_self", register_dup is container)
        self.check(
            "register.service.duplicate_failure",
            service_before_dup
            and service_after_dup
            and (
                (
                    container.resolve(service_name, type_cls=int).value
                    if container.resolve(service_name, type_cls=int).success
                    else -1
                )
                == service_value
            ),
        )
        self.check(
            "register.service.empty_name_returns_self",
            register_empty is container,
        )
        self.check(
            "register.service.empty_name_failure",
            not empty_before_register and (not empty_after_register),
        )
        factory_calls = {"count": 0}

        def _factory_counter() -> int:
            factory_calls["count"] += 1
            return factory_calls["count"]

        register_factory_ok = container.factory(factory_name, _factory_counter)
        register_factory_dup = container.factory(factory_name, _factory_counter)

        def _factory_raises() -> int:
            error_message = self.rand_str(10)
            raise RuntimeError(error_message)

        register_factory_bad = container.factory(bad_factory_name, _factory_raises)
        self.check("register.factory.returns_self", register_factory_ok is container)
        self.check("register.factory.success", container.resolve(factory_name).success)
        self.check(
            "register.factory.duplicate_failure",
            register_factory_dup is container
            and container.resolve(factory_name).success,
        )
        self.check(
            "register.factory.bad_registration_success",
            register_factory_bad is container
            and container.resolve(bad_factory_name).success,
        )
        resource_calls = {"count": 0}

        def _resource_data() -> t.IntMapping:
            resource_calls["count"] += 1
            return {self.rand_str(4): resource_calls["count"]}

        register_resource_ok = container.resource(resource_name, _resource_data)
        register_resource_dup = container.resource(resource_name, _resource_data)
        self.check("register.resource.returns_self", register_resource_ok is container)
        self.check(
            "register.resource.success", container.resolve(resource_name).success
        )
        self.check(
            "register.resource.duplicate_failure",
            register_resource_dup is container
            and container.resolve(resource_name).success,
        )
        get_service = container.resolve(service_name)
        get_factory = container.resolve(factory_name)
        get_resource = container.resolve(resource_name)
        get_missing = container.resolve(missing_name)
        get_bad_factory = container.resolve(bad_factory_name)
        self.check("get.service.success", get_service.success)
        self.check(
            "get.service.value_matches",
            (
                container.resolve(service_name, type_cls=int).value
                if container.resolve(service_name, type_cls=int).success
                else -1
            )
            == service_value,
        )
        self.check("get.factory.success", get_factory.success)
        self.check(
            "get.factory.value_first_call",
            (
                container.resolve(factory_name, type_cls=int).value
                if container.resolve(factory_name, type_cls=int).success
                else -1
            )
            == 1,
        )
        self.check("get.resource.success", get_resource.success)
        self.check("get.resource.call_count_is_one", resource_calls["count"] == 1)
        self.check("get.missing.failure", get_missing.failure)
        self.check("get.bad_factory.failure", get_bad_factory.failure)
        get_typed_service = container.resolve(service_name, type_cls=int)
        get_typed_service_bad = container.resolve(service_name, type_cls=str)
        get_typed_factory = container.resolve(factory_name, type_cls=int)
        get_typed_missing = container.resolve(missing_name, type_cls=int)
        self.check("get_typed.service.success", get_typed_service.success)
        self.check(
            "get_typed.service.value_matches",
            (get_typed_service.value if get_typed_service.success else -1)
            == service_value,
        )
        self.check(
            "get_typed.service.type_mismatch_failure",
            get_typed_service_bad.failure,
        )
        self.check("get_typed.factory.success", get_typed_factory.success)
        self.check(
            "get_typed.resource_via_get.success",
            container.resolve(resource_name).success,
        )
        self.check("get_typed.missing.failure", get_typed_missing.failure)
        self.check("has_service.service.true", container.has(service_name))
        self.check("has_service.factory.true", container.has(factory_name))
        self.check("has_service.resource.true", container.has(resource_name))
        self.check("has_service.missing.false", container.has(missing_name))
        service_list = list(container.names())
        self.check("list_services.contains.service", service_name in service_list)
        self.check("list_services.contains.factory", factory_name in service_list)
        self.check("list_services.contains.resource", resource_name in service_list)

    def _exercise_singleton_and_creation(self) -> p.ContainerLifecycle:
        """Exercise fetch_global/create entrypoints and singleton semantics."""
        self.section("singleton_and_creation")
        FlextContainer.reset_for_testing()
        root = FlextContainer.shared()
        self.check("fetch_global.type", type(root).__name__)
        self.check("fetch_global.context.type", type(root.context).__name__)
        self.check("fetch_global.settings.type", type(root.settings).__name__)
        self.check(
            "fetch_global.same_instance",
            root is FlextContainer.shared(),
        )
        created_false = FlextContainer.shared(auto_register_factories=False)
        created_true = FlextContainer.shared(auto_register_factories=True)
        self.check("create.false.same_instance", created_false is root)
        self.check("create.true.same_instance", created_true is root)
        random_ok_val = self.rand_int(1, 1000)
        self.check(
            "result.ok.roundtrip",
            r[int].ok(random_ok_val).value == random_ok_val,
        )
        self.check("runtime.normalize.bool", u.normalize_to_container(True))
        self.check("constants.default_max_services", c.DEFAULT_SIZE)
        return root

    def _exercise_wiring_and_scoped(
        self,
        container: p.ContainerLifecycle,
    ) -> p.ContainerLifecycle:
        """Exercise wire_modules and scoped with all supported parameter styles."""
        self.section("wiring_and_scoped")
        this_module: ModuleType = sys.modules[__name__]
        container.wire(modules=[this_module])
        container.wire(packages=[])
        container.wire(classes=[_WireProbe])
        self.check("wire_modules.calls_completed", True)
        scoped_default = container.scope()
        subproject_alpha = self.rand_str(6)
        subproject_beta = self.rand_str(6)
        scoped_subproject = container.scope(subproject=subproject_alpha)
        explicit_context = container.context
        explicit_settings = container.settings.model_copy(
            update={"app_name": f"scoped.{self.rand_str(8)}"},
        )
        scoped_service_name = f"svc.{self.rand_str(6)}"
        scoped_factory_name = f"svc.{self.rand_str(6)}"
        scoped_resource_name = f"svc.{self.rand_str(6)}"
        scoped_service_value = self.rand_str(8)
        scoped_factory_value = self.rand_int(1, 1000)
        scoped_resource_value = self.rand_str(8)
        scoped_full = container.scope(
            subproject=subproject_beta,
            registration=m.ServiceRegistrationSpec.model_validate({
                "settings": explicit_settings,
                "context": explicit_context,
                "services": {scoped_service_name: scoped_service_value},
                "factories": {scoped_factory_name: lambda: scoped_factory_value},
                "resources": {
                    scoped_resource_name: lambda: {"res": scoped_resource_value},
                },
            }),
        )
        self.check("scoped.default.new_instance", scoped_default is not container)
        self.check(
            "scoped.default.inherits_service",
            scoped_default.has(self._registered_service_name),
        )
        self.check(
            "scoped.default.get_typed_service_matches",
            (
                scoped_default.resolve(
                    self._registered_service_name, type_cls=int
                ).value
                if scoped_default.resolve(
                    self._registered_service_name, type_cls=int
                ).success
                else -1
            )
            == self._registered_service_value,
        )
        self.check(
            "scoped.subproject.app_name_suffix",
            scoped_subproject.settings.app_name.endswith(f".{subproject_alpha}"),
        )
        self.check("scoped.full.new_instance", scoped_full is not container)
        self.check("scoped.full.settings_app_name", scoped_full.settings.app_name)
        self.check(
            "scoped.full.uses_explicit_context",
            scoped_full.context is explicit_context,
        )
        self.check(
            "scoped.full.has_service",
            scoped_full.has(scoped_service_name),
        )
        self.check(
            "scoped.full.has_factory",
            scoped_full.has(scoped_factory_name),
        )
        self.check(
            "scoped.full.has_resource",
            scoped_full.has(scoped_resource_name),
        )
        self.check(
            "scoped.full.get_service_matches",
            (
                scoped_full.resolve(scoped_service_name, type_cls=str).value
                if scoped_full.resolve(scoped_service_name, type_cls=str).success
                else ""
            )
            == scoped_service_value,
        )
        self.check(
            "scoped.full.get_factory_matches",
            (
                scoped_full.resolve(scoped_factory_name, type_cls=int).value
                if scoped_full.resolve(scoped_factory_name, type_cls=int).success
                else -1
            )
            == scoped_factory_value,
        )
        self.check(
            "scoped.full.get_resource.success",
            scoped_full.resolve(scoped_resource_name).success,
        )
        return scoped_full


if __name__ == "__main__":
    Ex08FlextContainer(caller_file=Path(__file__)).run()
