"""Golden-file example for FlextContainer public APIs."""

from __future__ import annotations

from pathlib import Path
from typing import override

from examples import c, p, u
from flext_core import FlextContainer, r

from .ex_08_container_lifecycle import Ex08ContainerLifecycle


class Ex08FlextContainer(Ex08ContainerLifecycle):
    """Exercise FlextContainer public APIs."""

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
            fluent_factory_name,
            lambda: fluent_factory_value,
        )
        with_resource_result = container.resource(
            fluent_resource_name,
            lambda: fluent_resource_value,
        )
        with_settings_result = container.apply({"max_factories": max_factories})
        self.audit_check("with_service.returns_self", with_service_result is container)
        self.audit_check("with_factory.returns_self", with_factory_result is container)
        self.audit_check(
            "with_resource.returns_self",
            with_resource_result is container,
        )
        self.audit_check(
            "with_settings.returns_self",
            with_settings_result is container,
        )
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
        self.audit_check(
            "configure.resolve_settings.max_services_matches",
            max_services_num == configured_max_services,
        )
        self.audit_check(
            "configure.resolve_settings.enable_factory_caching_matches",
            factory_cache_flag == configured_factory_caching,
        )
        self.audit_check(
            "with_service.get.value_matches",
            (
                container.resolve(fluent_service_name, type_cls=str).value
                if container.resolve(fluent_service_name, type_cls=str).success
                else ""
            )
            == fluent_service_value,
        )
        self.audit_check(
            "with_factory.get.value_matches",
            (
                container.resolve(fluent_factory_name, type_cls=str).value
                if container.resolve(fluent_factory_name, type_cls=str).success
                else ""
            )
            == fluent_factory_value,
        )
        self.audit_check(
            "with_resource.get.value_matches",
            (
                container.resolve(fluent_resource_name, type_cls=str).value
                if container.resolve(fluent_resource_name, type_cls=str).success
                else ""
            )
            == fluent_resource_value,
        )

    def _exercise_singleton_and_creation(self) -> p.ContainerLifecycle:
        """Exercise fetch_global/create entrypoints and singleton semantics."""
        self.section("singleton_and_creation")
        FlextContainer.reset_for_testing()
        root = FlextContainer.shared()
        self.audit_check("fetch_global.type", type(root).__name__)
        self.audit_check("fetch_global.context.type", type(root.context).__name__)
        self.audit_check("fetch_global.settings.type", type(root.settings).__name__)
        self.audit_check(
            "fetch_global.same_instance",
            root is FlextContainer.shared(),
        )
        created_false = FlextContainer.shared(auto_register_factories=False)
        created_true = FlextContainer.shared(auto_register_factories=True)
        self.audit_check("create.false.same_instance", created_false is root)
        self.audit_check("create.true.same_instance", created_true is root)
        random_ok_val = self.rand_int(1, 1000)
        self.audit_check(
            "result.ok.roundtrip",
            r[int].ok(random_ok_val).value == random_ok_val,
        )
        self.audit_check("runtime.normalize.bool", u.normalize_to_container(True))
        self.audit_check("constants.default_max_services", c.DEFAULT_SIZE)
        return root


if __name__ == "__main__":
    Ex08FlextContainer(caller_file=Path(__file__)).run()
