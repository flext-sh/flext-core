"""Container lifecycle example section."""

from __future__ import annotations

from examples import m, p
from flext_core import FlextContainer

from .ex_08_container_scoped import Ex08ContainerScoped


class Ex08ContainerLifecycle(Ex08ContainerScoped):
    """Lifecycle and cleanup checks for the container example."""

    def _exercise_internal_and_cleanup(
        self,
        container: p.ContainerLifecycle,
        root: p.ContainerLifecycle,
    ) -> None:
        """Exercise lifecycle helpers and cleanup APIs."""
        self.section("internal_and_cleanup")
        container.initialize_di_components()
        self.audit_check(
            "initialize_di_components.bridge_exists",
            hasattr(container, "_di_bridge"),
        )
        self.audit_check(
            "initialize_di_components.container_exists",
            hasattr(container, "_di_container"),
        )
        container.initialize_registrations(
            registration=m.ServiceRegistrationSpec(
                settings=root.settings.clone(),
                context=root.context,
            ),
        )
        self.audit_check(
            "initialize_registrations.list_services_empty",
            len(container.names()),
        )
        container.sync_config_to_di()
        container.register_existing_providers()
        container.register_core_services()
        self.audit_check(
            "sync_settings_to_di.service_settings_present",
            container.has("settings"),
        )
        self.audit_check(
            "register_core_services.logger_present",
            container.has("logger"),
        )
        self.audit_check(
            "register_core_services.command_bus_present",
            container.has("command_bus"),
        )
        logger_default = container.logger()
        logger_custom = container.logger(f"examples.{self.rand_str(6)}")
        self.audit_check(
            "create_module_logger.default.type", type(logger_default).__name__
        )
        self.audit_check(
            "create_module_logger.custom.type", type(logger_custom).__name__
        )
        removable_name = f"svc.{self.rand_str(6)}"
        missing_remove_name = f"svc.{self.rand_str(6)}"
        _ = container.bind(removable_name, self.rand_int(1, 1000))
        unregister_ok = container.drop(removable_name)
        unregister_missing = container.drop(missing_remove_name)
        self.audit_check("unregister.existing.success", unregister_ok.success)
        self.audit_check("unregister.missing.failure", unregister_missing.failure)
        container.clear()
        self.audit_check("clear_all.count", len(container.names()))
        before_reset = root
        FlextContainer.reset_for_testing()
        after_reset = FlextContainer.shared()
        self.audit_check(
            "reset_singleton.new_instance", before_reset is not after_reset
        )
        self.audit_check(
            "reset_singleton.fetch_global.same_after_reset",
            after_reset is FlextContainer.shared(),
        )
