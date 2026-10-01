"""Container scope example section."""

from __future__ import annotations

from examples.models import m
from examples.protocols import p
from flext_core import FlextSettings

from .ex_08_container_registration import Ex08ContainerRegistration


class Ex08ContainerScoped(Ex08ContainerRegistration):
    """Scoped container checks for the container example."""

    def _exercise_scoped(self, container: p.ContainerLifecycle) -> p.ContainerLifecycle:
        """Exercise scope with all supported parameter styles."""
        self.section("scoped")
        scoped_default = container.scope()
        subproject_alpha = self.rand_str(6)
        subproject_beta = self.rand_str(6)
        scoped_subproject = container.scope(subproject=subproject_alpha)
        explicit_context = container.context
        explicit_settings = container.settings.clone(
            timezone=f"scoped/{self.rand_str(8)}",
        )
        scoped_service_name = f"svc.{self.rand_str(6)}"
        scoped_factory_name = f"svc.{self.rand_str(6)}"
        scoped_resource_name = f"svc.{self.rand_str(6)}"
        scoped_service_value = self.rand_str(8)
        scoped_factory_value = self.rand_int(1, 1000)
        scoped_resource_value = self.rand_str(8)
        scoped_full = container.scope(
            subproject=subproject_beta,
            registration=m.ServiceRegistrationSpec(
                settings=explicit_settings,
                context=explicit_context,
                services={scoped_service_name: scoped_service_value},
                factories={scoped_factory_name: lambda: scoped_factory_value},
                resources={
                    scoped_resource_name: lambda: {"res": scoped_resource_value},
                },
            ),
        )
        self.audit_check("scoped.default.new_instance", scoped_default is not container)
        self.audit_check(
            "scoped.default.logger_stays_internal",
            not scoped_default.has("logger")
            and scoped_default.resolve("logger").success,
        )
        self.audit_check(
            "scoped.default.settings_service_is_own",
            scoped_default.resolve("settings").unwrap() is scoped_default.settings,
        )
        self.audit_check(
            "scoped.default.inherits_service",
            scoped_default.has(self._registered_service_name),
        )
        self.audit_check(
            "scoped.default.get_typed_service_matches",
            (
                scoped_default.resolve(
                    self._registered_service_name,
                    type_cls=int,
                ).value
                if scoped_default.resolve(
                    self._registered_service_name,
                    type_cls=int,
                ).success
                else -1
            )
            == self._registered_service_value,
        )

        scoped_full_settings = scoped_full.settings
        self.audit_check(
            "scoped.subproject.context_marker",
            scoped_subproject.context.get("subproject").unwrap_or("")
            == subproject_alpha,
        )
        self.audit_check("scoped.full.new_instance", scoped_full is not container)
        self.audit_check(
            "scoped.full.settings_timezone",
            scoped_full_settings.timezone
            if isinstance(scoped_full_settings, FlextSettings)
            else "",
        )
        self.audit_check(
            "scoped.full.uses_explicit_context",
            scoped_full.context is explicit_context,
        )
        self.audit_check(
            "scoped.full.has_service",
            scoped_full.has(scoped_service_name),
        )
        self.audit_check(
            "scoped.full.has_factory",
            scoped_full.has(scoped_factory_name),
        )
        self.audit_check(
            "scoped.full.has_resource",
            scoped_full.has(scoped_resource_name),
        )
        self.audit_check(
            "scoped.full.get_service_matches",
            (
                scoped_full.resolve(scoped_service_name, type_cls=str).value
                if scoped_full.resolve(scoped_service_name, type_cls=str).success
                else ""
            )
            == scoped_service_value,
        )
        self.audit_check(
            "scoped.full.get_factory_matches",
            (
                scoped_full.resolve(scoped_factory_name, type_cls=int).value
                if scoped_full.resolve(scoped_factory_name, type_cls=int).success
                else -1
            )
            == scoped_factory_value,
        )
        self.audit_check(
            "scoped.full.get_resource.success",
            scoped_full.resolve(scoped_resource_name).success,
        )
        return scoped_full
