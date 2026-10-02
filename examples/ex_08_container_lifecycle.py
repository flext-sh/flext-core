"""Container lifecycle example section.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from examples.ex_08_container_scoped import Ex08ContainerScoped
from examples.models import m
from examples.protocols import p
from flext_core import FlextContainer


class Ex08ContainerLifecycle(Ex08ContainerScoped):
    """Lifecycle and cleanup checks for the container example."""

    def _exercise_internal_and_cleanup(
        self,
        container: p.ContainerLifecycle,
        root: p.ContainerLifecycle,
    ) -> None:
        """Exercise lifecycle helpers and cleanup APIs."""
        self.section("internal_and_cleanup")
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
        self.audit_check(
            "core_services.settings_internal",
            not container.has("settings") and container.resolve("settings").success,
        )
        self.audit_check(
            "core_services.logger_internal",
            not container.has("logger") and container.resolve("logger").success,
        )
        self.audit_check(
            "core_services.command_bus_internal",
            not container.has("command_bus") and container.dispatcher().success,
        )
        logger_default = container.logger(f"examples.{self.rand_str(6)}")
        logger_custom = container.logger(f"examples.{self.rand_str(6)}")
        self.audit_check(
            "create_module_logger.explicit.type",
            type(logger_default).__name__,
        )
        self.audit_check(
            "create_module_logger.explicit_custom.type",
            type(logger_custom).__name__,
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
            "reset_singleton.new_instance",
            before_reset is not after_reset,
        )
        self.audit_check(
            "reset_singleton.fetch_global.same_after_reset",
            after_reset is FlextContainer.shared(),
        )
