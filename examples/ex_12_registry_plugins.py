"""Plugin lifecycle, validation, and explicitly shared registry examples.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from examples import c, m, p, r, u
from examples.ex_12_registry_flow import ExamplesFlextRegistryFlow
from examples.ex_12_registry_support import RegistryLookup


class ExamplesFlextRegistryPlugins(ExamplesFlextRegistryFlow):
    """Use real plugin stores without a parallel fixture or adapter engine."""

    def _exercise_plugins(self, registry: p.Registry) -> None:
        self.section("plugin_lifecycle")
        category, name = self.rand_str(8), self.rand_str(8)
        payload = m.Metadata(attributes={"owner": self.rand_str(6)})
        replacement = m.Metadata(attributes={"owner": self.rand_str(6)})
        for scope in (c.RegistrationScope.INSTANCE, c.RegistrationScope.CLASS):
            self.audit_check(
                "plugin.register",
                registry.register_plugin(category, name, payload, scope=scope).value,
            )
            self.audit_check(
                "plugin.duplicate",
                registry.register_plugin(category, name, payload, scope=scope).value,
            )
            self.audit_check(
                "plugin.identity",
                registry.fetch_plugin(category, name, scope=scope).value is payload,
            )
            self.audit_check(
                "plugin.list",
                name in registry.list_plugins(category, scope=scope).value,
            )
            conflict = registry.register_plugin(
                category, name, replacement, scope=scope
            )
            self.audit_check("plugin.conflict", conflict.failure)
            self.audit_check(
                "plugin.conflict_preserves_identity",
                registry.fetch_plugin(category, name, scope=scope).value is payload,
            )
            self.audit_check(
                "plugin.unregister",
                registry.unregister_plugin(category, name, scope=scope).value,
            )
            self.audit_check(
                "plugin.missing",
                registry.fetch_plugin(category, name, scope=scope).failure,
            )
            self.audit_check(
                "plugin.remove_missing",
                registry.unregister_plugin(category, name, scope=scope).failure,
            )
            replaced = registry.register_plugin(
                category, name, replacement, scope=scope
            )
            self.audit_check("plugin.reregister", replaced.value)
            self.audit_check(
                "plugin.replacement_identity",
                registry.fetch_plugin(category, name, scope=scope).value is replacement,
            )
            self.audit_check(
                "plugin.cleanup",
                registry.unregister_plugin(category, name, scope=scope).value,
            )

        self.section("plugin_validation")
        cause = RuntimeError("Plugin validation rejected the payload")

        def rejected(_plugin: p.RegisterableService) -> p.Result[bool]:
            """Return a domain failure carrying the original cause."""
            return r[bool].fail(str(cause), exception=cause)

        def raised(_plugin: p.RegisterableService) -> p.Result[bool]:
            """Raise a real validation failure at the public registration boundary."""
            raise cause

        for scope in (c.RegistrationScope.INSTANCE, c.RegistrationScope.CLASS):
            for invalid_category, invalid_name in (
                (category, ""),
                ("", name),
                (c.REGISTRY_KEY_SEPARATOR, name),
                (category, c.REGISTRY_KEY_SEPARATOR),
            ):
                invalid = registry.register_plugin(
                    invalid_category, invalid_name, payload, scope=scope
                )
                self.audit_check("plugin.invalid_key", invalid.failure)
            for validate in (rejected, raised):
                failure = registry.register_plugin(
                    category, name, payload, validate=validate, scope=scope
                )
                self.audit_check("validation.failure", failure.failure)
                self.audit_check("validation.cause", failure.exception is cause)
                self.audit_check(
                    "validation.no_registration",
                    registry.fetch_plugin(category, name, scope=scope).failure,
                )
            false = registry.register_plugin(
                category,
                name,
                payload,
                validate=lambda _plugin: r[bool].ok(value=False),
                scope=scope,
            )
            accepted = registry.register_plugin(
                category,
                name,
                payload,
                validate=lambda _plugin: r[bool].ok(value=True),
                scope=scope,
            )
            self.audit_check("validation.false", false.failure)
            self.audit_check("validation.accepted", accepted.value)
            self.audit_check(
                "validation.cleanup",
                registry.unregister_plugin(category, name, scope=scope).value,
            )

        self.section("explicit_sharing")
        registry.register_plugin(category, name, payload).unwrap()
        lookup = RegistryLookup(registry=registry, category=category, plugin_name=name)
        shared = RegistryLookup(registry=registry, category=category, plugin_name=name)
        isolated = RegistryLookup(
            registry=u.build_registry(), category=category, plugin_name=name
        )
        self.audit_check("service.shared_registry", lookup.registry is shared.registry)
        self.audit_check(
            "service.shared_payload",
            lookup.execute().value is shared.execute().value is payload,
        )
        self.audit_check("service.isolated", isolated.execute().failure)
        self.audit_check(
            "service.cleanup", registry.unregister_plugin(category, name).value
        )
        self.audit_check("service.removal_visible", shared.execute().failure)
        self.audit_check(
            "register.service", registry.register(self.rand_str(8), payload).value
        )
        self.audit_check("register.invalid", registry.register("", payload).failure)
