"""Behavioral contract tests for the public FlextRegistry runtime surface.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from flext_core import FlextRegistry
from tests import c, h, m, p, r, t, tm, u


class TestsFlextCoreRegistry:
    """Assert observable behavior of the registry public API."""

    @staticmethod
    def test_registry_consumes_one_validated_runtime() -> None:
        """Registration and plugin storage share the explicit composition graph."""
        runtime = u.build_service_runtime()
        registry = FlextRegistry(registry_runtime=runtime)
        tm.that(registry.dispatcher is runtime.dispatcher, truth=True)
        tm.that(registry.container is runtime.container, truth=True)
        tm.that("registry_runtime" not in registry.model_dump(), truth=True)
        tm.that(
            registry.model_json_schema().get("properties", {}),
            lacks="registry_runtime",
        )

    @staticmethod
    @pytest.mark.parametrize("payload", [{}, {"registry_runtime": "invalid"}])
    def test_registry_rejects_missing_or_invalid_runtime(payload: t.JsonDict) -> None:
        """Invalid ingress never bootstraps or silently replaces a required port."""
        with pytest.raises((TypeError, ValueError)):
            FlextRegistry.model_validate(payload)

    @staticmethod
    def test_registry_validates_runtime_assignment() -> None:
        """Assignment cannot invalidate the composed graph after construction."""
        runtime = u.build_service_runtime()
        registry = FlextRegistry(registry_runtime=runtime)
        field_name = "registry_runtime"
        with pytest.raises((TypeError, ValueError)):
            setattr(registry, field_name, "invalid")
        tm.that(registry.registry_runtime is runtime, truth=True)

    @staticmethod
    @pytest.fixture
    def registry() -> p.Registry:
        """Build a registry backed by an accepting dispatcher.

        Returns:
            The resulting ``p.Registry``.

        """
        return u.build_registry(dispatcher=u.build_dispatcher())

    @staticmethod
    def test_execute_succeeds_when_dispatcher_present() -> None:
        """Test execute succeeds when dispatcher present."""
        registry = u.build_registry(dispatcher=u.build_dispatcher())

        outcome = registry.execute()

        tm.that(outcome.success, truth=True)
        tm.that(outcome.value is True, truth=True)

    @staticmethod
    def test_register_handler_returns_registration_details(
        registry: p.Registry,
    ) -> None:
        """Test register handler returns registration details."""
        handler = u.Tests.Handler()
        registration = registry.register_handler(handler)

        tm.that(registration.success, truth=True)
        details = registration.value
        tm.that(details.registration_id, truth=True)
        tm.that(
            details.status,
            eq=m.RegistrationDetails(registration_id=details.registration_id).status,
        )
        tm.that(details.handler_mode, eq=handler.mode)

    @staticmethod
    def test_register_handler_propagates_dispatcher_failure(
        registry: p.Registry,
    ) -> None:
        """Test register handler propagates dispatcher failure."""

        def unroutable(message: p.Routable) -> None:
            _ = message

        registration = registry.register_handler(unroutable)

        tm.that(registration.failure, truth=True)
        tm.that(
            c.ERR_HANDLER_ROUTE_DISCOVERY_REQUIRED in (registration.error or ""),
            truth=True,
        )

    @staticmethod
    @pytest.mark.parametrize("kind", ["handlers", "bindings"])
    def test_register_batch_reports_every_success(
        kind: str,
        registry: p.Registry,
    ) -> None:
        """Handler and binding batches report every success and route each one."""

        def first(_message: p.Routable) -> p.Result[t.JsonPayload]:
            return r[t.JsonPayload].ok("first")

        def second(_message: p.Routable) -> p.Result[t.JsonPayload]:
            return r[t.JsonPayload].ok("second")

        if kind == "handlers":
            batch = registry.register_handlers((
                h.handler(command=m.Command)(first),
                h.handler(command=m.Tests.RegistryQuery)(second),
            ))
            routes = (
                (m.Command(), "first"),
                (m.Tests.RegistryQuery(), "second"),
            )
        else:
            batch = registry.register_bindings({
                "bound-first": first,
                "bound-second": second,
            })
            routes = (
                (m.Command(command_type="bound-first"), "first"),
                (m.Command(command_type="bound-second"), "second"),
            )

        tm.that(batch.success, truth=True)
        summary = batch.value
        tm.that(summary.success is True, truth=True)
        tm.that(summary.failure is False, truth=True)
        tm.that(len(summary.registered), eq=2)
        tm.that(list(summary.errors), eq=[])
        for message, expected in routes:
            tm.that(registry.dispatcher.dispatch(message).value, eq=expected)

    @staticmethod
    def test_register_service_is_idempotent(registry: p.Registry) -> None:
        """Test register service is idempotent."""
        first = registry.register("service-name", "service-value")
        duplicate = registry.register("service-name", "service-value")

        tm.that(first.success, truth=True)
        tm.that(duplicate.success, truth=True)

    @staticmethod
    def test_instance_plugin_roundtrips_then_unregisters(
        registry: p.Registry,
    ) -> None:
        """Test instance plugin roundtrips then unregisters."""
        tm.that(
            registry.register_plugin("validators", "local", "plugin").success,
            truth=True,
        )

        tm.that(registry.fetch_plugin("validators", "local").value, eq="plugin")
        tm.that(list(registry.list_plugins("validators").value), eq=["local"])

        tm.that(registry.unregister_plugin("validators", "local").success, truth=True)
        tm.that(registry.fetch_plugin("validators", "local").failure, truth=True)
        tm.that(
            registry.register_plugin("validators", "local", "replacement").success,
            truth=True,
        )
        tm.that(registry.fetch_plugin("validators", "local").value, eq="replacement")

    @staticmethod
    @pytest.mark.parametrize("scope", tuple(c.RegistrationScope))
    def test_plugin_identity_and_conflicts(scope: c.RegistrationScope) -> None:
        """Models remain live collaborators, not dumps or equality-based aliases."""
        registry = u.build_registry()
        plugin = m.ConfigMap(root={"plugin": "original"})
        equal_but_distinct = m.ConfigMap(root={"plugin": "original"})
        try:
            tm.that(
                registry.register_plugin(
                    "identity", "payload", plugin, scope=scope
                ).success,
                truth=True,
            )
            tm.that(
                registry.register_plugin(
                    "identity", "payload", plugin, scope=scope
                ).success,
                truth=True,
            )
            tm.that(
                registry.register_plugin(
                    "identity", "payload", equal_but_distinct, scope=scope
                ).failure,
                truth=True,
            )
            tm.that(
                registry.fetch_plugin("identity", "payload", scope=scope).value,
                where=lambda value: value is plugin,
            )
        finally:
            tm.that(
                registry.unregister_plugin("identity", "payload", scope=scope).success,
                truth=True,
            )

    @staticmethod
    @pytest.mark.parametrize("scope", tuple(c.RegistrationScope))
    def test_bound_plugin_method_is_idempotent(scope: c.RegistrationScope) -> None:
        """Repeated bound-method access preserves the same receiver and function."""
        registry = u.build_registry()
        payload = m.ConfigMap(root={})
        original = payload.model_dump
        try:
            tm.that(
                registry.register_plugin(
                    "bound", "method", original, scope=scope
                ).success,
                truth=True,
            )
            tm.that(
                registry.register_plugin(
                    "bound", "method", payload.model_dump, scope=scope
                ).success,
                truth=True,
            )
            tm.that(
                registry.fetch_plugin("bound", "method", scope=scope).value,
                where=lambda value: value is original,
            )
        finally:
            tm.that(
                registry.unregister_plugin("bound", "method", scope=scope).success,
                truth=True,
            )

    @staticmethod
    def test_instance_plugins_do_not_leak_between_compositions() -> None:
        """Two independently composed APIs do not inherit another registry's plugins."""
        first, second = u.build_registry(), u.build_registry()
        tm.that(first.register_plugin("isolated", "payload", "one").success, truth=True)
        tm.that(second.fetch_plugin("isolated", "payload").failure, truth=True)

    @staticmethod
    def test_conflicting_batch_does_not_publish_earlier_binding(
        registry: p.Registry,
    ) -> None:
        """A conflicting batch fails; publish semantics live in the dispatcher suite."""

        def original(_message: p.Routable) -> p.Result[t.JsonPayload]:
            return r[t.JsonPayload].ok("original")

        def replacement(_message: p.Routable) -> p.Result[t.JsonPayload]:
            return r[t.JsonPayload].ok("replacement")

        tm.that(
            registry.register_bindings({"conflicting-batch": original}).success,
            truth=True,
        )
        tm.that(
            registry.register_bindings({
                "new-route": replacement,
                "conflicting-batch": replacement,
            }).failure,
            truth=True,
        )

    @staticmethod
    def test_class_scope_plugin_is_visible_across_instances() -> None:
        """Test class scope plugin is visible across instances."""
        writer = u.build_registry(dispatcher=u.build_dispatcher())
        reader = u.build_registry(dispatcher=u.build_dispatcher())

        registration = writer.register_plugin(
            "validators",
            "shared",
            "plugin",
            scope=c.RegistrationScope.CLASS,
        )
        tm.that(registration.success, truth=True)

        fetched = reader.fetch_plugin(
            "validators",
            "shared",
            scope=c.RegistrationScope.CLASS,
        )
        tm.that(fetched.value, eq="plugin")

        tm.that(
            reader.unregister_plugin(
                "validators", "shared", scope=c.RegistrationScope.CLASS
            ).success,
            truth=True,
        )
        tm.that(
            writer.fetch_plugin(
                "validators", "shared", scope=c.RegistrationScope.CLASS
            ).failure,
            truth=True,
        )

    @staticmethod
    def test_register_plugin_rejects_empty_name(registry: p.Registry) -> None:
        """Test register plugin rejects empty name."""
        result = registry.register_plugin("validators", "", "plugin")

        tm.that(result.failure, truth=True)
        tm.that(result.error, truth=True)

    @staticmethod
    def test_fetch_unknown_plugin_fails(registry: p.Registry) -> None:
        """Test fetch unknown plugin fails."""
        tm.that(registry.fetch_plugin("validators", "absent").failure, truth=True)

    @staticmethod
    def test_unregister_unknown_plugin_fails(registry: p.Registry) -> None:
        """Test unregister unknown plugin fails."""
        tm.that(registry.unregister_plugin("validators", "absent").failure, truth=True)

    @staticmethod
    @pytest.mark.parametrize(
        ("errors", "expected_success"),
        [((), True), (("boom",), False)],
    )
    def test_summary_success_reflects_error_state(
        errors: t.VariadicTuple[str],
        *,
        expected_success: bool,
    ) -> None:
        """Test summary success reflects error state."""
        detail = m.RegistrationDetails(
            registration_id="handler-a",
            handler_mode=c.HandlerType.COMMAND,
            status=c.Status.ACTIVE,
        )
        summary = m.RegistrySummary(registered=[detail], errors=list(errors))

        tm.that(summary.success is expected_success, truth=True)
        tm.that(summary.failure is (not expected_success), truth=True)
        tm.that(len(summary.registered), eq=1)
