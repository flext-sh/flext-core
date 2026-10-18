"""Behavioral contract tests for the public FlextRegistry runtime surface.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from flext_core import FlextRegistry
from tests import c, h, m, r, t, u

if TYPE_CHECKING:
    from tests import p


class TestsFlextCoreRegistry:
    """Assert observable behavior of the registry public API."""

    @staticmethod
    def test_registry_consumes_one_validated_runtime() -> None:
        """Registration and plugin storage share the explicit composition graph."""
        runtime = u.build_service_runtime()
        registry = FlextRegistry(registry_runtime=runtime)
        assert registry.dispatcher is runtime.dispatcher
        assert registry.container is runtime.container
        assert "registry_runtime" not in registry.model_dump()
        assert "registry_runtime" not in registry.model_json_schema().get(
            "properties", {}
        )

    @staticmethod
    @pytest.mark.parametrize("payload", [{}, {"registry_runtime": "invalid"}])
    def test_registry_rejects_missing_or_invalid_runtime(payload: t.JsonDict) -> None:
        """A required port is never bootstrapped or silently replaced on invalid ingress."""
        with pytest.raises((TypeError, ValueError)):
            FlextRegistry.model_validate(payload)

    @staticmethod
    def test_registry_validates_runtime_assignment() -> None:
        """Assignment cannot invalidate the composed graph after construction."""
        runtime = u.build_service_runtime()
        registry = FlextRegistry(registry_runtime=runtime)
        with pytest.raises((TypeError, ValueError)):
            registry.registry_runtime = "invalid"
        assert registry.registry_runtime is runtime

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

        assert outcome.success
        assert outcome.value is True

    @staticmethod
    def test_register_handler_returns_registration_details(
        registry: p.Registry,
    ) -> None:
        """Test register handler returns registration details."""
        handler = u.Tests.Handler()
        registration = registry.register_handler(handler)

        assert registration.success
        details = registration.value
        assert details.registration_id
        assert (
            details.status
            == m.RegistrationDetails(registration_id=details.registration_id).status
        )
        assert details.handler_mode == handler.mode

    @staticmethod
    def test_register_handler_propagates_dispatcher_failure(
        registry: p.Registry,
    ) -> None:
        """Test register handler propagates dispatcher failure."""

        def unroutable(message: p.Routable) -> None:
            _ = message

        registration = registry.register_handler(unroutable)

        assert registration.failure
        assert c.ERR_HANDLER_ROUTE_DISCOVERY_REQUIRED in (registration.error or "")

    @staticmethod
    def test_register_handlers_batch_reports_every_success(
        registry: p.Registry,
    ) -> None:
        """Test register handlers batch reports every success."""

        @h.handler(command=m.Command)
        def first(_message: p.Routable) -> p.Result[t.JsonPayload]:
            return r[t.JsonPayload].ok("first")

        @h.handler(command=m.Tests.RegistryQuery)
        def second(_message: p.Routable) -> p.Result[t.JsonPayload]:
            return r[t.JsonPayload].ok("second")

        batch = registry.register_handlers((first, second))

        assert batch.success
        summary = batch.value
        assert summary.success is True
        assert summary.failure is False
        assert len(summary.registered) == 2
        assert list(summary.errors) == []
        assert registry.dispatcher.dispatch(m.Command()).value == "first"
        assert registry.dispatcher.dispatch(m.Tests.RegistryQuery()).value == "second"

    @staticmethod
    def test_register_bindings_batch_reports_every_success(
        registry: p.Registry,
    ) -> None:
        """Test register bindings batch reports every success."""

        def first(_message: p.Routable) -> p.Result[t.JsonPayload]:
            return r[t.JsonPayload].ok("first")

        def second(_message: p.Routable) -> p.Result[t.JsonPayload]:
            return r[t.JsonPayload].ok("second")

        batch = registry.register_bindings({
            "bound-first": first,
            "bound-second": second,
        })

        assert batch.success
        summary = batch.value
        assert summary.success is True
        assert len(summary.registered) == 2
        assert list(summary.errors) == []
        assert (
            registry.dispatcher.dispatch(m.Command(command_type="bound-first")).value
            == "first"
        )
        assert (
            registry.dispatcher.dispatch(m.Command(command_type="bound-second")).value
            == "second"
        )

    @staticmethod
    def test_register_service_is_idempotent(registry: p.Registry) -> None:
        """Test register service is idempotent."""
        first = registry.register("service-name", "service-value")
        duplicate = registry.register("service-name", "service-value")

        assert first.success
        assert duplicate.success

    @staticmethod
    def test_instance_plugin_roundtrips_then_unregisters(
        registry: p.Registry,
    ) -> None:
        """Test instance plugin roundtrips then unregisters."""
        assert registry.register_plugin("validators", "local", "plugin").success

        assert registry.fetch_plugin("validators", "local").value == "plugin"
        assert list(registry.list_plugins("validators").value) == ["local"]

        assert registry.unregister_plugin("validators", "local").success
        assert registry.fetch_plugin("validators", "local").failure
        assert registry.register_plugin("validators", "local", "replacement").success
        assert registry.fetch_plugin("validators", "local").value == "replacement"

    @staticmethod
    @pytest.mark.parametrize("scope", tuple(c.RegistrationScope))
    def test_plugin_identity_and_conflicts(scope: c.RegistrationScope) -> None:
        """Models remain live collaborators, not dumps or equality-based aliases."""
        registry = u.build_registry()
        plugin = m.ConfigMap(root={"plugin": "original"})
        equal_but_distinct = m.ConfigMap(root={"plugin": "original"})
        try:
            assert registry.register_plugin(
                "identity", "payload", plugin, scope=scope
            ).success
            assert registry.register_plugin(
                "identity", "payload", plugin, scope=scope
            ).success
            assert registry.register_plugin(
                "identity", "payload", equal_but_distinct, scope=scope
            ).failure
            assert (
                registry.fetch_plugin("identity", "payload", scope=scope).value
                is plugin
            )
        finally:
            assert registry.unregister_plugin(
                "identity", "payload", scope=scope
            ).success

    @staticmethod
    @pytest.mark.parametrize("scope", tuple(c.RegistrationScope))
    def test_bound_plugin_method_is_idempotent(scope: c.RegistrationScope) -> None:
        """Repeated bound-method access preserves the same receiver and function."""
        registry = u.build_registry()
        payload = m.ConfigMap(root={})
        original = payload.model_dump
        try:
            assert registry.register_plugin(
                "bound", "method", original, scope=scope
            ).success
            assert registry.register_plugin(
                "bound", "method", payload.model_dump, scope=scope
            ).success
            assert (
                registry.fetch_plugin("bound", "method", scope=scope).value is original
            )
        finally:
            assert registry.unregister_plugin("bound", "method", scope=scope).success

    @staticmethod
    def test_instance_plugins_do_not_leak_between_compositions() -> None:
        """Two independently composed APIs do not inherit another registry's plugins."""
        first, second = u.build_registry(), u.build_registry()
        assert first.register_plugin("isolated", "payload", "one").success
        assert second.fetch_plugin("isolated", "payload").failure

    @staticmethod
    def test_conflicting_batch_does_not_publish_earlier_binding(
        registry: p.Registry,
    ) -> None:
        """A conflict in the last candidate cannot leave earlier routes registered."""

        def original(_message: p.Routable) -> p.Result[t.JsonPayload]:
            return r[t.JsonPayload].ok("original")

        def replacement(_message: p.Routable) -> p.Result[t.JsonPayload]:
            return r[t.JsonPayload].ok("replacement")

        assert registry.register_bindings({"conflicting-batch": original}).success
        assert registry.register_bindings({
            "new-route": replacement,
            "conflicting-batch": replacement,
        }).failure
        assert registry.dispatcher.dispatch(m.Command(command_type="new-route")).failure
        assert (
            registry.dispatcher.dispatch(
                m.Command(command_type="conflicting-batch")
            ).value
            == "original"
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
        assert registration.success

        fetched = reader.fetch_plugin(
            "validators",
            "shared",
            scope=c.RegistrationScope.CLASS,
        )
        assert fetched.value == "plugin"

        assert reader.unregister_plugin(
            "validators",
            "shared",
            scope=c.RegistrationScope.CLASS,
        ).success
        assert writer.fetch_plugin(
            "validators",
            "shared",
            scope=c.RegistrationScope.CLASS,
        ).failure

    @staticmethod
    def test_register_plugin_rejects_empty_name(registry: p.Registry) -> None:
        """Test register plugin rejects empty name."""
        result = registry.register_plugin("validators", "", "plugin")

        assert result.failure
        assert result.error

    @staticmethod
    def test_fetch_unknown_plugin_fails(registry: p.Registry) -> None:
        """Test fetch unknown plugin fails."""
        assert registry.fetch_plugin("validators", "absent").failure

    @staticmethod
    def test_unregister_unknown_plugin_fails(registry: p.Registry) -> None:
        """Test unregister unknown plugin fails."""
        assert registry.unregister_plugin("validators", "absent").failure

    @pytest.mark.parametrize(
        ("errors", "expected_success"),
        [((), True), (("boom",), False)],
    )
    @staticmethod
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

        assert summary.success is expected_success
        assert summary.failure is (not expected_success)
        assert len(summary.registered) == 1
