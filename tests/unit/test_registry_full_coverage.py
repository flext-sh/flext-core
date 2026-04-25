"""Behavior tests for registry DSL public runtime flows."""

from __future__ import annotations

from tests import c, m, u


class TestsFlextCoreRegistryFullCoverage:
    def test_registry_summary_reports_public_success_state(self) -> None:
        detail = m.RegistrationDetails(
            registration_id="handler-a",
            handler_mode=c.HandlerType.COMMAND,
            status=c.Status.ACTIVE,
        )
        summary = m.RegistrySummary(registered=[detail])
        success_attr = summary.success
        failure_attr = summary.failure
        success_value = success_attr() if callable(success_attr) else bool(success_attr)
        failure_value = failure_attr() if callable(failure_attr) else bool(failure_attr)
        assert success_value
        assert not failure_value
        assert len(summary.registered) == 1

    def test_registry_runtime_exposes_dispatcher_from_central_dsl(self) -> None:
        registry = u.build_registry()
        assert registry.execute().success
        registration = registry.register_handler(u.Core.Tests.Handler())
        assert registration.success
        assert registration.value is not None

    def test_registry_respects_explicit_dispatcher_behavior(self) -> None:
        unavailable_registry = u.build_registry(
            dispatcher=u.Core.Tests.FalseyDispatcher()
        )
        assert unavailable_registry.execute().success

        failing_registry = u.build_registry(dispatcher=u.Core.Tests.FailDispatcher())
        failed_registration = failing_registry.register_handler(u.Core.Tests.Handler())
        assert failed_registration.failure
        assert "dispatcher-fail" in (failed_registration.error or "")

        ready_registry = u.build_registry(dispatcher=u.Core.Tests.OkDispatcher())
        successful_registration = ready_registry.register_handler(
            u.Core.Tests.Handler()
        )
        assert successful_registration.success
        assert successful_registration.value.registration_id != ""
        assert successful_registration.value.status == c.Status.ACTIVE

    def test_registry_registers_batches_through_public_methods(self) -> None:
        registry = u.build_registry(dispatcher=u.Core.Tests.OkDispatcher())

        handler_batch = registry.register_handlers([
            u.Core.Tests.Handler(),
            u.Core.Tests.Handler(),
        ])
        assert handler_batch.success
        assert len(handler_batch.value.registered) == 2
        assert handler_batch.value.errors == []

        bindings_batch = registry.register_bindings({
            str: u.Core.Tests.Handler(),
            int: u.Core.Tests.Handler(),
        })
        assert bindings_batch.success
        assert len(bindings_batch.value.registered) == 2
        assert bindings_batch.value.errors == []

    def test_registry_plugin_scopes_roundtrip_via_public_api(self) -> None:
        registry = u.build_registry(dispatcher=u.Core.Tests.OkDispatcher())

        instance_registration = registry.register_plugin(
            "validators", "local", "plugin"
        )
        assert instance_registration.success
        assert registry.fetch_plugin("validators", "local").value == "plugin"
        assert registry.list_plugins("validators").value == ["local"]
        assert registry.unregister_plugin("validators", "local").success
        assert registry.fetch_plugin("validators", "local").failure

        registry_a = u.build_registry(dispatcher=u.Core.Tests.OkDispatcher())
        registry_b = u.build_registry(dispatcher=u.Core.Tests.OkDispatcher())
        class_registration = registry_a.register_plugin(
            "validators",
            "shared",
            "plugin",
            scope=c.RegistrationScope.CLASS,
        )
        assert class_registration.success
        assert (
            registry_b.fetch_plugin(
                "validators",
                "shared",
                scope=c.RegistrationScope.CLASS,
            ).value
            == "plugin"
        )
        assert registry_b.unregister_plugin(
            "validators",
            "shared",
            scope=c.RegistrationScope.CLASS,
        ).success
        assert registry_a.fetch_plugin(
            "validators",
            "shared",
            scope=c.RegistrationScope.CLASS,
        ).failure

    def test_registry_register_public_service(self) -> None:
        registry = u.build_registry(dispatcher=u.Core.Tests.OkDispatcher())
        registration = registry.register("service-name", "service-value")
        assert registration.success
        duplicate_registration = registry.register("service-name", "service-value")
        assert duplicate_registration.success
