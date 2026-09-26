"""Behavioral contract tests for the public FlextRegistry runtime surface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.constants import c
from tests.models import m
from tests.utilities import u

if TYPE_CHECKING:
    from tests.protocols import p
    from tests.typings import t


class TestsFlextCoreRegistry:
    """Assert observable behavior of the registry public API."""

    @pytest.fixture
    def registry(self) -> p.Registry:
        """Build a registry backed by an accepting dispatcher."""
        return u.build_registry(dispatcher=u.build_dispatcher())

    def test_execute_succeeds_when_dispatcher_present(self) -> None:
        registry = u.build_registry(dispatcher=u.build_dispatcher())

        outcome = registry.execute()

        assert outcome.success
        assert outcome.value is True

    def test_register_handler_returns_registration_details(
        self, registry: p.Registry
    ) -> None:
        registration = registry.register_handler(u.Tests.Handler())

        assert registration.success
        details = registration.value
        assert details.registration_id != ""
        assert details.status == c.Status.ACTIVE
        assert details.handler_mode == c.HandlerType.COMMAND

    def test_register_handler_propagates_dispatcher_failure(
        self, registry: p.Registry
    ) -> None:
        def unroutable(message: p.Routable) -> None:
            _ = message

        registration = registry.register_handler(unroutable)

        assert registration.failure
        assert c.ERR_HANDLER_ROUTE_DISCOVERY_REQUIRED in (registration.error or "")

    def test_register_handlers_batch_reports_every_success(
        self, registry: p.Registry
    ) -> None:
        batch = registry.register_handlers([u.Tests.Handler(), u.Tests.Handler()])

        assert batch.success
        summary = batch.value
        assert summary.success is True
        assert summary.failure is False
        assert len(summary.registered) == 2
        assert list(summary.errors) == []

    def test_register_bindings_batch_reports_every_success(
        self, registry: p.Registry
    ) -> None:
        batch = registry.register_bindings({
            str: u.Tests.Handler(),
            int: u.Tests.Handler(),
        })

        assert batch.success
        summary = batch.value
        assert summary.success is True
        assert len(summary.registered) == 2
        assert list(summary.errors) == []

    def test_register_service_is_idempotent(self, registry: p.Registry) -> None:
        first = registry.register("service-name", "service-value")
        duplicate = registry.register("service-name", "service-value")

        assert first.success
        assert duplicate.success

    def test_instance_plugin_roundtrips_then_unregisters(
        self, registry: p.Registry
    ) -> None:
        assert registry.register_plugin("validators", "local", "plugin").success

        assert registry.fetch_plugin("validators", "local").value == "plugin"
        assert list(registry.list_plugins("validators").value) == ["local"]

        assert registry.unregister_plugin("validators", "local").success
        assert registry.fetch_plugin("validators", "local").failure

    def test_class_scope_plugin_is_visible_across_instances(self) -> None:
        writer = u.build_registry(dispatcher=u.build_dispatcher())
        reader = u.build_registry(dispatcher=u.build_dispatcher())

        registration = writer.register_plugin(
            "validators", "shared", "plugin", scope=c.RegistrationScope.CLASS
        )
        assert registration.success

        fetched = reader.fetch_plugin(
            "validators", "shared", scope=c.RegistrationScope.CLASS
        )
        assert fetched.value == "plugin"

        assert reader.unregister_plugin(
            "validators", "shared", scope=c.RegistrationScope.CLASS
        ).success
        assert writer.fetch_plugin(
            "validators", "shared", scope=c.RegistrationScope.CLASS
        ).failure

    def test_register_plugin_rejects_empty_name(self, registry: p.Registry) -> None:
        result = registry.register_plugin("validators", "", "plugin")

        assert result.failure
        assert result.error

    def test_fetch_unknown_plugin_fails(self, registry: p.Registry) -> None:
        assert registry.fetch_plugin("validators", "absent").failure

    def test_unregister_unknown_plugin_fails(self, registry: p.Registry) -> None:
        assert registry.unregister_plugin("validators", "absent").failure

    @pytest.mark.parametrize(
        ("errors", "expected_success"), [((), True), (("boom",), False)]
    )
    def test_summary_success_reflects_error_state(
        self, errors: t.VariadicTuple[str], *, expected_success: bool
    ) -> None:
        detail = m.RegistrationDetails(
            registration_id="handler-a",
            handler_mode=c.HandlerType.COMMAND,
            status=c.Status.ACTIVE,
        )
        summary = m.RegistrySummary(registered=[detail], errors=list(errors))

        assert summary.success is expected_success
        assert summary.failure is (not expected_success)
        assert len(summary.registered) == 1


__all__: t.MutableSequenceOf[str] = ["TestsFlextCoreRegistry"]
