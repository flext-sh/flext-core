"""Behavioral contract tests for the public FlextRegistry runtime surface.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests import c, m
from tests.utilities import u

if TYPE_CHECKING:
    from tests import p, t


class TestsFlextCoreRegistry:
    """Assert observable behavior of the registry public API."""

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
        registration = registry.register_handler(u.Tests.Handler())

        assert registration.success
        details = registration.value
        assert details.registration_id
        assert details.status == c.Status.ACTIVE
        assert details.handler_mode == c.HandlerType.COMMAND

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
        batch = registry.register_handlers([u.Tests.Handler(), u.Tests.Handler()])

        assert batch.success
        summary = batch.value
        assert summary.success is True
        assert summary.failure is False
        assert len(summary.registered) == 2
        assert list(summary.errors) == []

    @staticmethod
    def test_register_bindings_batch_reports_every_success(
        registry: p.Registry,
    ) -> None:
        """Test register bindings batch reports every success."""
        batch = registry.register_bindings({
            str: u.Tests.Handler(),
            int: u.Tests.Handler(),
        })

        assert batch.success
        summary = batch.value
        assert summary.success is True
        assert len(summary.registered) == 2
        assert list(summary.errors) == []

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
