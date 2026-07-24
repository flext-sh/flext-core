"""Behavioral tests for FlextContainer registration and resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from flext_core.container import FlextContainer
from flext_tests import tm
from tests.models import m
from tests.protocols import p
from tests.utilities import u

if TYPE_CHECKING:
    from tests.typings import t


class TestsFlextCoreContainerRegistration:
    """Exercise public service registration and removal behavior."""

    def test_fresh_container_exposes_no_public_services(
        self, clean_container: p.Container
    ) -> None:
        """A fresh container starts with no publicly registered names."""
        tm.that(
            clean_container.names(),
            empty=True,
            msg="Fresh container must expose no registered services",
        )

    def test_container_is_process_singleton(self) -> None:
        """FlextContainer() always yields the one shared singleton instance."""
        first = FlextContainer()
        second = FlextContainer()
        tm.that(
            first is second,
            eq=True,
            msg="FlextContainer must return the same singleton instance",
        )

    @pytest.mark.parametrize(
        "scenario", m.Tests.ContainerScenarios.SERVICE_SCENARIOS, ids=lambda s: s.name
    )
    def test_bind_registers_and_resolves_service(
        self, scenario: m.Tests.ServiceScenario, clean_container: p.Container
    ) -> None:
        """Bind makes a service discoverable and resolvable to its value."""
        result = clean_container.bind(scenario.name, scenario.service)

        tm.that(
            result is clean_container,
            eq=True,
            msg="bind must return the container for fluent chaining",
        )
        tm.that(
            clean_container.has(scenario.name),
            eq=True,
            msg=f"Container must report {scenario.name} as registered",
        )
        tm.that(
            scenario.name in clean_container.names(),
            eq=True,
            msg=f"names() must list {scenario.name} after bind",
        )
        u.Tests.assert_success(
            clean_container.resolve(scenario.name), expected_value=scenario.service
        )

    def test_bind_duplicate_name_preserves_first_binding(
        self, clean_container: p.Container
    ) -> None:
        """Re-binding an existing name is a no-op; the first value wins."""
        _ = clean_container.bind("service1", "value1")
        _ = clean_container.bind("service1", "value2")

        u.Tests.assert_success(
            clean_container.resolve("service1"), expected_value="value1"
        )

    def test_bind_empty_name_is_rejected(self, clean_container: p.Container) -> None:
        """An empty service name is never registered."""
        result = clean_container.bind("", "service")

        tm.that(
            result is clean_container,
            eq=True,
            msg="bind must stay fluent even when rejecting an empty name",
        )
        tm.that(
            clean_container.has(""),
            eq=False,
            msg="Empty-named service must not be registered",
        )

    def test_resolve_unknown_name_fails(self, clean_container: p.Container) -> None:
        """Resolving an unregistered name yields a failing result naming it."""
        u.Tests.assert_failure(
            clean_container.resolve("missing_service"), expected_error="missing_service"
        )

    @pytest.mark.parametrize(
        "return_value", ["created_string", 42], ids=["string", "int"]
    )
    def test_factory_registers_and_produces_value(
        self, return_value: t.JsonValue, clean_container: p.Container
    ) -> None:
        """Factory registers a callable whose result is returned on resolve."""
        factory = u.Tests.create_factory(return_value)
        name = f"factory_{type(return_value).__name__}"

        result = clean_container.factory(name, factory)

        tm.that(
            result is clean_container,
            eq=True,
            msg="factory must return the container for fluent chaining",
        )
        tm.that(
            clean_container.has(name),
            eq=True,
            msg=f"Container must report factory {name} as registered",
        )
        u.Tests.assert_success(
            clean_container.resolve(name), expected_value=return_value
        )

    def test_factory_is_invoked_lazily_on_resolve(
        self, clean_container: p.Container
    ) -> None:
        """A factory is not called at registration, only when resolved."""
        factory, call_count = u.Tests.create_counting_factory("lazy_value")
        clean_container.factory("lazy", factory)

        tm.that(
            call_count(),
            eq=0,
            msg="Factory must not run before the service is resolved",
        )
        u.Tests.assert_success(
            clean_container.resolve("lazy"), expected_value="lazy_value"
        )
        tm.that(
            call_count() >= 1,
            eq=True,
            msg="Factory must be invoked to produce the resolved value",
        )

    def test_factory_duplicate_name_preserves_first_factory(
        self, clean_container: p.Container
    ) -> None:
        """Re-registering a factory name keeps the original callable."""
        clean_container.factory("factory1", u.Tests.create_factory("value1"))
        clean_container.factory("factory1", u.Tests.create_factory("value2"))

        u.Tests.assert_success(
            clean_container.resolve("factory1"), expected_value="value1"
        )

    def test_factory_empty_name_is_rejected(self, clean_container: p.Container) -> None:
        """An empty factory name is never registered."""
        result = clean_container.factory("", u.Tests.create_factory("value"))

        tm.that(
            clean_container.has(""),
            eq=False,
            msg="Empty-named factory must not be registered",
        )
        tm.that(
            result is clean_container,
            eq=True,
            msg="factory must stay fluent when rejecting an empty name",
        )

    def test_drop_removes_registered_service(
        self, clean_container: p.Container
    ) -> None:
        """Drop removes a service and reports success; it becomes unresolvable."""
        clean_container.bind("temp", "temp_value")

        u.Tests.assert_success(clean_container.drop("temp"), expected_value=True)

        tm.that(
            clean_container.has("temp"),
            eq=False,
            msg="Dropped service must no longer be registered",
        )
        u.Tests.assert_failure(clean_container.resolve("temp"), expected_error="temp")

    def test_drop_unknown_name_fails(self, clean_container: p.Container) -> None:
        """Dropping an unregistered name yields a failing result."""
        u.Tests.assert_failure(
            clean_container.drop("never_registered"), expected_error="never_registered"
        )
