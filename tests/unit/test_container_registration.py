"""Behavioral tests for FlextContainer registration and resolution.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from flext_core import FlextContainer
from tests import c, e, m, p, t, tm, u


class TestsFlextCoreContainerRegistration:
    """Exercise public service registration and removal behavior."""

    @staticmethod
    def test_fresh_container_exposes_no_public_services(
        clean_container: p.Container,
    ) -> None:
        """A fresh container starts with no publicly registered names."""
        tm.that(
            clean_container.names(),
            empty=True,
            msg="Fresh container must expose no registered services",
        )

    @staticmethod
    def test_container_is_process_singleton() -> None:
        """FlextContainer() always yields the one shared singleton instance."""
        first = FlextContainer()
        second = FlextContainer()
        tm.that(
            first is second,
            eq=True,
            msg="FlextContainer must return the same singleton instance",
        )

    @staticmethod
    def test_container_satisfies_container_protocol() -> None:
        """A default FlextContainer honors the public Container protocol."""
        tm.that(FlextContainer(), is_=p.Container)

    @staticmethod
    @pytest.mark.parametrize(
        "scenario",
        m.Tests.ContainerScenarios.SERVICE_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_bind_registers_and_resolves_service(
        scenario: m.Tests.ServiceScenario,
        clean_container: p.Container,
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
            clean_container.resolve(scenario.name),
            expected_value=scenario.service,
        )

    @staticmethod
    def test_bind_empty_name_raises(clean_container: p.Container) -> None:
        """An empty service name raises and registers nothing."""
        with pytest.raises(e.ValidationError, match=c.ERR_CONTAINER_NAME_EMPTY):
            _ = clean_container.bind("", "service")

        tm.that(clean_container.names(), empty=True)

    @staticmethod
    def test_resolve_unknown_name_fails(clean_container: p.Container) -> None:
        """Resolving an unregistered name yields a failing result naming it."""
        u.Tests.assert_failure(
            clean_container.resolve("missing_service"),
            expected_error="missing_service",
        )

    @staticmethod
    @pytest.mark.parametrize(
        "return_value",
        ["created_string", 42],
        ids=["string", "int"],
    )
    def test_factory_registers_and_produces_value(
        return_value: t.JsonValue,
        clean_container: p.Container,
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
            clean_container.resolve(name),
            expected_value=return_value,
        )

    @staticmethod
    def test_factory_empty_name_raises(clean_container: p.Container) -> None:
        """An empty factory name raises and registers nothing."""
        with pytest.raises(e.ValidationError, match=c.ERR_CONTAINER_NAME_EMPTY):
            clean_container.factory("", u.Tests.create_factory("value"))

        tm.that(clean_container.names(), empty=True)

    @staticmethod
    def test_drop_removes_registered_service(
        clean_container: p.Container,
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

    @staticmethod
    def test_drop_unknown_name_fails(clean_container: p.Container) -> None:
        """Dropping an unregistered name yields a failing result."""
        u.Tests.assert_failure(
            clean_container.drop("never_registered"),
            expected_error="never_registered",
        )
