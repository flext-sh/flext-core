"""Container resolution and listing tests.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from tests import m, p, tm, u


class TestsFlextContainerResolution:
    """Exercise public service resolution and discovery behavior."""

    @staticmethod
    def test_get_factory_called_each_time(clean_container: p.Container) -> None:
        """A factory is lazy at registration and runs on each resolve."""
        factory, get_count = u.Tests.create_counting_factory("service_value")
        clean_container.factory("factory_service", factory)
        tm.that(get_count(), eq=0, msg="Factory must not run before first resolve")
        result1: p.Result[p.RegisterableService] = clean_container.resolve(
            "factory_service",
        )
        _ = u.Tests.assert_success(result1)
        tm.that(get_count(), eq=1, msg="Factory must be called once after first get()")
        result2: p.Result[p.RegisterableService] = clean_container.resolve(
            "factory_service",
        )
        _ = u.Tests.assert_success(result2)
        tm.that(
            get_count(),
            eq=2,
            msg="Factory must be called twice after second get()",
        )

    @staticmethod
    @pytest.mark.parametrize(
        "scenario",
        m.Tests.ContainerScenarios.TYPED_RETRIEVAL_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_get_typed_correct(
        scenario: m.Tests.TypedRetrievalScenario,
        clean_container: p.Container,
    ) -> None:
        """Test typed retrieval with correct types using fixtures."""
        container = clean_container
        _ = container.bind(scenario.name, scenario.service)
        if scenario.should_pass:
            resolved_service: str | int = u.Tests.assert_success(
                container.resolve(scenario.name, type_cls=scenario.expected_type),
            )
            tm.that(
                str(resolved_service),
                eq=str(scenario.service),
                msg=f"Typed result value must match service for {scenario.name}",
            )
            tm.that(
                isinstance(resolved_service, scenario.expected_type),
                eq=True,
                msg=(
                    f"Typed result must be instance of "
                    f"{scenario.expected_type.__name__}"
                ),
            )
        else:
            _ = u.Tests.assert_failure(
                container.resolve(scenario.name, type_cls=scenario.expected_type),
            )

    @staticmethod
    def test_get_typed_wrong_type(clean_container: p.Container) -> None:
        """Test typed retrieval with wrong type fails using fixtures."""
        clean_container.bind("string_service", "test_value")
        result = clean_container.resolve("string_service", type_cls=int)
        _ = u.Tests.assert_failure(result)

    @staticmethod
    def test_get_typed_nonexistent(clean_container: p.Container) -> None:
        """Test typed retrieval of non-existent service using fixtures."""
        result = clean_container.resolve("nonexistent", type_cls=int)
        u.Tests.assert_failure(result, expected_error="not found")

    @staticmethod
    @pytest.mark.parametrize(
        ("has_service", "expected"),
        [(True, True), (False, False)],
        ids=["exists", "not_exists"],
    )
    def test_has_service(
        *,
        has_service: bool,
        expected: bool,
        clean_container: p.Container,
    ) -> None:
        """Test has_service returns correct value using fixtures."""
        container = clean_container
        service_name = "test_service" if has_service else "nonexistent"
        if has_service:
            _ = container.bind(service_name, "value")
        tm.that(
            container.has(service_name),
            eq=expected,
            msg=f"has_service must return {expected} for {service_name}",
        )

    @staticmethod
    def test_has_service_factory(clean_container: p.Container) -> None:
        """Test has_service returns True for factories using fixtures."""
        container = clean_container
        factory = u.Tests.create_factory("value")
        _ = container.factory("factory_service", factory)
        tm.that(
            container.has("factory_service"),
            eq=True,
            msg="Container must have factory_service after registration",
        )

    @staticmethod
    def test_list_services_empty(clean_container: p.Container) -> None:
        """Test listing services when none registered using fixtures."""
        container = clean_container
        services = container.names()
        tm.that(services, is_=list, msg="list_services must return a list")
        tm.that(
            len(services),
            eq=0,
            msg="Empty container must return empty services list",
        )
        tm.that(
            services,
            empty=True,
            msg="Empty container must have empty services list",
        )

    @staticmethod
    def test_list_services_mixed(clean_container: p.Container) -> None:
        """Test listing mix of registered services and factories using fixtures."""
        container = clean_container
        _ = container.bind("service1", "value1")
        _ = container.bind("service2", "value2")
        factory = u.Tests.create_factory("value3")
        _ = container.factory("factory1", factory)
        services = container.names()
        tm.that(len(services), eq=3, msg="Container must list 3 registered services")
        required_keys = ["service1", "service2", "factory1"]
        for key in required_keys:
            tm.that(key in services, eq=True, msg=f"Services list must contain {key}")
