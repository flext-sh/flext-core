"""Container resolution and listing tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from flext_tests import tm

from tests import m
from tests import u

if TYPE_CHECKING:
    from tests import p
    from tests import p, t


class TestsFlextContainerResolution:
    """Exercise public service resolution and discovery behavior."""

    @pytest.mark.parametrize(
        "scenario", m.Tests.ContainerScenarios.SERVICE_SCENARIOS, ids=lambda s: s.name
    )
    def test_get_service(
        self, scenario: p.Tests.ServiceScenario, clean_container: p.Container
    ) -> None:
        """Test service retrieval using fixtures."""
        clean_container.bind(scenario.name, scenario.service)
        result: p.Result[t.RegisterableService] = clean_container.resolve(scenario.name)
        u.Tests.assert_success(result, expected_value=scenario.service)

    def test_get_nonexistent_service(self, clean_container: p.Container) -> None:
        """Test getting non-existent service using fixtures."""
        result: p.Result[t.RegisterableService] = clean_container.resolve("nonexistent")
        u.Tests.assert_failure(result, expected_error="not found")

    def test_get_factory_service(self, clean_container: p.Container) -> None:
        """Test retrieving service created by factory using fixtures."""
        factory_result = {"created": "by_factory"}
        factory = u.Tests.create_factory(factory_result)
        clean_container.factory("factory_service", factory)
        result: p.Result[t.RegisterableService] = clean_container.resolve(
            "factory_service"
        )
        u.Tests.assert_success(result, expected_value=factory_result)

    def test_get_factory_called_each_time(self, clean_container: p.Container) -> None:
        """Test that factory is called each time get() is invoked using fixtures."""
        factory, get_count = u.Tests.create_counting_factory("service_value")
        clean_container.factory("factory_service", factory)
        result1: p.Result[t.RegisterableService] = clean_container.resolve(
            "factory_service"
        )
        _ = u.Tests.assert_success(result1)
        tm.that(get_count(), eq=1, msg="Factory must be called once after first get()")
        result2: p.Result[t.RegisterableService] = clean_container.resolve(
            "factory_service"
        )
        _ = u.Tests.assert_success(result2)
        tm.that(
            get_count(), eq=2, msg="Factory must be called twice after second get()"
        )

    @pytest.mark.parametrize(
        "scenario",
        m.Tests.ContainerScenarios.TYPED_RETRIEVAL_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_get_typed_correct(
        self, scenario: p.Tests.TypedRetrievalScenario, clean_container: p.Container
    ) -> None:
        """Test typed retrieval with correct types using fixtures."""
        container = clean_container
        _ = container.bind(scenario.name, scenario.service)
        if scenario.should_pass:
            resolved_service: str | int = u.Tests.assert_success(
                container.resolve(scenario.name, type_cls=scenario.expected_type)
            )
            tm.that(
                str(resolved_service),
                eq=str(scenario.service),
                msg=f"Typed result value must match service for {scenario.name}",
            )
            tm.that(
                isinstance(resolved_service, scenario.expected_type),
                eq=True,
                msg=f"Typed result must be instance of {scenario.expected_type.__name__}",
            )
        else:
            _ = u.Tests.assert_failure(
                container.resolve(scenario.name, type_cls=scenario.expected_type)
            )

    def test_get_typed_wrong_type(self, clean_container: p.Container) -> None:
        """Test typed retrieval with wrong type fails using fixtures."""
        clean_container.bind("string_service", "test_value")
        result = clean_container.resolve("string_service", type_cls=dict)
        _ = u.Tests.assert_failure(result)

    def test_get_typed_nonexistent(self, clean_container: p.Container) -> None:
        """Test typed retrieval of non-existent service using fixtures."""
        result = clean_container.resolve("nonexistent", type_cls=dict)
        u.Tests.assert_failure(result, expected_error="not found")

    @pytest.mark.parametrize(
        ("has_service", "expected"),
        [(True, True), (False, False)],
        ids=["exists", "not_exists"],
    )
    def test_has_service(
        self, *, has_service: bool, expected: bool, clean_container: p.Container
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

    def test_has_service_factory(self, clean_container: p.Container) -> None:
        """Test has_service returns True for factories using fixtures."""
        container = clean_container
        factory = u.Tests.create_factory("value")
        _ = container.factory("factory_service", factory)
        tm.that(
            container.has("factory_service"),
            eq=True,
            msg="Container must have factory_service after registration",
        )

    def test_list_services_empty(self, clean_container: p.Container) -> None:
        """Test listing services when none registered using fixtures."""
        container = clean_container
        services = container.names()
        tm.that(services, is_=list, msg="list_services must return a list")
        tm.that(
            len(services), eq=0, msg="Empty container must return empty services list"
        )
        tm.that(
            services, empty=True, msg="Empty container must have empty services list"
        )

    def test_list_services_mixed(self, clean_container: p.Container) -> None:
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
