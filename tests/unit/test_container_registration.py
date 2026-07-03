"""Container registration tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from flext_tests import tm

from flext_core.container import FlextContainer
from tests.models import TestsFlextModels
from tests.protocols import p
from tests.typings import t
from tests.utilities import u


class TestsFlextContainerRegistration:
    def test_container_initialization(self, clean_container: p.Container) -> None:
        """Test container initialization creates valid instance using fixtures."""
        assert isinstance(clean_container, p.Container), (
            "Container must be FlextContainer instance"
        )
        tm.that(
            clean_container.names(),
            empty=True,
            msg="Fresh container must start without registered services",
        )

    def test_container_singleton(self) -> None:
        """Test that FlextContainer returns singleton instance."""
        container1 = FlextContainer()
        container2 = FlextContainer()
        assert isinstance(container1, FlextContainer)
        assert isinstance(container2, FlextContainer)
        tm.that(
            container1 is container2,
            eq=True,
            msg="Containers must be the same singleton instance",
        )

    @pytest.mark.parametrize(
        "scenario",
        TestsFlextModels.Tests.ContainerScenarios.SERVICE_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_register_service(
        self,
        scenario: TestsFlextModels.Tests.ServiceScenario,
        clean_container: p.Container,
    ) -> None:
        """Test service registration with various types using fixtures."""
        result = clean_container.bind(scenario.name, scenario.service)
        assert result is clean_container
        tm.that(
            clean_container.has(scenario.name),
            eq=True,
            msg=f"Container must expose {scenario.name} after registration",
        )
        u.Tests.assert_success(
            clean_container.resolve(scenario.name), expected_value=scenario.service
        )

    @pytest.mark.parametrize(
        "scenario",
        TestsFlextModels.Tests.ContainerScenarios.SERVICE_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_with_service_fluent(
        self,
        scenario: TestsFlextModels.Tests.ServiceScenario,
        clean_container: p.Container,
    ) -> None:
        """Test fluent interface for service registration using fixtures."""
        container = clean_container
        result = container.bind(scenario.name, scenario.service)
        tm.that(
            result is container,
            eq=True,
            msg="with_service must return self for fluent interface",
        )
        tm.that(
            container.has(scenario.name),
            eq=True,
            msg=f"Container must have service {scenario.name} after registration",
        )

    def test_register_duplicate_service(self, clean_container: p.Container) -> None:
        """Test that registering duplicate service name preserves original using fixtures."""
        container = clean_container
        _ = container.bind("service1", "value1")
        _ = container.bind("service1", "value2")
        service_result = container.resolve("service1")
        u.Tests.assert_success(service_result, expected_value="value1")

    def test_register_with_empty_name(self, clean_container: p.Container) -> None:
        """Test that empty name is rejected using fixtures."""
        clean_container.bind("", "service")
        tm.that(
            clean_container.has(""),
            eq=False,
            msg="Empty name service must not be registered",
        )

    @pytest.mark.parametrize(
        "return_value",
        ["created_string", 42],
        ids=["string", "int"],
    )
    def test_register_factory(
        self,
        return_value: t.JsonValue,
        clean_container: p.Container,
    ) -> None:
        """Test factory registration using fixtures."""
        factory = u.Tests.create_factory(return_value)
        factory_name = f"factory_{type(return_value).__name__}"
        clean_container.factory(factory_name, factory)
        tm.that(
            clean_container.has(factory_name),
            eq=True,
            msg=f"Factory {factory_name} must be registered",
        )
        u.Tests.assert_success(
            clean_container.resolve(factory_name), expected_value=return_value
        )

    @pytest.mark.parametrize(
        "return_value",
        [{"created": "by_factory"}, "created_string"],
        ids=["dict", "string"],
    )
    def test_with_factory_fluent(
        self,
        return_value: t.RegisterableService,
        clean_container: p.Container,
    ) -> None:
        """Test fluent interface for factory using fixtures."""
        container = clean_container
        factory = u.Tests.create_factory(return_value)
        factory_typed: Callable[[], t.RegisterableService] = factory
        result = container.factory(
            f"factory_{type(return_value).__name__}", factory_typed
        )
        tm.that(
            result is container,
            eq=True,
            msg="with_factory must return self for fluent interface",
        )
        factory_name = f"factory_{type(return_value).__name__}"
        tm.that(
            container.has(factory_name),
            eq=True,
            msg=f"Container must have factory {factory_name} after registration",
        )

    def test_register_duplicate_factory(self, clean_container: p.Container) -> None:
        """Test that registering duplicate factory name preserves original using fixtures."""
        factory1 = u.Tests.create_factory("value1")
        clean_container.factory("factory1", factory1)
        factory2 = u.Tests.create_factory("value2")
        clean_container.factory("factory1", factory2)
        u.Tests.assert_success(
            clean_container.resolve("factory1"), expected_value="value1"
        )
