"""Comprehensive tests for FlextContainer - Dependency Injection Container.

Module: flext_core.container
Scope: FlextContainer - dependency injection, service registration, factory patterns

Tests FlextContainer functionality including:
- Container initialization and singleton pattern
- Service registration and retrieval
- Factory registration and execution
- Typed service retrieval
- Service existence checks and listing
- Service unregistration
- Configuration management
- Complex workflows

Uses Python 3.13 patterns, u, c,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from typing import Annotated, ClassVar

import pytest
from hypothesis import given, settings, strategies as st
from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextContainer, FlextContext, FlextSettings
from flext_tests import tm
from tests import c, p, t, u


class _ServiceScenario(BaseModel):
    """Test scenario for service registration and retrieval."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )
    name: Annotated[str, Field(description="Service scenario name")]
    service: Annotated[t.Primitives, Field(description="Service value to register")]
    description: Annotated[
        str,
        Field(default="", description="Scenario description"),
    ] = ""


class _TypedRetrievalScenario(BaseModel):
    """Test scenario for typed service retrieval."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )
    name: Annotated[str, Field(description="Typed retrieval scenario name")]
    service: Annotated[t.Primitives, Field(description="Registered service value")]
    expected_type: Annotated[type, Field(description="Expected service type")]
    should_pass: Annotated[
        bool,
        Field(description="Whether typed retrieval should succeed"),
    ]
    description: Annotated[
        str,
        Field(default="", description="Scenario description"),
    ] = ""


class _ContainerScenarios:
    """Centralized container test scenarios using c."""

    SERVICE_SCENARIOS: ClassVar[Sequence[_ServiceScenario]] = [
        _ServiceScenario(
            name="test_service",
            service="test_service_value",
            description="Simple string service",
        ),
        _ServiceScenario(
            name="service_instance",
            service=42,
            description="Integer service instance",
        ),
        _ServiceScenario(
            name="string_service",
            service="test_value",
            description="String service",
        ),
    ]
    TYPED_RETRIEVAL_SCENARIOS: ClassVar[Sequence[_TypedRetrievalScenario]] = [
        _TypedRetrievalScenario(
            name="dict_service",
            service="test_dict_service",
            expected_type=str,
            should_pass=True,
            description="String service",
        ),
        _TypedRetrievalScenario(
            name="string_service",
            service="test_string",
            expected_type=str,
            should_pass=True,
            description="String service",
        ),
        _TypedRetrievalScenario(
            name="list_service",
            service=123,
            expected_type=int,
            should_pass=True,
            description="Integer service for typed retrieval",
        ),
    ]
    CONFIG_SCENARIOS: ClassVar[Sequence[t.ScalarMapping]] = [
        {"enable_singleton": False, "max_services": 8},
        {"invalid_key": "value", "another_invalid": 42},
        {},
    ]


ContainerScenarios = _ContainerScenarios


class TestFlextContainer:
    @pytest.fixture(autouse=True)
    def _reset_container_state(self) -> Generator[None]:
        FlextContainer.reset_for_testing()
        yield
        FlextContainer.reset_for_testing()

    def test_container_initialization(self, clean_container: p.Container) -> None:
        """Test container initialization creates valid instance using fixtures."""
        assert isinstance(clean_container, p.Container), (
            "Container must be FlextContainer instance"
        )
        tm.that(
            clean_container.list_services(),
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
        ContainerScenarios.SERVICE_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_register_service(
        self,
        scenario: _ServiceScenario,
        clean_container: p.Container,
    ) -> None:
        """Test service registration with various types using fixtures."""
        result = clean_container.register(scenario.name, scenario.service)
        assert result is clean_container
        tm.that(
            clean_container.has_service(scenario.name),
            eq=True,
            msg=f"Container must expose {scenario.name} after registration",
        )
        u.Core.Tests.assert_success_with_value(
            clean_container.get(scenario.name),
            scenario.service,
        )

    @pytest.mark.parametrize(
        "scenario",
        ContainerScenarios.SERVICE_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_with_service_fluent(
        self,
        scenario: _ServiceScenario,
        clean_container: p.Container,
    ) -> None:
        """Test fluent interface for service registration using fixtures."""
        container = clean_container
        result = container.register(scenario.name, scenario.service)
        tm.that(
            result is container,
            eq=True,
            msg="with_service must return self for fluent interface",
        )
        tm.that(
            container.has_service(scenario.name),
            eq=True,
            msg=f"Container must have service {scenario.name} after registration",
        )

    def test_register_duplicate_service(self, clean_container: p.Container) -> None:
        """Test that registering duplicate service name preserves original using fixtures."""
        container = clean_container
        _ = container.register("service1", "value1")
        _ = container.register("service1", "value2")
        service_result = container.get("service1")
        u.Core.Tests.assert_success_with_value(service_result, "value1")

    def test_register_with_empty_name(self, clean_container: p.Container) -> None:
        """Test that empty name is rejected using fixtures."""
        clean_container.register("", "service")
        tm.that(
            clean_container.has_service(""),
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
        return_value: t.Scalar,
        clean_container: p.Container,
    ) -> None:
        """Test factory registration using fixtures."""
        factory = u.Core.Tests.create_factory(return_value)
        factory_name = f"factory_{type(return_value).__name__}"
        clean_container.register(factory_name, factory, kind=c.CONTAINER_KIND_FACTORY)
        tm.that(
            clean_container.has_service(factory_name),
            eq=True,
            msg=f"Factory {factory_name} must be registered",
        )
        u.Core.Tests.assert_success_with_value(
            clean_container.get(factory_name),
            return_value,
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
        factory = u.Core.Tests.create_factory(return_value)
        factory_typed: Callable[[], t.RegisterableService] = factory
        result = container.register(
            f"factory_{type(return_value).__name__}",
            factory_typed,
            kind=c.CONTAINER_KIND_FACTORY,
        )
        tm.that(
            result is container,
            eq=True,
            msg="with_factory must return self for fluent interface",
        )
        factory_name = f"factory_{type(return_value).__name__}"
        tm.that(
            container.has_service(factory_name),
            eq=True,
            msg=f"Container must have factory {factory_name} after registration",
        )

    def test_register_factory_non_callable(self, clean_container: p.Container) -> None:
        """Test that registering non-callable with factory kind handles gracefully."""
        non_callable = "not_callable"
        clean_container.register("invalid", non_callable, kind=c.CONTAINER_KIND_FACTORY)
        tm.that(
            clean_container.has_service("invalid"),
            eq=False,
            msg="Non-callable factory should not be registered",
        )

    def test_register_duplicate_factory(self, clean_container: p.Container) -> None:
        """Test that registering duplicate factory name preserves original using fixtures."""
        factory1 = u.Core.Tests.create_factory("value1")
        clean_container.register("factory1", factory1, kind=c.CONTAINER_KIND_FACTORY)
        factory2 = u.Core.Tests.create_factory("value2")
        clean_container.register("factory1", factory2, kind=c.CONTAINER_KIND_FACTORY)
        u.Core.Tests.assert_success_with_value(
            clean_container.get("factory1"), "value1"
        )

    @pytest.mark.parametrize(
        "scenario",
        ContainerScenarios.SERVICE_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_get_service(
        self,
        scenario: _ServiceScenario,
        clean_container: p.Container,
    ) -> None:
        """Test service retrieval using fixtures."""
        clean_container.register(scenario.name, scenario.service)
        result: p.Result[t.RegisterableService] = clean_container.get(scenario.name)
        u.Core.Tests.assert_success_with_value(result, scenario.service)

    def test_get_nonexistent_service(self, clean_container: p.Container) -> None:
        """Test getting non-existent service using fixtures."""
        result: p.Result[t.RegisterableService] = clean_container.get("nonexistent")
        u.Core.Tests.assert_result_failure_with_error(
            result,
            expected_error="not found",
        )

    def test_get_factory_service(self, clean_container: p.Container) -> None:
        """Test retrieving service created by factory using fixtures."""
        factory_result = {"created": "by_factory"}
        factory = u.Core.Tests.create_factory(factory_result)
        clean_container.register(
            "factory_service", factory, kind=c.CONTAINER_KIND_FACTORY
        )
        result: p.Result[t.RegisterableService] = clean_container.get("factory_service")
        u.Core.Tests.assert_success_with_value(result, factory_result)

    def test_get_factory_called_each_time(self, clean_container: p.Container) -> None:
        """Test that factory is called each time get() is invoked using fixtures."""
        factory, get_count = u.Core.Tests.create_counting_factory(
            "service_value",
        )
        clean_container.register(
            "factory_service", factory, kind=c.CONTAINER_KIND_FACTORY
        )
        result1: p.Result[t.RegisterableService] = clean_container.get(
            "factory_service"
        )
        _ = u.Core.Tests.assert_success(result1)
        tm.that(get_count(), eq=1, msg="Factory must be called once after first get()")
        result2: p.Result[t.RegisterableService] = clean_container.get(
            "factory_service"
        )
        _ = u.Core.Tests.assert_success(result2)
        tm.that(
            get_count(),
            eq=2,
            msg="Factory must be called twice after second get()",
        )

    @pytest.mark.parametrize(
        "scenario",
        ContainerScenarios.TYPED_RETRIEVAL_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_get_typed_correct(
        self,
        scenario: _TypedRetrievalScenario,
        clean_container: p.Container,
    ) -> None:
        """Test typed retrieval with correct types using fixtures."""
        container = clean_container
        _ = container.register(scenario.name, scenario.service)
        typed_result = container.get(
            scenario.name,
            type_cls=scenario.expected_type,
        )
        if scenario.should_pass:
            _ = u.Core.Tests.assert_success(typed_result)
            tm.that(
                str(typed_result.value),
                eq=str(scenario.service),
                msg=f"Typed result value must match service for {scenario.name}",
            )
            tm.that(
                isinstance(typed_result.value, scenario.expected_type),
                eq=True,
                msg=f"Typed result must be instance of {scenario.expected_type.__name__}",
            )
        else:
            _ = u.Core.Tests.assert_failure(typed_result)

    def test_get_typed_wrong_type(self, clean_container: p.Container) -> None:
        """Test typed retrieval with wrong type fails using fixtures."""
        clean_container.register("string_service", "test_value")
        result = clean_container.get("string_service", type_cls=dict)
        _ = u.Core.Tests.assert_failure(result)

    def test_get_typed_nonexistent(self, clean_container: p.Container) -> None:
        """Test typed retrieval of non-existent service using fixtures."""
        result = clean_container.get("nonexistent", type_cls=dict)
        u.Core.Tests.assert_result_failure_with_error(
            result,
            expected_error="not found",
        )

    @pytest.mark.parametrize(
        ("has_service", "expected"),
        [(True, True), (False, False)],
        ids=["exists", "not_exists"],
    )
    def test_has_service(
        self,
        has_service: bool,
        expected: bool,
        clean_container: p.Container,
    ) -> None:
        """Test has_service returns correct value using fixtures."""
        container = clean_container
        service_name = "test_service" if has_service else "nonexistent"
        if has_service:
            _ = container.register(service_name, "value")
        tm.that(
            container.has_service(service_name),
            eq=expected,
            msg=f"has_service must return {expected} for {service_name}",
        )

    def test_has_service_factory(self, clean_container: p.Container) -> None:
        """Test has_service returns True for factories using fixtures."""
        container = clean_container
        factory = u.Core.Tests.create_factory("value")
        _ = container.register(
            "factory_service", factory, kind=c.CONTAINER_KIND_FACTORY
        )
        tm.that(
            container.has_service("factory_service"),
            eq=True,
            msg="Container must have factory_service after registration",
        )

    def test_list_services_empty(self, clean_container: p.Container) -> None:
        """Test listing services when none registered using fixtures."""
        container = clean_container
        services = container.list_services()
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

    def test_list_services_mixed(self, clean_container: p.Container) -> None:
        """Test listing mix of registered services and factories using fixtures."""
        container = clean_container
        _ = container.register("service1", "value1")
        _ = container.register("service2", "value2")
        factory = u.Core.Tests.create_factory("value3")
        _ = container.register("factory1", factory, kind=c.CONTAINER_KIND_FACTORY)
        services = container.list_services()
        tm.that(len(services), eq=3, msg="Container must list 3 registered services")
        required_keys = ["service1", "service2", "factory1"]
        for key in required_keys:
            tm.that(key in services, eq=True, msg=f"Services list must contain {key}")

    @pytest.mark.parametrize("config", ContainerScenarios.CONFIG_SCENARIOS, ids=str)
    def test_configure_container(
        self,
        config: t.ScalarMapping,
        clean_container: p.Container,
    ) -> None:
        """Test container configuration."""
        container = clean_container
        original_config = container.get_config()
        container.configure(config)
        config_result = container.get_config()
        tm.that(
            config_result,
            is_=t.ConfigMap,
            none=False,
            msg="Container config must be a ConfigMap",
        )
        for key, value in config.items():
            if key in original_config.root:
                tm.that(
                    config_result.root.get(key),
                    eq=value,
                    msg=f"Config key {key} must be updated through configure()",
                )
            else:
                tm.that(
                    key in config_result.root,
                    eq=False,
                    msg=f"Unknown config key {key} must not leak into public config",
                )
        if not config:
            tm.that(
                config_result.root,
                eq=original_config.root,
                msg="Empty configure() input must preserve existing config",
            )

    def test_with_config_fluent(self, clean_container: p.Container) -> None:
        """Test fluent interface for configuration."""
        container = clean_container
        config: t.ScalarMapping = {"max_services": 32}
        result = container.configure(config)
        tm.that(
            result is container,
            eq=True,
            msg="with_config must return self for fluent interface",
        )
        config_result = container.get_config()
        tm.that(
            config_result,
            is_=t.ConfigMap,
            none=False,
            msg="get_config must return a ConfigMap",
        )
        tm.that(
            config_result.root,
            none=False,
            msg="Config must be accessible after with_config",
        )
        tm.that(
            config_result.root.get("max_services"),
            eq=32,
            msg="configure() must expose applied public config values",
        )

    def test_get_config(self) -> None:
        """Test retrieving current configuration."""
        container = FlextContainer()
        config = container.get_config()
        tm.that(
            config,
            is_=t.ConfigMap,
            none=False,
            msg="get_config must return ConfigMap",
        )
        tm.that(
            "enable_singleton" in config.root,
            eq=True,
            msg="Config must contain enable_singleton",
        )
        tm.that(
            "max_services" in config.root,
            eq=True,
            msg="Config must contain max_services",
        )

    def test_config_property(self) -> None:
        """Test accessing config via property."""
        container = FlextContainer()
        config = container.config
        tm.that(
            config,
            is_=FlextSettings,
            msg="Container config property must expose FlextSettings",
        )
        tm.that(
            config.app_name,
            eq=FlextSettings.get_global().app_name,
            msg="Container config property must reflect the bound public settings",
        )

    def test_clear_all_services(self, clean_container: p.Container) -> None:
        """Test clearing all services and factories using fixtures."""
        container = clean_container
        _ = container.register("service1", "value1")
        _ = container.register("service2", "value2")
        factory = u.Core.Tests.create_factory("value3")
        _ = container.register("factory1", factory, kind=c.CONTAINER_KIND_FACTORY)
        tm.that(
            len(container.list_services()),
            eq=3,
            msg="Container must have 3 services before clear_all",
        )
        container.clear_all()
        tm.that(
            len(container.list_services()),
            eq=0,
            msg="Container must have 0 services after clear_all",
        )
        tm.that(
            container.list_services(),
            empty=True,
            msg="Container services list must be empty after clear_all",
        )

    def test_clear_all_empty(self, clean_container: p.Container) -> None:
        """Test clearing when no services exist using fixtures."""
        container = clean_container
        container.clear_all()
        tm.that(
            len(container.list_services()),
            eq=0,
            msg="Empty container must have 0 services after clear_all",
        )
        tm.that(
            container.list_services(),
            empty=True,
            msg="Empty container services list must be empty after clear_all",
        )

    def test_full_workflow(self, clean_container: p.Container) -> None:
        """Test complete container workflow using fixtures."""
        container = clean_container
        _ = container.register("db_connection", {"host": c.LOCALHOST})
        _ = container.register("cache", {"type": "redis"})
        factory = u.Core.Tests.create_factory({"logger": "instance"})
        _ = container.register("logger", factory, kind=c.CONTAINER_KIND_FACTORY)
        required_services = ["db_connection", "cache", "logger"]
        for service_name in required_services:
            tm.that(
                container.has_service(service_name),
                eq=True,
                msg=f"Container must have {service_name} after registration",
            )
        for name in required_services:
            result: p.Result[t.RegisterableService] = container.get(name)
            _ = u.Core.Tests.assert_success(result)
        tm.that(
            len(container.list_services()),
            eq=3,
            msg="Container must have 3 services in full workflow",
        )
        container.clear_all()
        tm.that(
            len(container.list_services()),
            eq=0,
            msg="Container must have 0 services after clear_all in workflow",
        )

    def test_factory_exception_handling(self, clean_container: p.Container) -> None:
        """Test handling of factory exceptions using fixtures."""
        container = clean_container
        error_msg = "Factory failed"

        def failing_factory() -> str:
            raise RuntimeError(error_msg)

        _ = container.register(
            "failing", failing_factory, kind=c.CONTAINER_KIND_FACTORY
        )
        result: p.Result[t.RegisterableService] = container.get("failing")
        _ = u.Core.Tests.assert_failure(result)

    def test_scoped_container_with_context(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test scoped container creation with FlextContext."""
        scoped = clean_container.scoped(
            context=FlextContext.create(),
            subproject="unit",
            services={"scoped_service": "scoped-value"},
        )
        tm.that(scoped.has_service("scoped_service"), eq=True)
        tm.ok(scoped.get("scoped_service", type_cls=str), eq="scoped-value")
        ctx_result = scoped.context.get("subproject")
        assert ctx_result.success
        assert ctx_result.value == "unit"

    @given(
        name=st.text(
            min_size=1,
            max_size=30,
            alphabet=st.characters(min_codepoint=48, max_codepoint=122),
        ),
    )
    @settings(max_examples=50)
    def test_register_get_roundtrip_property(self, name: str) -> None:
        """Property: register then get roundtrips for any valid name."""
        container = FlextContainer.create()
        sanitized = "".join(ch for ch in name if ch.isalnum()) or "svc"

        def dynamic_factory() -> str:
            return sanitized

        _ = container.register(
            sanitized, dynamic_factory, kind=c.CONTAINER_KIND_FACTORY
        )
        tm.ok(container.get(sanitized, type_cls=str), eq=sanitized)
        FlextContainer.reset_for_testing()

    __all__ = ["TestFlextContainer"]


__all__ = ["TestFlextContainer"]
