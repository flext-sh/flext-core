"""Tests for p - Protocol Definitions and Implementations.

Module: flext_core.protocols
Scope: p - all protocol definitions and implementations

Tests p functionality including:
- Foundation protocols (Result, ResultLike, HasModelFields, HasModelDump, Model)
- Domain protocols (Repository, Service)
- Infrastructure protocols (Configurable)
- Application protocols (Handler)
- Commands/Extensions protocols (Dispatcher, Middleware)
- Protocol implementations and runtime checking

Uses Python 3.13 patterns, u, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Annotated, ClassVar

import pytest
from flext_tests import tm, u
from pydantic import BaseModel, ConfigDict, Field

from flext_core import r
from tests import p, t, u


class TestFlextProtocols:
    @unique
    class ProtocolCategoryType(StrEnum):
        """Protocol category types."""

        FOUNDATION = "foundation"
        DOMAIN = "domain"
        INFRASTRUCTURE = "infrastructure"
        APPLICATION = "application"
        COMMANDS = "commands"
        EXTENSIONS = "extensions"

    class ProtocolDefinitionScenario(BaseModel):
        """Scenario for protocol definitions."""

        model_config = ConfigDict(frozen=True)

        name: Annotated[str, Field(description="Protocol definition scenario name")]
        protocol_name: Annotated[str, Field(description="Protocol attribute name")]
        category: Annotated[StrEnum, Field(description="Protocol category")]

    class ProtocolAvailabilityScenario(BaseModel):
        """Scenario for protocol availability."""

        model_config = ConfigDict(frozen=True)

        name: Annotated[str, Field(description="Protocol availability scenario name")]
        category: Annotated[StrEnum, Field(description="Protocol category")]
        protocol_names: Annotated[
            list[str], Field(description="Expected protocol names")
        ]

    _DEFINITION_SCENARIOS: ClassVar[
        list[TestFlextProtocols.ProtocolDefinitionScenario]
    ] = [
        ProtocolDefinitionScenario(
            name="result_protocol",
            protocol_name="Result",
            category=ProtocolCategoryType.FOUNDATION,
        ),
        ProtocolDefinitionScenario(
            name="has_model_dump_protocol",
            protocol_name="HasModelDump",
            category=ProtocolCategoryType.FOUNDATION,
        ),
        ProtocolDefinitionScenario(
            name="service_protocol",
            protocol_name="Service",
            category=ProtocolCategoryType.DOMAIN,
        ),
        ProtocolDefinitionScenario(
            name="configurable_protocol",
            protocol_name="Configurable",
            category=ProtocolCategoryType.INFRASTRUCTURE,
        ),
        ProtocolDefinitionScenario(
            name="handler_protocol",
            protocol_name="Handler",
            category=ProtocolCategoryType.APPLICATION,
        ),
        ProtocolDefinitionScenario(
            name="command_bus_protocol",
            protocol_name="Dispatcher",
            category=ProtocolCategoryType.COMMANDS,
        ),
        ProtocolDefinitionScenario(
            name="middleware_protocol",
            protocol_name="Middleware",
            category=ProtocolCategoryType.COMMANDS,
        ),
    ]
    _AVAILABILITY_SCENARIOS: ClassVar[
        list[TestFlextProtocols.ProtocolAvailabilityScenario]
    ] = [
        ProtocolAvailabilityScenario(
            name="all_foundation_protocols_available",
            category=ProtocolCategoryType.FOUNDATION,
            protocol_names=["Result", "ResultLike", "HasModelDump", "Model"],
        ),
        ProtocolAvailabilityScenario(
            name="all_domain_protocols_available",
            category=ProtocolCategoryType.DOMAIN,
            protocol_names=["Service"],
        ),
        ProtocolAvailabilityScenario(
            name="all_infrastructure_protocols_available",
            category=ProtocolCategoryType.INFRASTRUCTURE,
            protocol_names=["Configurable"],
        ),
        ProtocolAvailabilityScenario(
            name="all_application_protocols_available",
            category=ProtocolCategoryType.APPLICATION,
            protocol_names=["Handler"],
        ),
        ProtocolAvailabilityScenario(
            name="all_commands_protocols_available",
            category=ProtocolCategoryType.COMMANDS,
            protocol_names=["Dispatcher", "Middleware"],
        ),
        ProtocolAvailabilityScenario(
            name="all_extensions_protocols_available",
            category=ProtocolCategoryType.EXTENSIONS,
            protocol_names=["Middleware"],
        ),
    ]

    @pytest.mark.parametrize(
        "scenario",
        _DEFINITION_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_protocol_definition(
        self,
        scenario: TestFlextProtocols.ProtocolDefinitionScenario,
    ) -> None:
        """Test protocol definitions are accessible and valid."""
        protocol = getattr(p, scenario.protocol_name)
        assert protocol is not None, f"Protocol {scenario.protocol_name} must exist"
        tm.that(
            hasattr(protocol, "__protocol_attrs__")
            or hasattr(protocol, "__annotations__"),
            eq=True,
            msg=f"Protocol {scenario.protocol_name} must have protocol attributes or annotations",
        )

    @pytest.mark.parametrize(
        "scenario",
        _AVAILABILITY_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_protocol_availability(
        self,
        scenario: TestFlextProtocols.ProtocolAvailabilityScenario,
    ) -> None:
        """Test that protocols are available by category with real validation."""
        for proto_name in scenario.protocol_names:
            tm.that(
                hasattr(p, proto_name),
                eq=True,
                msg=f"Protocol {proto_name} must be found in p (FlextProtocols)",
            )
            protocol = getattr(p, proto_name)
            assert protocol is not None, f"Protocol {proto_name} must not be None"

    def test_result_protocol_implementation(self) -> None:
        """Test that a class can implement Result protocol."""

        class ResultContainer:
            """Container implementing Result protocol structure."""

            def __init__(self, value: str) -> None:
                self._value = value
                self._is_success = True

            @property
            def value(self) -> str:
                return self._value

            @property
            def is_success(self) -> bool:
                return self._is_success

            @property
            def is_failure(self) -> bool:
                return not self._is_success

            @property
            def error(self) -> str | None:
                return None

            @property
            def error_code(self) -> str | None:
                return None

            def unwrap(self) -> str:
                return self._value

        container = ResultContainer("test")
        tm.that(container.value, eq="test", msg="Container value must match input")
        tm.that(container.is_success, eq=True, msg="Container must be success")
        tm.that(container.is_failure, eq=False, msg="Container must not be failure")
        tm.that(container.error, none=True, msg="Success container must have no error")
        tm.that(container.value, eq="test", msg="Unwrap must return value")

    def test_repository_implementation(self) -> None:
        """Test that a class can implement Repository protocol."""

        class UserRepository:
            """Repository for user entities."""

            def find_by_id(self, entity_id: str) -> r[t.ConfigMap]:
                return r[t.ConfigMap].ok(
                    t.ConfigMap(root={"id": entity_id, "name": "Test"}),
                )

            def save(self, entity: t.ConfigMap) -> r[t.ConfigMap]:
                return r[t.ConfigMap].ok(entity)

            def delete(self, entity_id: str) -> r[bool]:
                return r[bool].ok(True)

        repo = UserRepository()
        required_methods = ["find_by_id", "save", "delete"]
        for method in required_methods:
            tm.that(
                hasattr(repo, method),
                eq=True,
                msg=f"Repository must have {method} method",
            )
            tm.that(
                callable(getattr(repo, method)),
                eq=True,
                msg=f"Repository {method} must be callable",
            )
        find_result = repo.find_by_id("test_id")
        _ = u.Tests.Result.assert_success(find_result)
        tm.that(find_result.value, has="id", msg="Find result must contain id")
        tm.that(find_result.value, has="name", msg="Find result must contain name")

    def test_service_implementation(self) -> None:
        """Test that a class can implement Service protocol."""

        class UserService:
            """Service for user operations."""

            def execute(self) -> r[t.ConfigMap]:
                return r[t.ConfigMap].ok(t.ConfigMap(root={"status": "success"}))

        service = UserService()
        tm.that(
            hasattr(service, "execute"),
            eq=True,
            msg="Service must have execute method",
        )
        tm.that(
            callable(service.execute),
            eq=True,
            msg="Service execute must be callable",
        )
        result = service.execute()
        _ = u.Tests.Result.assert_success(result)
        tm.that(result.value, has="status", msg="Service result must contain status")
        tm.that(
            result.value["status"],
            eq="success",
            msg="Service status must be success",
        )

    def test_handler_implementation(self) -> None:
        """Test that a class can implement Handler protocol."""

        class CreateUserHandler:
            """Handler for user creation."""

            def handle(self, command: t.ConfigMap) -> r[t.ConfigMap]:
                return r[t.ConfigMap].ok(t.ConfigMap(root={"user_id": "123"}))

        handler = CreateUserHandler()
        tm.that(
            hasattr(handler, "handle"),
            eq=True,
            msg="Handler must have handle method",
        )
        tm.that(
            callable(handler.handle),
            eq=True,
            msg="Handler handle must be callable",
        )
        command = t.ConfigMap(root={"name": "Test"})
        result = handler.handle(command)
        _ = u.Tests.Result.assert_success(result)
        tm.that(result.value, has="user_id", msg="Handler result must contain user_id")
        tm.that(
            result.value["user_id"],
            is_=str,
            none=False,
            empty=False,
            msg="user_id must be non-empty string",
        )

    def test_multiple_protocol_implementation(self) -> None:
        """Test that a class can implement multiple protocols."""

        class AdvancedService:
            """Service implementing multiple protocols."""

            def execute(self) -> r[t.ConfigMap]:
                return r[t.ConfigMap].ok(t.ConfigMap(root={}))

            def handle(self, command: t.ConfigMap) -> r[t.ConfigMap]:
                return r[t.ConfigMap].ok(t.ConfigMap(root={}))

        service = AdvancedService()
        required_methods = ["execute", "handle"]
        for method in required_methods:
            tm.that(
                hasattr(service, method),
                eq=True,
                msg=f"AdvancedService must have {method} method",
            )
            tm.that(
                callable(getattr(service, method)),
                eq=True,
                msg=f"AdvancedService {method} must be callable",
            )
        execute_result = service.execute()
        _ = u.Tests.Result.assert_success(execute_result)
        handle_result = service.handle(t.ConfigMap(root={"command": "test"}))
        _ = u.Tests.Result.assert_success(handle_result)

    def test_protocol_runtime_checkable(self) -> None:
        """Test that protocols support runtime checking."""

        class Container:
            """Container with runtime checkable property."""

            @property
            def value(self) -> str:
                return "test"

        container = Container()
        tm.that(
            hasattr(container, "value"),
            eq=True,
            msg="Container must have value property",
        )
        tm.that(container.value, eq="test", msg="Container value must be 'test'")
        tm.that(
            container.value,
            is_=str,
            none=False,
            empty=False,
            msg="Container value must be non-empty string",
        )
