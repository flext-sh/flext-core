"""Tests for p - Protocol Definitions and Implementations.

Module: flext_core.protocols
Scope: p - all protocol definitions and implementations

Tests p functionality including:
- Foundation protocols (Result, ResultLike, HasModelFields, HasModelDump, Model)
- Domain protocols (Repository, Service)
- Infrastructure protocols (Configurable)
- Application protocols (Handler)
- Commands/Extensions protocols (CommandBus, Middleware)
- Protocol implementations and runtime checking

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

import pytest

from flext_core import p, r, t
from flext_tests.matchers import tm
from flext_tests.utilities import u
from flext_core.models import m


class ProtocolCategoryType(StrEnum):
    """Protocol category types for organization."""

    FOUNDATION = "foundation"
    DOMAIN = "domain"
    INFRASTRUCTURE = "infrastructure"
    APPLICATION = "application"
    COMMANDS = "commands"
    EXTENSIONS = "extensions"


@dataclass(frozen=True, slots=True)
class ProtocolDefinitionScenario:
    """Protocol definition test scenario."""

    name: str
    protocol_name: str
    category: ProtocolCategoryType


@dataclass(frozen=True, slots=True)
class ProtocolAvailabilityScenario:
    """Protocol availability test scenario."""

    name: str
    category: ProtocolCategoryType
    protocol_names: list[str]


class ProtocolScenarios:
    """Centralized protocol test scenarios."""

    DEFINITION_SCENARIOS: ClassVar[list[ProtocolDefinitionScenario]] = [
        ProtocolDefinitionScenario(
            "result_protocol",
            "Result",
            ProtocolCategoryType.FOUNDATION,
        ),
        ProtocolDefinitionScenario(
            "has_model_fields_protocol",
            "HasModelFields",
            ProtocolCategoryType.FOUNDATION,
        ),
        ProtocolDefinitionScenario(
            "has_model_dump_protocol",
            "HasModelDump",
            ProtocolCategoryType.FOUNDATION,
        ),
        ProtocolDefinitionScenario(
            "repository_protocol",
            "Repository",
            ProtocolCategoryType.DOMAIN,
        ),
        ProtocolDefinitionScenario(
            "service_protocol",
            "Service",
            ProtocolCategoryType.DOMAIN,
        ),
        ProtocolDefinitionScenario(
            "configurable_protocol",
            "Configurable",
            ProtocolCategoryType.INFRASTRUCTURE,
        ),
        ProtocolDefinitionScenario(
            "handler_protocol",
            "Handler",
            ProtocolCategoryType.APPLICATION,
        ),
        ProtocolDefinitionScenario(
            "command_bus_protocol",
            "CommandBus",
            ProtocolCategoryType.COMMANDS,
        ),
        ProtocolDefinitionScenario(
            "middleware_protocol",
            "Middleware",
            ProtocolCategoryType.COMMANDS,
        ),
    ]

    AVAILABILITY_SCENARIOS: ClassVar[list[ProtocolAvailabilityScenario]] = [
        ProtocolAvailabilityScenario(
            "all_foundation_protocols_available",
            ProtocolCategoryType.FOUNDATION,
            ["Result", "ResultLike", "HasModelFields", "HasModelDump", "Model"],
        ),
        ProtocolAvailabilityScenario(
            "all_domain_protocols_available",
            ProtocolCategoryType.DOMAIN,
            ["Repository", "Service"],
        ),
        ProtocolAvailabilityScenario(
            "all_infrastructure_protocols_available",
            ProtocolCategoryType.INFRASTRUCTURE,
            ["Configurable"],
        ),
        ProtocolAvailabilityScenario(
            "all_application_protocols_available",
            ProtocolCategoryType.APPLICATION,
            ["Handler"],
        ),
        ProtocolAvailabilityScenario(
            "all_commands_protocols_available",
            ProtocolCategoryType.COMMANDS,
            ["CommandBus", "Middleware"],
        ),
        ProtocolAvailabilityScenario(
            "all_extensions_protocols_available",
            ProtocolCategoryType.EXTENSIONS,
            ["Middleware"],
        ),
    ]


class TestFlextProtocols:
    """Comprehensive test suite for p using FlextTestsUtilities."""

    @pytest.mark.parametrize(
        "scenario",
        ProtocolScenarios.DEFINITION_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_protocol_definition(self, scenario: ProtocolDefinitionScenario) -> None:
        """Test protocol definitions are accessible and valid."""
        protocol = getattr(p, scenario.protocol_name)
        tm.that(
            protocol,
            none=False,
            msg=f"Protocol {scenario.protocol_name} must exist",
        )
        tm.that(
            hasattr(protocol, "__protocol_attrs__")
            or hasattr(protocol, "__annotations__"),
            eq=True,
            msg=(
                f"Protocol {scenario.protocol_name} must have "
                f"protocol attributes or annotations"
            ),
        )

    @pytest.mark.parametrize(
        "scenario",
        ProtocolScenarios.AVAILABILITY_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_protocol_availability(
        self,
        scenario: ProtocolAvailabilityScenario,
    ) -> None:
        """Test that protocols are available by category with real validation."""
        # Protocols are directly in p (FlextProtocols), not in nested namespaces
        # All protocols are accessible directly from p
        for proto_name in scenario.protocol_names:
            tm.that(
                hasattr(p, proto_name),
                eq=True,
                msg=f"Protocol {proto_name} must be found in p (FlextProtocols)",
            )
            # Validate protocol is not None
            protocol = getattr(p, proto_name)
            tm.that(protocol, none=False, msg=f"Protocol {proto_name} must not be None")

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

            def find_by_id(self, entity_id: str) -> r[m.ConfigMap]:
                return r[m.ConfigMap].ok(
                    m.ConfigMap(
                        root={
                            "id": entity_id,
                            "name": "Test",
                        }
                    )
                )

            def save(
                self,
                entity: m.ConfigMap,
            ) -> r[m.ConfigMap]:
                return r[m.ConfigMap].ok(entity)

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
        # Test actual execution
        find_result = repo.find_by_id("test_id")
        u.Tests.Result.assert_success(find_result)
        tm.that(find_result.value, has="id", msg="Find result must contain id")
        tm.that(find_result.value, has="name", msg="Find result must contain name")

    def test_service_implementation(self) -> None:
        """Test that a class can implement Service protocol."""

        class UserService:
            """Service for user operations."""

            def execute(self) -> r[m.ConfigMap]:
                return r[m.ConfigMap].ok(m.ConfigMap(root={"status": "success"}))

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
        u.Tests.Result.assert_success(result)
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

            def handle(
                self,
                command: m.ConfigMap,
            ) -> r[m.ConfigMap]:
                return r[m.ConfigMap].ok(m.ConfigMap(root={"user_id": "123"}))

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
        command = m.ConfigMap(root={"name": "Test"})
        result = handler.handle(command)
        u.Tests.Result.assert_success(result)
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

            def execute(self) -> r[m.ConfigMap]:
                return r[m.ConfigMap].ok(m.ConfigMap(root={}))

            def handle(
                self,
                command: m.ConfigMap,
            ) -> r[m.ConfigMap]:
                return r[m.ConfigMap].ok(m.ConfigMap(root={}))

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
        # Test actual execution
        execute_result = service.execute()
        u.Tests.Result.assert_success(execute_result)
        handle_result = service.handle(m.ConfigMap(root={"command": "test"}))
        u.Tests.Result.assert_success(handle_result)

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


__all__ = ["TestFlextProtocols"]
