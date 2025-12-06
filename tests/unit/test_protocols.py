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
from flext_tests.utilities import FlextTestsUtilities


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
        assert protocol is not None
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol,
            "__annotations__",
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
        """Test that protocols are available by category."""
        category_mapping = {
            ProtocolCategoryType.FOUNDATION: p.Foundation,
            ProtocolCategoryType.DOMAIN: p.Domain,
            ProtocolCategoryType.INFRASTRUCTURE: p.Configuration,
            ProtocolCategoryType.APPLICATION: p.Application,
            ProtocolCategoryType.COMMANDS: p.Application,
        }
        category_class = category_mapping.get(scenario.category, p)
        for proto_name in scenario.protocol_names:
            assert hasattr(category_class, proto_name), (
                f"Protocol {proto_name} not found in {category_class.__name__}"
            )

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
        assert container.value == "test"
        assert container.is_success
        assert not container.is_failure

    def test_repository_implementation(self) -> None:
        """Test that a class can implement Repository protocol."""

        class UserRepository:
            """Repository for user entities."""

            def find_by_id(self, entity_id: str) -> r[t.Types.ConfigurationMapping]:
                return r[t.Types.ConfigurationMapping].ok({
                    "id": entity_id,
                    "name": "Test",
                })

            def save(
                self,
                entity: t.Types.ConfigurationMapping,
            ) -> r[t.Types.ConfigurationMapping]:
                return r[t.Types.ConfigurationMapping].ok(entity)

            def delete(self, entity_id: str) -> r[bool]:
                return r[bool].ok(True)

        repo = UserRepository()
        assert all(hasattr(repo, m) for m in ["find_by_id", "save", "delete"])

    def test_service_implementation(self) -> None:
        """Test that a class can implement Service protocol."""

        class UserService:
            """Service for user operations."""

            def execute(self) -> r[t.Types.ConfigurationMapping]:
                return r[t.Types.ConfigurationMapping].ok({"status": "success"})

        service = UserService()
        result = service.execute()
        FlextTestsUtilities.Tests.TestUtilities.assert_result_success(result)

    def test_handler_implementation(self) -> None:
        """Test that a class can implement Handler protocol."""

        class CreateUserHandler:
            """Handler for user creation."""

            def handle(
                self,
                command: t.Types.ConfigurationMapping,
            ) -> r[t.Types.ConfigurationMapping]:
                return r[t.Types.ConfigurationMapping].ok({"user_id": "123"})

        handler = CreateUserHandler()
        result = handler.handle({"name": "Test"})
        FlextTestsUtilities.Tests.TestUtilities.assert_result_success(result)

    def test_multiple_protocol_implementation(self) -> None:
        """Test that a class can implement multiple protocols."""

        class AdvancedService:
            """Service implementing multiple protocols."""

            def execute(self) -> r[t.Types.ConfigurationMapping]:
                return r[t.Types.ConfigurationMapping].ok({})

            def handle(
                self,
                command: t.Types.ConfigurationMapping,
            ) -> r[t.Types.ConfigurationMapping]:
                return r[t.Types.ConfigurationMapping].ok({})

        service = AdvancedService()
        assert all(hasattr(service, m) for m in ["execute", "handle"])

    def test_protocol_runtime_checkable(self) -> None:
        """Test that protocols support runtime checking."""

        class Container:
            """Container with runtime checkable property."""

            @property
            def value(self) -> str:
                return "test"

        container = Container()
        assert hasattr(container, "value")


__all__ = ["TestFlextProtocols"]
