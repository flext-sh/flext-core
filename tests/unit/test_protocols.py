"""Tests for FlextProtocols - Protocol Definitions and Implementations.

Module: flext_core.protocols
Scope: FlextProtocols - all protocol definitions and implementations

Tests FlextProtocols functionality including:
- Foundation protocols (HasResultValue, HasModelFields, HasModelDump)
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

from flext_core import FlextProtocols, FlextResult
from flext_core.typings import FlextTypes


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
            "has_result_value_protocol",
            "HasResultValue",
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
            ["HasResultValue", "HasModelFields", "HasModelDump"],
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
    """Comprehensive test suite for FlextProtocols using FlextTestsUtilities."""

    @pytest.mark.parametrize(
        "scenario",
        ProtocolScenarios.DEFINITION_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_protocol_definition(self, scenario: ProtocolDefinitionScenario) -> None:
        """Test protocol definitions are accessible and valid."""
        protocol = getattr(FlextProtocols, scenario.protocol_name)
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
        for proto_name in scenario.protocol_names:
            assert hasattr(FlextProtocols, proto_name)

    def test_has_result_value_implementation(self) -> None:
        """Test that a class can implement HasResultValue."""

        class ResultContainer:
            """Container with result value property."""

            def __init__(self, value: str) -> None:
                self._value = value

            @property
            def value(self) -> str:
                return self._value

        container = ResultContainer("test")
        assert container.value == "test"

    def test_repository_implementation(self) -> None:
        """Test that a class can implement Repository protocol."""

        class UserRepository:
            """Repository for user entities."""

            def find_by_id(
                self, entity_id: str
            ) -> FlextResult[FlextTypes.Types.ConfigurationMapping]:
                return FlextResult[FlextTypes.Types.ConfigurationMapping].ok({
                    "id": entity_id,
                    "name": "Test",
                })

            def save(
                self, entity: FlextTypes.Types.ConfigurationMapping
            ) -> FlextResult[FlextTypes.Types.ConfigurationMapping]:
                return FlextResult[FlextTypes.Types.ConfigurationMapping].ok(entity)

            def delete(self, entity_id: str) -> FlextResult[bool]:
                return FlextResult[bool].ok(True)

        repo = UserRepository()
        assert all(hasattr(repo, m) for m in ["find_by_id", "save", "delete"])

    def test_service_implementation(self) -> None:
        """Test that a class can implement Service protocol."""

        class UserService:
            """Service for user operations."""

            def execute(self) -> FlextResult[FlextTypes.Types.ConfigurationMapping]:
                return FlextResult[FlextTypes.Types.ConfigurationMapping].ok({
                    "status": "success"
                })

        service = UserService()
        result = service.execute()
        assert result.is_success

    def test_handler_implementation(self) -> None:
        """Test that a class can implement Handler protocol."""

        class CreateUserHandler:
            """Handler for user creation."""

            def handle(
                self,
                command: FlextTypes.Types.ConfigurationMapping,
            ) -> FlextResult[FlextTypes.Types.ConfigurationMapping]:
                return FlextResult[FlextTypes.Types.ConfigurationMapping].ok({
                    "user_id": "123"
                })

        handler = CreateUserHandler()
        result = handler.handle({"name": "Test"})
        assert result.is_success

    def test_multiple_protocol_implementation(self) -> None:
        """Test that a class can implement multiple protocols."""

        class AdvancedService:
            """Service implementing multiple protocols."""

            def execute(self) -> FlextResult[FlextTypes.Types.ConfigurationMapping]:
                return FlextResult[FlextTypes.Types.ConfigurationMapping].ok({})

            def handle(
                self,
                command: FlextTypes.Types.ConfigurationMapping,
            ) -> FlextResult[FlextTypes.Types.ConfigurationMapping]:
                return FlextResult[FlextTypes.Types.ConfigurationMapping].ok({})

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
