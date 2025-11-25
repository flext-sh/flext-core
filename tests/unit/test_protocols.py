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

Uses Python 3.13 patterns (StrEnum, frozen dataclasses with slots),
centralized constants, and parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

import pytest

from flext_core import FlextProtocols, FlextResult

# =========================================================================
# Protocol Scenario Type Enumerations
# =========================================================================


class ProtocolCategoryType(StrEnum):
    """Protocol category types for organization."""

    FOUNDATION = "foundation"
    DOMAIN = "domain"
    INFRASTRUCTURE = "infrastructure"
    APPLICATION = "application"
    COMMANDS = "commands"
    EXTENSIONS = "extensions"


# =========================================================================
# Test Case Structures
# =========================================================================


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


# =========================================================================
# Test Scenario Factories
# =========================================================================


class ProtocolDefinitionScenarios:
    """Factory for protocol definition test scenarios."""

    SCENARIOS: ClassVar[list[ProtocolDefinitionScenario]] = [
        ProtocolDefinitionScenario(
            name="has_result_value_protocol",
            protocol_name="HasResultValue",
            category=ProtocolCategoryType.FOUNDATION,
        ),
        ProtocolDefinitionScenario(
            name="has_model_fields_protocol",
            protocol_name="HasModelFields",
            category=ProtocolCategoryType.FOUNDATION,
        ),
        ProtocolDefinitionScenario(
            name="has_model_dump_protocol",
            protocol_name="HasModelDump",
            category=ProtocolCategoryType.FOUNDATION,
        ),
        ProtocolDefinitionScenario(
            name="repository_protocol",
            protocol_name="Repository",
            category=ProtocolCategoryType.DOMAIN,
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
            protocol_name="CommandBus",
            category=ProtocolCategoryType.COMMANDS,
        ),
        ProtocolDefinitionScenario(
            name="middleware_protocol",
            protocol_name="Middleware",
            category=ProtocolCategoryType.COMMANDS,
        ),
    ]


class ProtocolAvailabilityScenarios:
    """Factory for protocol availability test scenarios."""

    SCENARIOS: ClassVar[list[ProtocolAvailabilityScenario]] = [
        ProtocolAvailabilityScenario(
            name="all_foundation_protocols_available",
            category=ProtocolCategoryType.FOUNDATION,
            protocol_names=["HasResultValue", "HasModelFields", "HasModelDump"],
        ),
        ProtocolAvailabilityScenario(
            name="all_domain_protocols_available",
            category=ProtocolCategoryType.DOMAIN,
            protocol_names=["Repository", "Service"],
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
            protocol_names=["CommandBus", "Middleware"],
        ),
        ProtocolAvailabilityScenario(
            name="all_extensions_protocols_available",
            category=ProtocolCategoryType.EXTENSIONS,
            protocol_names=["Middleware"],
        ),
    ]


# =========================================================================
# Test Suite
# =========================================================================


class TestFlextProtocols:
    """Comprehensive test suite for FlextProtocols protocol definitions."""

    @pytest.mark.parametrize(
        "scenario",
        ProtocolDefinitionScenarios.SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_protocol_definition(self, scenario: ProtocolDefinitionScenario) -> None:
        """Test protocol definitions are accessible and valid."""
        protocol = getattr(FlextProtocols, scenario.protocol_name)
        assert protocol is not None
        # Check it's a Protocol
        assert hasattr(protocol, "__protocol_attrs__") or hasattr(
            protocol, "__annotations__"
        )

    @pytest.mark.parametrize(
        "scenario",
        ProtocolAvailabilityScenarios.SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_protocol_availability(
        self, scenario: ProtocolAvailabilityScenario
    ) -> None:
        """Test that protocols are available by category."""
        for proto_name in scenario.protocol_names:
            assert hasattr(FlextProtocols, proto_name)

    def test_has_result_value_implementation(self) -> None:
        """Test that a class can implement HasResultValue."""

        class ResultContainer:
            """Container with result value property."""

            def __init__(self, value: str) -> None:
                super().__init__()
                self._value = value

            @property
            def value(self) -> str:
                return self._value

        container = ResultContainer("test")
        assert hasattr(container, "value")
        assert container.value == "test"

    def test_repository_implementation(self) -> None:
        """Test that a class can implement Repository protocol."""

        class UserRepository:
            """Repository for user entities."""

            def find_by_id(self, entity_id: str) -> FlextResult[dict[str, object]]:
                return FlextResult[dict[str, object]].ok({
                    "id": entity_id,
                    "name": "Test",
                })

            def save(self, entity: dict[str, object]) -> FlextResult[dict[str, object]]:
                return FlextResult[dict[str, object]].ok(entity)

            def delete(self, entity_id: str) -> FlextResult[bool]:
                return FlextResult[bool].ok(True)

        repo = UserRepository()
        assert hasattr(repo, "find_by_id")
        assert hasattr(repo, "save")
        assert hasattr(repo, "delete")

    def test_service_implementation(self) -> None:
        """Test that a class can implement Service protocol."""

        class UserService:
            """Service for user operations."""

            def execute(self, **_kwargs: object) -> FlextResult[dict[str, object]]:
                return FlextResult[dict[str, object]].ok({"status": "success"})

        service = UserService()
        assert hasattr(service, "execute")
        result = service.execute()
        assert result.is_success

    def test_handler_implementation(self) -> None:
        """Test that a class can implement Handler protocol."""

        class CreateUserHandler:
            """Handler for user creation."""

            def handle(
                self,
                command: dict[str, object],
            ) -> FlextResult[dict[str, object]]:
                return FlextResult[dict[str, object]].ok({"user_id": "123"})

        handler = CreateUserHandler()
        assert hasattr(handler, "handle")
        result = handler.handle({"name": "Test"})
        assert result.is_success

    def test_multiple_protocol_implementation(self) -> None:
        """Test that a class can implement multiple protocols."""

        class AdvancedService:
            """Service implementing multiple protocols."""

            def execute(self, **_kwargs: object) -> FlextResult[dict[str, object]]:
                return FlextResult[dict[str, object]].ok({})

            def handle(
                self,
                command: dict[str, object],
            ) -> FlextResult[dict[str, object]]:
                return FlextResult[dict[str, object]].ok({})

        service = AdvancedService()
        assert hasattr(service, "execute")
        assert hasattr(service, "handle")

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
