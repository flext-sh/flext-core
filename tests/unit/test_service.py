"""FlextService core functionality tests.

Module: flext_core.service
Scope: FlextService abstract base class - execution, validation, metadata

Tests core FlextService functionality including:
- Service creation and Pydantic configuration
- Service immutability (frozen model)
- Abstract execute method implementation
- Basic service execution with FlextResult
- Business rules validation (success, failure, exception handling)
- Service metadata retrieval

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
from pydantic import ValidationError

from flext_core import FlextResult, FlextService


class ServiceScenarioType(StrEnum):
    """Service scenario types for parametrized testing."""

    BASIC_USER = "basic_user"
    COMPLEX_VALID = "complex_valid"
    COMPLEX_INVALID = "complex_invalid"
    FAILING = "failing"
    EXCEPTION = "exception"


@dataclass(frozen=True, slots=True)
class ServiceScenario:
    """Service test scenario definition."""

    name: str
    scenario_type: ServiceScenarioType
    is_valid_expected: bool
    service_kwargs: dict[str, bool | str] | None = None


class UserService(FlextService[dict]):
    """Basic user service for standard testing."""

    def execute(self, **_kwargs: object) -> FlextResult[dict]:
        """Execute service and return data."""
        return FlextResult[dict].ok({"user_id": 1, "name": "test_user"})


class ComplexService(FlextService[object]):
    """Service with custom validation rules."""

    name: str = "test"
    amount: int = 0
    enabled: bool = True

    def validate_business_rules(self) -> FlextResult[bool]:
        """Validate business rules."""
        if not self.name:
            return FlextResult[bool].fail("Missing value")
        if self.amount < 0:
            return FlextResult[bool].fail("Value too low")
        return FlextResult[bool].ok(True)

    def execute(self, **_kwargs: object) -> FlextResult[object]:
        """Execute operation."""
        if not self.name:
            return FlextResult[object].fail("Missing value")
        return FlextResult[object].ok(f"Processed: {self.name}")


class FailingService(FlextService[bool]):
    """Service that fails validation."""

    def validate_business_rules(self) -> FlextResult[bool]:
        """Always fail validation."""
        return FlextResult[bool].fail("Processing error")

    def execute(self, **_kwargs: object) -> FlextResult[bool]:
        """Execute failing operation."""
        return FlextResult[bool].fail("Processing error")


class ExceptionService(FlextService[str]):
    """Service that raises exceptions during validation."""

    should_raise: bool = False

    def validate_business_rules(self) -> FlextResult[bool]:
        """Validation that can raise exceptions."""
        if self.should_raise:
            error_msg = "Processing error"
            raise ValueError(error_msg)
        return FlextResult[bool].ok(True)

    def execute(self, **_kwargs: object) -> FlextResult[str]:
        """Execute operation that can raise."""
        if self.should_raise:
            error_msg = "Processing error"
            raise RuntimeError(error_msg)
        return FlextResult[str].ok("test_value")


class ServiceScenarios:
    """Centralized service test scenarios using FlextConstants."""

    SCENARIOS: ClassVar[list[ServiceScenario]] = [
        ServiceScenario("basic_user_service", ServiceScenarioType.BASIC_USER, True),
        ServiceScenario(
            "complex_valid", ServiceScenarioType.COMPLEX_VALID, True, {"name": "test"}
        ),
        ServiceScenario(
            "complex_invalid", ServiceScenarioType.COMPLEX_INVALID, False, {"name": ""}
        ),
        ServiceScenario("failing_service", ServiceScenarioType.FAILING, False),
        ServiceScenario(
            "exception_handling",
            ServiceScenarioType.EXCEPTION,
            False,
            {"should_raise": True},
        ),
    ]

    @staticmethod
    def create_service(scenario: ServiceScenario) -> FlextService[object]:
        """Create service instance for scenario."""
        kwargs = scenario.service_kwargs or {}
        if scenario.scenario_type == ServiceScenarioType.BASIC_USER:
            return UserService()
        if scenario.scenario_type == ServiceScenarioType.COMPLEX_VALID:
            return ComplexService(**kwargs)
        if scenario.scenario_type == ServiceScenarioType.COMPLEX_INVALID:
            return ComplexService(**kwargs)
        if scenario.scenario_type == ServiceScenarioType.FAILING:
            return FailingService()
        if scenario.scenario_type == ServiceScenarioType.EXCEPTION:
            return ExceptionService(**kwargs)
        error_msg = f"Unknown scenario type: {scenario.scenario_type}"
        raise ValueError(error_msg)


class TestFlextServiceCore:
    """Unified test suite for FlextService using FlextTestsUtilities."""

    def test_basic_service_creation(self) -> None:
        """Test basic service creation and Pydantic configuration."""
        service = UserService()
        assert isinstance(service, FlextService)
        assert isinstance(service.model_config, dict)
        assert service.model_config.get("validate_assignment") is True

    def test_service_immutability(self) -> None:
        """Test service immutability (frozen model)."""
        service = UserService()
        with pytest.raises(ValidationError):
            service.new_field = "test"

    def test_execute_abstract_method(self) -> None:
        """Test execute method implementation."""

        class ConcreteService(FlextService[str]):
            def execute(self, **_kwargs: object) -> FlextResult[str]:
                return FlextResult[str].ok("test_value")

        service = ConcreteService()
        result = service.execute()
        assert result.is_success
        assert result.value == "test_value"

    def test_basic_execution(self) -> None:
        """Test basic service execution returns expected type."""
        service = UserService()
        result = service.execute()
        assert result.is_success
        data = result.value
        assert isinstance(data, dict)
        assert "user_id" in data

    @pytest.mark.parametrize(
        "scenario", ServiceScenarios.SCENARIOS, ids=lambda s: s.name
    )
    def test_is_valid_scenarios(self, scenario: ServiceScenario) -> None:
        """Test is_valid with various service scenarios."""
        service = ServiceScenarios.create_service(scenario)
        assert service.is_valid() is scenario.is_valid_expected

    def test_validate_business_rules_default(self) -> None:
        """Test default business rules validation."""
        service = UserService()
        result = service.validate_business_rules()
        assert result.is_success

    def test_validate_business_rules_custom_success(self) -> None:
        """Test custom business rules validation success."""
        service = ComplexService(name="test")
        result = service.validate_business_rules()
        assert result.is_success

    def test_validate_business_rules_custom_failure(self) -> None:
        """Test custom business rules validation failure."""
        service = ComplexService(name="")
        result = service.validate_business_rules()
        assert result.is_failure
        assert result.error is not None
        assert "Missing value" in result.error

    def test_get_service_info(self) -> None:
        """Test get_service_info returns proper metadata."""
        service = UserService()
        info = service.get_service_info()
        assert isinstance(info, dict)
        assert "service_type" in info


__all__ = ["TestFlextServiceCore"]
