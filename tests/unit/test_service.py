"""s core functionality tests.

Module: flext_core.service
Scope: s abstract base class - execution, validation, metadata

Tests core s functionality including:
- Service creation and Pydantic configuration
- Service immutability (frozen model)
- Abstract execute method implementation
- Basic service execution with r
- Business rules validation (success, failure, exception handling)
- Service metadata retrieval

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

import pytest

from flext_core import r, s, t
from flext_tests import FlextTestsUtilities, u


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
    service_kwargs: Mapping[str, t.ScalarValue] | None = None


class UserService(s[t.Types.ConfigurationMapping]):
    """Basic user service for standard testing."""

    def execute(self) -> r[t.Types.ConfigurationMapping]:
        """Execute service and return data."""
        return r[t.Types.ConfigurationMapping].ok(
            {
                "user_id": 1,
                "name": "test_user",
            }
        )


class ComplexService(s[str]):
    """Service with custom validation rules."""

    name: str = "test"
    amount: int = 0
    enabled: bool = True

    def validate_business_rules(self) -> r[bool]:
        """Validate business rules."""
        if not self.name:
            return r[bool].fail("Missing value")
        if self.amount < 0:
            return r[bool].fail("Value too low")
        return r[bool].ok(True)

    def execute(self) -> r[str]:
        """Execute operation."""
        if not self.name:
            return r[str].fail("Missing value")
        return r[str].ok(f"Processed: {self.name}")


class FailingService(s[bool]):
    """Service that fails validation."""

    def validate_business_rules(self) -> r[bool]:
        """Always fail validation."""
        return r[bool].fail("Processing error")

    def execute(self) -> r[bool]:
        """Execute failing operation."""
        return r[bool].fail("Processing error")


class ExceptionService(s[str]):
    """Service that raises exceptions during validation."""

    should_raise: bool = False

    def validate_business_rules(self) -> r[bool]:
        """Validation that can raise exceptions."""
        if self.should_raise:
            error_msg = "Processing error"
            raise ValueError(error_msg)
        return r[bool].ok(True)

    def execute(self) -> r[str]:
        """Execute operation that can raise."""
        if self.should_raise:
            error_msg = "Processing error"
            raise RuntimeError(error_msg)
        return r[str].ok("test_value")


class ServiceScenarios:
    """Centralized service test scenarios using FlextConstants."""

    SCENARIOS: ClassVar[list[ServiceScenario]] = [
        ServiceScenario("basic_user_service", ServiceScenarioType.BASIC_USER, True),
        ServiceScenario(
            "complex_valid",
            ServiceScenarioType.COMPLEX_VALID,
            True,
            {"name": "test"},
        ),
        ServiceScenario(
            "complex_invalid",
            ServiceScenarioType.COMPLEX_INVALID,
            False,
            {"name": ""},
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
    def create_service(
        scenario: ServiceScenario,
    ) -> s[t.Types.ConfigurationMapping] | s[str] | s[bool]:
        """Create service instance for scenario."""
        kwargs_raw: Mapping[str, t.ScalarValue] = scenario.service_kwargs or {}

        if scenario.scenario_type == ServiceScenarioType.BASIC_USER:
            return UserService()

        if scenario.scenario_type in {
            ServiceScenarioType.COMPLEX_VALID,
            ServiceScenarioType.COMPLEX_INVALID,
        }:
            # Extract and convert types for ComplexService with safe defaults
            name_val = kwargs_raw.get("name", "test")
            name = str(name_val) if name_val is not None else "test"

            amount_val = kwargs_raw.get("amount", 0)
            amount = int(amount_val) if isinstance(amount_val, (int, float)) else 0

            enabled_val = kwargs_raw.get("enabled", True)
            enabled = bool(enabled_val) if enabled_val is not None else True

            return ComplexService(name=name, amount=amount, enabled=enabled)

        if scenario.scenario_type == ServiceScenarioType.FAILING:
            return FailingService()

        if scenario.scenario_type == ServiceScenarioType.EXCEPTION:
            should_raise_val = kwargs_raw.get("should_raise", False)
            should_raise = (
                bool(should_raise_val) if should_raise_val is not None else False
            )
            return ExceptionService(should_raise=should_raise)

        error_msg = f"Unknown scenario type: {scenario.scenario_type}"
        raise ValueError(error_msg)


class TestsCore:
    """Unified test suite for s using FlextTestsUtilities."""

    def test_basic_service_creation(self) -> None:
        """Test basic service creation and Pydantic configuration."""
        service = UserService()
        assert isinstance(service, s)
        assert isinstance(service.model_config, Mapping)
        assert service.model_config.get("validate_assignment") is True

    def test_service_immutability(self) -> None:
        """Test service mutability (frozen removed for compatibility with FlextMixins)."""
        service = UserService()
        # frozen=True was removed from FlextService to allow direct attribute assignment
        # compatible with FlextMixins pattern (e.g., _runtime assignment)
        # Service is now mutable to support runtime initialization
        assert (
            service.model_config.get("frozen") is None
            or service.model_config.get("frozen") is False
        )

    def test_execute_abstract_method(self) -> None:
        """Test execute method implementation."""

        class ConcreteService(s[str]):
            def execute(self) -> r[str]:
                return r[str].ok("test_value")

        service = ConcreteService()
        result = service.execute()
        u.Tests.Result.assert_success_with_value(
            result,
            "test_value",
        )

    def test_basic_execution(self) -> None:
        """Test basic service execution returns expected type."""
        service = UserService()
        result = service.execute()
        u.Tests.Result.assert_result_success(result)
        data = result.value
        assert isinstance(data, Mapping)
        assert "user_id" in data

    @pytest.mark.parametrize(
        "scenario",
        ServiceScenarios.SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_is_valid_scenarios(self, scenario: ServiceScenario) -> None:
        """Test is_valid with various service scenarios."""
        service = ServiceScenarios.create_service(scenario)
        assert service.is_valid() is scenario.is_valid_expected

    def test_validate_business_rules_default(self) -> None:
        """Test default business rules validation."""
        service = UserService()
        result = service.validate_business_rules()
        u.Tests.Result.assert_result_success(result)

    def test_validate_business_rules_custom_success(self) -> None:
        """Test custom business rules validation success."""
        service = ComplexService(name="test")
        result = service.validate_business_rules()
        u.Tests.Result.assert_result_success(result)

    def test_validate_business_rules_custom_failure(self) -> None:
        """Test custom business rules validation failure."""
        service = ComplexService(name="")
        result = service.validate_business_rules()
        u.Tests.Result.assert_failure_with_error(
            result,
            "Missing value",
        )

    def test_get_service_info(self) -> None:
        """Test get_service_info returns proper metadata."""
        service = UserService()
        info = service.get_service_info()
        assert isinstance(info, Mapping)
        assert "service_type" in info

    def test_service_validation_using_generic_helpers(self) -> None:
        """Test service validation using generic helpers - real behavior."""
        # Use generic helper to validate model attributes
        service = ComplexService(name="test", amount=10, enabled=True)
        validation_result = (
            FlextTestsUtilities.Tests.GenericHelpers.validate_model_attributes(
                service,
                required_attrs=["name", "amount", "enabled"],
                optional_attrs=["validate_business_rules"],
            )
        )
        assert validation_result.is_success

    def test_service_validation_failure_limits(self) -> None:
        """Test service validation failure - limit cases."""
        # Test with missing required attributes
        service = ComplexService(name="", amount=-1, enabled=False)
        validation_result = (
            FlextTestsUtilities.Tests.GenericHelpers.validate_model_attributes(
                service,
                required_attrs=["name"],
            )
        )
        # Should pass attribute check, but business rules should fail
        u.Tests.Result.assert_result_success(
            validation_result,
        )  # Attributes exist
        # But business rules validation should fail
        business_result = service.validate_business_rules()
        u.Tests.Result.assert_result_failure(business_result)


__all__ = ["TestsCore"]
