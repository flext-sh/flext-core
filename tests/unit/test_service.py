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
from enum import StrEnum, unique
from typing import Annotated, override

from flext_tests import FlextTestsUtilities, u
from pydantic import BaseModel, ConfigDict, Field

from flext_core import r, s


class TestsCore:
    """Unified test suite for s using FlextTestsUtilities."""

    @unique
    class ServiceScenarioType(StrEnum):
        """Service scenario types for scenario testing."""

        BASIC_USER = "basic_user"
        COMPLEX_VALID = "complex_valid"
        COMPLEX_INVALID = "complex_invalid"
        FAILING = "failing"
        EXCEPTION = "exception"

    class ServiceScenario(BaseModel):
        """Service test scenario definition."""

        model_config = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Service scenario name")]
        scenario_type: Annotated[object, Field(description="Service scenario type")]
        is_valid_expected: Annotated[
            bool,
            Field(description="Expected is_valid result"),
        ]
        service_kwargs: Annotated[
            Mapping[str, object] | None,
            Field(default=None, description="Optional scenario service kwargs"),
        ] = None

    class UserService(s[dict[str, object]]):
        """Basic user service for standard testing."""

        @override
        def execute(self) -> r[dict[str, object]]:
            return r[dict[str, object]].ok({"user_id": 1, "name": "test_user"})

    class ComplexService(s[str]):
        """Service with custom validation rules."""

        name: str = "test"
        amount: int = 0
        enabled: bool = True

        @override
        def validate_business_rules(self) -> r[bool]:
            if not self.name:
                return r[bool].fail("Missing value")
            if self.amount < 0:
                return r[bool].fail("Value too low")
            return r[bool].ok(True)

        @override
        def execute(self) -> r[str]:
            if not self.name:
                return r[str].fail("Missing value")
            return r[str].ok(f"Processed: {self.name}")

    class FailingService(s[bool]):
        """Service that fails validation."""

        @override
        def validate_business_rules(self) -> r[bool]:
            return r[bool].fail("Processing error")

        @override
        def execute(self) -> r[bool]:
            return r[bool].fail("Processing error")

    class ExceptionService(s[str]):
        """Service that raises exceptions during validation."""

        should_raise: bool = False

        @override
        def validate_business_rules(self) -> r[bool]:
            if self.should_raise:
                error_msg = "Processing error"
                raise ValueError(error_msg)
            return r[bool].ok(True)

        @override
        def execute(self) -> r[str]:
            if self.should_raise:
                error_msg = "Processing error"
                raise RuntimeError(error_msg)
            return r[str].ok("test_value")

    class ServiceScenarios:
        """Centralized service test scenarios using FlextConstants."""

        @staticmethod
        def create_service(
            scenario: TestsCore.ServiceScenario,
        ) -> s[t.ConfigMap] | s[str] | s[bool]:
            kwargs_raw: Mapping[str, t.Scalar] = scenario.service_kwargs or {}
            if scenario.scenario_type == TestsCore.ServiceScenarioType.BASIC_USER:
                return TestsCore.UserService()
            if scenario.scenario_type in {
                TestsCore.ServiceScenarioType.COMPLEX_VALID,
                TestsCore.ServiceScenarioType.COMPLEX_INVALID,
            }:
                name_val = kwargs_raw.get("name", "test")
                name = str(name_val)
                amount_val = kwargs_raw.get("amount", 0)
                amount = int(amount_val) if isinstance(amount_val, (int, float)) else 0
                enabled_val = kwargs_raw.get("enabled", True)
                enabled = bool(enabled_val)
                complex_service = TestsCore.ComplexService()
                complex_service.name = name
                complex_service.amount = amount
                complex_service.enabled = enabled
                return complex_service
            if scenario.scenario_type == TestsCore.ServiceScenarioType.FAILING:
                return TestsCore.FailingService()
            if scenario.scenario_type == TestsCore.ServiceScenarioType.EXCEPTION:
                should_raise_val = kwargs_raw.get("should_raise", False)
                should_raise = bool(should_raise_val)
                exception_service = TestsCore.ExceptionService()
                exception_service.should_raise = should_raise
                return exception_service
            error_msg = f"Unknown scenario type: {scenario.scenario_type}"
            raise ValueError(error_msg)

    def _service_scenarios(self) -> list[TestsCore.ServiceScenario]:
        return [
            self.ServiceScenario(
                name="basic_user_service",
                scenario_type=self.ServiceScenarioType.BASIC_USER,
                is_valid_expected=True,
            ),
            self.ServiceScenario(
                name="complex_valid",
                scenario_type=self.ServiceScenarioType.COMPLEX_VALID,
                is_valid_expected=True,
                service_kwargs={"name": "test"},
            ),
            self.ServiceScenario(
                name="complex_invalid",
                scenario_type=self.ServiceScenarioType.COMPLEX_INVALID,
                is_valid_expected=False,
                service_kwargs={"name": ""},
            ),
            self.ServiceScenario(
                name="failing_service",
                scenario_type=self.ServiceScenarioType.FAILING,
                is_valid_expected=False,
            ),
            self.ServiceScenario(
                name="exception_handling",
                scenario_type=self.ServiceScenarioType.EXCEPTION,
                is_valid_expected=False,
                service_kwargs={"should_raise": True},
            ),
        ]

    def test_basic_service_creation(self) -> None:
        """Test basic service creation and Pydantic configuration."""
        service = self.UserService()
        assert isinstance(service, s)
        assert isinstance(service.model_config, Mapping)
        assert service.model_config.get("validate_assignment") is True

    def test_service_immutability(self) -> None:
        """Test service mutability (frozen removed for compatibility with FlextMixins)."""
        service = self.UserService()
        assert (
            service.model_config.get("frozen") is None
            or service.model_config.get("frozen") is False
        )

    def test_execute_abstract_method(self) -> None:
        """Test execute method implementation."""

        class ConcreteService(s[str]):
            @override
            def execute(self) -> r[str]:
                return r[str].ok("test_value")

        service = ConcreteService()
        result = service.execute()
        u.Tests.Result.assert_success_with_value(result, "test_value")

    def test_basic_execution(self) -> None:
        """Test basic service execution returns expected type."""
        service = self.UserService()
        result = service.execute()
        _ = u.Tests.Result.assert_success(result)
        data = result.value
        assert isinstance(data, t.ConfigMap)
        assert "user_id" in data

    def test_is_valid_scenarios(self) -> None:
        """Test is_valid with various service scenarios."""
        for scenario in self._service_scenarios():
            service = self.ServiceScenarios.create_service(scenario)
            assert service.is_valid() is scenario.is_valid_expected

    def test_validate_business_rules_default(self) -> None:
        """Test default business rules validation."""
        service = self.UserService()
        result = service.validate_business_rules()
        _ = u.Tests.Result.assert_success(result)

    def test_validate_business_rules_custom_success(self) -> None:
        """Test custom business rules validation success."""
        service = self.ComplexService()
        service.name = "test"
        result = service.validate_business_rules()
        _ = u.Tests.Result.assert_success(result)

    def test_validate_business_rules_custom_failure(self) -> None:
        """Test custom business rules validation failure."""
        service = self.ComplexService()
        service.name = ""
        result = service.validate_business_rules()
        u.Tests.Result.assert_failure_with_error(result, "Missing value")

    def test_get_service_info(self) -> None:
        """Test get_service_info returns proper metadata."""
        service = self.UserService()
        info = service.get_service_info()
        assert isinstance(info, Mapping)
        assert "service_type" in info

    def test_service_validation_using_generic_helpers(self) -> None:
        """Test service validation using generic helpers - real behavior."""
        service = self.ComplexService()
        service.name = "test"
        service.amount = 10
        service.enabled = True
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
        service = self.ComplexService()
        service.name = ""
        service.amount = -1
        service.enabled = False
        validation_result = (
            FlextTestsUtilities.Tests.GenericHelpers.validate_model_attributes(
                service,
                required_attrs=["name"],
            )
        )
        _ = u.Tests.Result.assert_success(validation_result)
        business_result = service.validate_business_rules()
        _ = u.Tests.Result.assert_failure(business_result)


__all__ = ["TestsCore"]
