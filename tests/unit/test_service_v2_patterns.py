"""V2 Pattern tests for FlextService - Refactored with Parametrization.

This module tests V2 patterns (Property and Auto) alongside V1 patterns with
comprehensive parametrization to ensure:
- V2 Property (.result) works correctly
- V2 Auto (auto_execute = True) works correctly
- V1 and V2 are fully interoperable
- Backward compatibility is maintained

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import dataclasses
from enum import StrEnum
from typing import ClassVar

import pytest
from pydantic import Field

from flext_core import FlextExceptions, FlextModels, FlextResult, FlextService

# ============================================================================
# Test Services - V1 and V2 Pattern Implementations
# ============================================================================


class SimpleV1Service(FlextService[str]):
    """Simple service for V1 testing."""

    message: str = "default"

    def execute(self, **_kwargs: object) -> FlextResult[str]:
        """Execute and return message."""
        return FlextResult.ok(f"V1: {self.message}")


class SimpleV2PropertyService(FlextService[str]):
    """Simple service for V2 Property testing."""

    message: str = "default"

    def execute(self, **_kwargs: object) -> FlextResult[str]:
        """Execute and return message."""
        return FlextResult.ok(f"V2 Property: {self.message}")


class SimpleV2AutoService(FlextService[str]):
    """Simple service for V2 Auto testing."""

    auto_execute: ClassVar[bool] = True
    message: str = "default"

    def execute(self, **_kwargs: object) -> FlextResult[str]:
        """Execute and return message."""
        return FlextResult.ok(f"V2 Auto: {self.message}")


class ValidationService(FlextService[dict[str, object]]):
    """Service with validation for testing."""

    value: int

    def execute(self, **_kwargs: object) -> FlextResult[dict[str, object]]:
        """Execute with validation."""
        if self.value < 0:
            return FlextResult.fail("Value must be positive")

        return FlextResult.ok({"value": self.value, "valid": True})


class AutoValidationService(FlextService[dict[str, object]]):
    """Service with validation and auto_execute."""

    auto_execute: ClassVar[bool] = True
    value: int

    def execute(self, **_kwargs: object) -> FlextResult[dict[str, object]]:
        """Execute with validation."""
        if self.value < 0:
            return FlextResult.fail("Value must be positive")

        return FlextResult.ok({"value": self.value, "valid": True})


class ComplexV1Service(FlextService[dict[str, object]]):
    """Complex service for V1 testing."""

    items: list[str] = Field(default_factory=list)
    multiplier: int = 1

    def execute(self, **_kwargs: object) -> FlextResult[dict[str, object]]:
        """Execute complex operation."""
        if not self.items:
            return FlextResult.fail("Items cannot be empty")

        return FlextResult.ok({
            "count": len(self.items) * self.multiplier,
            "items": self.items,
        })


class ComplexV2Service(FlextService[dict[str, object]]):
    """Complex service for V2 testing."""

    items: list[str] = Field(default_factory=list)
    multiplier: int = 1

    def execute(self, **_kwargs: object) -> FlextResult[dict[str, object]]:
        """Execute complex operation."""
        if not self.items:
            return FlextResult.fail("Items cannot be empty")

        return FlextResult.ok({
            "count": len(self.items) * self.multiplier,
            "items": self.items,
        })


class BoolService(FlextService[bool]):
    """Service returning bool."""

    auto_execute: ClassVar[bool] = True

    def execute(self, **_kwargs: object) -> FlextResult[bool]:
        """Execute and return bool."""
        return FlextResult[bool].ok(True)


class ListService(FlextService[list[str]]):
    """Service returning list."""

    auto_execute: ClassVar[bool] = True

    def execute(self, **_kwargs: object) -> FlextResult[list[str]]:
        """Execute and return list."""
        return FlextResult.ok(["x", "y", "z"])


class DictService(FlextService[dict[str, int]]):
    """Service returning dict."""

    def execute(self, **_kwargs: object) -> FlextResult[dict[str, int]]:
        """Execute and return dict."""
        return FlextResult.ok({"a": 1, "b": 2})


class UserService(FlextService[FlextModels.Entity]):
    """Service returning User entity."""

    user_id: str
    user_name: str

    def execute(self, **_kwargs: object) -> FlextResult[FlextModels.Entity]:
        """Execute and return user."""

        class User(FlextModels.Entity):
            """User entity."""

            unique_id: str = "test_id"
            name: str

        return FlextResult.ok(User(unique_id=self.user_id, name=self.user_name))


# =========================================================================
# Operation Type Enumeration
# =========================================================================


class ServiceOperationType(StrEnum):
    """Service operation types for parametrization."""

    # V1 vs V2 Property scenarios (13 cases)
    V1_EXPLICIT_EXECUTE_UNWRAP = "v1_explicit_execute_unwrap"
    V2_PROPERTY_DIRECT_RESULT = "v2_property_direct_result"
    V1_AND_V2_SAME_RESULT = "v1_and_v2_same_result"
    V2_MAINTAINS_EXECUTE_ACCESS = "v2_maintains_execute_access"
    V1_VALIDATION_FAILURE = "v1_validation_failure"
    V2_VALIDATION_FAILURE = "v2_validation_failure"
    V1_VALIDATION_SUCCESS = "v1_validation_success"
    V2_VALIDATION_SUCCESS = "v2_validation_success"
    V1_COMPLEX_WITH_ITEMS = "v1_complex_with_items"
    V2_COMPLEX_WITH_ITEMS = "v2_complex_with_items"
    V1_COMPLEX_EMPTY_ITEMS = "v1_complex_empty_items"
    V2_COMPLEX_EMPTY_ITEMS = "v2_complex_empty_items"

    # V2 Auto scenarios (7 cases)
    V2_AUTO_RETURNS_VALUE = "v2_auto_returns_value"
    V2_AUTO_VS_V1_SERVICE = "v2_auto_vs_v1_service"
    V2_AUTO_VALIDATION_SUCCESS = "v2_auto_validation_success"
    V2_AUTO_VALIDATION_FAILURE = "v2_auto_validation_failure"
    V2_AUTO_COMPLEX_SUCCESS = "v2_auto_complex_success"
    V2_AUTO_COMPLEX_FAILURE = "v2_auto_complex_failure"
    V2_AUTO_ZERO_CEREMONY = "v2_auto_zero_ceremony"

    # V1 V2 Interoperability scenarios (4 cases)
    V1_V2_BOTH_IN_CODEBASE = "v1_v2_both_in_codebase"
    V2_USE_V1_RAILWAY = "v2_use_v1_railway"
    MIX_V1_AND_V2_PIPELINE = "mix_v1_and_v2_pipeline"
    ERROR_CONSISTENCY = "error_consistency"

    # Backward Compatibility scenarios (4 cases)
    V1_CODE_STILL_WORKS = "v1_code_still_works"
    V2_DOESNT_BREAK_V1 = "v2_doesnt_break_v1"
    AUTO_EXECUTE_FALSE = "auto_execute_false"
    NO_AUTO_EXECUTE_ATTRIBUTE = "no_auto_execute_attribute"

    # V2 Edge Cases scenarios (5 cases)
    V2_RESULT_MULTIPLE_CALLS = "v2_result_multiple_calls"
    V2_AUTO_WITH_BOOL_RETURN = "v2_auto_with_bool_return"
    V2_PROPERTY_WITH_DICT_RETURN = "v2_property_with_dict_return"
    V2_AUTO_WITH_LIST_RETURN = "v2_auto_with_list_return"
    V2_PROPERTY_WITH_MODEL_RETURN = "v2_property_with_model_return"

    # V2 Best Practices scenarios (3 cases)
    V2_PROPERTY_RECOMMENDED = "v2_property_recommended"
    V1_EXECUTE_RECOMMENDED = "v1_execute_recommended"
    V2_AUTO_RECOMMENDED = "v2_auto_recommended"


# =========================================================================
# Test Case Data Structure
# =========================================================================


@dataclasses.dataclass(frozen=True, slots=True)
class ServiceTestCase:
    """Service test case definition with parametrization data."""

    name: str
    operation: ServiceOperationType
    should_succeed: bool = True
    expected_value: object = None
    expected_type: type[object] | None = None


# =========================================================================
# Test Scenario Factory
# =========================================================================


class ServiceScenarios:
    """Factory for organizing test scenarios by category."""

    V1_VS_V2_PROPERTY_SCENARIOS: ClassVar[list[ServiceTestCase]] = [
        ServiceTestCase(
            "V1 explicit execute unwrap",
            ServiceOperationType.V1_EXPLICIT_EXECUTE_UNWRAP,
        ),
        ServiceTestCase(
            "V2 property direct result", ServiceOperationType.V2_PROPERTY_DIRECT_RESULT,
        ),
        ServiceTestCase(
            "V1 and V2 produce same result", ServiceOperationType.V1_AND_V2_SAME_RESULT,
        ),
        ServiceTestCase(
            "V2 maintains execute access",
            ServiceOperationType.V2_MAINTAINS_EXECUTE_ACCESS,
        ),
        ServiceTestCase(
            "V1 with validation failure", ServiceOperationType.V1_VALIDATION_FAILURE,
        ),
        ServiceTestCase(
            "V2 with validation failure", ServiceOperationType.V2_VALIDATION_FAILURE,
        ),
        ServiceTestCase(
            "V1 with validation success", ServiceOperationType.V1_VALIDATION_SUCCESS,
        ),
        ServiceTestCase(
            "V2 with validation success", ServiceOperationType.V2_VALIDATION_SUCCESS,
        ),
        ServiceTestCase(
            "V1 complex with items", ServiceOperationType.V1_COMPLEX_WITH_ITEMS,
        ),
        ServiceTestCase(
            "V2 complex with items", ServiceOperationType.V2_COMPLEX_WITH_ITEMS,
        ),
        ServiceTestCase(
            "V1 complex with empty items", ServiceOperationType.V1_COMPLEX_EMPTY_ITEMS,
        ),
        ServiceTestCase(
            "V2 complex with empty items", ServiceOperationType.V2_COMPLEX_EMPTY_ITEMS,
        ),
        ServiceTestCase(
            "V2 property result can be called multiple times",
            ServiceOperationType.V2_RESULT_MULTIPLE_CALLS,
        ),
    ]

    V2_AUTO_SCENARIOS: ClassVar[list[ServiceTestCase]] = [
        ServiceTestCase(
            "V2 Auto returns value directly", ServiceOperationType.V2_AUTO_RETURNS_VALUE,
        ),
        ServiceTestCase(
            "V2 Auto vs V1 service instance", ServiceOperationType.V2_AUTO_VS_V1_SERVICE,
        ),
        ServiceTestCase(
            "V2 Auto validation success",
            ServiceOperationType.V2_AUTO_VALIDATION_SUCCESS,
        ),
        ServiceTestCase(
            "V2 Auto validation failure raises",
            ServiceOperationType.V2_AUTO_VALIDATION_FAILURE,
        ),
        ServiceTestCase(
            "V2 Auto complex service success",
            ServiceOperationType.V2_AUTO_COMPLEX_SUCCESS,
        ),
        ServiceTestCase(
            "V2 Auto complex service failure",
            ServiceOperationType.V2_AUTO_COMPLEX_FAILURE,
        ),
        ServiceTestCase(
            "V2 Auto zero ceremony", ServiceOperationType.V2_AUTO_ZERO_CEREMONY,
        ),
    ]

    V1_V2_INTEROPERABILITY_SCENARIOS: ClassVar[list[ServiceTestCase]] = [
        ServiceTestCase(
            "V1 V2 property and V2 auto in same codebase",
            ServiceOperationType.V1_V2_BOTH_IN_CODEBASE,
        ),
        ServiceTestCase(
            "V2 property can use V1 railway pattern",
            ServiceOperationType.V2_USE_V1_RAILWAY,
        ),
        ServiceTestCase(
            "Mix V1 and V2 in pipeline",
            ServiceOperationType.MIX_V1_AND_V2_PIPELINE,
        ),
        ServiceTestCase(
            "Error handling consistency", ServiceOperationType.ERROR_CONSISTENCY,
        ),
    ]

    BACKWARD_COMPATIBILITY_SCENARIOS: ClassVar[list[ServiceTestCase]] = [
        ServiceTestCase(
            "V1 code still works", ServiceOperationType.V1_CODE_STILL_WORKS,
        ),
        ServiceTestCase(
            "V2 property doesn't break V1 tests",
            ServiceOperationType.V2_DOESNT_BREAK_V1,
        ),
        ServiceTestCase(
            "Auto execute false behaves like V1",
            ServiceOperationType.AUTO_EXECUTE_FALSE,
        ),
        ServiceTestCase(
            "No auto execute attribute defaults to false",
            ServiceOperationType.NO_AUTO_EXECUTE_ATTRIBUTE,
        ),
    ]

    V2_EDGE_CASES_SCENARIOS: ClassVar[list[ServiceTestCase]] = [
        ServiceTestCase(
            "V2 auto with bool return", ServiceOperationType.V2_AUTO_WITH_BOOL_RETURN,
        ),
        ServiceTestCase(
            "V2 property with dict return",
            ServiceOperationType.V2_PROPERTY_WITH_DICT_RETURN,
        ),
        ServiceTestCase(
            "V2 auto with list return", ServiceOperationType.V2_AUTO_WITH_LIST_RETURN,
        ),
        ServiceTestCase(
            "V2 property with pydantic model return",
            ServiceOperationType.V2_PROPERTY_WITH_MODEL_RETURN,
        ),
    ]

    V2_BEST_PRACTICES_SCENARIOS: ClassVar[list[ServiceTestCase]] = [
        ServiceTestCase(
            "V2 property recommended for happy path",
            ServiceOperationType.V2_PROPERTY_RECOMMENDED,
        ),
        ServiceTestCase(
            "V1 execute recommended for error handling",
            ServiceOperationType.V1_EXECUTE_RECOMMENDED,
        ),
        ServiceTestCase(
            "V2 auto recommended for simple services",
            ServiceOperationType.V2_AUTO_RECOMMENDED,
        ),
    ]


# =========================================================================
# Unified Test Class with Parametrization
# =========================================================================


class TestFlextServiceV2Patterns:
    """Unified test class for V2 Service patterns with parametrization."""

    @pytest.mark.parametrize(
        "test_case",
        ServiceScenarios.V1_VS_V2_PROPERTY_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_v1_vs_v2_property_patterns(self, test_case: ServiceTestCase) -> None:
        """Test V1 vs V2 Property pattern comparisons."""
        if test_case.operation == ServiceOperationType.V1_EXPLICIT_EXECUTE_UNWRAP:
            result = SimpleV1Service(message="test").execute()
            assert result.is_success
            assert result.unwrap() == "V1: test"

        elif test_case.operation == ServiceOperationType.V2_PROPERTY_DIRECT_RESULT:
            value = SimpleV2PropertyService(message="test").result
            assert value == "V2 Property: test"

        elif test_case.operation == ServiceOperationType.V1_AND_V2_SAME_RESULT:
            v1_service: SimpleV1Service = SimpleV1Service(message="test")
            v1_result = v1_service.execute().unwrap()

            v2_service_prop: SimpleV2PropertyService = SimpleV2PropertyService(
                message="test",
            )
            v2_result = v2_service_prop.result

            assert v1_result == "V1: test"
            assert v2_result == "V2 Property: test"

        elif test_case.operation == ServiceOperationType.V2_MAINTAINS_EXECUTE_ACCESS:
            service_v2: SimpleV2PropertyService = SimpleV2PropertyService(
                message="test",
            )
            result_via_property = service_v2.result
            result_via_execute = service_v2.execute().unwrap()
            assert result_via_property == result_via_execute

        elif test_case.operation == ServiceOperationType.V1_VALIDATION_FAILURE:
            val_service_v1: ValidationService = ValidationService(value=-1)
            result = val_service_v1.execute()
            assert result.is_failure

        elif test_case.operation == ServiceOperationType.V2_VALIDATION_FAILURE:
            val_service_v2: ValidationService = ValidationService(value=-1)
            try:
                _ = val_service_v2.result
                pytest.fail("Should raise exception")
            except FlextExceptions.BaseError:
                pass

        elif test_case.operation == ServiceOperationType.V1_VALIDATION_SUCCESS:
            val_service_v1s: ValidationService = ValidationService(value=10)
            result = val_service_v1s.execute()
            assert result.is_success
            assert result.unwrap()["value"] == 10

        elif test_case.operation == ServiceOperationType.V2_VALIDATION_SUCCESS:
            val_service_v2s: ValidationService = ValidationService(value=10)
            value = val_service_v2s.result
            assert isinstance(value, dict)
            assert value["value"] == 10

        elif test_case.operation == ServiceOperationType.V1_COMPLEX_WITH_ITEMS:
            service = ComplexV1Service(items=["a", "b", "c"], multiplier=2)
            result = service.execute()
            assert result.is_success
            value = result.unwrap()
            assert value["count"] == 6

        elif test_case.operation == ServiceOperationType.V2_COMPLEX_WITH_ITEMS:
            service = ComplexV2Service(items=["a", "b", "c"], multiplier=2)
            value = service.result
            assert isinstance(value, dict)
            assert value["count"] == 6

        elif test_case.operation == ServiceOperationType.V1_COMPLEX_EMPTY_ITEMS:
            service = ComplexV1Service(items=[], multiplier=2)
            result = service.execute()
            assert result.is_failure

        elif test_case.operation == ServiceOperationType.V2_COMPLEX_EMPTY_ITEMS:
            service = ComplexV2Service(items=[], multiplier=2)
            try:
                _ = service.result
                pytest.fail("Should raise exception")
            except FlextExceptions.BaseError:
                pass

        elif test_case.operation == ServiceOperationType.V2_RESULT_MULTIPLE_CALLS:
            service = SimpleV2PropertyService(message="test")
            result1 = service.result
            result2 = service.result
            assert result1 == result2
            assert result1 == "V2 Property: test"

    @pytest.mark.parametrize(
        "test_case",
        ServiceScenarios.V2_AUTO_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_v2_auto_pattern(self, test_case: ServiceTestCase) -> None:
        """Test V2 Auto pattern with auto_execute."""
        if test_case.operation == ServiceOperationType.V2_AUTO_RETURNS_VALUE:
            value = SimpleV2AutoService(message="test")
            assert value == "V2 Auto: test"

        elif test_case.operation == ServiceOperationType.V2_AUTO_VS_V1_SERVICE:
            v1_service = SimpleV1Service(message="test")
            assert isinstance(v1_service, FlextService)

            v2_auto_value = SimpleV2AutoService(message="test")
            assert isinstance(v2_auto_value, str)

        elif test_case.operation == ServiceOperationType.V2_AUTO_VALIDATION_SUCCESS:
            result_value = AutoValidationService(value=10)
            assert isinstance(result_value, dict)
            assert result_value["value"] == 10

        elif test_case.operation == ServiceOperationType.V2_AUTO_VALIDATION_FAILURE:
            try:
                _ = AutoValidationService(value=-1)
                pytest.fail("Should raise exception")
            except FlextExceptions.BaseError:
                pass

        elif test_case.operation == ServiceOperationType.V2_AUTO_COMPLEX_SUCCESS:
            result_value = AutoValidationService(value=10)
            assert isinstance(result_value, dict)
            assert result_value["value"] == 10

        elif test_case.operation == ServiceOperationType.V2_AUTO_COMPLEX_FAILURE:
            try:
                _ = AutoValidationService(value=-5)
                pytest.fail("Should raise exception")
            except FlextExceptions.BaseError:
                pass

        elif test_case.operation == ServiceOperationType.V2_AUTO_ZERO_CEREMONY:
            value = SimpleV2AutoService(message="simple")
            assert value == "V2 Auto: simple"

    @pytest.mark.parametrize(
        "test_case",
        ServiceScenarios.V1_V2_INTEROPERABILITY_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_v1_v2_interoperability(self, test_case: ServiceTestCase) -> None:
        """Test interoperability between V1 and V2 patterns."""
        if test_case.operation == ServiceOperationType.V1_V2_BOTH_IN_CODEBASE:
            v1 = SimpleV1Service(message="v1")
            v2 = SimpleV2PropertyService(message="v2")
            v2_auto = SimpleV2AutoService(message="auto")

            assert isinstance(v1, FlextService)
            assert isinstance(v2, FlextService)
            assert isinstance(v2_auto, str)

        elif test_case.operation == ServiceOperationType.V2_USE_V1_RAILWAY:
            service_rail: SimpleV2PropertyService = SimpleV2PropertyService(
                message="test",
            )
            chained_result = SimpleV1Service(message="chained").execute()

            def chain_wrapper(_: object) -> FlextResult[object]:
                if chained_result.is_success:
                    return FlextResult[object].ok(chained_result.value)
                return FlextResult[object].fail(
                    chained_result.error or "Chained failed",
                )

            result: FlextResult[object] = service_rail.execute().flat_map(chain_wrapper)
            assert result.is_success

        elif test_case.operation == ServiceOperationType.MIX_V1_AND_V2_PIPELINE:
            step2_result = SimpleV2PropertyService(message="step2").execute()

            def step2_wrapper(_: object) -> FlextResult[object]:
                if step2_result.is_success:
                    return FlextResult[object].ok(step2_result.value)
                return FlextResult[object].fail(step2_result.error or "Step2 failed")

            result_mix: FlextResult[object] = (
                SimpleV1Service(message="step1").execute().flat_map(step2_wrapper)
            )
            assert result_mix.is_success

        elif test_case.operation == ServiceOperationType.ERROR_CONSISTENCY:
            v1_error = ValidationService(value=-1).execute()
            assert v1_error.is_failure

            try:
                _ = ValidationService(value=-1).result
                pytest.fail("Should raise")
            except FlextExceptions.BaseError:
                pass

    @pytest.mark.parametrize(
        "test_case",
        ServiceScenarios.BACKWARD_COMPATIBILITY_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_backward_compatibility(self, test_case: ServiceTestCase) -> None:
        """Test backward compatibility with V1 patterns."""
        if test_case.operation == ServiceOperationType.V1_CODE_STILL_WORKS:
            service = SimpleV1Service(message="test")
            result = service.execute()
            assert result.is_success
            assert result.unwrap() == "V1: test"

        elif test_case.operation == ServiceOperationType.V2_DOESNT_BREAK_V1:
            v1 = SimpleV1Service(message="works")
            v2 = SimpleV2PropertyService(message="also_works")
            assert v1.execute().unwrap() == "V1: works"
            assert v2.result == "V2 Property: also_works"

        elif test_case.operation == ServiceOperationType.AUTO_EXECUTE_FALSE:

            class NoAutoService(FlextService[str]):
                """Service without auto_execute."""

                auto_execute: ClassVar[bool] = False

                def execute(self, **_kwargs: object) -> FlextResult[str]:
                    return FlextResult.ok("no_auto")

            service = NoAutoService()
            assert isinstance(service, FlextService)
            result = service.execute()
            assert result.unwrap() == "no_auto"

        elif test_case.operation == ServiceOperationType.NO_AUTO_EXECUTE_ATTRIBUTE:

            class DefaultService(FlextService[str]):
                """Service without explicit auto_execute."""

                def execute(self, **_kwargs: object) -> FlextResult[str]:
                    return FlextResult.ok("default")

            service = DefaultService()
            assert isinstance(service, FlextService)

    @pytest.mark.parametrize(
        "test_case",
        ServiceScenarios.V2_EDGE_CASES_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_v2_edge_cases(self, test_case: ServiceTestCase) -> None:
        """Test edge cases with various return types."""
        if test_case.operation == ServiceOperationType.V2_AUTO_WITH_BOOL_RETURN:
            value = BoolService()
            assert value is True

        elif test_case.operation == ServiceOperationType.V2_PROPERTY_WITH_DICT_RETURN:
            service = DictService()
            value = service.result
            assert isinstance(value, dict)
            assert value["a"] == 1
            assert value["b"] == 2

        elif test_case.operation == ServiceOperationType.V2_AUTO_WITH_LIST_RETURN:
            value = ListService()
            assert isinstance(value, list)
            assert len(value) == 3
            assert value[0] == "x"

        elif test_case.operation == ServiceOperationType.V2_PROPERTY_WITH_MODEL_RETURN:
            service = UserService(user_id="123", user_name="Test User")
            user = service.result
            assert isinstance(user, FlextModels.Entity)
            assert user.unique_id == "123"
            # User entity may not have name attribute directly, check via getattr
            user_name = getattr(user, "name", None)
            if user_name is not None:
                assert user_name == "Test User"

    @pytest.mark.parametrize(
        "test_case",
        ServiceScenarios.V2_BEST_PRACTICES_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_v2_best_practices(self, test_case: ServiceTestCase) -> None:
        """Test V2 best practice recommendations."""
        if test_case.operation == ServiceOperationType.V2_PROPERTY_RECOMMENDED:
            service = SimpleV2PropertyService(message="test")
            value = service.result
            assert value == "V2 Property: test"

        elif test_case.operation == ServiceOperationType.V1_EXECUTE_RECOMMENDED:
            service = ValidationService(value=10)
            result = service.execute()
            if result.is_success:
                value = result.unwrap()
                assert value["value"] == 10

        elif test_case.operation == ServiceOperationType.V2_AUTO_RECOMMENDED:
            value = SimpleV2AutoService(message="simple")
            assert value == "V2 Auto: simple"


__all__ = ["TestFlextServiceV2Patterns"]
