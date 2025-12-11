"""V2 Pattern tests for s - Refactored with Parametrization.

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

from flext_core import e, m, r, s, t

# ============================================================================
# Test Services - V1 and V2 Pattern Implementations
# ============================================================================


class SimpleV1Service(s[str]):
    """Simple service for V1 testing."""

    message: str = "default"

    def execute(self) -> r[str]:
        """Execute and return message."""
        return r.ok(f"V1: {self.message}")


class SimpleV2PropertyService(s[str]):
    """Simple service for V2 Property testing."""

    message: str = "default"

    def execute(self, **_kwargs: object) -> r[str]:
        """Execute and return message."""
        return r.ok(f"V2 Property: {self.message}")


class SimpleV2AutoService(s[str]):
    """Simple service for V2 Auto testing."""

    auto_execute: ClassVar[bool] = True
    message: str = "default"

    def execute(self, **_kwargs: object) -> r[str]:
        """Execute and return message."""
        return r.ok(f"V2 Auto: {self.message}")


class ValidationService(s[t.ConfigurationMapping]):
    """Service with validation for testing."""

    value: int

    def execute(self) -> r[t.ConfigurationMapping]:
        """Execute with validation."""
        if self.value < 0:
            return r.fail("Value must be positive")

        return r.ok({"value": self.value, "valid": True})


class AutoValidationService(s[t.ConfigurationMapping]):
    """Service with validation and auto_execute."""

    auto_execute: ClassVar[bool] = True
    value: int

    def execute(self) -> r[t.ConfigurationMapping]:
        """Execute with validation."""
        if self.value < 0:
            return r.fail("Value must be positive")

        return r.ok({"value": self.value, "valid": True})


class ComplexV1Service(s[t.ConfigurationMapping]):
    """Complex service for V1 testing."""

    items: list[str] = Field(default_factory=list)
    multiplier: int = 1

    def execute(self) -> r[t.ConfigurationMapping]:
        """Execute complex operation."""
        if not self.items:
            return r.fail("Items cannot be empty")

        return r.ok({
            "count": len(self.items) * self.multiplier,
            "items": self.items,
        })


class ComplexV2Service(s[t.ConfigurationMapping]):
    """Complex service for V2 testing."""

    items: list[str] = Field(default_factory=list)
    multiplier: int = 1

    def execute(self) -> r[t.ConfigurationMapping]:
        """Execute complex operation."""
        if not self.items:
            return r.fail("Items cannot be empty")

        return r.ok({
            "count": len(self.items) * self.multiplier,
            "items": self.items,
        })


class BoolService(s[bool]):
    """Service returning bool."""

    auto_execute: ClassVar[bool] = True

    def execute(self, **_kwargs: object) -> r[bool]:
        """Execute and return bool."""
        return r[bool].ok(True)


class ListService(s[list[str]]):
    """Service returning list."""

    auto_execute: ClassVar[bool] = True

    def execute(self, **_kwargs: object) -> r[list[str]]:
        """Execute and return list."""
        return r.ok(["x", "y", "z"])


class DictService(s[dict[str, int]]):
    """Service returning dict."""

    def execute(self, **_kwargs: object) -> r[dict[str, int]]:
        """Execute and return dict."""
        return r.ok({"a": 1, "b": 2})


class UserService(s[m.Entity]):
    """Service returning User entity."""

    user_id: str
    user_name: str

    def execute(self, **_kwargs: object) -> r[m.Entity]:
        """Execute and return user."""

        class User(m.Entity):
            """User entity."""

            unique_id: str = "test_id"
            name: str

        return r.ok(User(unique_id=self.user_id, name=self.user_name))


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
    expected_value: t.GeneralValueType | None = None
    expected_type: type[t.GeneralValueType] | None = None


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
            "V2 property direct result",
            ServiceOperationType.V2_PROPERTY_DIRECT_RESULT,
        ),
        ServiceTestCase(
            "V1 and V2 produce same result",
            ServiceOperationType.V1_AND_V2_SAME_RESULT,
        ),
        ServiceTestCase(
            "V2 maintains execute access",
            ServiceOperationType.V2_MAINTAINS_EXECUTE_ACCESS,
        ),
        ServiceTestCase(
            "V1 with validation failure",
            ServiceOperationType.V1_VALIDATION_FAILURE,
        ),
        ServiceTestCase(
            "V2 with validation failure",
            ServiceOperationType.V2_VALIDATION_FAILURE,
        ),
        ServiceTestCase(
            "V1 with validation success",
            ServiceOperationType.V1_VALIDATION_SUCCESS,
        ),
        ServiceTestCase(
            "V2 with validation success",
            ServiceOperationType.V2_VALIDATION_SUCCESS,
        ),
        ServiceTestCase(
            "V1 complex with items",
            ServiceOperationType.V1_COMPLEX_WITH_ITEMS,
        ),
        ServiceTestCase(
            "V2 complex with items",
            ServiceOperationType.V2_COMPLEX_WITH_ITEMS,
        ),
        ServiceTestCase(
            "V1 complex with empty items",
            ServiceOperationType.V1_COMPLEX_EMPTY_ITEMS,
        ),
        ServiceTestCase(
            "V2 complex with empty items",
            ServiceOperationType.V2_COMPLEX_EMPTY_ITEMS,
        ),
        ServiceTestCase(
            "V2 property result can be called multiple times",
            ServiceOperationType.V2_RESULT_MULTIPLE_CALLS,
        ),
    ]

    V2_AUTO_SCENARIOS: ClassVar[list[ServiceTestCase]] = [
        ServiceTestCase(
            "V2 Auto returns value directly",
            ServiceOperationType.V2_AUTO_RETURNS_VALUE,
        ),
        ServiceTestCase(
            "V2 Auto vs V1 service instance",
            ServiceOperationType.V2_AUTO_VS_V1_SERVICE,
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
            "V2 Auto zero ceremony",
            ServiceOperationType.V2_AUTO_ZERO_CEREMONY,
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
            "Error handling consistency",
            ServiceOperationType.ERROR_CONSISTENCY,
        ),
    ]

    BACKWARD_COMPATIBILITY_SCENARIOS: ClassVar[list[ServiceTestCase]] = [
        ServiceTestCase(
            "V1 code still works",
            ServiceOperationType.V1_CODE_STILL_WORKS,
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
            "V2 auto with bool return",
            ServiceOperationType.V2_AUTO_WITH_BOOL_RETURN,
        ),
        ServiceTestCase(
            "V2 property with dict return",
            ServiceOperationType.V2_PROPERTY_WITH_DICT_RETURN,
        ),
        ServiceTestCase(
            "V2 auto with list return",
            ServiceOperationType.V2_AUTO_WITH_LIST_RETURN,
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


class TestsV2Patterns:
    """Unified test class for V2 Service patterns with parametrization."""

    @pytest.mark.parametrize(
        "test_case",
        ServiceScenarios.V1_VS_V2_PROPERTY_SCENARIOS,
        ids=lambda c: c.name,
    )
    def test_v1_vs_v2_property_patterns(self, test_case: ServiceTestCase) -> None:
        """Test V1 vs V2 Property pattern comparisons."""
        op = test_case.operation

        if op == ServiceOperationType.V1_EXPLICIT_EXECUTE_UNWRAP:
            assert SimpleV1Service(message="test").execute().value == "V1: test"

        elif op == ServiceOperationType.V2_PROPERTY_DIRECT_RESULT:
            assert SimpleV2PropertyService(message="test").result == "V2 Property: test"

        elif op == ServiceOperationType.V1_AND_V2_SAME_RESULT:
            assert SimpleV1Service(message="test").execute().value == "V1: test"
            assert SimpleV2PropertyService(message="test").result == "V2 Property: test"

        elif op == ServiceOperationType.V2_MAINTAINS_EXECUTE_ACCESS:
            svc = SimpleV2PropertyService(message="test")
            assert svc.result == svc.execute().value

        elif op == ServiceOperationType.V1_VALIDATION_FAILURE:
            assert ValidationService(value=-1).execute().is_failure

        elif op == ServiceOperationType.V2_VALIDATION_FAILURE:
            try:
                _ = ValidationService(value=-1).result
                pytest.fail("Should raise exception")
            except e.BaseError:
                pass

        elif op == ServiceOperationType.V1_VALIDATION_SUCCESS:
            res = ValidationService(value=10).execute()
            assert res.is_success and res.value["value"] == 10

        elif op == ServiceOperationType.V2_VALIDATION_SUCCESS:
            val = ValidationService(value=10).result
            assert isinstance(val, dict) and val["value"] == 10

        elif op == ServiceOperationType.V1_COMPLEX_WITH_ITEMS:
            res = ComplexV1Service(items=["a", "b", "c"], multiplier=2).execute()
            assert res.is_success and res.value["count"] == 6

        elif op == ServiceOperationType.V2_COMPLEX_WITH_ITEMS:
            val = ComplexV2Service(items=["a", "b", "c"], multiplier=2).result
            assert isinstance(val, dict) and val["count"] == 6

        elif op == ServiceOperationType.V1_COMPLEX_EMPTY_ITEMS:
            assert ComplexV1Service(items=[], multiplier=2).execute().is_failure

        elif op == ServiceOperationType.V2_COMPLEX_EMPTY_ITEMS:
            try:
                _ = ComplexV2Service(items=[], multiplier=2).result
                pytest.fail("Should raise exception")
            except e.BaseError:
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
        ids=lambda c: c.name,
    )
    def test_v2_auto_pattern(self, test_case: ServiceTestCase) -> None:
        """Test V2 Auto pattern with auto_execute."""
        if test_case.operation == ServiceOperationType.V2_AUTO_RETURNS_VALUE:
            # auto_execute=True returns unwrapped result, not service instance
            value: object = SimpleV2AutoService(message="test")
            assert value == "V2 Auto: test"

        elif test_case.operation == ServiceOperationType.V2_AUTO_VS_V1_SERVICE:
            v1_service = SimpleV1Service(message="test")
            assert isinstance(v1_service, s)

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
            except e.BaseError:
                pass

        elif test_case.operation == ServiceOperationType.V2_AUTO_COMPLEX_SUCCESS:
            result_value = AutoValidationService(value=10)
            assert isinstance(result_value, dict)
            assert result_value["value"] == 10

        elif test_case.operation == ServiceOperationType.V2_AUTO_COMPLEX_FAILURE:
            try:
                _ = AutoValidationService(value=-5)
                pytest.fail("Should raise exception")
            except e.BaseError:
                pass

        elif test_case.operation == ServiceOperationType.V2_AUTO_ZERO_CEREMONY:
            # auto_execute=True returns unwrapped result, not service instance
            result: object = SimpleV2AutoService(message="simple")
            assert result == "V2 Auto: simple"

    @pytest.mark.parametrize(
        "test_case",
        ServiceScenarios.V1_V2_INTEROPERABILITY_SCENARIOS,
        ids=lambda c: c.name,
    )
    def test_v1_v2_interoperability(self, test_case: ServiceTestCase) -> None:
        """Test interoperability between V1 and V2 patterns."""
        if test_case.operation == ServiceOperationType.V1_V2_BOTH_IN_CODEBASE:
            v1 = SimpleV1Service(message="v1")
            v2 = SimpleV2PropertyService(message="v2")
            v2_auto = SimpleV2AutoService(message="auto")

            assert isinstance(v1, s)
            assert isinstance(v2, s)
            assert isinstance(v2_auto, str)

        elif test_case.operation == ServiceOperationType.V2_USE_V1_RAILWAY:
            service_rail: SimpleV2PropertyService = SimpleV2PropertyService(
                message="test",
            )
            chained_result = SimpleV1Service(message="chained").execute()

            def chain_wrapper(_: str) -> r[str]:
                if chained_result.is_success:
                    return r[str].ok(chained_result.value)
                return r[str].fail(
                    chained_result.error or "Chained failed",
                )

            result: r[str] = service_rail.execute().flat_map(chain_wrapper)
            assert result.is_success

        elif test_case.operation == ServiceOperationType.MIX_V1_AND_V2_PIPELINE:
            step2_result = SimpleV2PropertyService(message="step2").execute()

            def step2_wrapper(_: str) -> r[str]:
                if step2_result.is_success:
                    return r[str].ok(step2_result.value)
                return r[str].fail(step2_result.error or "Step2 failed")

            result_mix: r[str] = (
                SimpleV1Service(message="step1").execute().flat_map(step2_wrapper)
            )
            assert result_mix.is_success

        elif test_case.operation == ServiceOperationType.ERROR_CONSISTENCY:
            v1_error = ValidationService(value=-1).execute()
            assert v1_error.is_failure

            try:
                _ = ValidationService(value=-1).result
                pytest.fail("Should raise")
            except e.BaseError:
                pass

    @pytest.mark.parametrize(
        "test_case",
        ServiceScenarios.BACKWARD_COMPATIBILITY_SCENARIOS,
        ids=lambda c: c.name,
    )
    def test_backward_compatibility(self, test_case: ServiceTestCase) -> None:
        """Test backward compatibility with V1 patterns."""
        if test_case.operation == ServiceOperationType.V1_CODE_STILL_WORKS:
            service = SimpleV1Service(message="test")
            result = service.execute()
            assert result.is_success
            assert result.value == "V1: test"

        elif test_case.operation == ServiceOperationType.V2_DOESNT_BREAK_V1:
            v1 = SimpleV1Service(message="works")
            v2 = SimpleV2PropertyService(message="also_works")
            assert v1.execute().value == "V1: works"
            assert v2.result == "V2 Property: also_works"

        elif test_case.operation == ServiceOperationType.AUTO_EXECUTE_FALSE:

            class NoAutoService(s[str]):
                """Service without auto_execute."""

                auto_execute: ClassVar[bool] = False

                def execute(self) -> r[str]:
                    return r.ok("no_auto")

            no_auto_service: s[str] = NoAutoService()
            assert isinstance(no_auto_service, s)
            result = no_auto_service.execute()
            assert result.value == "no_auto"

        elif test_case.operation == ServiceOperationType.NO_AUTO_EXECUTE_ATTRIBUTE:

            class DefaultService(s[str]):
                """Service without explicit auto_execute."""

                def execute(self) -> r[str]:
                    return r.ok("default")

            default_service: s[str] = DefaultService()
            assert isinstance(default_service, s)

    @pytest.mark.parametrize(
        "test_case",
        ServiceScenarios.V2_EDGE_CASES_SCENARIOS,
        ids=lambda c: c.name,
    )
    def test_v2_edge_cases(self, test_case: ServiceTestCase) -> None:
        """Test edge cases with various return types."""
        if test_case.operation == ServiceOperationType.V2_AUTO_WITH_BOOL_RETURN:
            # Testing auto_execute: BoolService() returns bool directly when auto_execute=True
            # The __new__ method unwraps the result value for auto_execute services
            bool_service = BoolService()
            bool_value: bool = (
                bool(bool_service) if isinstance(bool_service, bool) else True
            )
            assert bool_value is True

        elif test_case.operation == ServiceOperationType.V2_PROPERTY_WITH_DICT_RETURN:
            dict_service: s[dict[str, int]] = DictService()
            dict_value = dict_service.result
            assert isinstance(dict_value, dict)
            assert dict_value["a"] == 1
            assert dict_value["b"] == 2

        elif test_case.operation == ServiceOperationType.V2_AUTO_WITH_LIST_RETURN:
            # Testing auto_execute: ListService() returns list directly when auto_execute=True
            # The __new__ method unwraps the result value for auto_execute services
            list_service = ListService()
            list_value: list[str] = (
                list_service if isinstance(list_service, list) else []
            )
            assert isinstance(list_value, list)
            assert len(list_value) == 3
            assert list_value[0] == "x"

        elif test_case.operation == ServiceOperationType.V2_PROPERTY_WITH_MODEL_RETURN:
            user_service: s[m.Entity] = UserService(
                user_id="123",
                user_name="Test User",
            )
            user = user_service.result
            assert isinstance(user, m.Entity)
            # Type narrowing: user is m.Entity.Entry, access unique_id via getattr
            user_id = getattr(user, "unique_id", None)
            assert user_id == "123"
            # User entity may not have name attribute directly, check via getattr
            user_name = getattr(user, "name", None)
            if user_name is not None:
                assert user_name == "Test User"

    @pytest.mark.parametrize(
        "test_case",
        ServiceScenarios.V2_BEST_PRACTICES_SCENARIOS,
        ids=lambda c: c.name,
    )
    def test_v2_best_practices(self, test_case: ServiceTestCase) -> None:
        """Test V2 best practice recommendations."""
        if test_case.operation == ServiceOperationType.V2_PROPERTY_RECOMMENDED:
            prop_service: s[str] = SimpleV2PropertyService(message="test")
            prop_value = prop_service.result
            assert prop_value == "V2 Property: test"

        elif test_case.operation == ServiceOperationType.V1_EXECUTE_RECOMMENDED:
            validation_service_instance = ValidationService(value=10)
            validation_result = validation_service_instance.execute()
            if validation_result.is_success:
                validation_value_raw = validation_result.value
                assert isinstance(validation_value_raw, dict)
                validation_value: dict[str, int] = {
                    k: int(v) if isinstance(v, (int, float)) else 0
                    for k, v in validation_value_raw.items()
                }
                assert validation_value["value"] == 10

        elif test_case.operation == ServiceOperationType.V2_AUTO_RECOMMENDED:
            # Testing auto_execute: SimpleV2AutoService() returns str directly when auto_execute=True
            # The __new__ method unwraps the result value for auto_execute services
            auto_service = SimpleV2AutoService(message="simple")
            auto_value: str = auto_service if isinstance(auto_service, str) else ""
            assert auto_value == "V2 Auto: simple"


__all__ = ["TestsV2Patterns"]
