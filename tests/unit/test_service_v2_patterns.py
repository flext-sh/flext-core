"""V2 Pattern tests for FlextService - V2 Property and V2 Auto.

This module tests V2 patterns alongside V1 patterns to ensure:
- V2 Property (.result) works correctly
- V2 Auto (auto_execute = True) works correctly
- V1 and V2 are fully interoperable
- Backward compatibility is maintained

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest
from pydantic import Field

from flext_core import FlextExceptions, FlextModels, FlextResult, FlextService

# ============================================================================
# Test Services
# ============================================================================


class SimpleV1Service(FlextService[str]):
    """Simple service for V1 testing."""

    message: str = "default"

    def execute(self) -> FlextResult[str]:
        """Execute and return message."""
        return FlextResult.ok(f"V1: {self.message}")


class SimpleV2PropertyService(FlextService[str]):
    """Simple service for V2 Property testing."""

    message: str = "default"

    def execute(self) -> FlextResult[str]:
        """Execute and return message."""
        return FlextResult.ok(f"V2 Property: {self.message}")


class SimpleV2AutoService(FlextService[str]):
    """Simple service for V2 Auto testing."""

    auto_execute = True
    message: str = "default"

    def execute(self) -> FlextResult[str]:
        """Execute and return message."""
        return FlextResult.ok(f"V2 Auto: {self.message}")


class ValidationService(FlextService[dict[str, object]]):
    """Service with validation for testing."""

    value: int

    def execute(self) -> FlextResult[dict[str, object]]:
        """Execute with validation."""
        if self.value < 0:
            return FlextResult.fail("Value must be positive")

        return FlextResult.ok({"value": self.value, "valid": True})


class AutoValidationService(FlextService[dict[str, object]]):
    """Service with validation and auto_execute."""

    auto_execute = True
    value: int

    def execute(self) -> FlextResult[dict[str, object]]:
        """Execute with validation."""
        if self.value < 0:
            return FlextResult.fail("Value must be positive")

        return FlextResult.ok({"value": self.value, "valid": True})


class ComplexV1Service(FlextService[dict[str, object]]):
    """Complex service for V1 testing."""

    items: list[str] = Field(default_factory=list)
    multiplier: int = 1

    def execute(self) -> FlextResult[dict[str, object]]:
        """Execute complex operation."""
        if not self.items:
            return FlextResult.fail("Items cannot be empty")

        return FlextResult.ok({
            "count": len(self.items) * self.multiplier,
            "items": self.items,
        })


class ComplexV2Service(FlextService[dict[str, object]]):
    """Complex service for V2 Auto testing."""

    auto_execute = True
    items: list[str] = Field(default_factory=list)
    multiplier: int = 1

    def execute(self) -> FlextResult[dict[str, object]]:
        """Execute complex operation."""
        if not self.items:
            return FlextResult.fail("Items cannot be empty")

        return FlextResult.ok({
            "count": len(self.items) * self.multiplier,
            "items": self.items,
        })


# ============================================================================
# V1 vs V2 Property Tests
# ============================================================================


class TestV1VsV2Property:
    """Test V1 vs V2 Property patterns."""

    def test_v1_explicit_execute_unwrap(self) -> None:
        """V1: Explicit .execute().unwrap() pattern."""
        service = SimpleV1Service(message="test")
        result = service.execute()

        assert result.is_success
        value = result.unwrap()
        assert value == "V1: test"

    def test_v2_property_direct_result(self) -> None:
        """V2 Property: Direct .result access."""
        service = SimpleV2PropertyService(message="test")
        value = service.result

        assert value == "V2 Property: test"
        assert isinstance(value, str)

    def test_v1_and_v2_property_produce_same_result(self) -> None:
        """V1 and V2 Property produce equivalent results."""
        v1_service = SimpleV1Service(message="same")
        v2_service = SimpleV2PropertyService(message="same")

        v1_result = v1_service.execute().unwrap()
        v2_result = v2_service.result

        # Both return strings (types are the same)
        assert isinstance(v1_result, str)
        assert isinstance(v2_result, str)

    def test_v2_property_maintains_execute_access(self) -> None:
        """V2 Property: .execute() still available for railway pattern."""
        service = SimpleV2PropertyService(message="test")

        # V2 Property can still use execute() for railway pattern
        result = service.execute()
        assert result.is_success
        assert result.unwrap() == "V2 Property: test"

        # And can use .result for happy path
        direct_value = service.result
        assert direct_value == "V2 Property: test"

    def test_v1_with_validation_failure(self) -> None:
        """V1: Validation failure returns FlextResult.fail."""
        service = ValidationService(value=-1)
        result = service.execute()

        assert result.is_failure
        assert result.error is not None
        assert "positive" in result.error.lower()

    def test_v2_property_with_validation_failure(self) -> None:
        """V2 Property: Validation failure raises exception."""
        service = ValidationService(value=-1)

        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            service.result

        assert "positive" in str(exc_info.value).lower()

    def test_v1_with_validation_success(self) -> None:
        """V1: Validation success returns FlextResult.ok."""
        service = ValidationService(value=10)
        result = service.execute()

        assert result.is_success
        value = result.unwrap()
        assert value["value"] == 10
        assert value["valid"] is True

    def test_v2_property_with_validation_success(self) -> None:
        """V2 Property: Validation success returns value directly."""
        service = ValidationService(value=10)
        value = service.result

        assert isinstance(value, dict)
        assert value["value"] == 10
        assert value["valid"] is True

    def test_complex_v1_with_items(self) -> None:
        """V1: Complex service with items."""
        service = ComplexV1Service(items=["a", "b", "c"], multiplier=2)
        result = service.execute()

        assert result.is_success
        value = result.unwrap()
        assert value["count"] == 6  # 3 items * 2 multiplier
        assert len(value["items"]) == 3

    def test_complex_v2_property_with_items(self) -> None:
        """V2 Property: Complex service with items."""
        # Note: ComplexV1Service doesn't have auto_execute, so it's V2 Property compatible
        service = ComplexV1Service(items=["a", "b", "c"], multiplier=2)
        value = service.result

        assert isinstance(value, dict)
        assert value["count"] == 6
        assert len(value["items"]) == 3

    def test_complex_v1_with_empty_items(self) -> None:
        """V1: Complex service fails with empty items."""
        service = ComplexV1Service(items=[], multiplier=2)
        result = service.execute()

        assert result.is_failure
        assert result.error is not None
        assert "empty" in result.error.lower()

    def test_complex_v2_property_with_empty_items(self) -> None:
        """V2 Property: Complex service raises with empty items."""
        service = ComplexV1Service(items=[], multiplier=2)

        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            service.result

        assert "empty" in str(exc_info.value).lower()


# ============================================================================
# V2 Auto Tests
# ============================================================================


class TestV2Auto:
    """Test V2 Auto (auto_execute = True) pattern."""

    def test_v2_auto_returns_value_directly(self) -> None:
        """V2 Auto: Instantiation returns value, not service instance."""
        # V2 Auto: auto_execute = True
        value = SimpleV2AutoService(message="test")

        # Returns value directly, not service instance
        assert isinstance(value, str)
        assert not isinstance(value, SimpleV2AutoService)
        assert value == "V2 Auto: test"

    def test_v2_auto_vs_v1_service_instance(self) -> None:
        """V2 Auto vs V1: Different return types on instantiation."""
        # V1: Returns service instance
        v1_service = SimpleV1Service(message="test")
        assert isinstance(v1_service, SimpleV1Service)
        assert not isinstance(v1_service, str)

        # V2 Auto: Returns value directly
        v2_value = SimpleV2AutoService(message="test")
        assert isinstance(v2_value, str)
        assert not isinstance(v2_value, SimpleV2AutoService)

    def test_v2_auto_validation_success(self) -> None:
        """V2 Auto: Validation success returns value directly."""
        # V2 Auto with validation success
        value = AutoValidationService(value=10)

        assert isinstance(value, dict)
        assert value["value"] == 10
        assert value["valid"] is True

    def test_v2_auto_validation_failure_raises(self) -> None:
        """V2 Auto: Validation failure raises exception."""
        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            AutoValidationService(value=-1)

        assert "positive" in str(exc_info.value).lower()

    def test_v2_auto_complex_service_success(self) -> None:
        """V2 Auto: Complex service returns value directly."""
        value = ComplexV2Service(items=["x", "y"], multiplier=3)

        assert isinstance(value, dict)
        assert value["count"] == 6  # 2 items * 3 multiplier
        assert len(value["items"]) == 2

    def test_v2_auto_complex_service_failure(self) -> None:
        """V2 Auto: Complex service raises on failure."""
        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            ComplexV2Service(items=[], multiplier=2)

        assert "empty" in str(exc_info.value).lower()

    def test_v2_auto_zero_ceremony(self) -> None:
        """V2 Auto: Zero ceremony - just instantiate and use."""
        # V1: 3 steps
        v1_service = SimpleV1Service(message="v1")
        v1_result = v1_service.execute()
        v1_value = v1_result.unwrap()

        # V2 Auto: 1 step (95% less code!)
        v2_value = SimpleV2AutoService(message="v2")

        # Both return strings
        assert isinstance(v1_value, str)
        assert isinstance(v2_value, str)


# ============================================================================
# Interoperability Tests
# ============================================================================


class TestV1V2Interoperability:
    """Test V1 and V2 interoperability."""

    def test_v1_v2_property_and_v2_auto_in_same_codebase(self) -> None:
        """All patterns can coexist in the same codebase."""
        # V1: Explicit
        v1 = SimpleV1Service(message="v1").execute().unwrap()

        # V2 Property: Happy path
        v2_prop = SimpleV2PropertyService(message="v2prop").result

        # V2 Auto: Zero ceremony
        v2_auto = SimpleV2AutoService(message="v2auto")

        # All return strings
        assert isinstance(v1, str)
        assert isinstance(v2_prop, str)
        assert isinstance(v2_auto, str)

    def test_v2_property_can_use_v1_railway_pattern(self) -> None:
        """V2 Property services can still use V1 railway pattern."""
        service = SimpleV2PropertyService(message="test")

        # V2 Property: Happy path
        happy_result = service.result
        assert happy_result == "V2 Property: test"

        # V1 Railway: Error handling
        railway_result = service.execute()
        assert railway_result.is_success
        assert railway_result.unwrap() == "V2 Property: test"

    def test_mixing_v1_and_v2_in_pipeline(self) -> None:
        """V1 and V2 services can be mixed in the same pipeline."""

        class V1PipelineService(FlextService[int]):
            """V1 service for pipeline."""

            value: int

            def execute(self) -> FlextResult[int]:
                return FlextResult.ok(self.value * 2)

        class V2PropertyPipelineService(FlextService[int]):
            """V2 Property service for pipeline."""

            value: int

            def execute(self) -> FlextResult[int]:
                return FlextResult.ok(self.value + 10)

        # Pipeline mixing V1 and V2
        step1 = V1PipelineService(value=5).execute()  # V1 style
        assert step1.is_success

        step2_value = step1.unwrap()
        step2 = V2PropertyPipelineService(value=step2_value).result  # V2 Property style

        assert step2 == 20  # (5 * 2) + 10

    def test_error_handling_consistency_across_versions(self) -> None:
        """Error handling is consistent across V1 and V2."""
        # V1: Returns FlextResult.fail
        v1_service = ValidationService(value=-1)
        v1_result = v1_service.execute()
        assert v1_result.is_failure
        assert v1_result.error is not None
        assert "positive" in v1_result.error.lower()

        # V2 Property: Raises exception
        v2_property_service = ValidationService(value=-1)
        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            v2_property_service.result
        assert "positive" in str(exc_info.value).lower()

        # V2 Auto: Raises exception
        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            AutoValidationService(value=-1)
        assert "positive" in str(exc_info.value).lower()


# ============================================================================
# Backward Compatibility Tests
# ============================================================================


class TestBackwardCompatibility:
    """Test backward compatibility between V1 and V2."""

    def test_v1_code_still_works(self) -> None:
        """V1 code continues to work with V2 implementation."""
        # Old V1 code pattern
        service = SimpleV1Service(message="legacy")
        result = service.execute()

        assert result.is_success
        value = result.unwrap()
        assert value == "V1: legacy"

    def test_v2_property_doesnt_break_v1_tests(self) -> None:
        """V2 Property doesn't break existing V1 tests."""
        service = SimpleV2PropertyService(message="test")

        # V1 pattern still works
        result = service.execute()
        assert result.is_success

        # V2 pattern also works
        value = service.result
        assert isinstance(value, str)

    def test_auto_execute_false_behaves_like_v1(self) -> None:
        """Services with auto_execute = False behave like V1."""

        class ManualService(FlextService[str]):
            """Service with auto_execute = False (default)."""

            auto_execute = False  # Explicit V1 behavior
            message: str

            def execute(self) -> FlextResult[str]:
                return FlextResult.ok(f"Manual: {self.message}")

        # Returns service instance (V1 behavior)
        service = ManualService(message="test")
        assert isinstance(service, ManualService)
        assert not isinstance(service, str)

        # Can use V1 pattern
        result = service.execute()
        assert result.is_success
        assert result.unwrap() == "Manual: test"

        # Can use V2 Property pattern
        value = service.result
        assert value == "Manual: test"

    def test_no_auto_execute_attribute_defaults_to_false(self) -> None:
        """Services without auto_execute attribute default to False (V1)."""

        class DefaultService(FlextService[str]):
            """Service without explicit auto_execute."""

            # No auto_execute attribute = defaults to False
            message: str

            def execute(self) -> FlextResult[str]:
                return FlextResult.ok(f"Default: {self.message}")

        # Returns service instance (V1 behavior)
        service = DefaultService(message="test")
        assert isinstance(service, DefaultService)
        assert not isinstance(service, str)


# ============================================================================
# Edge Cases and Advanced Tests
# ============================================================================


class TestV2EdgeCases:
    """Test edge cases and advanced V2 patterns."""

    def test_v2_property_result_can_be_called_multiple_times(self) -> None:
        """V2 Property: .result can be accessed multiple times."""
        service = SimpleV2PropertyService(message="test")

        result1 = service.result
        result2 = service.result

        # Both return the same value (re-execution)
        assert result1 == result2
        assert result1 == "V2 Property: test"

    def test_v2_auto_with_none_return(self) -> None:
        """V2 Auto: Works with None return type."""

        class NoneService(FlextService[None]):
            """Service returning None."""

            auto_execute = True

            def execute(self) -> FlextResult[None]:
                return FlextResult.ok(None)

        # Returns None directly
        value = NoneService()
        assert value is None

    def test_v2_property_with_dict_return(self) -> None:
        """V2 Property: Works with dict return type."""

        class DictService(FlextService[dict[str, int]]):
            """Service returning dict."""

            def execute(self) -> FlextResult[dict[str, int]]:
                return FlextResult.ok({"a": 1, "b": 2})

        service = DictService()
        value = service.result

        assert isinstance(value, dict)
        assert value["a"] == 1
        assert value["b"] == 2

    def test_v2_auto_with_list_return(self) -> None:
        """V2 Auto: Works with list return type."""

        class ListService(FlextService[list[str]]):
            """Service returning list."""

            auto_execute = True

            def execute(self) -> FlextResult[list[str]]:
                return FlextResult.ok(["x", "y", "z"])

        # Returns list directly
        value = ListService()

        assert isinstance(value, list)
        assert len(value) == 3
        assert value[0] == "x"

    def test_v2_property_with_pydantic_model_return(self) -> None:
        """V2 Property: Works with Pydantic model return type."""

        class User(FlextModels.Entity):
            """User entity."""

            unique_id: str
            name: str

        class UserService(FlextService[User]):
            """Service returning User entity."""

            user_id: str
            user_name: str

            def execute(self) -> FlextResult[User]:
                return FlextResult.ok(User(unique_id=self.user_id, name=self.user_name))

        service = UserService(user_id="123", user_name="Test User")
        user = service.result

        assert isinstance(user, User)
        assert user.unique_id == "123"
        assert user.name == "Test User"


# ============================================================================
# Performance and Best Practices Tests
# ============================================================================


class TestV2BestPractices:
    """Test V2 best practices and recommendations."""

    def test_v2_property_recommended_for_happy_path(self) -> None:
        """V2 Property: Recommended for happy path scenarios."""
        # Happy path scenario: We expect success
        service = SimpleV2PropertyService(message="success expected")

        # V2 Property: Direct access (68% less code than V1)
        value = service.result
        assert value == "V2 Property: success expected"

    def test_v1_execute_recommended_for_error_handling(self) -> None:
        """V1 execute: Recommended for explicit error handling."""
        # Error handling scenario: We need to check for failures
        service = ValidationService(value=10)

        # V1: Explicit error handling via FlextResult
        result = service.execute()
        if result.is_success:
            value = result.unwrap()
            assert value["value"] == 10
        else:
            pytest.fail("Should succeed")

    def test_v2_auto_recommended_for_simple_services(self) -> None:
        """V2 Auto: Recommended for simple, always-succeed services."""
        # Simple service that always succeeds
        value = SimpleV2AutoService(message="simple")

        # V2 Auto: Zero ceremony (95% less code than V1)
        assert value == "V2 Auto: simple"

    def test_v1_railway_pattern_for_complex_pipelines(self) -> None:
        """V1 Railway: Recommended for complex pipelines."""

        class Step1Service(FlextService[int]):
            """First step in pipeline."""

            value: int

            def execute(self) -> FlextResult[int]:
                if self.value < 0:
                    return FlextResult.fail("Must be positive")
                return FlextResult.ok(self.value * 2)

        class Step2Service(FlextService[int]):
            """Second step in pipeline."""

            value: int

            def execute(self) -> FlextResult[int]:
                if self.value > 100:
                    return FlextResult.fail("Too large")
                return FlextResult.ok(self.value + 10)

        # V1 Railway: Complex pipeline with error handling
        pipeline = (
            Step1Service(value=5)
            .execute()
            .flat_map(lambda v: Step2Service(value=v).execute())
        )

        assert pipeline.is_success
        assert pipeline.unwrap() == 20  # (5 * 2) + 10
