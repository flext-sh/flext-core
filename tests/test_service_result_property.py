"""Tests for FlextService .result property (V2 zero-ceremony pattern).

Module: flext_core.service.FlextService[T]
Scope: .result property, V2 pattern, V1 backward compatibility, property behavior
Pattern: Railway-Oriented, zero-ceremony result access with exception handling

Tests validate:
- Success cases with direct result unwrapping
- Failure handling with exception raising
- Validation scenarios with multiple inputs
- Type inference for generic services
- Lazy evaluation and property behavior
- V1 and V2 pattern equivalence

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import cast

import pytest

from flext_core import FlextExceptions, FlextResult
from tests.fixtures.factories import (
    FailingService,
    GetUserService,
    ServiceTestCase,
    ServiceTestCases,
    User,
    ValidatingService,
)

# =========================================================================
# Test Suite - Result Property Pattern
# =========================================================================


class TestServiceResultProperty:
    """Unified test suite for FlextService .result property.

    Tests cover:
    - V2 zero-ceremony pattern (.result property)
    - V1 backward compatibility (.execute() method)
    - Property behavior (computed_field)
    - Type inference and lazy evaluation
    - Edge cases and error handling
    """

    # =====================================================================
    # V2 Pattern Tests - Zero-Ceremony Result Access
    # =====================================================================

    @pytest.mark.parametrize("case", ServiceTestCases.USER_SUCCESS)
    def test_result_property_returns_unwrapped_value(
        self, case: ServiceTestCase
    ) -> None:
        """V2: .result returns unwrapped domain result directly."""
        service = ServiceTestCases.create_service(case)
        assert isinstance(service, GetUserService)
        user = cast("User", service.result)

        assert isinstance(user, User)
        assert user.user_id == case.input_value

    @pytest.mark.parametrize("case", ServiceTestCases.VALIDATE_SUCCESS)
    def test_result_property_with_validation_success(
        self, case: ServiceTestCase
    ) -> None:
        """V2: Validation success returns unwrapped value."""
        service = ServiceTestCases.create_service(case)
        assert isinstance(service, ValidatingService)
        result = service.result

        assert isinstance(result, str)
        assert result == case.input_value.upper()

    @pytest.mark.parametrize("case", ServiceTestCases.VALIDATE_FAILURE)
    def test_result_property_with_validation_failure(
        self, case: ServiceTestCase
    ) -> None:
        """V2: Validation failure raises exception."""
        service = ServiceTestCases.create_service(case)
        assert isinstance(service, ValidatingService)

        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            _ = service.result

        assert case.expected_error and case.expected_error in str(exc_info.value)

    def test_result_property_raises_on_failure(self) -> None:
        """V2: Failures raise exceptions immediately."""
        service = FailingService(error_message="Test error")
        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            _ = service.result

        assert "Test error" in str(exc_info.value)

    @pytest.mark.parametrize("case", ServiceTestCases.USER_SUCCESS)
    def test_result_property_type_inference(self, case: ServiceTestCase) -> None:
        """V2: Type checkers infer correct type."""
        service = ServiceTestCases.create_service(case)
        assert isinstance(service, GetUserService)
        user: User = cast("User", service.result)

        assert isinstance(user, User)
        assert user.user_id == case.input_value

    @pytest.mark.parametrize("case", ServiceTestCases.USER_SUCCESS)
    def test_result_property_lazy_evaluation(self, case: ServiceTestCase) -> None:
        """V2: Property is lazily evaluated (executes only when accessed)."""
        service = ServiceTestCases.create_service(case)
        assert isinstance(service, GetUserService)
        assert hasattr(service, "result")

        user = cast("User", service.result)
        assert isinstance(user, User)
        assert user.user_id == case.input_value

    # =====================================================================
    # V1 Pattern Tests - Backward Compatibility
    # =====================================================================

    @pytest.mark.parametrize("case", ServiceTestCases.USER_SUCCESS)
    def test_v1_execute_still_works(self, case: ServiceTestCase) -> None:
        """V1: .execute() continues to work."""
        service = ServiceTestCases.create_service(case)
        assert isinstance(service, GetUserService)
        result = service.execute()

        assert isinstance(result, FlextResult)
        assert result.is_success

        user = result.unwrap()
        assert isinstance(user, User)
        assert user.user_id == case.input_value

    def test_v1_error_handling_with_flext_result(self) -> None:
        """V1: Error handling via FlextResult pattern."""
        service = FailingService(error_message="Test failure")
        result = service.execute()

        assert isinstance(result, FlextResult)
        assert result.is_failure
        assert "Test failure" in str(result.error)

    @pytest.mark.parametrize("case", ServiceTestCases.USER_SUCCESS)
    def test_v1_railway_pattern_composition(self, case: ServiceTestCase) -> None:
        """V1: Railway pattern composition with map."""
        service = ServiceTestCases.create_service(case)
        assert isinstance(service, GetUserService)
        result = (
            service.execute()
            .map(lambda user: user.name)
            .map(lambda name: str(name).upper())
            .map(lambda name: f"Hello, {name}!")
        )

        assert result.is_success
        greeting = result.unwrap()
        assert greeting == f"Hello, USER {case.input_value}!"

    @pytest.mark.parametrize("case", ServiceTestCases.USER_SUCCESS)
    def test_v2_and_v1_return_same_result(self, case: ServiceTestCase) -> None:
        """V2 and V1 should return equivalent results."""
        service1 = ServiceTestCases.create_service(case)
        service2 = ServiceTestCases.create_service(case)
        assert isinstance(service1, GetUserService)
        assert isinstance(service2, GetUserService)

        user_v2 = cast("User", service1.result)
        user_v1 = service2.execute().unwrap()

        assert user_v2.user_id == user_v1.user_id
        assert user_v2.name == user_v1.name
        assert user_v2.email == user_v1.email

    def test_v1_compatibility_edge_cases(self) -> None:
        """Test V1 compatibility edge cases."""
        v1_result = ValidatingService(value_input="hello").execute()
        assert v1_result.is_success
        assert v1_result.unwrap() == "HELLO"

        fail_result = FailingService(error_message="V1 fail").execute()
        assert fail_result.is_failure
        assert fail_result.error is not None

        railway = (
            GetUserService(user_id="railway")
            .execute()
            .map(lambda u: u.email)
            .filter(lambda e: "@" in str(e))
        )
        assert railway.is_success

    # =====================================================================
    # Property Behavior Tests - Computed Field
    # =====================================================================

    @pytest.mark.parametrize("case", ServiceTestCases.USER_SUCCESS)
    def test_result_is_computed_field(self, case: ServiceTestCase) -> None:
        """Verify .result is a Pydantic computed_field."""
        service = ServiceTestCases.create_service(case)
        assert isinstance(service, GetUserService)
        assert hasattr(service, "result")

        user = cast("User", service.result)
        assert isinstance(user, User)
        assert user.user_id == case.input_value

    @pytest.mark.parametrize("case", ServiceTestCases.USER_SUCCESS)
    def test_result_property_in_model_dump(self, case: ServiceTestCase) -> None:
        """Computed fields behavior in model_dump."""
        service = ServiceTestCases.create_service(case)
        assert isinstance(service, GetUserService)
        dump = service.model_dump()
        assert "user_id" in dump

        user = cast("User", service.result)
        assert isinstance(user, User)
        assert user.user_id == case.input_value

    def test_property_behavior_edge_cases(self) -> None:
        """Test property behavior edge cases."""
        service = GetUserService(user_id="prop")

        user1 = service.result
        user2 = service.result
        assert isinstance(user1, User)
        assert isinstance(user2, User)
        assert user1.user_id == user2.user_id

        assert hasattr(service, "result")

        dump = service.model_dump()
        assert isinstance(dump, dict)
        assert "user_id" in dump

    # =====================================================================
    # Edge Case Tests
    # =====================================================================

    def test_result_property_comprehensive_edge_cases(self) -> None:
        """Test comprehensive edge cases."""
        # Different service types
        user_result = GetUserService(user_id="edge").result
        assert isinstance(user_result, User)

        validation_result = ValidatingService(value_input="edge").result
        assert isinstance(validation_result, str)
        assert validation_result == "EDGE"

        # Failure edge case
        with pytest.raises(FlextExceptions.BaseError):
            FailingService(error_message="").result

        # Empty service operations
        service = FailingService(error_message="fail")
        assert isinstance(service.execute(), FlextResult)
        with pytest.raises(FlextExceptions.BaseError):
            _ = service.result
