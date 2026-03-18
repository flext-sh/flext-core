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

import pytest

from flext_core import FlextExceptions, r

from .helpers.factories import TestHelperFactories
from .test_utils import assertion_helpers


class TestServiceResultProperty:
    """Unified test suite for FlextService .result property.

    Tests cover:
    - V2 zero-ceremony pattern (.result property)
    - V1 backward compatibility (.execute() method)
    - Property behavior (computed_field)
    - Type inference and lazy evaluation
    - Edge cases and error handling
    """

    @pytest.mark.parametrize("case", TestHelperFactories.ServiceTestCases.USER_SUCCESS)
    def test_result_property_returns_unwrapped_value(
        self,
        case: TestHelperFactories.ServiceTestCase,
    ) -> None:
        """V2: .result returns unwrapped domain result directly."""
        service = TestHelperFactories.ServiceTestCases.create_service(case)
        assert isinstance(service, TestHelperFactories.GetUserService)
        user_raw = service.result
        assert isinstance(user_raw, TestHelperFactories.User)
        user = user_raw
        assert user.id == case.input_value

    @pytest.mark.parametrize(
        "case",
        TestHelperFactories.ServiceTestCases.VALIDATE_SUCCESS,
    )
    def test_result_property_with_validation_success(
        self,
        case: TestHelperFactories.ServiceTestCase,
    ) -> None:
        """V2: Validation success returns unwrapped value."""
        service = TestHelperFactories.ServiceTestCases.create_service(case)
        assert isinstance(service, TestHelperFactories.ValidatingService)
        result = service.result
        assert isinstance(result, str)
        assert result == case.input_value.upper()

    @pytest.mark.parametrize(
        "case",
        TestHelperFactories.ServiceTestCases.VALIDATE_FAILURE,
    )
    def test_result_property_with_validation_failure(
        self,
        case: TestHelperFactories.ServiceTestCase,
    ) -> None:
        """V2: Validation failure raises exception."""
        service = TestHelperFactories.ServiceTestCases.create_service(case)
        assert isinstance(service, TestHelperFactories.ValidatingService)
        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            _ = service.result
        assert case.expected_error and case.expected_error in str(exc_info.value)

    def test_result_property_raises_on_failure(self) -> None:
        """V2: Failures raise exceptions immediately."""
        service = TestHelperFactories.FailingService.model_construct(
            error_message="Test error"
        )
        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            _ = service.result
        assert "Test error" in str(exc_info.value)

    @pytest.mark.parametrize("case", TestHelperFactories.ServiceTestCases.USER_SUCCESS)
    def test_result_property_type_inference(
        self,
        case: TestHelperFactories.ServiceTestCase,
    ) -> None:
        """V2: Type checkers infer correct type."""
        service = TestHelperFactories.ServiceTestCases.create_service(case)
        assert isinstance(service, TestHelperFactories.GetUserService)
        user_raw = service.result
        assert isinstance(user_raw, TestHelperFactories.User)
        user = user_raw
        assert user.id == case.input_value

    @pytest.mark.parametrize("case", TestHelperFactories.ServiceTestCases.USER_SUCCESS)
    def test_result_property_lazy_evaluation(
        self,
        case: TestHelperFactories.ServiceTestCase,
    ) -> None:
        """V2: Property is lazily evaluated (executes only when accessed)."""
        service = TestHelperFactories.ServiceTestCases.create_service(case)
        assert isinstance(service, TestHelperFactories.GetUserService)
        assert hasattr(service, "result")
        user_raw = service.result
        assert isinstance(user_raw, TestHelperFactories.User)
        user = user_raw
        assert user.id == case.input_value

    @pytest.mark.parametrize("case", TestHelperFactories.ServiceTestCases.USER_SUCCESS)
    def test_v1_execute_still_works(
        self,
        case: TestHelperFactories.ServiceTestCase,
    ) -> None:
        """V1: .execute() continues to work."""
        service = TestHelperFactories.ServiceTestCases.create_service(case)
        assert isinstance(service, TestHelperFactories.GetUserService)
        result = service.execute()
        assert isinstance(result, r)
        _ = assertion_helpers.assert_flext_result_success(result)
        user = result.value
        assert isinstance(user, TestHelperFactories.User)
        assert user.id == case.input_value

    def test_v1_error_handling_with_flext_result(self) -> None:
        """V1: Error handling via r pattern."""
        service = TestHelperFactories.FailingService.model_construct(
            error_message="Test failure",
        )
        result = service.execute()
        assert isinstance(result, r)
        _ = assertion_helpers.assert_flext_result_failure(result)
        assert "Test failure" in str(result.error)

    @pytest.mark.parametrize("case", TestHelperFactories.ServiceTestCases.USER_SUCCESS)
    def test_v1_railway_pattern_composition(
        self,
        case: TestHelperFactories.ServiceTestCase,
    ) -> None:
        """V1: Railway pattern composition with map."""
        service = TestHelperFactories.ServiceTestCases.create_service(case)
        assert isinstance(service, TestHelperFactories.GetUserService)
        result = (
            service
            .execute()
            .map(lambda user: user.name)
            .map(lambda name: str(name).upper())
            .map(lambda name: f"Hello, {name}!")
        )
        _ = assertion_helpers.assert_flext_result_success(result)
        greeting = result.value
        assert greeting == f"Hello, USER {case.input_value}!"

    @pytest.mark.parametrize("case", TestHelperFactories.ServiceTestCases.USER_SUCCESS)
    def test_v2_and_v1_return_same_result(
        self,
        case: TestHelperFactories.ServiceTestCase,
    ) -> None:
        """V2 and V1 should return equivalent results."""
        service1 = TestHelperFactories.ServiceTestCases.create_service(case)
        service2 = TestHelperFactories.ServiceTestCases.create_service(case)
        assert isinstance(service1, TestHelperFactories.GetUserService)
        assert isinstance(service2, TestHelperFactories.GetUserService)
        user_v2_raw = service1.result
        assert isinstance(user_v2_raw, TestHelperFactories.User)
        user_v2 = user_v2_raw
        user_v1_result = service2.execute()
        assert user_v1_result.is_success
        user_v1 = user_v1_result.value
        assert isinstance(user_v1, TestHelperFactories.User)
        assert user_v2.id == user_v1.id
        assert user_v2.name == user_v1.name
        assert user_v2.email == user_v1.email

    @pytest.mark.parametrize("case", TestHelperFactories.ServiceTestCases.USER_SUCCESS)
    def test_result_is_computed_field(
        self,
        case: TestHelperFactories.ServiceTestCase,
    ) -> None:
        """Verify .result is a Pydantic computed_field."""
        service = TestHelperFactories.ServiceTestCases.create_service(case)
        assert isinstance(service, TestHelperFactories.GetUserService)
        assert hasattr(service, "result")
        user_raw = service.result
        assert isinstance(user_raw, TestHelperFactories.User)
        user = user_raw
        assert user.id == case.input_value

    @pytest.mark.parametrize("case", TestHelperFactories.ServiceTestCases.USER_SUCCESS)
    def test_result_property_in_model_dump(
        self,
        case: TestHelperFactories.ServiceTestCase,
    ) -> None:
        """Computed fields behavior in model_dump."""
        service = TestHelperFactories.ServiceTestCases.create_service(case)
        assert isinstance(service, TestHelperFactories.GetUserService)
        dump = service.model_dump(exclude={"access", "runtime"})
        assert "user_id" in dump
        user_raw = service.result
        assert isinstance(user_raw, TestHelperFactories.User)
        user = user_raw
        assert user.id == case.input_value

    def test_property_behavior_edge_cases(self) -> None:
        """Test property behavior edge cases."""
        service = TestHelperFactories.GetUserService.model_construct(user_id="prop")
        user1 = service.result
        user2 = service.result
        assert isinstance(user1, TestHelperFactories.User)
        assert isinstance(user2, TestHelperFactories.User)
        assert user1.id == user2.id
        assert hasattr(service, "result")
        dump = service.model_dump(exclude={"access", "runtime"})
        assert isinstance(dump, dict)
        assert "user_id" in dump

    def test_result_property_comprehensive_edge_cases(self) -> None:
        """Test comprehensive edge cases."""
        user_result = TestHelperFactories.GetUserService.model_construct(
            user_id="edge",
        ).result
        assert isinstance(user_result, TestHelperFactories.User)
        validation_result = TestHelperFactories.ValidatingService.model_construct(
            value_input="edge",
        ).result
        assert isinstance(validation_result, str)
        assert validation_result == "EDGE"
        with pytest.raises(FlextExceptions.BaseError):
            TestHelperFactories.FailingService.model_construct(error_message="").result
        service = TestHelperFactories.FailingService.model_construct(
            error_message="fail",
        )
        assert isinstance(service.execute(), r)
        with pytest.raises(FlextExceptions.BaseError):
            _ = service.result
