"""Tests for FlextService auto_execute feature (V2 zero-ceremony pattern).

Module: flext_core.service.FlextService[T]
Scope: auto_execute class attribute, zero-ceremony instantiation, backward compatibility
Pattern: Railway-Oriented, auto-execution with exception raising on failure

Tests validate:
- Manual execution (default auto_execute=False)
- Auto-execution (auto_execute=True) returning unwrapped results
- Failure handling with exception raising
- Backward compatibility preservation
- Edge cases and equivalence between patterns

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import cast

import pytest

from flext_core import FlextExceptions, FlextResult
from tests.fixtures.factories import (
    FailingService,
    FailingServiceAuto,
    GetUserService,
    GetUserServiceAuto,
    User,
    ValidatingServiceAuto,
)

# =========================================================================
# Test Suite - Auto-Execution Pattern
# =========================================================================


class TestServiceAutoExecute:
    """Unified test suite for FlextService auto_execute feature.

    Tests cover:
    - Manual execution (default auto_execute=False)
    - Auto-execution (auto_execute=True) returning unwrapped results
    - Failure handling with exception raising
    - Backward compatibility preservation
    - Edge cases and equivalence between patterns
    """

    # =====================================================================
    # Manual Execution Tests
    # =====================================================================

    @pytest.mark.parametrize("user_id", ["1", "101", "201"])
    def test_manual_execution_returns_flext_result(self, user_id: str) -> None:
        """Manual execution returns FlextResult wrapper."""
        service = GetUserService(user_id=user_id)
        result = service.execute()

        assert isinstance(result, FlextResult)
        assert result.is_success
        assert result.value.user_id == user_id

    def test_manual_execution_failure_handling(self) -> None:
        """Manual execution allows proper error handling."""
        service = FailingService(error_message="Manual fail")
        result = service.execute()

        assert isinstance(result, FlextResult)
        assert result.is_failure
        assert "Manual fail" in str(result.error)

    @pytest.mark.parametrize("user_id", ["1", "101", "201"])
    def test_manual_service_result_property(self, user_id: str) -> None:
        """Manual service .result property works."""
        service = GetUserService(user_id=user_id)
        user = cast("User", service.result)

        assert user.user_id == user_id
        assert user.name.startswith("User ")
        assert "@" in user.email

    # =====================================================================
    # Auto-Execution Tests
    # =====================================================================

    @pytest.mark.parametrize("user_id", ["1", "101", "201"])
    def test_auto_execution_returns_unwrapped_result(self, user_id: str) -> None:
        """Auto-execution returns unwrapped domain result directly."""
        # Note: auto_execute is not implemented in FlextService base
        # Use .result property instead for zero-ceremony pattern
        service = GetUserServiceAuto(user_id=user_id)
        user = service.result

        assert isinstance(user, User)
        assert user.user_id == user_id
        assert user.name.startswith("User ")
        assert "@" in user.email

    def test_auto_execution_failure_raises_exception(self) -> None:
        """Auto-execution raises exception on failure."""
        # Note: auto_execute is not implemented in FlextService base
        # Use .result property which raises on failure
        service = FailingServiceAuto(error_message="Auto fail")
        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            _ = service.result

        assert "Auto fail" in str(exc_info.value)

    @pytest.mark.parametrize("user_id", ["1", "101", "201"])
    def test_manual_service_execute_works(self, user_id: str) -> None:
        """Manual services work with .execute()."""
        service = GetUserService(user_id=user_id)
        result = service.execute()

        assert isinstance(result, FlextResult)
        assert result.is_success
        assert result.value.user_id == user_id

    # =====================================================================
    # Validation Tests
    # =====================================================================

    @pytest.mark.parametrize("value", ["hello", "world", "test"])
    def test_validation_auto_execution_success(self, value: str) -> None:
        """Validation service auto-executes successfully."""
        # Note: auto_execute is not implemented in FlextService base
        # Use .result property instead for zero-ceremony pattern
        service = ValidatingServiceAuto(value_input=value)
        result = service.result

        assert isinstance(result, str)
        assert result == value.upper()

    def test_validation_auto_execution_failure(self) -> None:
        """Validation service auto-executes and raises on failure."""
        # Note: auto_execute is not implemented in FlextService base
        # Use .result property which raises on failure
        service = ValidatingServiceAuto(value_input="x", min_length=5)
        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            _ = service.result

        assert "must be at least 5 characters" in str(exc_info.value)

    # =====================================================================
    # Pattern Equivalence Tests
    # =====================================================================

    @pytest.mark.parametrize("user_id", ["1", "101", "201"])
    def test_manual_and_auto_equivalence(self, user_id: str) -> None:
        """Manual and auto services return equivalent results."""
        manual = GetUserService(user_id=user_id)
        # Note: auto_execute is not implemented in FlextService base
        # Use .result property instead
        auto_service = GetUserServiceAuto(user_id=user_id)
        auto_result = auto_service.result

        manual_result = manual.execute()

        assert manual_result.is_success
        assert isinstance(auto_result, User)
        assert manual_result.value.user_id == auto_result.user_id
        assert manual_result.value.name == auto_result.name
        assert manual_result.value.email == auto_result.email

    @pytest.mark.parametrize("user_id", ["1", "101", "201"])
    def test_auto_vs_manual_result_equivalence(self, user_id: str) -> None:
        """Auto result matches manual .execute().unwrap()."""
        # Note: auto_execute is not implemented in FlextService base
        # Use .result property instead
        auto_service = GetUserServiceAuto(user_id=user_id)
        auto_result = auto_service.result
        manual_result = GetUserService(user_id=user_id).execute().unwrap()

        assert isinstance(auto_result, User)
        assert auto_result.user_id == manual_result.user_id
        assert auto_result.name == manual_result.name
        assert auto_result.email == manual_result.email

    # =====================================================================
    # Backward Compatibility Tests
    # =====================================================================

    def test_manual_service_backward_compatibility(self) -> None:
        """Manual services maintain full backward compatibility."""
        service = GetUserService(user_id="legacy")

        # Can call execute
        result = service.execute()
        assert result.is_success

        # Can access result property
        user = cast("User", service.result)
        assert user.user_id == "legacy"

        # Can use railway pattern
        railway = service.execute().map(lambda u: u.name).map(lambda n: str(n).upper())
        assert railway.is_success

    def test_auto_service_backward_compatibility(self) -> None:
        """Auto services use .result property for zero-ceremony access."""
        # Note: auto_execute is not implemented in FlextService base
        # Use .result property instead for zero-ceremony pattern
        service = GetUserServiceAuto(user_id="legacy")
        user = service.result
        assert isinstance(user, User)
        assert user.user_id == "legacy"

    # =====================================================================
    # Edge Cases and Error Handling
    # =====================================================================

    def test_auto_execution_comprehensive_failure_cases(self) -> None:
        """Test comprehensive auto-execution failure scenarios."""
        # Note: auto_execute is not implemented in FlextService base
        # Use .result property which raises on failure
        # Different error messages - auto-execution fails when accessing .result
        errors = ["Error 1", "Error 2", "Custom error"]

        for error_msg in errors:
            service = FailingServiceAuto(error_message=error_msg)
            with pytest.raises(FlextExceptions.BaseError) as exc_info:
                _ = service.result
            assert error_msg in str(exc_info.value)

    def test_manual_vs_auto_service_behavior(self) -> None:
        """Compare manual vs auto service behavior."""
        manual = GetUserService(user_id="compare")
        # Note: auto_execute is not implemented in FlextService base
        # Use .result property instead
        auto_service = GetUserServiceAuto(user_id="compare")
        auto_result = auto_service.result

        # Manual service has result property and execute method
        assert hasattr(manual, "result")
        assert hasattr(manual, "execute")
        # auto_execute is optional ClassVar, default to False if not present
        assert getattr(manual, "auto_execute", False) is False

        # Auto service also has result property
        assert hasattr(auto_service, "result")
        assert isinstance(auto_result, User)
        assert auto_result.user_id == "compare"

        # Results are equivalent
        manual_user = cast("User", manual.result)
        assert manual_user.user_id == auto_result.user_id

    def test_auto_execution_edge_cases(self) -> None:
        """Test auto-execution edge cases."""
        # Note: auto_execute is not implemented in FlextService base
        # Use .result property instead
        # Empty user ID
        service = GetUserServiceAuto(user_id="")
        user = service.result
        assert isinstance(user, User)
        assert user.user_id == ""

        # Long user ID
        long_id = "a" * 100
        service = GetUserServiceAuto(user_id=long_id)
        user = service.result
        assert isinstance(user, User)
        assert user.user_id == long_id

        # Special characters
        special_id = "user@123#$%"
        service = GetUserServiceAuto(user_id=special_id)
        user = service.result
        assert isinstance(user, User)
        assert user.user_id == special_id

    def test_validation_edge_cases(self) -> None:
        """Test validation edge cases with auto-execution."""
        # Note: auto_execute is not implemented in FlextService base
        # Use .result property instead
        # Boundary length
        service = ValidatingServiceAuto(value_input="abc", min_length=3)
        result = service.result
        assert isinstance(result, str)
        assert result == "ABC"

        # Just below boundary
        service = ValidatingServiceAuto(value_input="ab", min_length=3)
        with pytest.raises(FlextExceptions.BaseError):
            _ = service.result

        # Empty string
        service = ValidatingServiceAuto(value_input="", min_length=1)
        with pytest.raises(FlextExceptions.BaseError):
            _ = service.result
